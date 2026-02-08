"""Knowledge Integrator — Merge and load facts from all sources.

Combines ConceptNet, WordNet, and SimpleWiki facts into a unified
knowledge base with deduplication, conflict resolution, and batch loading.

Example:
    >>> integrator = KnowledgeIntegrator()
    >>> integrator.integrate_all_sources()
    >>> integrator.load_into_memory(memory)
    >>> print(integrator.get_statistics())
"""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory


# Fact tuple: (subject, relation, object, confidence)
Fact = tuple[str, str, str, float]


class KnowledgeIntegrator:
    """Merge facts from multiple knowledge sources.
    
    Handles deduplication, conflict resolution, and efficient
    batch loading into VectorMemory.
    
    Args:
        cache_dir: Directory for cached data
        data_dir: Directory for output knowledge base
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        data_dir: str = "data",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.data_dir = Path(data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._kb_path = self.data_dir / "knowledge_base.json.gz"
        
        # Unified facts
        self._facts: list[Fact] = []
        self._source_counts: dict[str, int] = {}
        self._conflicts: list[tuple[Fact, Fact, str]] = []
    
    @property
    def facts(self) -> list[Fact]:
        return self._facts
    
    @property
    def fact_count(self) -> int:
        return len(self._facts)
    
    def integrate_all_sources(
        self,
        sources: list[str] | None = None,
        max_facts_per_source: int | None = None,
        use_cache: bool = True,
        wordnet_max_synsets: int = 10_000,
        conceptnet_min_confidence: float = 1.0,
    ) -> None:
        """Load and merge facts from all specified sources.
        
        Args:
            sources: Which sources to load ('conceptnet', 'wordnet', 'simplewiki')
                     None = all sources
            max_facts_per_source: Limit per source (for testing)
            use_cache: Use cached data from individual loaders
            wordnet_max_synsets: Max WordNet synsets to process (default 10000)
            conceptnet_min_confidence: Min confidence for ConceptNet facts (default 1.0)
        """
        if sources is None:
            sources = ['conceptnet', 'wordnet', 'simplewiki']
        
        all_facts: list[tuple[Fact, str]] = []  # (fact, source_name)
        
        t0 = time.time()
        
        for source in sources:
            st = time.time()
            loader_facts = self._load_source(
                source, max_facts_per_source, use_cache,
                wordnet_max_synsets=wordnet_max_synsets,
                conceptnet_min_confidence=conceptnet_min_confidence,
            )
            elapsed = time.time() - st
            
            self._source_counts[source] = len(loader_facts)
            for f in loader_facts:
                all_facts.append((f, source))
            
            print(f"  [{source}] {len(loader_facts):,} facts in {elapsed:.1f}s")
        
        # Merge and deduplicate
        self._facts, self._conflicts = self._merge_facts(all_facts)
        
        total_time = time.time() - t0
        print(f"\nIntegration complete:")
        print(f"  Total facts: {len(self._facts):,}")
        print(f"  Conflicts: {len(self._conflicts):,}")
        print(f"  Time: {total_time:.1f}s")
    
    def integrate_sample(self) -> None:
        """Load sample facts from all sources (no downloads)."""
        from prism.data.loaders.conceptnet_loader import ConceptNetLoader
        from prism.data.loaders.wordnet_loader import WordNetLoader
        from prism.data.loaders.simplewiki_loader import SimpleWikiLoader
        
        all_facts: list[tuple[Fact, str]] = []
        
        cn_facts = ConceptNetLoader().get_sample(100)
        self._source_counts['conceptnet'] = len(cn_facts)
        for f in cn_facts:
            all_facts.append((f, 'conceptnet'))
        
        # WordNet sample requires NLTK — wrap in try
        try:
            wn_facts = WordNetLoader().get_sample(100)
            self._source_counts['wordnet'] = len(wn_facts)
            for f in wn_facts:
                all_facts.append((f, 'wordnet'))
        except Exception as e:
            print(f"  WordNet sample skipped: {e}")
            self._source_counts['wordnet'] = 0
        
        sw_facts = SimpleWikiLoader().get_sample(50)
        self._source_counts['simplewiki'] = len(sw_facts)
        for f in sw_facts:
            all_facts.append((f, 'simplewiki'))
        
        self._facts, self._conflicts = self._merge_facts(all_facts)
    
    def _load_source(
        self,
        source: str,
        max_facts: int | None,
        use_cache: bool,
        wordnet_max_synsets: int = 10_000,
        conceptnet_min_confidence: float = 1.0,
    ) -> list[Fact]:
        """Load facts from a single source."""
        if source == 'conceptnet':
            from prism.data.loaders.conceptnet_loader import ConceptNetLoader
            loader = ConceptNetLoader(cache_dir=str(self.cache_dir))
            return loader.load(
                min_confidence=conceptnet_min_confidence,
                max_facts=max_facts,
                use_cache=use_cache,
            )
        
        elif source == 'wordnet':
            from prism.data.loaders.wordnet_loader import WordNetLoader
            loader = WordNetLoader()
            return loader.load(
                max_facts=max_facts,
                max_synsets=wordnet_max_synsets,
            )
        
        elif source == 'simplewiki':
            from prism.data.loaders.simplewiki_loader import SimpleWikiLoader
            loader = SimpleWikiLoader(cache_dir=str(self.cache_dir))
            return loader.load(max_facts=max_facts, use_cache=use_cache)
        
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _merge_facts(
        self,
        tagged_facts: list[tuple[Fact, str]],
    ) -> tuple[list[Fact], list[tuple[Fact, Fact, str]]]:
        """Merge facts with deduplication and conflict detection.
        
        Returns:
            (merged_facts, conflicts)
        """
        # Group by (subject, relation, object)
        groups: dict[tuple[str, str, str], list[tuple[float, str]]] = defaultdict(list)
        
        for (subj, rel, obj, conf), source in tagged_facts:
            key = (subj.lower().strip(), rel.upper().strip(), obj.lower().strip())
            groups[key].append((conf, source))
        
        merged: list[Fact] = []
        conflicts: list[tuple[Fact, Fact, str]] = []
        
        for (subj, rel, obj), entries in groups.items():
            if len(entries) == 1:
                conf, _ = entries[0]
                merged.append((subj, rel, obj, conf))
            else:
                # Multiple sources — average confidence
                avg_conf = sum(c for c, _ in entries) / len(entries)
                merged.append((subj, rel, obj, min(avg_conf, 1.0)))
        
        # Detect contradictions (same subject+object but opposite relations)
        OPPOSITE_RELS = {
            ('IS-A', 'IS-NOT-A'), ('CAN', 'CANNOT'),
            ('HAS', 'DOES-NOT-HAVE'),
        }
        
        subj_obj_groups: dict[tuple[str, str], list[Fact]] = defaultdict(list)
        for fact in merged:
            key = (fact[0], fact[2])
            subj_obj_groups[key].append(fact)
        
        for key, group_facts in subj_obj_groups.items():
            if len(group_facts) < 2:
                continue
            rels = {f[1] for f in group_facts}
            for pos, neg in OPPOSITE_RELS:
                if pos in rels and neg in rels:
                    pos_fact = next(f for f in group_facts if f[1] == pos)
                    neg_fact = next(f for f in group_facts if f[1] == neg)
                    conflicts.append((pos_fact, neg_fact, "contradiction"))
        
        return merged, conflicts
    
    def load_into_memory(
        self,
        memory: 'VectorMemory',
        batch_size: int = 1000,
        max_facts: int | None = None,
        adjust: bool = False,
    ) -> int:
        """Load all integrated facts into VectorMemory.
        
        Args:
            memory: Target VectorMemory instance
            batch_size: Facts per batch
            max_facts: Limit total facts loaded
            adjust: Whether to adjust vectors (slow for large datasets)
            
        Returns:
            Number of facts loaded
        """
        facts = self._facts[:max_facts] if max_facts else self._facts
        total = len(facts)
        loaded = 0
        
        print(f"Loading {total:,} facts into memory (batch_size={batch_size})...")
        t0 = time.time()
        
        for i in range(0, total, batch_size):
            batch = facts[i : i + batch_size]
            
            try:
                if hasattr(memory, 'batch_store'):
                    memory.batch_store(batch, adjust=adjust)
                else:
                    for subj, rel, obj, conf in batch:
                        memory.store(subj, rel, obj, importance=conf, adjust=adjust)
                
                loaded += len(batch)
            except Exception as e:
                print(f"  Error at batch {i//batch_size}: {e}")
                # Continue with next batch
                continue
            
            if (i + batch_size) % 10000 == 0 or i + batch_size >= total:
                elapsed = time.time() - t0
                rate = loaded / elapsed if elapsed > 0 else 0
                pct = min(100, loaded * 100 // total)
                print(f"  [{pct:3d}%] {loaded:,}/{total:,} facts ({rate:.0f} facts/s)")
        
        elapsed = time.time() - t0
        print(f"  Loaded {loaded:,} facts in {elapsed:.1f}s")
        return loaded
    
    def save_knowledge_base(self, filepath: str | None = None) -> None:
        """Save unified knowledge base to compressed JSON."""
        path = Path(filepath) if filepath else self._kb_path
        
        data = {
            'facts': self._facts,
            'source_counts': self._source_counts,
            'conflict_count': len(self._conflicts),
        }
        
        print(f"Saving knowledge base to {path}...")
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  Saved ({size_mb:.1f} MB)")
    
    def load_knowledge_base(self, filepath: str | None = None) -> None:
        """Load unified knowledge base from disk."""
        path = Path(filepath) if filepath else self._kb_path
        
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        
        print(f"Loading knowledge base from {path}...")
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        self._facts = [tuple(f) for f in data['facts']]
        self._source_counts = data.get('source_counts', {})
        print(f"  Loaded {len(self._facts):,} facts")
    
    def get_statistics(self) -> dict:
        """Get statistics about the integrated knowledge base."""
        if not self._facts:
            return {'total_facts': 0}
        
        # Relation distribution
        rel_counts = Counter(f[1] for f in self._facts)
        
        # Top entities (by fact count)
        entity_counts: Counter = Counter()
        for f in self._facts:
            entity_counts[f[0]] += 1
            entity_counts[f[2]] += 1
        
        # Confidence distribution
        confs = [f[3] for f in self._facts]
        avg_conf = sum(confs) / len(confs)
        
        return {
            'total_facts': len(self._facts),
            'source_counts': dict(self._source_counts),
            'relation_distribution': dict(rel_counts.most_common(20)),
            'top_entities': dict(entity_counts.most_common(20)),
            'unique_subjects': len(set(f[0] for f in self._facts)),
            'unique_objects': len(set(f[2] for f in self._facts)),
            'unique_relations': len(set(f[1] for f in self._facts)),
            'avg_confidence': round(avg_conf, 3),
            'conflicts': len(self._conflicts),
        }
    
    def print_statistics(self) -> None:
        """Print formatted statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*50}")
        print(f"Knowledge Base Statistics")
        print(f"{'='*50}")
        print(f"Total facts: {stats['total_facts']:,}")
        print(f"Unique subjects: {stats.get('unique_subjects', 0):,}")
        print(f"Unique objects: {stats.get('unique_objects', 0):,}")
        print(f"Unique relations: {stats.get('unique_relations', 0):,}")
        print(f"Avg confidence: {stats.get('avg_confidence', 0):.3f}")
        print(f"Conflicts: {stats.get('conflicts', 0):,}")
        
        if 'source_counts' in stats:
            print(f"\nFacts per source:")
            for src, cnt in stats['source_counts'].items():
                print(f"  {src}: {cnt:,}")
        
        if 'relation_distribution' in stats:
            print(f"\nTop relations:")
            for rel, cnt in list(stats['relation_distribution'].items())[:10]:
                print(f"  {rel}: {cnt:,}")
        
        if 'top_entities' in stats:
            print(f"\nTop entities:")
            for ent, cnt in list(stats['top_entities'].items())[:10]:
                print(f"  {ent}: {cnt:,}")
        print()
