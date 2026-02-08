"""Enhanced Vector Memory with semantic adjustment during learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator
import numpy as np

from gunter.core import VSAConfig, DEFAULT_CONFIG
from gunter.core.vector_ops import VectorOps, HVector
from gunter.core.lexicon import Lexicon
from gunter.memory.episode_index import EpisodeIndex


@dataclass
class EpisodicMemory:
    """A single episodic memory."""
    
    id: int = 0
    vector: HVector | None = None
    text: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0
    access_count: int = 0
    
    subject: str = ""
    relation: str = ""
    obj: str = ""
    
    agent: str = ""
    action: str = ""
    patient: str = ""


# Adjustment rates for different relationship types
ADJUSTMENT_RATES = {
    "IS-A": 0.15,      # Strong: cat IS-A animal
    "MEANS": 0.20,     # Synonym: kitty MEANS cat
    "HAS": 0.08,       # Possession: cat HAS whiskers
    "CAN": 0.05,       # Capability: cat CAN jump
    "LIKES": 0.03,     # Preference: cat LIKES fish
    "NEEDS": 0.05,     # Requirement
    "IS": 0.10,        # Property: cat IS furry
}


class VectorMemory:
    """Vector-based memory with semantic adjustment during learning.
    
    When facts are stored, vectors are adjusted to reflect relationships:
    - "cat IS-A animal" → cat moves closer to animal
    - "kitten IS-A cat" → kitten moves closer to cat
    
    This builds semantic structure over time.
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        config: VSAConfig | None = None,
    ) -> None:
        """Initialize memory."""
        self.config = config or DEFAULT_CONFIG
        self.ops = VectorOps(self.config)
        self.lexicon = lexicon
        
        self._semantic: HVector = self.ops.zero_vector()
        self._episodes: list[EpisodicMemory] = []
        self._next_id = 1
        self._index = EpisodeIndex()
    
    def store(
        self,
        subject: str,
        relation: str,
        obj: str,
        importance: float = 1.0,
        adjust: bool = True,
    ) -> EpisodicMemory:
        """Store a fact and optionally adjust vectors.
        
        Args:
            subject: Subject of the fact
            relation: Relation type
            obj: Object of the fact
            importance: Weight of this fact
            adjust: Whether to adjust vectors for similarity
            
        Returns:
            The created memory
        """
        s_vec = self.lexicon.get(subject)
        r_vec = self.lexicon.get(relation)
        o_vec = self.lexicon.get(obj)
        
        # Encode: bind(bind(S, R), O)
        fact_vec = self.ops.bind(self.ops.bind(s_vec, r_vec), o_vec)
        
        # Add to semantic memory
        self._semantic = self._semantic + (fact_vec * importance)
        
        # Adjust vectors to build semantic structure
        if adjust:
            rate = ADJUSTMENT_RATES.get(relation.upper(), 0.05)
            self.lexicon.adjust_similarity(subject, obj, rate)
        
        # Create episodic record
        episode = EpisodicMemory(
            id=self._next_id,
            vector=fact_vec,
            text=f"{subject} {relation} {obj}",
            importance=importance,
            subject=subject,
            relation=relation,
            obj=obj,
        )
        self._episodes.append(episode)
        self._index.add(episode)
        self._next_id += 1
        
        return episode
    
    def store_event(
        self,
        agent: str,
        action: str,
        patient: str,
        importance: float = 1.0,
        adjust: bool = True,
    ) -> EpisodicMemory:
        """Store an event with semantic roles."""
        agent_vec = self.lexicon.get(agent)
        action_vec = self.lexicon.get(action)
        patient_vec = self.lexicon.get(patient)
        
        role_agent = self.lexicon.get("AGENT")
        role_action = self.lexicon.get("ACTION")
        role_patient = self.lexicon.get("PATIENT")
        
        event_vec = self.ops.bundle([
            self.ops.bind(agent_vec, role_agent),
            self.ops.bind(action_vec, role_action),
            self.ops.bind(patient_vec, role_patient),
        ])
        
        self._semantic = self._semantic + (event_vec * importance)
        
        # Adjust: agent and action co-occur
        if adjust:
            self.lexicon.adjust_similarity(agent, action, 0.03)
            self.lexicon.adjust_similarity(action, patient, 0.03)
        
        episode = EpisodicMemory(
            id=self._next_id,
            vector=event_vec,
            text=f"{agent} {action} {patient}",
            importance=importance,
            agent=agent,
            action=action,
            patient=patient,
        )
        self._episodes.append(episode)
        self._next_id += 1
        
        return episode
    
    def store_episode(
        self,
        text: str,
        vector: HVector | None = None,
        importance: float = 1.0,
    ) -> EpisodicMemory:
        """Store a raw episode."""
        if vector is None:
            words = text.lower().split()
            word_vecs = [self.lexicon.get(w) for w in words if len(w) > 2]
            vector = self.ops.bundle(word_vecs) if word_vecs else self.ops.zero_vector()
        
        self._semantic = self._semantic + (vector * importance)
        
        episode = EpisodicMemory(
            id=self._next_id,
            vector=vector,
            text=text,
            importance=importance,
        )
        self._episodes.append(episode)
        self._next_id += 1
        
        return episode
    
    # =========================================================================
    # Queries (with min_score filtering)
    # =========================================================================
    
    def query_subject(
        self,
        relation: str,
        obj: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Query: what [relation] [object]?"""
        r_vec = self.lexicon.get(relation)
        o_vec = self.lexicon.get(obj)
        
        key = self.ops.bind(r_vec, o_vec)
        query_result = self.ops.unbind(self._semantic, key)
        
        return self.lexicon.find_nearest(
            query_result, top_k=top_k,
            exclude={relation.upper(), obj.lower()},
            min_score=min_score,
        )
    
    def query_object(
        self,
        subject: str,
        relation: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Query: [subject] [relation] what?"""
        s_vec = self.lexicon.get(subject)
        r_vec = self.lexicon.get(relation)
        
        key = self.ops.bind(s_vec, r_vec)
        query_result = self.ops.unbind(self._semantic, key)
        
        return self.lexicon.find_nearest(
            query_result, top_k=top_k,
            exclude={subject.lower(), relation.upper()},
            min_score=min_score,
        )
    
    def query_agent(self, action: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Query: who [action]?"""
        # Check episodes directly first
        for ep in self._episodes:
            if ep.action.lower() == action.lower() and ep.agent:
                return [(ep.agent, 0.95)]
        
        # Fallback to vector search
        role_agent = self.lexicon.get("AGENT")
        result = self.ops.unbind(self._semantic, role_agent)
        
        return self.lexicon.find_nearest(result, top_k=top_k)
    
    def query_action(self, agent: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Query: what did [agent] do?"""
        for ep in self._episodes:
            if ep.agent.lower() == agent.lower():
                return [(f"{ep.action} {ep.patient}", 0.95)]
        
        role_action = self.lexicon.get("ACTION")
        result = self.ops.unbind(self._semantic, role_action)
        return self.lexicon.find_nearest(result, top_k=top_k)
    
    def check_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> float:
        """Check if a fact is in memory."""
        s_vec = self.lexicon.get(subject)
        r_vec = self.lexicon.get(relation)
        o_vec = self.lexicon.get(obj)
        
        fact_vec = self.ops.bind(self.ops.bind(s_vec, r_vec), o_vec)
        return self.ops.similarity(self._semantic, fact_vec)
    
    # =========================================================================
    # Similarity Search
    # =========================================================================
    
    def find_similar(
        self,
        query: str | HVector,
        k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Find words similar to a query."""
        if isinstance(query, str):
            query_vec = self.lexicon.get(query)
            exclude = {query.lower()}
        else:
            query_vec = query
            exclude = set()
        
        return self.lexicon.find_nearest(
            query_vec, top_k=k, exclude=exclude, min_score=min_score
        )
    
    def find_similar_episodes(
        self,
        query: str | HVector,
        k: int = 5,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Find episodic memories similar to query."""
        if isinstance(query, str):
            query_vec = self.lexicon.get(query)
        else:
            query_vec = query
        
        scores = []
        for ep in self._episodes:
            if ep.vector is not None:
                sim = self.ops.similarity(query_vec, ep.vector)
                scores.append((ep, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    def consolidate(self, similarity_threshold: float = 0.8) -> int:
        """Merge similar memories."""
        if len(self._episodes) < 2:
            return 0
        
        merged_count = 0
        i = 0
        
        while i < len(self._episodes):
            j = i + 1
            while j < len(self._episodes):
                ep_i = self._episodes[i]
                ep_j = self._episodes[j]
                
                if ep_i.vector is not None and ep_j.vector is not None:
                    sim = self.ops.similarity(ep_i.vector, ep_j.vector)
                    
                    if sim > similarity_threshold:
                        ep_i.vector = self.ops.bundle([ep_i.vector, ep_j.vector])
                        ep_i.importance = max(ep_i.importance, ep_j.importance)
                        ep_i.access_count += ep_j.access_count
                        ep_i.text = f"{ep_i.text}; {ep_j.text}"
                        
                        self._episodes.pop(j)
                        merged_count += 1
                        continue
                
                j += 1
            i += 1
        
        return merged_count
    
    def get_all_facts(self) -> list[str]:
        """Get all stored fact texts."""
        return [ep.text for ep in self._episodes if ep.text]
    
    def get_episodes(self) -> list[EpisodicMemory]:
        """Get all episodes."""
        return list(self._episodes)
    
    # =========================================================================
    # Phase 18: Indexed Search (replaces O(n) substring scans)
    # =========================================================================
    
    def search_facts(
        self,
        entity: str,
        top_k: int = 10,
    ) -> list[EpisodicMemory]:
        """Search for facts mentioning an entity.
        
        Uses the inverted index for O(1) exact lookup.
        Falls back to fuzzy embedding search if no exact matches found.
        
        Args:
            entity: Entity to search for
            top_k: Maximum results
            
        Returns:
            Matching episodes sorted by importance
        """
        results = self._index.search(entity, top_k=top_k)
        if results:
            return results
        
        # Fuzzy fallback: try embedding similarity
        fuzzy = self._index.search_fuzzy(entity, top_k=top_k, min_similarity=0.7)
        return [ep for ep, score in fuzzy]
    
    def search_facts_by_relation(
        self,
        entity: str,
        relation: str,
        top_k: int = 10,
    ) -> list[EpisodicMemory]:
        """Search for facts with specific entity + relation.
        
        E.g., search_facts_by_relation("cat", "IS-A") returns
        all "cat IS-A ..." facts.
        """
        return self._index.search_by_relation(entity, relation, top_k=top_k)
    
    def get_entity_relations(self, entity: str) -> dict[str, list[str]]:
        """Get all relations for an entity, grouped by relation type.
        
        Returns:
            {"IS-A": ["mammal", "animal"], "CAN": ["purr", "climb"]}
        """
        return self._index.get_relations_for(entity)
    
    def clear(self) -> None:
        """Clear all memory."""
        self._semantic = self.ops.zero_vector()
        self._episodes.clear()
        self._index.clear()
        self._next_id = 1
    
    # =========================================================================
    # Phase 17: Bulk Operations
    # =========================================================================
    
    def batch_store(
        self,
        facts: list[tuple[str, str, str, float]],
        adjust: bool = False,
    ) -> int:
        """Store multiple facts efficiently.
        
        Uses vectorized NumPy operations for 10-100× speedup
        over individual store() calls.
        
        Args:
            facts: List of (subject, relation, object, importance) tuples
            adjust: Whether to adjust vectors (slow for bulk)
            
        Returns:
            Number of facts stored
        """
        if not facts:
            return 0
        
        dim = self.config.dimension
        batch_semantic = np.zeros(dim, dtype=np.float32)
        stored = 0
        
        for subj, rel, obj, importance in facts:
            try:
                s_vec = self.lexicon.get(subj)
                r_vec = self.lexicon.get(rel)
                o_vec = self.lexicon.get(obj)
                
                # bind(bind(S, R), O)
                fact_vec = self.ops.bind(self.ops.bind(s_vec, r_vec), o_vec)
                
                # Accumulate into batch semantic vector
                batch_semantic += fact_vec * importance
                
                # Create episodic record
                episode = EpisodicMemory(
                    id=self._next_id,
                    vector=fact_vec,
                    text=f"{subj} {rel} {obj}",
                    importance=importance,
                    subject=subj,
                    relation=rel,
                    obj=obj,
                )
                self._episodes.append(episode)
                self._index.add(episode)
                self._next_id += 1
                stored += 1
                
                if adjust:
                    rate = ADJUSTMENT_RATES.get(rel.upper(), 0.05)
                    self.lexicon.adjust_similarity(subj, obj, rate)
                    
            except Exception:
                continue
        
        # Add batch to semantic memory in one operation
        self._semantic = self._semantic + batch_semantic
        
        return stored
    
    def save_to_disk(self, filepath: str) -> None:
        """Serialize memory to disk for fast loading.
        
        Saves semantic vector, episodes (without individual vectors to save
        space), and lexicon state using pickle + numpy.
        
        Args:
            filepath: Path to save to (will create .npz + .pkl files)
        """
        import pickle
        from pathlib import Path
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save semantic vector as numpy
        np.savez_compressed(
            str(path.with_suffix('.npz')),
            semantic=np.array(self._semantic, dtype=np.float32),
        )
        
        # Save episodes and lexicon (without large vectors)
        ep_data = []
        for ep in self._episodes:
            ep_data.append({
                'id': ep.id,
                'text': ep.text,
                'importance': ep.importance,
                'subject': ep.subject,
                'relation': ep.relation,
                'obj': ep.obj,
                'agent': ep.agent,
                'action': ep.action,
                'patient': ep.patient,
            })
        
        meta = {
            'episodes': ep_data,
            'next_id': self._next_id,
            'lexicon_words': dict(self.lexicon._vectors),
            'config_dimension': self.config.dimension,
        }
        
        pkl_path = str(path.with_suffix('.pkl'))
        with open(pkl_path, 'wb') as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        total_size = (
            path.with_suffix('.npz').stat().st_size +
            Path(pkl_path).stat().st_size
        )
        print(f"Saved memory to {filepath} ({total_size / 1024 / 1024:.1f} MB)")
    
    @classmethod
    def load_from_disk(
        cls,
        filepath: str,
        lexicon: 'Lexicon | None' = None,
        config: 'VSAConfig | None' = None,
    ) -> 'VectorMemory':
        """Load memory from disk.
        
        Args:
            filepath: Path to load from
            lexicon: Lexicon to use (creates new if None)
            config: VSA config (inferred from saved data if None)
            
        Returns:
            Restored VectorMemory
        """
        import pickle
        from pathlib import Path
        
        path = Path(filepath)
        
        # Load semantic vector
        npz_path = path.with_suffix('.npz')
        if not npz_path.exists():
            raise FileNotFoundError(f"Memory file not found: {npz_path}")
        
        data = np.load(str(npz_path))
        semantic = data['semantic']
        
        # Load metadata
        pkl_path = path.with_suffix('.pkl')
        with open(str(pkl_path), 'rb') as f:
            meta = pickle.load(f)
        
        # Create config from saved dimension
        if config is None:
            dim = meta.get('config_dimension', len(semantic))
            from gunter.core import VSAConfig
            config = VSAConfig(dimension=dim)
        
        # Create or restore lexicon
        if lexicon is None:
            from gunter.core.lexicon import Lexicon
            lexicon = Lexicon(config)
        
        # Restore lexicon vectors
        if 'lexicon_words' in meta:
            for word, vec in meta['lexicon_words'].items():
                lexicon._vectors[word] = vec
        
        # Create memory instance
        memory = cls(lexicon, config)
        memory._semantic = semantic
        memory._next_id = meta.get('next_id', 1)
        
        # Restore episodes (without individual vectors)
        for ep_dict in meta.get('episodes', []):
            ep = EpisodicMemory(
                id=ep_dict.get('id', 0),
                text=ep_dict.get('text', ''),
                importance=ep_dict.get('importance', 1.0),
                subject=ep_dict.get('subject', ''),
                relation=ep_dict.get('relation', ''),
                obj=ep_dict.get('obj', ''),
                agent=ep_dict.get('agent', ''),
                action=ep_dict.get('action', ''),
                patient=ep_dict.get('patient', ''),
            )
            memory._episodes.append(ep)
        
        # Rebuild the episode index
        memory._index.add_batch(memory._episodes)
        
        print(f"Loaded memory: {len(memory._episodes):,} episodes")
        return memory
    
    def get_statistics(self) -> dict:
        """Get memory statistics."""
        from collections import Counter
        
        relations = Counter(ep.relation for ep in self._episodes if ep.relation)
        subjects = Counter(ep.subject for ep in self._episodes if ep.subject)
        
        return {
            'total_episodes': len(self._episodes),
            'total_facts': sum(1 for ep in self._episodes if ep.subject),
            'unique_subjects': len(set(ep.subject for ep in self._episodes if ep.subject)),
            'unique_objects': len(set(ep.obj for ep in self._episodes if ep.obj)),
            'unique_relations': len(relations),
            'lexicon_size': len(self.lexicon._vectors),
            'top_relations': dict(relations.most_common(10)),
            'top_subjects': dict(subjects.most_common(10)),
        }
    
    def __len__(self) -> int:
        """Number of episodes."""
        return len(self._episodes)

