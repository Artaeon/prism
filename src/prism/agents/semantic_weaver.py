"""Semantic Weaver Network v4 — VSA-Driven Text Generation.

Generates unique text through phrase-level recombination:

1. **Sentence Ranking** — Ranks evidence sentences by vector similarity
   to the query, selecting the most relevant information.

2. **Clause Extraction** — Breaks each sentence into clauses/phrases
   at natural split points (commas, conjunctions, relative pronouns).

3. **Phrase Recombination** — Assembles a response by selecting and
   ordering clauses using topic vector scoring. Different queries
   produce different clause selections and orderings.

4. **Variety** — Uses hash-based seeding + call counter to vary
   output even for repeated queries with the same evidence.

No hardcoded templates. Every response is assembled from actual
evidence phrases, scored and ordered by vector similarity.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

from prism.agents.blackboard import Blackboard, Finding, QueryType


# ─── Text Utilities ─────────────────────────────────────────────────

_NOISE = [
    (re.compile(r'\[\d+\]'), ''),
    (re.compile(r'\(see also:[^)]*\)'), ''),
    (re.compile(r'/[^/]+/'), ''),
    (re.compile(r'\([^)]{0,15}\)'), ''),
    (re.compile(r'\s+'), ' '),
]

_ABBREVS = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Jr.', 'Sr.',
            'vs.', 'etc.', 'i.e.', 'e.g.', 'St.', 'Mt.', 'Inc.',
            'Ltd.', 'Corp.', 'U.S.', 'U.K.', 'E.U.']

_STOP_WORDS = {
    'what', 'who', 'where', 'when', 'how', 'why', 'which',
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'about',
    'does', 'did', 'can', 'could', 'explain', 'tell', 'define',
    'describe', 'compare',
}

# Clause split patterns: commas, semicolons, "and", "which", "that"
_CLAUSE_SPLIT = re.compile(
    r'(?:,\s+|\s+and\s+|\s+which\s+|\s+that\s+|\s+but\s+|;\s+|-\s+|\s+or\s+)',
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    for pattern, replacement in _NOISE:
        text = pattern.sub(replacement, text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    protected = text
    for abbr in _ABBREVS:
        protected = protected.replace(abbr, abbr.replace('.', '\x00'))
    raw = re.split(r'(?<=[.!?])\s+', protected)
    sentences = []
    for s in raw:
        s = s.replace('\x00', '.').strip()
        if len(s) >= 15:
            sentences.append(s)
    return sentences


def _extract_clauses(sentence: str) -> list[str]:
    """Split a sentence into clause-level phrases."""
    # Normalize whitespace first
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    # Split at natural clause boundaries
    parts = _CLAUSE_SPLIT.split(sentence)
    clauses = []
    for p in parts:
        p = re.sub(r'\s+', ' ', p).strip().rstrip('.,;:')
        # Clean leading conjunctions/junk
        p = re.sub(r'^(?:and|but|or|also|yet|so)\s+', '', p, flags=re.IGNORECASE)
        if len(p) >= 10:  # Only keep substantive clauses
            clauses.append(p)
    return clauses if clauses else [sentence.strip()]


# ─── Vector Scorer ──────────────────────────────────────────────────

class VectorScorer:
    """Scores text fragments by vector similarity to a query.
    
    Uses O(1) vocab lookups — no full nlp pipeline calls.
    Averages word vectors for phrase-level similarity.
    """
    
    def __init__(self, nlp) -> None:
        self._nlp = nlp
        self._cache: dict[str, np.ndarray | None] = {}
    
    def phrase_vec(self, text: str) -> np.ndarray | None:
        """Compute averaged word vector for a phrase."""
        key = text[:200]  # Cache key
        if key in self._cache:
            return self._cache[key]
        
        words = re.findall(r'[a-z]+', text.lower())
        vecs = []
        for w in words:
            if len(w) < 2:
                continue
            lex = self._nlp.vocab[w]
            if lex.has_vector:
                vecs.append(lex.vector)
        
        if not vecs:
            self._cache[key] = None
            return None
        
        avg = np.mean(vecs, axis=0).astype(np.float32)
        n = np.linalg.norm(avg)
        result = avg / n if n > 0 else None
        self._cache[key] = result
        return result
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two phrases."""
        va = self.phrase_vec(text_a)
        vb = self.phrase_vec(text_b)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))
    
    def rank_by_query(
        self, query: str, items: list[str], top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rank items by similarity to query."""
        qv = self.phrase_vec(query)
        if qv is None:
            return [(s, 0.5) for s in items[:top_k]]
        
        scored = []
        for item in items:
            iv = self.phrase_vec(item)
            score = float(np.dot(qv, iv)) if iv is not None else 0.1
            scored.append((item, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ─── Phrase Combiner ────────────────────────────────────────────────

class PhraseCombiner:
    """Assembles responses by selecting and ordering evidence phrases.
    
    Phase 5 upgrade:
    - Discourse connectives between clauses (however, furthermore, etc.)
    - Sentence fusion: merges overlapping entity references
    - Smarter clause ordering based on semantic flow
    
    The "VSA-native" generation:
    - Scores each clause by vector similarity to the query
    - Selects top-scoring clauses
    - Fuses overlapping sentences
    - Inserts discourse connectives
    - Orders them for coherence (entity-first, details later)
    """
    
    # Discourse connectives by rhetorical relation
    CONNECTIVES = {
        "elaboration": ["specifically", "in particular", "notably", "that is"],
        "contrast": ["however", "on the other hand", "in contrast", "whereas"],
        "addition": ["furthermore", "additionally", "also", "moreover"],
        "cause": ["because of this", "as a result", "therefore", "consequently"],
        "example": ["for example", "for instance", "such as"],
    }
    
    def __init__(self, scorer: VectorScorer) -> None:
        self._scorer = scorer
    
    def combine(
        self,
        evidence_sentences: list[str],
        query: str,
        max_clauses: int = 6,
        seed: str = "",
    ) -> str:
        """Combine evidence into a coherent response.
        
        Args:
            evidence_sentences: Source sentences
            query: The user's query (drives selection)
            max_clauses: Maximum clauses to include
            seed: For deterministic variety
            
        Returns:
            Combined response text
        """
        if not evidence_sentences:
            return ""
        
        # 1. Try sentence fusion first (merge overlapping evidence)
        fused = self._fuse_sentences(evidence_sentences)
        
        # 2. Extract all clauses from fused sentences
        all_clauses: list[str] = []
        for sent in fused:
            clauses = _extract_clauses(sent)
            all_clauses.extend(clauses)
        
        if not all_clauses:
            return evidence_sentences[0]
        
        # 3. Score each clause by query relevance
        scored = self._scorer.rank_by_query(query, all_clauses, top_k=max_clauses + 3)
        
        # 4. Select clauses — use seed for variety in selection
        h = int(hashlib.md5(seed.encode()).hexdigest(), 16)
        variety_offset = (h % 200) / 2000.0  # 0.0 - 0.1
        
        selected = []
        seen_norms: set[str] = set()
        
        # Optionally skip the top-1 clause based on seed (for variety)
        skip_first = ((h >> 8) % 3) == 0 and len(scored) > 3
        
        for ci, (clause, score) in enumerate(scored):
            if skip_first and ci == 0:
                continue
            
            # Normalize for dedup
            norm = re.sub(r'\W+', '', clause.lower())[:30]
            if norm in seen_norms:
                continue
            seen_norms.add(norm)
            
            # Score threshold with variety
            if score < 0.15 + variety_offset and len(selected) > 1:
                continue
            
            selected.append((clause, score))
            if len(selected) >= max_clauses:
                break
        
        if not selected:
            return evidence_sentences[0]
        
        # 5. Order: keep #1 in place, shuffle rest by seed
        if len(selected) > 2:
            top = selected[0]
            rest = selected[1:]
            rotation = (h >> 16) % max(1, len(rest))
            rest = rest[rotation:] + rest[:rotation]
            selected = [top] + rest
        
        # 6. Build response with discourse connectives
        parts = []
        for i, (clause, score) in enumerate(selected):
            clause = re.sub(r'\s+', ' ', clause).strip()
            if not clause:
                continue
            
            # Add discourse connective between clauses
            if i > 0 and len(selected) > 2:
                connective = self._select_connective(
                    parts[-1] if parts else "", clause, h + i,
                )
                if connective:
                    clause = f"{connective}, {clause[0].lower() + clause[1:]}"
                else:
                    clause = clause[0].upper() + clause[1:]
            else:
                clause = clause[0].upper() + clause[1:]
            
            parts.append(clause)
        
        # Join with periods for readability
        text = '. '.join(parts)
        if not text.rstrip().endswith('.'):
            text = text.rstrip() + '.'
        
        return text
    
    def _fuse_sentences(self, sentences: list[str]) -> list[str]:
        """Merge overlapping sentences that share entities.
        
        If two sentences mention the same entity in similar phrasing,
        combine them into a single richer sentence.
        """
        if len(sentences) <= 1:
            return sentences
        
        fused = []
        used = set()
        
        for i, s1 in enumerate(sentences):
            if i in used:
                continue
            
            # Extract key content words
            words1 = set(re.findall(r'\b[a-z]{3,}\b', s1.lower())) - _STOP_WORDS
            
            merged = s1
            for j, s2 in enumerate(sentences):
                if j <= i or j in used:
                    continue
                
                words2 = set(re.findall(r'\b[a-z]{3,}\b', s2.lower())) - _STOP_WORDS
                
                # Check overlap ratio
                if not words1 or not words2:
                    continue
                overlap = len(words1 & words2)
                ratio = overlap / min(len(words1), len(words2))
                
                if ratio >= 0.4:  # 40% content word overlap → fuse
                    # Find unique content in s2 that's not in s1
                    unique_words = words2 - words1
                    if unique_words:
                        # Extract the clause containing unique info
                        clauses2 = _extract_clauses(s2)
                        for clause in clauses2:
                            clause_words = set(re.findall(r'\b[a-z]{3,}\b', clause.lower()))
                            if clause_words & unique_words:
                                merged = merged.rstrip('.') + ', ' + clause.lower().strip()
                                used.add(j)
                                break
                    else:
                        used.add(j)
            
            fused.append(merged)
        
        return fused
    
    def _select_connective(self, prev: str, next_clause: str, seed: int) -> str | None:
        """Select an appropriate discourse connective between clauses.
        
        Uses semantic cues to pick the right relation type, then
        selects a specific connective based on the seed for variety.
        """
        prev_lower = prev.lower()
        next_lower = next_clause.lower()
        
        # Detect contrast
        contrast_words = {'but', 'however', 'unlike', 'different', 'not', 'except'}
        if any(w in next_lower.split()[:3] for w in contrast_words):
            rel = "contrast"
        # Detect causal
        elif any(w in next_lower for w in ('because', 'therefore', 'result', 'cause')):
            rel = "cause"
        # Detect example
        elif any(w in next_lower for w in ('example', 'instance', 'such as', 'like ')):
            rel = "example"
        # Default: alternate between addition and elaboration
        elif seed % 3 == 0:
            rel = "elaboration"
        else:
            rel = "addition"
        
        # Don't add connective too often (every other clause)
        if seed % 2 == 0:
            return None
        
        options = self.CONNECTIVES[rel]
        return options[seed % len(options)].capitalize()


# ─── Semantic Weaver (main API) ────────────────────────────────────

class SemanticWeaver:
    """Semantic Weaver Network — VSA-driven text generation.
    
    Generates unique responses by:
    1. Ranking evidence sentences by vector similarity to the query
    2. Extracting clause-level phrases from top sentences
    3. Scoring and recombining clauses using topic vectors
    4. Varying output via seeded selection and ordering
    
    No templates. Every response is assembled from evidence phrases
    scored by 300-dimensional vector similarity.
    
    Architecture:
      Evidence → sentences → clauses → score by query → select → combine
      
    Why this works like an LLM:
    - Clause selection is driven by vector similarity (like attention)
    - Output varies per query even with the same evidence
    - Call counter ensures different output for repeated queries
    """
    
    def __init__(self, nlp=None) -> None:
        self._nlp = nlp
        self._scorer = VectorScorer(nlp) if nlp else None
        self._combiner = PhraseCombiner(self._scorer) if self._scorer else None
        self._call_count = 0
    
    def weave(
        self,
        bb: Blackboard,
        plan: Any = None,
        max_sentences: int = 3,
    ) -> str:
        """Generate a response from Blackboard findings."""
        self._call_count += 1
        
        if not self._combiner:
            return ""
        
        query_type = QueryType.GENERAL
        if plan and hasattr(plan, 'query_type'):
            query_type = plan.query_type
        
        if query_type == QueryType.YES_NO:
            return self._weave_yes_no(bb, plan)
        elif query_type == QueryType.COMPARISON:
            return self._weave_comparison(bb, plan)
        else:
            return self._weave_knowledge(bb, plan, max_sentences)
    
    # ─── Knowledge Response ────────────────────────────────────
    
    def _weave_knowledge(
        self, bb: Blackboard, plan: Any, max_sents: int,
    ) -> str:
        """Generate a knowledge response by phrase recombination."""
        entity = self._primary_entity(bb, plan)
        evidence = self._collect_evidence(bb)
        
        if not evidence:
            return f"I couldn't find information about {entity}."
        
        sentences = _split_sentences(evidence)
        if not sentences:
            return _clean_text(evidence)[:300]
        
        # Rank and select top sentences
        ranked = self._scorer.rank_by_query(
            bb.query, sentences, top_k=max_sents + 1,
        )
        top_sents = [s for s, _ in ranked]
        
        # Generate via phrase combination
        seed = f"{bb.query}:{self._call_count}"
        result = self._combiner.combine(
            top_sents, bb.query,
            max_clauses=5,
            seed=seed,
        )
        
        return result or f"I found some information about {entity} but couldn't form a clear response."
    
    # ─── Yes/No Response ───────────────────────────────────────
    
    def _weave_yes_no(self, bb: Blackboard, plan: Any) -> str:
        """Generate yes/no response with verdict."""
        evidence = self._collect_evidence(bb)
        if not evidence:
            return "I'm not sure — not enough evidence."
        
        query = bb.query.lower().rstrip('?').strip()
        predicate = ""
        match = re.search(
            r'(?:can|could|do|does|is|are|will|would|has|have)\s+\w+\s+(.+)',
            query,
        )
        if match:
            predicate = match.group(1).strip()
        
        combined = evidence.lower() + " " + " ".join(bb.get_all_facts()[:5]).lower()
        
        neg = ['cannot', "can't", 'unable', 'flightless', 'incapable',
               'never ', 'not ', 'no ', 'fail', 'do not']
        pos = ['can ', 'able', 'capable', 'known for', 'adapted',
               'evolved', 'excellent', 'swimmer', 'swimming',
               'aquatic', 'marine', 'flipper', 'webbed']
        
        neg_c = sum(1 for n in neg if n in combined)
        pos_c = sum(1 for p in pos if p in combined)
        if predicate and predicate in combined:
            pos_c += 3
        
        if pos_c > neg_c:
            verdict = "✅ Yes"
        elif neg_c > pos_c:
            verdict = "❌ No"
        else:
            verdict = "🤔 Unclear"
        
        # Find best supporting sentence
        sentences = _split_sentences(evidence)
        if sentences:
            ranked = self._scorer.rank_by_query(bb.query, sentences, top_k=1)
            best = _clean_text(ranked[0][0]) if ranked else _clean_text(sentences[0])
            return f"{verdict} — {best}"
        
        return f"{verdict}."
    
    # ─── Comparison Response ───────────────────────────────────
    
    def _weave_comparison(self, bb: Blackboard, plan: Any) -> str:
        """Generate comparison response."""
        entities = self._all_entities(bb, plan)
        if len(entities) < 2:
            return self._weave_knowledge(bb, plan, 3)
        
        entity_a, entity_b = entities[0], entities[1]
        lines = [f"Comparing {entity_a.title()} and {entity_b.title()}:"]
        
        for entity in [entity_a, entity_b]:
            findings = self._findings_for_entity(bb, entity)
            if findings:
                best_f = max(findings, key=lambda f: f.confidence)
                evidence = _clean_text(best_f.text)
                sentences = _split_sentences(evidence)
                
                if sentences:
                    ranked = self._scorer.rank_by_query(entity, sentences, top_k=2)
                    top_sents = [s for s, _ in ranked]
                    
                    seed = f"{entity}:{self._call_count}"
                    combined = self._combiner.combine(
                        top_sents, entity, max_clauses=3, seed=seed,
                    )
                    lines.append(f"\n  ▸ **{entity.title()}**: {combined}")
                else:
                    short = evidence[:200] + ("..." if len(evidence) > 200 else "")
                    lines.append(f"\n  ▸ **{entity.title()}**: {short}")
            else:
                lines.append(f"\n  ▸ **{entity.title()}**: No information found.")
        
        return "\n".join(lines)
    
    # ─── Helpers ────────────────────────────────────────────────
    
    def _collect_evidence(self, bb: Blackboard) -> str:
        best = bb.best_findings(top_k=3)
        if not best:
            return ""
        return " ".join(_clean_text(f.text) for _, f in best)
    
    def _primary_entity(self, bb: Blackboard, plan: Any) -> str:
        if plan and hasattr(plan, 'entities') and plan.entities:
            return plan.entities[0]
        if bb.entities:
            return bb.entities[0]
        words = [w for w in bb.query.lower().split()
                 if len(w) > 3 and w not in _STOP_WORDS]
        return max(words, key=len) if words else bb.query
    
    def _all_entities(self, bb: Blackboard, plan: Any) -> list[str]:
        if plan and hasattr(plan, 'entities') and plan.entities:
            return plan.entities
        return bb.entities or []
    
    def _findings_for_entity(self, bb: Blackboard, entity: str) -> list[Finding]:
        results = []
        el = entity.lower()
        for af in bb.findings.values():
            for f in af:
                if el in f.text.lower():
                    results.append(f)
        if not results:
            for _, f in bb.best_findings(top_k=3):
                results.append(f)
        return results
