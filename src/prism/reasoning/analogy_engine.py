"""Advanced Analogy Engine — Vector + fact-based analogical reasoning.

Combines the existing vector arithmetic approach (king - man + woman = queen)
with structural fact-based reasoning for more grounded analogies.

Example:
    >>> engine = AdvancedAnalogyReasoner(lexicon, memory)
    >>> result = engine.solve_analogy("lion", "savanna", "polar bear")
    >>> result.answer  # "arctic"
    >>> result.method  # "relation_based"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory
    from prism.core.lexicon import Lexicon

from prism.reasoning import AnalogyReasoner


@dataclass
class AnalogyResult:
    """Result of an analogy computation."""
    
    answer: str = ""
    candidates: list[tuple[str, float]] = field(default_factory=list)
    method: str = ""  # "vector", "relation_based", "structural"
    confidence: float = 0.0
    relation_found: str = ""
    explanation: str = ""


class AdvancedAnalogyReasoner:
    """Analogical reasoning combining vector arithmetic and fact structure.
    
    Strategy:
    1. Try vector arithmetic (fastest, works for semantic relationships)
    2. Try relation-based (find relation R where A R B, then C R ?)
    3. Try structural similarity (compare property patterns)
    
    Example:
        >>> reasoner = AdvancedAnalogyReasoner(lexicon, memory)
        >>> result = reasoner.solve_analogy("king", "queen", "prince")
        >>> result.answer  # "princess"
    """
    
    def __init__(self, lexicon: 'Lexicon', memory: 'VectorMemory') -> None:
        self.lexicon = lexicon
        self.memory = memory
        self.ops = lexicon.ops
        self.vector_reasoner = AnalogyReasoner(lexicon)
    
    def solve_analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 5,
    ) -> AnalogyResult:
        """Solve analogy: A is to B as C is to ?
        
        Args:
            a: First term (e.g., "king")
            b: Second term (e.g., "queen")
            c: Third term (e.g., "prince")
            top_k: Number of candidates
            
        Returns:
            AnalogyResult with best answer and all candidates
        """
        # Strategy 1: Relation-based (most reliable with stored facts)
        relation_result = self._try_relation_based(a, b, c)
        if relation_result and relation_result.confidence >= 0.5:
            return relation_result
        
        # Strategy 2: Vector arithmetic
        vector_result = self._try_vector_arithmetic(a, b, c, top_k)
        
        # Pick the best overall result
        if relation_result and vector_result:
            if relation_result.confidence >= vector_result.confidence:
                return relation_result
            return vector_result
        
        return relation_result or vector_result or AnalogyResult(
            answer="unknown",
            explanation=f"Couldn't solve: {a} is to {b} as {c} is to ?",
        )
    
    def find_structural_similarity(
        self,
        entity1: str,
        entity2: str,
    ) -> tuple[float, list[str], list[str]]:
        """Compare property structures between two entities.
        
        Returns:
            Tuple of (similarity_score, shared_properties, different_properties)
        """
        props1 = self._get_properties(entity1)
        props2 = self._get_properties(entity2)
        
        shared = []
        different = []
        
        # Find shared relations and values
        all_rels = set(props1.keys()) | set(props2.keys())
        for rel in all_rels:
            vals1 = set(props1.get(rel, []))
            vals2 = set(props2.get(rel, []))
            
            common = vals1 & vals2
            for v in common:
                shared.append(f"{rel} {v}")
            
            only1 = vals1 - vals2
            only2 = vals2 - vals1
            
            for v in only1:
                different.append(f"{entity1} {rel} {v}")
            for v in only2:
                different.append(f"{entity2} {rel} {v}")
        
        # Also check semantic similarity of properties
        for rel in set(props1.keys()) & set(props2.keys()):
            for v1 in props1[rel]:
                for v2 in props2[rel]:
                    if v1 != v2:
                        vec1 = self.lexicon.get(v1)
                        vec2 = self.lexicon.get(v2)
                        if vec1 is not None and vec2 is not None:
                            sim = self.ops.similarity(vec1, vec2)
                            if sim > 0.6:
                                shared.append(f"{rel} ~{v1}/{v2}~")
        
        total = len(shared) + len(different)
        score = len(shared) / total if total > 0 else 0.0
        
        # Factor in vector similarity
        vec1 = self.lexicon.get(entity1)
        vec2 = self.lexicon.get(entity2)
        if vec1 is not None and vec2 is not None:
            vec_sim = max(0.0, self.ops.similarity(vec1, vec2))
            score = score * 0.6 + vec_sim * 0.4
        
        return score, shared, different
    
    def complete_pattern(self, pattern: str) -> AnalogyResult:
        """Handle patterns like "X is to Y as Z is to ?"
        
        Parses natural language pattern and delegates to solve_analogy.
        """
        import re
        
        # "X is to Y as Z is to ?"
        m = re.search(
            r'(\w+)\s+(?:is|are)\s+to\s+(\w+)\s+as\s+(\w+)\s+(?:is|are)\s+to\s+\?',
            pattern.lower(),
        )
        if m:
            return self.solve_analogy(m.group(1), m.group(2), m.group(3))
        
        # "X : Y :: Z : ?"
        m = re.search(r'(\w+)\s*:\s*(\w+)\s*::\s*(\w+)\s*:\s*\?', pattern.lower())
        if m:
            return self.solve_analogy(m.group(1), m.group(2), m.group(3))
        
        return AnalogyResult(
            answer="unknown",
            explanation="Couldn't parse the analogy pattern.",
        )
    
    # ── Internal strategies ──
    
    def _try_vector_arithmetic(
        self, a: str, b: str, c: str, top_k: int
    ) -> AnalogyResult | None:
        """Strategy 1: d = b - a + c (vector arithmetic)."""
        try:
            results = self.vector_reasoner.solve(a, b, c, top_k=top_k)
        except Exception:
            return None
        
        if not results:
            return None
        
        best_word, best_score = results[0]
        
        return AnalogyResult(
            answer=best_word,
            candidates=results,
            method="vector",
            confidence=best_score,
            explanation=(
                f"{a} is to {b} as {c} is to {best_word} "
                f"(vector arithmetic, {best_score:.0%})"
            ),
        )
    
    def _try_relation_based(
        self, a: str, b: str, c: str
    ) -> AnalogyResult | None:
        """Strategy 2: Find relation R where A R B, then find C R ?"""
        # Find relations between A and B
        a_lower = a.lower()
        b_lower = b.lower()
        
        episodes = self.memory.get_episodes()
        
        # Direct: A R B
        relations = []
        for ep in episodes:
            if not ep.subject or not ep.obj:
                continue
            if (ep.subject.lower() == a_lower and ep.obj.lower() == b_lower):
                relations.append(ep.relation)
            elif (ep.subject.lower() == b_lower and ep.obj.lower() == a_lower):
                relations.append(ep.relation)
        
        if not relations:
            return None
        
        # Apply best relation to C
        c_lower = c.lower()
        for rel in relations:
            for ep in episodes:
                if not ep.subject or not ep.obj:
                    continue
                if ep.subject.lower() == c_lower and ep.relation == rel:
                    return AnalogyResult(
                        answer=ep.obj,
                        candidates=[(ep.obj, 0.85)],
                        method="relation_based",
                        confidence=0.85,
                        relation_found=rel,
                        explanation=(
                            f"{a} {rel} {b}, "
                            f"{c} {rel} {ep.obj}"
                        ),
                    )
        
        # Relation found but no C match — try vector similarity
        for rel in relations:
            # Find all objects of C
            c_objects = [
                ep.obj for ep in episodes
                if ep.subject and ep.subject.lower() == c_lower
            ]
            if c_objects:
                # Return most similar to B
                vec_b = self.lexicon.get(b)
                if vec_b is not None:
                    scored = []
                    for obj in c_objects:
                        vec_o = self.lexicon.get(obj)
                        if vec_o is not None:
                            sim = self.ops.similarity(vec_b, vec_o)
                            scored.append((obj, sim))
                    if scored:
                        scored.sort(key=lambda x: x[1], reverse=True)
                        best = scored[0]
                        return AnalogyResult(
                            answer=best[0],
                            candidates=scored[:5],
                            method="relation_based",
                            confidence=best[1] * 0.7,
                            relation_found=relations[0],
                            explanation=(
                                f"{a} {relations[0]} {b}, "
                                f"{c} → {best[0]} (semantic match)"
                            ),
                        )
        
        return None
    
    def _get_properties(self, entity: str) -> dict[str, list[str]]:
        """Get properties of an entity grouped by relation."""
        entity_lower = entity.lower()
        props: dict[str, list[str]] = {}
        
        for ep in self.memory.get_episodes():
            if ep.subject and ep.subject.lower() == entity_lower:
                props.setdefault(ep.relation, []).append(ep.obj.lower())
        
        return props
