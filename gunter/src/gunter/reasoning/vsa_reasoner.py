"""VSA Reasoner — Execute reasoning patterns over stored facts.

Given a matched pattern type and extracted entities, searches Gunter's
memory and performs pattern-specific reasoning (comparison, similarity,
causation, etc.) to produce structured results.

Phase 16: Enhanced with advanced reasoners (multihop, analogy, causal, temporal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gunter.reasoning.pattern_library import PatternType


@dataclass
class ReasoningResult:
    """Result of executing a reasoning pattern."""
    
    pattern_type: PatternType
    answer: str  # Yes / No / partial
    confidence: float = 0.0
    facts_used: list[str] = field(default_factory=list)
    shared_properties: list[str] = field(default_factory=list)
    different_properties: list[str] = field(default_factory=list)
    similarity_score: float = 0.0
    reasoning_chain: list[str] = field(default_factory=list)
    explanation: str = ""


class VSAReasoner:
    """Execute reasoning patterns over Gunter's knowledge base.
    
    For each of the 15 pattern types, implements specific logic
    to search facts, compare entities, and produce structured results.
    
    Example:
        >>> reasoner = VSAReasoner(gunter)
        >>> result = reasoner.execute_pattern(
        ...     PatternType.SIMILARITY, ["cats", "dogs"]
        ... )
        >>> result.similarity_score  # 0.73
    """
    
    def __init__(self, gunter: Any) -> None:
        self.gunter = gunter
        self.memory = gunter.memory
        self.lexicon = gunter.lexicon
        self.ops = gunter.lexicon.ops
        
        # Advanced reasoners (lazy-initialized)
        self._multihop = None
        self._analogy = None
        self._causal = None
    
    def _get_multihop(self):
        """Lazy-init MultiHopReasoner."""
        if self._multihop is None:
            from gunter.reasoning.multihop import MultiHopReasoner
            self._multihop = MultiHopReasoner(self.memory, self.lexicon)
        return self._multihop
    
    def _get_analogy(self):
        """Lazy-init AdvancedAnalogyReasoner."""
        if self._analogy is None:
            from gunter.reasoning.analogy_engine import AdvancedAnalogyReasoner
            self._analogy = AdvancedAnalogyReasoner(self.lexicon, self.memory)
        return self._analogy
    
    def _get_causal(self):
        """Lazy-init CausalReasoner."""
        if self._causal is None:
            from gunter.reasoning.causality import CausalReasoner
            self._causal = CausalReasoner(self.memory, self.lexicon)
        return self._causal
    
    def execute_pattern(
        self,
        pattern_type: PatternType,
        entities: list[str],
    ) -> ReasoningResult:
        """Dispatch to the appropriate reasoning method."""
        handlers = {
            PatternType.SAMENESS: self._reason_sameness,
            PatternType.SIMILARITY: self._reason_similarity,
            PatternType.DIFFERENCE: self._reason_difference,
            PatternType.COMPARISON: self._reason_comparison,
            PatternType.CAPABILITY: self._reason_capability,
            PatternType.CAUSATION: self._reason_causation,
            PatternType.IDENTITY: self._reason_identity,
            PatternType.POSSESSION: self._reason_possession,
            PatternType.LOCATION: self._reason_location,
            PatternType.PURPOSE: self._reason_purpose,
            PatternType.COMPOSITION: self._reason_composition,
            PatternType.PROPERTY: self._reason_property,
            PatternType.QUANTITY: self._reason_quantity,
            PatternType.TIME: self._reason_time,
            PatternType.RELATION: self._reason_relation,
        }
        
        handler = handlers.get(pattern_type, self._reason_property)
        return handler(entities)
    
    # ── Core helpers ──
    
    def _get_facts_for(self, entity: str) -> list[str]:
        """Get all stored facts mentioning an entity."""
        episodes = self.memory.search_facts(entity, top_k=20)
        return [ep.text for ep in episodes]
    
    def _get_properties(self, entity: str) -> dict[str, list[str]]:
        """Get properties of an entity grouped by relation type."""
        return self.memory.get_entity_relations(entity)
    
    def _find_shared_properties(self, x: str, y: str) -> list[str]:
        """Find properties shared between two entities."""
        props_x = self._get_properties(x)
        props_y = self._get_properties(y)
        
        shared = []
        for rel in set(props_x.keys()) & set(props_y.keys()):
            for val in props_x[rel]:
                if val in props_y[rel]:
                    shared.append(f"{rel} {val}")
        
        # Also check semantic similarity of properties
        for rel in set(props_x.keys()) & set(props_y.keys()):
            for vx in props_x[rel]:
                for vy in props_y[rel]:
                    if vx != vy:
                        vec_x = self.lexicon.get(vx)
                        vec_y = self.lexicon.get(vy)
                        if vec_x is not None and vec_y is not None:
                            sim = self.ops.similarity(vec_x, vec_y)
                            if sim > 0.6:
                                shared.append(f"{rel} {vx}/{vy}")
        
        return shared
    
    def _find_different_properties(self, x: str, y: str) -> tuple[list[str], list[str]]:
        """Find properties unique to each entity."""
        props_x = self._get_properties(x)
        props_y = self._get_properties(y)
        
        unique_x = []
        unique_y = []
        
        for rel, vals in props_x.items():
            for val in vals:
                if rel not in props_y or val not in props_y.get(rel, []):
                    unique_x.append(f"{rel} {val}")
        
        for rel, vals in props_y.items():
            for val in vals:
                if rel not in props_x or val not in props_x.get(rel, []):
                    unique_y.append(f"{rel} {val}")
        
        return unique_x, unique_y
    
    def _compute_similarity_score(self, x: str, y: str) -> float:
        """Compute similarity between two entities using embeddings."""
        vec_x = self.lexicon.get(x)
        vec_y = self.lexicon.get(y)
        if vec_x is not None and vec_y is not None:
            return max(0.0, self.ops.similarity(vec_x, vec_y))
        return 0.0
    
    def _check_transitive(
        self, subject: str, relation: str, obj: str, max_depth: int = 3
    ) -> list[str]:
        """Check for transitive chain: subject → ... → obj."""
        chain = self.gunter.transitive.infer(subject, obj)
        if chain and chain.confidence >= 0.3:
            return [
                f"{s.subject} {s.relation} {s.obj}" for s in chain.steps
            ]
        return []
    
    # ── Pattern-specific reasoning ──
    
    def _reason_sameness(self, entities: list[str]) -> ReasoningResult:
        if len(entities) < 2:
            return ReasoningResult(
                pattern_type=PatternType.SAMENESS,
                answer="unclear",
                explanation="I need two things to compare.",
            )
        
        x, y = entities[0], entities[1]
        sim = self._compute_similarity_score(x, y)
        shared = self._find_shared_properties(x, y)
        unique_x, unique_y = self._find_different_properties(x, y)
        
        is_same = sim > 0.85 and not unique_x and not unique_y
        
        return ReasoningResult(
            pattern_type=PatternType.SAMENESS,
            answer="yes" if is_same else "no",
            confidence=sim,
            shared_properties=shared,
            different_properties=unique_x + unique_y,
            similarity_score=sim,
            facts_used=self._get_facts_for(x)[:3] + self._get_facts_for(y)[:3],
            explanation=(
                f"{x} and {y} are {'the same' if is_same else 'different'}. "
                f"Similarity: {sim:.0%}."
            ),
        )
    
    def _reason_similarity(self, entities: list[str]) -> ReasoningResult:
        if len(entities) < 2:
            return ReasoningResult(
                pattern_type=PatternType.SIMILARITY,
                answer="unclear",
                explanation="I need two things to compare.",
            )
        
        x, y = entities[0], entities[1]
        sim = self._compute_similarity_score(x, y)
        shared = self._find_shared_properties(x, y)
        unique_x, unique_y = self._find_different_properties(x, y)
        
        return ReasoningResult(
            pattern_type=PatternType.SIMILARITY,
            answer=f"{sim:.0%}",
            confidence=sim,
            shared_properties=shared,
            different_properties=unique_x[:3] + unique_y[:3],
            similarity_score=sim,
            facts_used=self._get_facts_for(x)[:3] + self._get_facts_for(y)[:3],
            explanation=f"{x} and {y} are {sim:.0%} similar.",
        )
    
    def _reason_difference(self, entities: list[str]) -> ReasoningResult:
        if len(entities) < 2:
            return ReasoningResult(
                pattern_type=PatternType.DIFFERENCE,
                answer="unclear",
                explanation="I need two things to compare.",
            )
        
        x, y = entities[0], entities[1]
        unique_x, unique_y = self._find_different_properties(x, y)
        sim = self._compute_similarity_score(x, y)
        
        return ReasoningResult(
            pattern_type=PatternType.DIFFERENCE,
            answer=f"{len(unique_x) + len(unique_y)} differences",
            confidence=1.0 - sim,
            different_properties=unique_x + unique_y,
            similarity_score=sim,
            facts_used=self._get_facts_for(x)[:3] + self._get_facts_for(y)[:3],
            explanation=f"{x} differs from {y} in {len(unique_x) + len(unique_y)} way(s).",
        )
    
    def _reason_comparison(self, entities: list[str]) -> ReasoningResult:
        if len(entities) < 2:
            return ReasoningResult(
                pattern_type=PatternType.COMPARISON,
                answer="unclear",
                explanation="I need two things to compare.",
            )
        
        x, y = entities[0], entities[1]
        sim = self._compute_similarity_score(x, y)
        shared = self._find_shared_properties(x, y)
        unique_x, unique_y = self._find_different_properties(x, y)
        
        return ReasoningResult(
            pattern_type=PatternType.COMPARISON,
            answer="compared",
            confidence=0.8,
            shared_properties=shared,
            different_properties=unique_x + unique_y,
            similarity_score=sim,
            facts_used=self._get_facts_for(x)[:3] + self._get_facts_for(y)[:3],
            explanation=f"Comparing {x} and {y}: {sim:.0%} similar.",
        )
    
    def _reason_capability(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.CAPABILITY,
                answer="unclear",
                explanation="I need to know what entity and ability to check.",
            )
        
        subject = entities[0]
        ability = entities[1] if len(entities) > 1 else None
        
        facts = self._get_facts_for(subject)
        can_facts = [f for f in facts if 'CAN' in f]
        does_facts = [f for f in facts if 'DOES' in f]
        
        if ability:
            # Check specific ability
            direct = [f for f in can_facts + does_facts if ability.lower() in f.lower()]
            if direct:
                return ReasoningResult(
                    pattern_type=PatternType.CAPABILITY,
                    answer="yes",
                    confidence=0.85,
                    facts_used=direct,
                    reasoning_chain=[f"Found: {d}" for d in direct],
                    explanation=f"Yes, {subject} can {ability}.",
                )
            
            # Try transitive
            chain = self._check_transitive(subject, "CAN", ability)
            if chain:
                return ReasoningResult(
                    pattern_type=PatternType.CAPABILITY,
                    answer="yes (inferred)",
                    confidence=0.6,
                    facts_used=chain,
                    reasoning_chain=chain,
                    explanation=f"{subject} can {ability} (inferred).",
                )
            
            # Check negation
            neg_facts = [f for f in facts if 'NOT' in f and ability.lower() in f.lower()]
            if neg_facts:
                return ReasoningResult(
                    pattern_type=PatternType.CAPABILITY,
                    answer="no",
                    confidence=0.9,
                    facts_used=neg_facts,
                    explanation=f"No, {subject} cannot {ability}.",
                )
            
            return ReasoningResult(
                pattern_type=PatternType.CAPABILITY,
                answer="unknown",
                confidence=0.0,
                explanation=f"I don't know if {subject} can {ability}.",
            )
        
        # List all capabilities
        return ReasoningResult(
            pattern_type=PatternType.CAPABILITY,
            answer=f"{len(can_facts)} abilities",
            confidence=0.7,
            facts_used=can_facts[:5],
            explanation=f"{subject} has {len(can_facts)} known abilities.",
        )
    
    def _reason_causation(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.CAUSATION,
                answer="unclear",
                explanation="I need to know what to explain.",
            )
        
        subject = entities[0]
        
        # Phase 16: Use CausalReasoner for deeper chains
        try:
            causal = self._get_causal()
            
            if len(entities) >= 2:
                # Explain causation between two entities
                result = causal.explain_causation(entities[0], entities[1])
            else:
                # Find causes of the subject
                result = causal.find_causes(subject)
            
            if result.best_chain:
                chain_steps = [
                    f"{s.cause} {s.relation} {s.effect}"
                    for s in result.best_chain.steps
                ]
                return ReasoningResult(
                    pattern_type=PatternType.CAUSATION,
                    answer="found",
                    confidence=result.confidence,
                    facts_used=chain_steps,
                    reasoning_chain=chain_steps,
                    explanation=result.explanation,
                )
        except Exception:
            pass
        
        # Fallback: search facts directly
        facts = self._get_facts_for(subject)
        causal_facts = [f for f in facts if any(
            kw in f.upper() for kw in ['CAUSES', 'BECAUSE', 'REASON', 'LEADS']
        )]
        
        if causal_facts:
            return ReasoningResult(
                pattern_type=PatternType.CAUSATION,
                answer="found",
                confidence=0.8,
                facts_used=causal_facts,
                reasoning_chain=causal_facts,
                explanation=f"Here's why about {subject}.",
            )
        
        # Try MultiHop to explain why
        try:
            mh = self._get_multihop()
            hop_result = mh.explain_why(subject, entities[1] if len(entities) > 1 else "")
            if hop_result.chain and hop_result.confidence > 0.3:
                chain_steps = [
                    f"{s.subject} {s.relation} {s.obj}"
                    for s in hop_result.chain.steps
                ]
                return ReasoningResult(
                    pattern_type=PatternType.CAUSATION,
                    answer="inferred",
                    confidence=hop_result.confidence,
                    facts_used=chain_steps,
                    reasoning_chain=chain_steps,
                    explanation=hop_result.explanation,
                )
        except Exception:
            pass
        
        return ReasoningResult(
            pattern_type=PatternType.CAUSATION,
            answer="partial",
            confidence=0.4,
            facts_used=facts[:5],
            explanation=f"I don't have specific causal information, but here's what I know about {subject}.",
        )
    
    def _reason_identity(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.IDENTITY,
                answer="unclear",
                explanation="I need to know what to identify.",
            )
        
        subject = entities[0]
        category = entities[1] if len(entities) > 1 else None
        
        facts = self._get_facts_for(subject)
        is_a_facts = [f for f in facts if 'IS-A' in f or 'IS' in f]
        
        if category:
            # Check specific identity
            matching = [
                f for f in is_a_facts
                if category.lower() in f.lower()
            ]
            if matching:
                return ReasoningResult(
                    pattern_type=PatternType.IDENTITY,
                    answer="yes",
                    confidence=0.9,
                    facts_used=matching,
                    explanation=f"Yes, {subject} is a {category}.",
                )
            
            # Try transitive
            chain = self._check_transitive(subject, "IS-A", category)
            if chain:
                return ReasoningResult(
                    pattern_type=PatternType.IDENTITY,
                    answer="yes (inferred)",
                    confidence=0.7,
                    facts_used=chain,
                    reasoning_chain=chain,
                    explanation=f"Yes, {subject} is a {category} (inferred through chain).",
                )
        
        # Return all identity facts
        return ReasoningResult(
            pattern_type=PatternType.IDENTITY,
            answer=f"{len(is_a_facts)} classifications",
            confidence=0.7,
            facts_used=is_a_facts[:5],
            explanation=f"Here's what {subject} is classified as.",
        )
    
    def _reason_possession(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.POSSESSION,
                answer="unclear",
                explanation="I need to know what entity to check.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        has_facts = [f for f in facts if 'HAS' in f]
        
        return ReasoningResult(
            pattern_type=PatternType.POSSESSION,
            answer=f"{len(has_facts)} possessions",
            confidence=0.7 if has_facts else 0.0,
            facts_used=has_facts[:5],
            explanation=f"{subject} has {len(has_facts)} known attributes/possessions.",
        )
    
    def _reason_location(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.LOCATION,
                answer="unclear",
                explanation="I need to know what to locate.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        loc_facts = [f for f in facts if any(
            kw in f.upper() for kw in ['LIVES-IN', 'IS-IN', 'IS-AT', 'IS-ON', 'LOCATION']
        )]
        
        if loc_facts:
            return ReasoningResult(
                pattern_type=PatternType.LOCATION,
                answer="found",
                confidence=0.8,
                facts_used=loc_facts,
                explanation=f"Here's where {subject} is located.",
            )
        
        return ReasoningResult(
            pattern_type=PatternType.LOCATION,
            answer="unknown",
            confidence=0.0,
            facts_used=facts[:3],
            explanation=f"I don't have location information for {subject}.",
        )
    
    def _reason_purpose(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.PURPOSE,
                answer="unclear",
                explanation="I need to know what to explain the purpose of.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        purpose_facts = [f for f in facts if any(
            kw in f.upper() for kw in ['USES', 'SERVES', 'PURPOSE', 'FOR']
        )]
        
        return ReasoningResult(
            pattern_type=PatternType.PURPOSE,
            answer=f"{len(purpose_facts)} uses",
            confidence=0.7 if purpose_facts else 0.3,
            facts_used=purpose_facts[:5] or facts[:3],
            explanation=f"{'Purpose' if purpose_facts else 'What I know about'} of {subject}.",
        )
    
    def _reason_composition(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.COMPOSITION,
                answer="unclear",
                explanation="I need to know what to analyze.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        comp_facts = [f for f in facts if any(
            kw in f.upper() for kw in ['HAS', 'CONTAINS', 'MADE', 'CONSISTS', 'COMPOSED']
        )]
        
        return ReasoningResult(
            pattern_type=PatternType.COMPOSITION,
            answer=f"{len(comp_facts)} components",
            confidence=0.7 if comp_facts else 0.0,
            facts_used=comp_facts[:5] or facts[:3],
            explanation=f"Composition of {subject}.",
        )
    
    def _reason_property(self, entities: list[str]) -> ReasoningResult:
        """Default: describe the entity's properties."""
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.PROPERTY,
                answer="unclear",
                explanation="I need to know what to describe.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        
        return ReasoningResult(
            pattern_type=PatternType.PROPERTY,
            answer=f"{len(facts)} facts",
            confidence=0.7 if facts else 0.0,
            facts_used=facts[:6],
            explanation=f"Properties of {subject}.",
        )
    
    def _reason_quantity(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.QUANTITY,
                answer="unclear",
                explanation="I need to know what to count.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        
        return ReasoningResult(
            pattern_type=PatternType.QUANTITY,
            answer=str(len(facts)),
            confidence=0.8,
            facts_used=facts[:5],
            explanation=f"I have {len(facts)} fact(s) about {subject}.",
        )
    
    def _reason_time(self, entities: list[str]) -> ReasoningResult:
        if not entities:
            return ReasoningResult(
                pattern_type=PatternType.TIME,
                answer="unclear",
                explanation="I need to know what to check the time for.",
            )
        
        subject = entities[0]
        facts = self._get_facts_for(subject)
        
        # Phase 16: Use enhanced TemporalReasoner
        try:
            temporal = self.gunter.temporal
            
            # Try WHEN query
            results = temporal.query_temporal(subject, "WHEN")
            if results:
                time_strs = [f"{v} [{c:.0%}]" for v, c in results]
                return ReasoningResult(
                    pattern_type=PatternType.TIME,
                    answer="found",
                    confidence=results[0][1],
                    facts_used=time_strs,
                    explanation=f"When {subject}: {', '.join(v for v, _ in results)}",
                )
            
            # Try DURATION
            if len(entities) > 1:
                duration = temporal.get_duration(subject, entities[1])
                if duration:
                    return ReasoningResult(
                        pattern_type=PatternType.TIME,
                        answer=duration,
                        confidence=0.85,
                        facts_used=[f"{subject} {entities[1]}: {duration}"],
                        explanation=f"{subject} {entities[1]} for {duration}.",
                    )
            else:
                duration = temporal.get_duration(subject)
                if duration:
                    return ReasoningResult(
                        pattern_type=PatternType.TIME,
                        answer=duration,
                        confidence=0.85,
                        facts_used=[f"{subject}: {duration}"],
                        explanation=f"{subject} lasts {duration}.",
                    )
            
            # Compare temporal if two entities
            if len(entities) >= 2:
                ordering = temporal.compare_temporal(entities[0], entities[1])
                if ordering != "UNKNOWN":
                    return ReasoningResult(
                        pattern_type=PatternType.TIME,
                        answer=ordering,
                        confidence=0.7,
                        facts_used=[f"{entities[0]} {ordering} {entities[1]}"],
                        explanation=f"{entities[0]} is {ordering.lower()} {entities[1]}.",
                    )
        except Exception:
            pass
        
        return ReasoningResult(
            pattern_type=PatternType.TIME,
            answer="unknown",
            confidence=0.0,
            facts_used=facts[:3],
            explanation=f"I don't have time information for {subject}.",
        )
    
    def _reason_relation(self, entities: list[str]) -> ReasoningResult:
        if len(entities) < 2:
            if entities:
                facts = self._get_facts_for(entities[0])
                return ReasoningResult(
                    pattern_type=PatternType.RELATION,
                    answer=f"{len(facts)} relations",
                    confidence=0.5,
                    facts_used=facts[:5],
                    explanation=f"Known relations for {entities[0]}.",
                )
            return ReasoningResult(
                pattern_type=PatternType.RELATION,
                answer="unclear",
                explanation="I need entities to check relations.",
            )
        
        x, y = entities[0], entities[1]
        facts_x = self._get_facts_for(x)
        facts_y = self._get_facts_for(y)
        
        # Find facts that mention both
        shared_facts = [f for f in facts_x if y.lower() in f.lower()]
        shared_facts += [f for f in facts_y if x.lower() in f.lower()]
        
        sim = self._compute_similarity_score(x, y)
        
        # Phase 16: Use MultiHopReasoner for connection finding
        try:
            mh = self._get_multihop()
            connection = mh.find_connection(x, y)
            if connection.chain and connection.confidence > 0.3:
                chain_steps = [
                    f"{s.subject} {s.relation} {s.obj}"
                    for s in connection.chain.steps
                ]
                return ReasoningResult(
                    pattern_type=PatternType.RELATION,
                    answer=f"connected ({connection.hops} hops)",
                    confidence=connection.confidence,
                    similarity_score=sim,
                    facts_used=chain_steps,
                    reasoning_chain=chain_steps,
                    explanation=connection.explanation,
                )
        except Exception:
            pass
        
        return ReasoningResult(
            pattern_type=PatternType.RELATION,
            answer=f"{len(shared_facts)} direct relations",
            confidence=sim,
            similarity_score=sim,
            facts_used=shared_facts[:5] or (facts_x[:2] + facts_y[:2]),
            explanation=f"Relation between {x} and {y}: {sim:.0%} similarity.",
        )
    
    # ── Phase 16: Advanced reasoning methods ──
    
    def solve_analogy(
        self, a: str, b: str, c: str, top_k: int = 5
    ) -> ReasoningResult:
        """Solve analogy: A is to B as C is to ?"""
        analogy = self._get_analogy()
        result = analogy.solve_analogy(a, b, c, top_k=top_k)
        
        return ReasoningResult(
            pattern_type=PatternType.RELATION,
            answer=result.answer,
            confidence=result.confidence,
            facts_used=[f"{a}:{b} :: {c}:{result.answer}"],
            reasoning_chain=[result.explanation],
            explanation=result.explanation,
        )
    
    def find_chain(self, start: str, end: str) -> ReasoningResult:
        """Find multi-hop reasoning chain between entities."""
        mh = self._get_multihop()
        result = mh.find_chain(start, end)
        
        chain_steps = []
        if result.chain:
            chain_steps = [
                f"{s.subject} {s.relation} {s.obj}"
                for s in result.chain.steps
            ]
        
        return ReasoningResult(
            pattern_type=PatternType.RELATION,
            answer=f"{result.hops} hops" if result.chain else "no path",
            confidence=result.confidence,
            facts_used=chain_steps,
            reasoning_chain=chain_steps,
            explanation=result.explanation,
        )
