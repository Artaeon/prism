"""Multi-Hop Reasoner — Deep inference chains through fact graphs.

Wraps TransitiveReasoner with higher-level methods:
- find_chain: directed BFS from start to end entity
- explain_why: find explanatory chain for why entity has property
- find_connection: bidirectional search for any conceptual link
- get_all_paths: enumerate multiple paths between entities

Example:
    >>> mhr = MultiHopReasoner(memory, lexicon)
    >>> chain = mhr.explain_why("cats", "purr")
    >>> # cats ARE mammals → mammals HAVE vocal-cords → vocal-cords PRODUCE sounds
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gunter.memory import VectorMemory
    from gunter.core.lexicon import Lexicon

from gunter.reasoning.transitive import (
    TransitiveReasoner,
    InferenceChain,
    InferenceStep,
)


@dataclass
class HopResult:
    """Result of a multi-hop reasoning query."""
    
    chain: InferenceChain | None = None
    all_chains: list[InferenceChain] = field(default_factory=list)
    explanation: str = ""
    hops: int = 0
    confidence: float = 0.0


class MultiHopReasoner:
    """Deep multi-hop inference over stored facts.
    
    Extends TransitiveReasoner with:
    - Explanatory chains (why does X have property Y?)
    - Connection finding (how are X and Y related?)
    - Multiple path enumeration
    - Semantic similarity fallback when BFS fails
    
    Example:
        >>> reasoner = MultiHopReasoner(memory, lexicon)
        >>> result = reasoner.explain_why("cats", "purr")
        >>> print(result.explanation)
        "cats → mammals → vocal-cords → sounds → purr"
    """
    
    # Maximum search depth
    MAX_DEPTH = 5
    # Minimum confidence to keep a chain
    MIN_CONFIDENCE = 0.3
    
    def __init__(self, memory: 'VectorMemory', lexicon: 'Lexicon') -> None:
        self.memory = memory
        self.lexicon = lexicon
        self.transitive = TransitiveReasoner(memory)
    
    def find_chain(
        self,
        start: str,
        end: str,
        max_depth: int = 5,
    ) -> HopResult:
        """Find reasoning path from start to end entity.
        
        Uses BFS through the fact graph with confidence pruning.
        
        Args:
            start: Starting entity
            end: Target entity
            max_depth: Maximum hops (default 5)
            
        Returns:
            HopResult with best chain and all found chains
        """
        max_depth = min(max_depth, self.MAX_DEPTH)
        
        # Try direct transitive inference
        chain = self.transitive.infer(start, end, max_hops=max_depth)
        
        if chain and chain.confidence >= self.MIN_CONFIDENCE:
            return HopResult(
                chain=chain,
                all_chains=[chain],
                explanation=self._format_chain(chain),
                hops=len(chain.steps),
                confidence=chain.confidence,
            )
        
        # Try with normalized forms
        start_norm = self.transitive._normalize(start)
        end_norm = self.transitive._normalize(end)
        
        if start_norm != start or end_norm != end:
            chain = self.transitive.infer(start_norm, end_norm, max_hops=max_depth)
            if chain and chain.confidence >= self.MIN_CONFIDENCE:
                return HopResult(
                    chain=chain,
                    all_chains=[chain],
                    explanation=self._format_chain(chain),
                    hops=len(chain.steps),
                    confidence=chain.confidence,
                )
        
        # Try semantic similarity bridge
        bridge_result = self._try_semantic_bridge(start, end, max_depth)
        if bridge_result:
            return bridge_result
        
        return HopResult(
            explanation=f"No reasoning path found from '{start}' to '{end}'.",
        )
    
    def explain_why(self, entity: str, prop: str) -> HopResult:
        """Find explanatory chain for why entity has a property.
        
        Strategy:
        1. Search for direct fact: entity HAS/IS/CAN prop
        2. If not found, explore all paths from entity
        3. Check if any path leads to prop or related concept
        
        Args:
            entity: The entity to explain about
            prop: The property/behavior to explain
            
        Returns:
            HopResult with explanatory chain
        """
        # 1. Check direct fact
        score = self.memory.check_fact(entity, "IS", prop)
        if score > 0.1:
            step = InferenceStep(entity, "IS", prop, score)
            chain = InferenceChain(steps=[step])
            return HopResult(
                chain=chain,
                all_chains=[chain],
                explanation=f"{entity} IS {prop} (direct fact)",
                hops=1,
                confidence=score,
            )
        
        # 2. Try direct chain
        direct = self.find_chain(entity, prop, max_depth=4)
        if direct.chain:
            return direct
        
        # 3. Explore all paths from entity, find ones mentioning prop
        all_chains = self.transitive.infer_all(entity, max_hops=3, min_confidence=0.3)
        
        matching = []
        prop_lower = prop.lower()
        for chain in all_chains:
            for step in chain.steps:
                if (prop_lower in step.obj.lower() or 
                    prop_lower in step.relation.lower() or
                    prop_lower in step.subject.lower()):
                    matching.append(chain)
                    break
        
        if matching:
            best = max(matching, key=lambda c: c.confidence)
            return HopResult(
                chain=best,
                all_chains=matching[:5],
                explanation=self._format_explanation(entity, prop, best),
                hops=len(best.steps),
                confidence=best.confidence,
            )
        
        # 4. Try semantic similarity to find related concepts
        related = self._find_related_facts(entity, prop)
        if related:
            return HopResult(
                chain=related,
                all_chains=[related],
                explanation=self._format_explanation(entity, prop, related),
                hops=len(related.steps),
                confidence=related.confidence,
            )
        
        return HopResult(
            explanation=f"I can't fully explain why {entity} {prop}.",
        )
    
    def find_connection(self, entity1: str, entity2: str) -> HopResult:
        """Find how two entities are connected.
        
        Uses bidirectional search: try from both directions
        and pick the shorter chain.
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            HopResult with connection chain
        """
        # Forward: entity1 → entity2
        forward = self.find_chain(entity1, entity2, max_depth=4)
        
        # Backward: entity2 → entity1
        backward = self.find_chain(entity2, entity1, max_depth=4)
        
        best = None
        if forward.chain and backward.chain:
            best = forward if forward.confidence >= backward.confidence else backward
        elif forward.chain:
            best = forward
        elif backward.chain:
            best = backward
        
        if best and best.chain:
            return HopResult(
                chain=best.chain,
                all_chains=([forward.chain] if forward.chain else []) + 
                           ([backward.chain] if backward.chain else []),
                explanation=f"Connection: {self._format_chain(best.chain)}",
                hops=best.hops,
                confidence=best.confidence,
            )
        
        # Try finding common ancestor
        common = self._find_common_ancestor(entity1, entity2)
        if common:
            return common
        
        return HopResult(
            explanation=f"No direct connection found between '{entity1}' and '{entity2}'.",
        )
    
    def get_chain_confidence(self, chain: InferenceChain) -> float:
        """Get overall confidence of a chain (product of step confidences)."""
        return chain.confidence
    
    # ── Internal helpers ──
    
    def _format_chain(self, chain: InferenceChain) -> str:
        """Format a chain as readable arrows."""
        if not chain.steps:
            return "(empty chain)"
        
        parts = []
        for step in chain.steps:
            parts.append(f"{step.subject} {step.relation} {step.obj}")
        return " → ".join(parts)
    
    def _format_explanation(
        self, entity: str, prop: str, chain: InferenceChain
    ) -> str:
        """Format an explanation chain."""
        if not chain.steps:
            return f"No explanation found for why {entity} {prop}."
        
        lines = [f"Why {entity} {prop}:"]
        for i, step in enumerate(chain.steps):
            prefix = "  " + "→ " * (i + 1)
            lines.append(f"{prefix}{step.subject} {step.relation} {step.obj}")
        return "\n".join(lines)
    
    def _try_semantic_bridge(
        self, start: str, end: str, max_depth: int
    ) -> HopResult | None:
        """Try to bridge start and end via semantic similarity."""
        vec_start = self.lexicon.get(start)
        vec_end = self.lexicon.get(end)
        if vec_start is None or vec_end is None:
            return None
        
        ops = self.lexicon.ops
        
        # Find intermediary concepts that are similar to both
        episodes = self.memory.get_episodes()
        candidates = set()
        for ep in episodes:
            if ep.subject:
                candidates.add(ep.subject.lower())
            if ep.obj:
                candidates.add(ep.obj.lower())
        
        bridges = []
        for candidate in candidates:
            if candidate in (start.lower(), end.lower()):
                continue
            vec_c = self.lexicon.get(candidate)
            if vec_c is None:
                continue
            sim_to_start = ops.similarity(vec_start, vec_c)
            sim_to_end = ops.similarity(vec_c, vec_end)
            if sim_to_start > 0.3 and sim_to_end > 0.3:
                bridges.append((candidate, sim_to_start, sim_to_end))
        
        if not bridges:
            return None
        
        # Pick best bridge (highest combined similarity)
        bridges.sort(key=lambda x: x[1] + x[2], reverse=True)
        bridge, sim1, sim2 = bridges[0]
        
        # Try: start → bridge → end
        chain1 = self.transitive.infer(start, bridge, max_hops=2)
        chain2 = self.transitive.infer(bridge, end, max_hops=2)
        
        if chain1 and chain2:
            combined = InferenceChain(steps=chain1.steps + chain2.steps)
            if combined.confidence >= self.MIN_CONFIDENCE:
                return HopResult(
                    chain=combined,
                    all_chains=[combined],
                    explanation=self._format_chain(combined),
                    hops=len(combined.steps),
                    confidence=combined.confidence,
                )
        
        # Create semantic bridge step
        avg_sim = (sim1 + sim2) / 2
        steps = [
            InferenceStep(start, "RELATED-TO", bridge, sim1),
            InferenceStep(bridge, "RELATED-TO", end, sim2),
        ]
        chain = InferenceChain(steps=steps)
        return HopResult(
            chain=chain,
            all_chains=[chain],
            explanation=f"Semantic bridge: {start} ~{sim1:.0%}~ {bridge} ~{sim2:.0%}~ {end}",
            hops=2,
            confidence=avg_sim,
        )
    
    def _find_related_facts(self, entity: str, prop: str) -> InferenceChain | None:
        """Find facts that relate entity to prop via semantic search."""
        entity_lower = entity.lower()
        prop_lower = prop.lower()
        
        episodes = self.memory.get_episodes()
        
        # Facts about entity
        entity_facts = [
            ep for ep in episodes
            if ep.subject and entity_lower in ep.subject.lower()
        ]
        
        # Search for prop in objects of entity's facts
        for ef in entity_facts:
            obj_lower = ef.obj.lower()
            # Check if this object has facts containing prop
            for ep in episodes:
                if (ep.subject and obj_lower in ep.subject.lower() and 
                    prop_lower in ep.obj.lower()):
                    steps = [
                        InferenceStep(entity, ef.relation, ef.obj, 0.9),
                        InferenceStep(ef.obj, ep.relation, ep.obj, 0.85),
                    ]
                    return InferenceChain(steps=steps)
        
        return None
    
    def _find_common_ancestor(
        self, entity1: str, entity2: str
    ) -> HopResult | None:
        """Find a common concept that both entities connect to."""
        chains1 = self.transitive.infer_all(entity1, max_hops=2, min_confidence=0.3)
        chains2 = self.transitive.infer_all(entity2, max_hops=2, min_confidence=0.3)
        
        # Collect all endpoints from entity1
        endpoints1: dict[str, InferenceChain] = {}
        for c in chains1:
            endpoints1[c.end] = c
        
        # Check if any endpoint from entity2 overlaps
        for c in chains2:
            if c.end in endpoints1:
                ancestor = c.end
                chain1 = endpoints1[ancestor]
                chain2 = c
                
                explanation = (
                    f"Both connect through '{ancestor}':\n"
                    f"  {entity1}: {self._format_chain(chain1)}\n"
                    f"  {entity2}: {self._format_chain(chain2)}"
                )
                
                avg_conf = (chain1.confidence + chain2.confidence) / 2
                return HopResult(
                    chain=chain1,
                    all_chains=[chain1, chain2],
                    explanation=explanation,
                    hops=len(chain1.steps) + len(chain2.steps),
                    confidence=avg_conf,
                )
        
        return None
