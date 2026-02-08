"""Causal Reasoner — Forward and backward causal chain inference.

Searches the fact graph for causal relations (CAUSES, LEADS-TO, etc.)
and builds explanatory chains from effects to causes or vice versa.

Example:
    >>> cr = CausalReasoner(memory, lexicon)
    >>> result = cr.find_causes("rain")
    >>> # evaporation CAUSES clouds → clouds CAUSE rain
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gunter.memory import VectorMemory
    from gunter.core.lexicon import Lexicon


# Relations that imply causation
CAUSAL_RELATIONS = {
    'CAUSES', 'CAUSED-BY', 'LEADS-TO', 'RESULTS-IN',
    'BECAUSE', 'DUE-TO', 'MAKES', 'PRODUCES',
    'CREATES', 'TRIGGERS', 'ENABLES', 'PREVENTS',
}

# Relations that are implicitly forward-causal
FORWARD_CAUSAL = {
    'CAUSES', 'LEADS-TO', 'RESULTS-IN', 'MAKES',
    'PRODUCES', 'CREATES', 'TRIGGERS', 'ENABLES',
}

# Relations that are implicitly backward-causal
BACKWARD_CAUSAL = {
    'CAUSED-BY', 'BECAUSE', 'DUE-TO',
}


@dataclass
class CausalStep:
    """A single step in a causal chain."""
    
    cause: str
    relation: str
    effect: str
    confidence: float = 0.9


@dataclass
class CausalChain:
    """A complete causal chain from cause to effect."""
    
    steps: list[CausalStep] = field(default_factory=list)
    
    @property
    def confidence(self) -> float:
        if not self.steps:
            return 0.0
        result = 1.0
        for step in self.steps:
            result *= step.confidence
        return result
    
    @property
    def root_cause(self) -> str:
        return self.steps[0].cause if self.steps else ""
    
    @property
    def final_effect(self) -> str:
        return self.steps[-1].effect if self.steps else ""
    
    def format(self) -> str:
        if not self.steps:
            return "(no causal chain)"
        parts = []
        for step in self.steps:
            parts.append(f"{step.cause} {step.relation} {step.effect}")
        return " → ".join(parts)


@dataclass
class CausalResult:
    """Result of a causal reasoning query."""
    
    chains: list[CausalChain] = field(default_factory=list)
    best_chain: CausalChain | None = None
    explanation: str = ""
    confidence: float = 0.0


class CausalReasoner:
    """Causal chain inference over stored facts.
    
    Builds forward (cause → effect) and backward (effect → cause)
    chains using explicitly causal relations. Falls back to general
    fact graph search when explicit causal relations aren't found.
    
    Example:
        >>> reasoner = CausalReasoner(memory, lexicon)
        >>> result = reasoner.find_causes("rain")
        >>> print(result.explanation)
        "evaporation CAUSES clouds → clouds CAUSE rain"
    """
    
    MAX_DEPTH = 3
    MIN_CONFIDENCE = 0.2
    
    def __init__(self, memory: 'VectorMemory', lexicon: 'Lexicon') -> None:
        self.memory = memory
        self.lexicon = lexicon
    
    def find_causes(self, effect: str, max_depth: int = 3) -> CausalResult:
        """Search backward from effect to find causes.
        
        Args:
            effect: The effect/outcome to explain
            max_depth: Maximum chain depth
            
        Returns:
            CausalResult with causal chains
        """
        max_depth = min(max_depth, self.MAX_DEPTH)
        graph = self._build_causal_graph()
        
        # Build reverse graph for backward search
        reverse_graph: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for cause, edges in graph.items():
            for rel, eff, conf in edges:
                reverse_graph[eff.lower()].append((rel, cause, conf))
        
        # BFS backward from effect
        chains = self._bfs_chains(
            effect.lower(), reverse_graph, max_depth, reverse=True
        )
        
        # Also search general fact graph for implicit causation
        general_chains = self._search_general_causation(effect, "backward", max_depth)
        chains.extend(general_chains)
        
        # Deduplicate and sort
        chains = self._deduplicate_chains(chains)
        chains.sort(key=lambda c: c.confidence, reverse=True)
        
        best = chains[0] if chains else None
        return CausalResult(
            chains=chains[:5],
            best_chain=best,
            explanation=self._format_causes(effect, chains),
            confidence=best.confidence if best else 0.0,
        )
    
    def find_effects(self, cause: str, max_depth: int = 3) -> CausalResult:
        """Search forward from cause to find effects.
        
        Args:
            cause: The cause to trace forward
            max_depth: Maximum chain depth
            
        Returns:
            CausalResult with effect chains
        """
        max_depth = min(max_depth, self.MAX_DEPTH)
        graph = self._build_causal_graph()
        
        chains = self._bfs_chains(cause.lower(), graph, max_depth, reverse=False)
        
        # Also search general facts
        general_chains = self._search_general_causation(cause, "forward", max_depth)
        chains.extend(general_chains)
        
        chains = self._deduplicate_chains(chains)
        chains.sort(key=lambda c: c.confidence, reverse=True)
        
        best = chains[0] if chains else None
        return CausalResult(
            chains=chains[:5],
            best_chain=best,
            explanation=self._format_effects(cause, chains),
            confidence=best.confidence if best else 0.0,
        )
    
    def explain_causation(self, cause: str, effect: str) -> CausalResult:
        """Find and explain the causal path between cause and effect.
        
        Tries direct chain first, then explores intermediate paths.
        
        Args:
            cause: The proposed cause
            effect: The proposed effect
            
        Returns:
            CausalResult with explanatory chain
        """
        graph = self._build_causal_graph()
        
        # Direct chain search
        chain = self._find_path(cause.lower(), effect.lower(), graph, self.MAX_DEPTH)
        if chain:
            return CausalResult(
                chains=[chain],
                best_chain=chain,
                explanation=f"{cause} causes {effect}: {chain.format()}",
                confidence=chain.confidence,
            )
        
        # Try via general fact graph
        from gunter.reasoning.transitive import TransitiveReasoner
        transitive = TransitiveReasoner(self.memory)
        inf_chain = transitive.infer(cause, effect, max_hops=4)
        
        if inf_chain and inf_chain.confidence >= 0.3:
            causal_chain = CausalChain(
                steps=[
                    CausalStep(s.subject, s.relation, s.obj, s.confidence)
                    for s in inf_chain.steps
                ]
            )
            return CausalResult(
                chains=[causal_chain],
                best_chain=causal_chain,
                explanation=f"Connection: {causal_chain.format()}",
                confidence=causal_chain.confidence,
            )
        
        return CausalResult(
            explanation=f"No causal path found from '{cause}' to '{effect}'.",
        )
    
    @staticmethod
    def detect_causal_relation(relation_name: str) -> bool:
        """Check if a relation name implies causation."""
        rel_upper = relation_name.upper()
        if rel_upper in CAUSAL_RELATIONS:
            return True
        # Fuzzy check
        causal_keywords = ['cause', 'lead', 'result', 'because', 'due',
                           'make', 'produce', 'create', 'trigger', 'prevent']
        return any(kw in rel_upper.lower() for kw in causal_keywords)
    
    # ── Internal helpers ──
    
    def _build_causal_graph(self) -> dict[str, list[tuple[str, str, float]]]:
        """Build a directed graph of causal relations from memory."""
        graph: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        
        for ep in self.memory.get_episodes():
            if not ep.subject or not ep.relation or not ep.obj:
                continue
            
            rel_upper = ep.relation.upper()
            subj = ep.subject.lower()
            obj = ep.obj.lower()
            
            if rel_upper in FORWARD_CAUSAL:
                graph[subj].append((ep.relation, obj, 0.9))
            elif rel_upper in BACKWARD_CAUSAL:
                # Reverse: the subject is caused by the object
                graph[obj].append((ep.relation, subj, 0.9))
            elif self.detect_causal_relation(ep.relation):
                graph[subj].append((ep.relation, obj, 0.7))
        
        return graph
    
    def _bfs_chains(
        self,
        start: str,
        graph: dict[str, list[tuple[str, str, float]]],
        max_depth: int,
        reverse: bool = False,
    ) -> list[CausalChain]:
        """BFS to find all causal chains from start."""
        chains: list[CausalChain] = []
        queue: deque[tuple[str, list[CausalStep]]] = deque()
        visited: set[str] = {start}
        
        for rel, target, conf in graph.get(start, []):
            if reverse:
                step = CausalStep(cause=target, relation=rel, effect=start, confidence=conf)
            else:
                step = CausalStep(cause=start, relation=rel, effect=target, confidence=conf)
            
            chain = CausalChain(steps=[step])
            if chain.confidence >= self.MIN_CONFIDENCE:
                chains.append(chain)
            
            if len(chain.steps) < max_depth and target not in visited:
                visited.add(target)
                queue.append((target, [step]))
        
        while queue:
            current, steps = queue.popleft()
            if len(steps) >= max_depth:
                continue
            
            for rel, target, conf in graph.get(current, []):
                if target in visited:
                    continue
                
                hop_conf = conf * (0.9 ** len(steps))
                if reverse:
                    step = CausalStep(cause=target, relation=rel, effect=current, confidence=hop_conf)
                else:
                    step = CausalStep(cause=current, relation=rel, effect=target, confidence=hop_conf)
                
                new_steps = steps + [step]
                chain = CausalChain(steps=new_steps)
                
                if chain.confidence >= self.MIN_CONFIDENCE:
                    chains.append(chain)
                
                if len(new_steps) < max_depth:
                    visited.add(target)
                    queue.append((target, new_steps))
        
        return chains
    
    def _find_path(
        self,
        start: str,
        end: str,
        graph: dict[str, list[tuple[str, str, float]]],
        max_depth: int,
    ) -> CausalChain | None:
        """Find a specific path from start to end in the causal graph."""
        queue: deque[tuple[str, list[CausalStep]]] = deque()
        visited: set[str] = {start}
        
        for rel, target, conf in graph.get(start, []):
            step = CausalStep(cause=start, relation=rel, effect=target, confidence=conf)
            if target == end:
                return CausalChain(steps=[step])
            if len([step]) < max_depth:
                visited.add(target)
                queue.append((target, [step]))
        
        while queue:
            current, steps = queue.popleft()
            if len(steps) >= max_depth:
                continue
            
            for rel, target, conf in graph.get(current, []):
                if target in visited:
                    continue
                
                hop_conf = conf * (0.9 ** len(steps))
                step = CausalStep(cause=current, relation=rel, effect=target, confidence=hop_conf)
                new_steps = steps + [step]
                
                if target == end:
                    return CausalChain(steps=new_steps)
                
                if len(new_steps) < max_depth:
                    visited.add(target)
                    queue.append((target, new_steps))
        
        return None
    
    def _search_general_causation(
        self, entity: str, direction: str, max_depth: int
    ) -> list[CausalChain]:
        """Search the general fact graph for implicit causal patterns."""
        chains = []
        entity_lower = entity.lower()
        
        episodes = self.memory.get_episodes()
        
        # Find facts mentioning the entity
        entity_facts = []
        for ep in episodes:
            if not ep.subject or not ep.obj:
                continue
            if (entity_lower in ep.subject.lower() or 
                entity_lower in ep.obj.lower()):
                entity_facts.append(ep)
        
        # For each fact, check if the linked entity has causal facts
        for ef in entity_facts:
            linked = ef.obj.lower() if entity_lower in ef.subject.lower() else ef.subject.lower()
            
            for ep2 in episodes:
                if not ep2.subject or not ep2.obj:
                    continue
                if (ep2.subject.lower() == linked and 
                    self.detect_causal_relation(ep2.relation)):
                    
                    if direction == "backward":
                        steps = [
                            CausalStep(ep2.subject, ep2.relation, ep2.obj, 0.7),
                            CausalStep(linked, ef.relation, entity, 0.6),
                        ]
                    else:
                        steps = [
                            CausalStep(entity, ef.relation, linked, 0.6),
                            CausalStep(ep2.subject, ep2.relation, ep2.obj, 0.7),
                        ]
                    
                    chain = CausalChain(steps=steps)
                    if chain.confidence >= self.MIN_CONFIDENCE:
                        chains.append(chain)
        
        return chains[:5]
    
    def _deduplicate_chains(self, chains: list[CausalChain]) -> list[CausalChain]:
        """Remove duplicate chains."""
        seen: set[str] = set()
        unique = []
        for chain in chains:
            key = chain.format()
            if key not in seen:
                seen.add(key)
                unique.append(chain)
        return unique
    
    def _format_causes(self, effect: str, chains: list[CausalChain]) -> str:
        """Format cause chains as readable text."""
        if not chains:
            return f"No known causes for '{effect}'."
        
        lines = [f"Causes of {effect}:"]
        for i, chain in enumerate(chains[:3], 1):
            lines.append(f"  {i}. {chain.format()} [{chain.confidence:.0%}]")
        return "\n".join(lines)
    
    def _format_effects(self, cause: str, chains: list[CausalChain]) -> str:
        """Format effect chains as readable text."""
        if not chains:
            return f"No known effects of '{cause}'."
        
        lines = [f"Effects of {cause}:"]
        for i, chain in enumerate(chains[:3], 1):
            lines.append(f"  {i}. {chain.format()} [{chain.confidence:.0%}]")
        return "\n".join(lines)
