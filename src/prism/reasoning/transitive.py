"""Transitive Reasoning — Multi-hop inference.

Given facts A→B and B→C, infer A→C with confidence scores.
Uses BFS through the fact graph to find inference chains.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory


@dataclass
class InferenceStep:
    """A single step in an inference chain."""
    
    subject: str
    relation: str
    obj: str
    confidence: float = 0.95  # Per-hop confidence


@dataclass  
class InferenceChain:
    """Complete inference chain from start to end."""
    
    steps: list[InferenceStep] = field(default_factory=list)
    
    @property
    def confidence(self) -> float:
        """Overall confidence = product of step confidences."""
        if not self.steps:
            return 0.0
        result = 1.0
        for step in self.steps:
            result *= step.confidence
        return result
    
    @property
    def start(self) -> str:
        """Start of chain."""
        return self.steps[0].subject if self.steps else ""
    
    @property
    def end(self) -> str:
        """End of chain."""
        return self.steps[-1].obj if self.steps else ""
    
    def format(self) -> str:
        """Format chain for display."""
        if not self.steps:
            return "(no chain)"
        
        parts = []
        for step in self.steps:
            parts.append(
                f"{step.subject} {step.relation} {step.obj} [{step.confidence:.2f}]"
            )
        
        chain_str = " → ".join(parts)
        return f"{chain_str} (overall: {self.confidence:.2f})"


class TransitiveReasoner:
    """Multi-hop inference over stored facts.
    
    Builds a graph from episodic memories and searches for
    paths between concepts using BFS.
    
    Example:
        >>> reasoner = TransitiveReasoner(memory)
        >>> # Given: cat IS-A animal, animal NEEDS water
        >>> chain = reasoner.infer("cat", "water")
        >>> # Returns: cat → animal → water
    """
    
    # Confidence per hop (decreases with distance)
    HOP_CONFIDENCE = 0.95
    
    def __init__(self, memory: VectorMemory) -> None:
        """Initialize with vector memory."""
        self.memory = memory
    
    def _normalize(self, word: str) -> str:
        """Normalize word for graph matching (simple stemming)."""
        w = word.lower().strip()
        # Simple plural stripping
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            return w[:-1]
        return w
    
    def _build_graph(self) -> dict[str, list[tuple[str, str, float]]]:
        """Build adjacency graph from episodic memories.
        
        Returns:
            Dict mapping subject → [(relation, object, confidence), ...]
        """
        graph: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        
        for episode in self.memory.get_episodes():
            if episode.subject and episode.relation and episode.obj:
                s_raw = episode.subject.lower()
                r = episode.relation
                o_raw = episode.obj.lower()
                
                s_norm = self._normalize(s_raw)
                o_norm = self._normalize(o_raw)
                
                # Add edges for both raw and normalized forms
                for s in {s_raw, s_norm}:
                    for o in {o_raw, o_norm}:
                        graph[s].append((r, o, self.HOP_CONFIDENCE))
                
                # For IS-A, also add reverse lookup
                if r == "IS-A":
                    for o in {o_raw, o_norm}:
                        for s in {s_raw, s_norm}:
                            graph[o].append(("HAS-INSTANCE", s, self.HOP_CONFIDENCE * 0.8))
        
        return graph
    
    def infer(
        self,
        start: str,
        end: str,
        max_hops: int = 3,
    ) -> InferenceChain | None:
        """Find inference chain from start to end concept.
        
        Uses BFS to find shortest path through fact graph.
        
        Args:
            start: Starting concept
            end: Target concept
            max_hops: Maximum inference depth
            
        Returns:
            InferenceChain if path found, None otherwise
        """
        start = self._normalize(start)
        end = self._normalize(end)
        
        if start == end:
            return None
        
        graph = self._build_graph()
        
        if start not in graph:
            return None
        
        # BFS: queue of (current_node, chain_so_far)
        queue: deque[tuple[str, list[InferenceStep]]] = deque()
        visited: set[str] = {start}
        
        # Initialize with all edges from start
        for relation, target, conf in graph.get(start, []):
            step = InferenceStep(
                subject=start, relation=relation, obj=target, confidence=conf
            )
            if target == end:
                return InferenceChain(steps=[step])
            if len([step]) < max_hops:
                queue.append((target, [step]))
                visited.add(target)
        
        # BFS
        while queue:
            current, chain = queue.popleft()
            
            if len(chain) >= max_hops:
                continue
            
            for relation, target, conf in graph.get(current, []):
                if target in visited:
                    continue
                
                # Confidence decreases with each hop
                hop_conf = conf * (0.95 ** len(chain))
                step = InferenceStep(
                    subject=current, relation=relation, obj=target,
                    confidence=hop_conf,
                )
                new_chain = chain + [step]
                
                if target == end:
                    return InferenceChain(steps=new_chain)
                
                if len(new_chain) < max_hops:
                    visited.add(target)
                    queue.append((target, new_chain))
        
        return None
    
    def infer_all(
        self,
        start: str,
        max_hops: int = 2,
        min_confidence: float = 0.5,
    ) -> list[InferenceChain]:
        """Find all reachable inferences from a concept.
        
        Args:
            start: Starting concept
            max_hops: Maximum depth
            min_confidence: Minimum chain confidence
            
        Returns:
            List of inference chains sorted by confidence
        """
        start = start.lower()
        graph = self._build_graph()
        
        if start not in graph:
            return []
        
        chains: list[InferenceChain] = []
        queue: deque[tuple[str, list[InferenceStep]]] = deque()
        visited: set[str] = {start}
        
        for relation, target, conf in graph.get(start, []):
            step = InferenceStep(
                subject=start, relation=relation, obj=target, confidence=conf
            )
            chain = InferenceChain(steps=[step])
            if chain.confidence >= min_confidence:
                chains.append(chain)
            if len(chain.steps) < max_hops:
                queue.append((target, [step]))
                visited.add(target)
        
        while queue:
            current, steps = queue.popleft()
            
            if len(steps) >= max_hops:
                continue
            
            for relation, target, conf in graph.get(current, []):
                if target in visited:
                    continue
                
                hop_conf = conf * (0.95 ** len(steps))
                step = InferenceStep(
                    subject=current, relation=relation, obj=target,
                    confidence=hop_conf,
                )
                new_steps = steps + [step]
                chain = InferenceChain(steps=new_steps)
                
                if chain.confidence >= min_confidence:
                    chains.append(chain)
                
                if len(new_steps) < max_hops:
                    visited.add(target)
                    queue.append((target, new_steps))
        
        chains.sort(key=lambda c: c.confidence, reverse=True)
        return chains
    
    def can_infer(self, start: str, end: str, max_hops: int = 3) -> bool:
        """Quick check if a path exists."""
        return self.infer(start, end, max_hops) is not None
