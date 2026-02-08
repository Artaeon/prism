"""Explanation Generation — Explain reasoning for answers.

Tracks the last answer's reasoning chain and provides
explanations on demand via the 'why' command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReasoningTrace:
    """Record of how an answer was derived."""
    
    question: str = ""
    answer: str = ""
    source_type: str = ""  # "direct", "transitive", "similarity", "profile"
    facts_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    chain_steps: list[str] = field(default_factory=list)
    
    def format_explanation(self) -> str:
        """Format a human-readable explanation."""
        if not self.answer:
            return "I haven't answered anything yet."
        
        lines = [f"Q: {self.question}", f"A: {self.answer}", ""]
        
        source_desc = {
            "direct": "I know this because you told me directly.",
            "transitive": "I inferred this through a chain of facts.",
            "similarity": "This is based on semantic similarity.",
            "profile": "This is from your personal profile.",
            "concept_search": "This is from searching my stored facts.",
        }
        
        lines.append(source_desc.get(self.source_type, "Source unknown."))
        
        if self.facts_used:
            lines.append(f"\nFacts used ({len(self.facts_used)}):")
            for fact in self.facts_used[:5]:
                lines.append(f"  • {fact}")
        
        if self.chain_steps:
            lines.append(f"\nReasoning chain:")
            for step in self.chain_steps:
                lines.append(f"  → {step}")
        
        if self.confidence > 0:
            lines.append(f"\nConfidence: {self.confidence:.0%}")
        
        return "\n".join(lines)


class Explainer:
    """Track and explain reasoning.
    
    Maintains the last answer's context so users can ask 'why'.
    
    Example:
        >>> explainer = Explainer()
        >>> explainer.record_direct("Is cat an animal?", "Yes", "cat IS-A animal")
        >>> explainer.explain()  # "I know this because you told me..."
    """
    
    def __init__(self) -> None:
        """Initialize explainer."""
        self.last_trace: ReasoningTrace = ReasoningTrace()
    
    def record_direct(
        self, question: str, answer: str, fact: str, confidence: float = 0.95
    ) -> None:
        """Record a direct fact answer."""
        self.last_trace = ReasoningTrace(
            question=question,
            answer=answer,
            source_type="direct",
            facts_used=[fact],
            confidence=confidence,
        )
    
    def record_transitive(
        self, question: str, answer: str, 
        chain_steps: list[str], confidence: float,
    ) -> None:
        """Record a transitive inference answer."""
        self.last_trace = ReasoningTrace(
            question=question,
            answer=answer,
            source_type="transitive",
            facts_used=chain_steps,
            chain_steps=chain_steps,
            confidence=confidence,
        )
    
    def record_similarity(
        self, question: str, answer: str,
        similar_items: list[str], confidence: float,
    ) -> None:
        """Record a similarity-based answer."""
        self.last_trace = ReasoningTrace(
            question=question,
            answer=answer,
            source_type="similarity",
            facts_used=similar_items,
            confidence=confidence,
        )
    
    def record_profile(
        self, question: str, answer: str, confidence: float = 0.90
    ) -> None:
        """Record a user profile answer."""
        self.last_trace = ReasoningTrace(
            question=question,
            answer=answer,
            source_type="profile",
            confidence=confidence,
        )
    
    def record_concept(
        self, question: str, answer: str,
        facts: list[str], confidence: float = 0.85,
    ) -> None:
        """Record a concept search answer."""
        self.last_trace = ReasoningTrace(
            question=question,
            answer=answer,
            source_type="concept_search",
            facts_used=facts,
            confidence=confidence,
        )
    
    def explain(self) -> str:
        """Explain the last answer."""
        return self.last_trace.format_explanation()
