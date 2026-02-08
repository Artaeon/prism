"""Answer Generator — Template-based natural language answer formatting.

Converts structured ReasoningResult objects into human-readable
responses with confidence indicators, fact citations, and reasoning chains.
"""

from __future__ import annotations

from typing import Any

from gunter.reasoning.pattern_library import PatternType
from gunter.reasoning.vsa_reasoner import ReasoningResult


def _confidence_icon(score: float) -> str:
    """Return a visual confidence indicator."""
    if score >= 0.8:
        return "●"
    elif score >= 0.6:
        return "◕"
    elif score >= 0.4:
        return "◐"
    elif score >= 0.2:
        return "◔"
    return "○"


def _format_confidence(score: float) -> str:
    """Format confidence as percentage with icon."""
    icon = _confidence_icon(score)
    return f"{icon} [{score:.0%}]"


def _format_fact_list(facts: list[str], max_items: int = 5) -> str:
    """Format a list of facts as bullet points."""
    if not facts:
        return "  (no facts found)"
    lines = []
    for f in facts[:max_items]:
        lines.append(f"  • {f}")
    if len(facts) > max_items:
        lines.append(f"  ... and {len(facts) - max_items} more")
    return "\n".join(lines)


def _format_reasoning_chain(steps: list[str]) -> str:
    """Format a reasoning chain as numbered steps."""
    if not steps:
        return ""
    lines = ["  Reasoning:"]
    for i, step in enumerate(steps, 1):
        lines.append(f"    {i}. {step}")
    return "\n".join(lines)


def _format_chain_arrows(steps: list[str]) -> str:
    """Format reasoning steps as arrow chain."""
    if not steps:
        return ""
    return "  Chain: " + " → ".join(steps)


class AnswerGenerator:
    """Generate natural language answers from reasoning results.
    
    Uses pattern-specific templates for each of the 15 question types.
    All answers include confidence indicators and supporting facts.
    
    Example:
        >>> gen = AnswerGenerator()
        >>> answer = gen.generate_answer(result)
        >>> print(answer)  # "No, cats and dogs are different..."
    """
    
    def generate_answer(self, result: ReasoningResult) -> str:
        """Generate a formatted answer from a reasoning result."""
        generators = {
            PatternType.SAMENESS: self._gen_sameness,
            PatternType.SIMILARITY: self._gen_similarity,
            PatternType.DIFFERENCE: self._gen_difference,
            PatternType.COMPARISON: self._gen_comparison,
            PatternType.CAPABILITY: self._gen_capability,
            PatternType.CAUSATION: self._gen_causation,
            PatternType.IDENTITY: self._gen_identity,
            PatternType.POSSESSION: self._gen_possession,
            PatternType.LOCATION: self._gen_location,
            PatternType.PURPOSE: self._gen_purpose,
            PatternType.COMPOSITION: self._gen_composition,
            PatternType.PROPERTY: self._gen_property,
            PatternType.QUANTITY: self._gen_quantity,
            PatternType.TIME: self._gen_time,
            PatternType.RELATION: self._gen_relation,
        }
        
        generator = generators.get(result.pattern_type, self._gen_property)
        return generator(result)
    
    def _gen_sameness(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        header = f"{'Yes' if r.answer == 'yes' else 'No'}, {r.explanation} {conf}"
        
        parts = [header]
        if r.shared_properties:
            parts.append(f"  Shared: {', '.join(r.shared_properties)}")
        if r.different_properties:
            parts.append(f"  Differ: {', '.join(r.different_properties[:5])}")
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_similarity(self, r: ReasoningResult) -> str:
        pct = int(r.similarity_score * 100)
        conf = _format_confidence(r.confidence)
        
        parts = [f"Similarity: {pct}% {conf}"]
        parts.append(f"  {r.explanation}")
        if r.shared_properties:
            parts.append(f"  Both: {', '.join(r.shared_properties)}")
        if r.different_properties:
            parts.append(f"  Differ: {', '.join(r.different_properties[:5])}")
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_difference(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        parts = [f"{r.explanation} {conf}"]
        if r.different_properties:
            parts.append(f"  Differences: {', '.join(r.different_properties[:6])}")
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_comparison(self, r: ReasoningResult) -> str:
        pct = int(r.similarity_score * 100)
        conf = _format_confidence(r.confidence)
        
        parts = [f"Comparison ({pct}% similar) {conf}"]
        if r.shared_properties:
            parts.append(f"  In common: {', '.join(r.shared_properties)}")
        if r.different_properties:
            parts.append(f"  Differences: {', '.join(r.different_properties[:5])}")
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_capability(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        if r.answer in ("yes", "yes (inferred)"):
            header = f"Yes! {r.explanation} {conf}"
        elif r.answer == "no":
            header = f"No. {r.explanation} {conf}"
        elif r.answer == "unknown":
            header = r.explanation
        else:
            header = f"Capabilities {conf}:"
        
        parts = [header]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        if r.reasoning_chain:
            parts.append(_format_reasoning_chain(r.reasoning_chain))
        return "\n".join(parts)
    
    def _gen_causation(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        parts = [f"{r.explanation} {conf}"]
        
        # Show chain as arrows for multi-step causation
        if r.reasoning_chain and len(r.reasoning_chain) > 1:
            parts.append(_format_chain_arrows(r.reasoning_chain))
        elif r.reasoning_chain:
            parts.append(_format_reasoning_chain(r.reasoning_chain))
        
        if r.facts_used and not r.reasoning_chain:
            parts.append(_format_fact_list(r.facts_used))
        
        return "\n".join(parts)
    
    def _gen_identity(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        if r.answer in ("yes", "yes (inferred)"):
            header = f"Yes! {r.explanation} {conf}"
        else:
            header = f"{r.explanation} {conf}"
        
        parts = [header]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        if r.reasoning_chain:
            parts.append(_format_reasoning_chain(r.reasoning_chain))
        return "\n".join(parts)
    
    def _gen_possession(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_location(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_purpose(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_composition(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_property(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_quantity(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        parts = [f"{r.explanation} {conf}"]
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_time(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        # Special formatting for temporal ordering
        if r.answer in ("BEFORE", "AFTER", "SIMULTANEOUS"):
            parts = [f"{r.explanation} {conf}"]
        elif r.answer not in ("unknown", "unclear", "found"):
            # Duration answer
            parts = [f"Duration: {r.answer} {conf}"]
            if r.explanation:
                parts.append(f"  {r.explanation}")
        else:
            parts = [f"{r.explanation} {conf}"]
        
        if r.facts_used:
            parts.append(_format_fact_list(r.facts_used))
        return "\n".join(parts)
    
    def _gen_relation(self, r: ReasoningResult) -> str:
        conf = _format_confidence(r.confidence)
        
        parts = [f"{r.explanation} {conf}"]
        
        # Show reasoning chain for multi-hop connections
        if r.reasoning_chain and len(r.reasoning_chain) > 1:
            parts.append(_format_chain_arrows(r.reasoning_chain))
        elif r.reasoning_chain:
            parts.append(_format_reasoning_chain(r.reasoning_chain))
        
        if r.facts_used and not r.reasoning_chain:
            parts.append(_format_fact_list(r.facts_used))
        
        return "\n".join(parts)
    
    # ── Phase 16: Advanced answer templates ──
    
    def generate_analogy_answer(
        self, a: str, b: str, c: str, result: 'ReasoningResult'
    ) -> str:
        """Format an analogy answer."""
        conf = _format_confidence(result.confidence)
        
        parts = [
            f"{a} is to {b} as {c} is to {result.answer} {conf}"
        ]
        if result.reasoning_chain:
            parts.append(_format_reasoning_chain(result.reasoning_chain))
        return "\n".join(parts)
    
    def generate_chain_answer(self, result: 'ReasoningResult') -> str:
        """Format a multi-hop chain answer."""
        conf = _format_confidence(result.confidence)
        
        if result.reasoning_chain:
            chain = " → ".join(result.reasoning_chain)
            parts = [
                f"Here's the reasoning chain: {conf}",
                f"  {chain}",
            ]
        else:
            parts = [result.explanation + f" {conf}"]
        
        return "\n".join(parts)

