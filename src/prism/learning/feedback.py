"""Feedback Learning — Self-improvement from user corrections.

Allows PRISM to learn from user feedback like:
- "correct: cats are NOT friendly"
- "wrong" (marks last answer as incorrect)
- "clarify: cats can be both friendly and mean"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory
    from prism.reasoning.confidence import ConfidenceScorer, ScoredAnswer


@dataclass
class FeedbackEntry:
    """A single feedback record."""
    
    feedback_type: str  # "correct", "wrong", "clarify"
    original_fact: str
    correction: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class FeedbackLearner:
    """Learn from user corrections and feedback.
    
    Features:
    - "correct: X" — replace a fact
    - "wrong" — mark last answer as wrong
    - "clarify: X" — add nuance to a fact
    - Confidence adjustment based on feedback
    
    Example:
        >>> learner = FeedbackLearner(memory)
        >>> learner.process_correction("cats are NOT big")
        "Noted! I'll update my knowledge about cats."
    """
    
    def __init__(self, memory: VectorMemory) -> None:
        """Initialize with memory."""
        self.memory = memory
        self.feedback_history: list[FeedbackEntry] = []
        self.last_answer: str = ""
        self.last_facts: list[str] = []
    
    def record_answer(self, answer: str, facts_used: list[str]) -> None:
        """Record the last answer for feedback reference."""
        self.last_answer = answer
        self.last_facts = facts_used
    
    def try_handle(self, text: str) -> str | None:
        """Try to handle text as feedback.
        
        Returns response if handled, None otherwise.
        """
        text_lower = text.lower().strip()
        
        # "wrong" / "that's wrong" / "no" / "incorrect"
        if text_lower in ("wrong", "no", "incorrect", "that's wrong", "thats wrong"):
            return self._handle_wrong()
        
        # "correct: X"
        match = re.search(r"correct(?:ion)?:?\s+(.+)", text, re.IGNORECASE)
        if match:
            return self._handle_correction(match.group(1))
        
        # "clarify: X"
        match = re.search(r"clarif(?:y|ication):?\s+(.+)", text, re.IGNORECASE)
        if match:
            return self._handle_clarification(match.group(1))
        
        # "actually, X" / "no, X"
        match = re.search(r"^(?:actually|no),?\s+(.+)", text, re.IGNORECASE)
        if match:
            return self._handle_correction(match.group(1))
        
        return None
    
    def _handle_wrong(self) -> str:
        """Handle 'wrong' feedback."""
        if not self.last_answer:
            return "I haven't said anything to correct yet."
        
        entry = FeedbackEntry(
            feedback_type="wrong",
            original_fact=self.last_answer,
        )
        self.feedback_history.append(entry)
        
        return (
            f"Sorry about that! I've noted that my answer was wrong.\n"
            f"You can teach me the correct information with 'correct: ...' or 'learn ...'"
        )
    
    def _handle_correction(self, correction: str) -> str:
        """Handle a correction."""
        entry = FeedbackEntry(
            feedback_type="correct",
            original_fact=self.last_answer or "(unknown)",
            correction=correction,
        )
        self.feedback_history.append(entry)
        
        return (
            f"Noted! I'll remember: {correction}\n"
            f"(Use 'learn {correction}' to formally store this as a fact.)"
        )
    
    def _handle_clarification(self, clarification: str) -> str:
        """Handle a clarification."""
        entry = FeedbackEntry(
            feedback_type="clarify",
            original_fact=self.last_answer or "(unknown)",
            correction=clarification,
        )
        self.feedback_history.append(entry)
        
        return f"Thanks for the clarification! I'll keep in mind: {clarification}"
    
    def get_feedback_summary(self) -> str:
        """Summarize feedback received."""
        if not self.feedback_history:
            return "No feedback received yet."
        
        by_type: dict[str, int] = {}
        for entry in self.feedback_history:
            by_type[entry.feedback_type] = by_type.get(entry.feedback_type, 0) + 1
        
        lines = [f"Feedback received ({len(self.feedback_history)} total):"]
        for ftype, count in sorted(by_type.items()):
            lines.append(f"  • {ftype}: {count}")
        
        return "\n".join(lines)
    
    def suggest_knowledge_gaps(self) -> str:
        """Suggest areas where PRISM needs more knowledge.
        
        Based on:
        - Questions that couldn't be answered
        - Facts that were corrected
        """
        if not self.feedback_history:
            return "No knowledge gaps identified yet. Keep asking questions!"
        
        wrong_count = sum(
            1 for e in self.feedback_history if e.feedback_type == "wrong"
        )
        
        if wrong_count > 0:
            return (
                f"I've been wrong {wrong_count} time(s). "
                f"Consider teaching me more facts to improve my accuracy!"
            )
        
        return "So far so good! Keep teaching me and I'll keep learning."
