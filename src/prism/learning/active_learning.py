"""Active Learning — Tracks uncertainty and learns from corrections.

Phase 6: Extends FeedbackLearner with active learning capabilities:
- Tracks unanswered questions as knowledge gaps
- Adjusts confidence based on user feedback patterns
- Suggests the most valuable questions to learn
- Applies confidence overrides for corrected facts
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory
    from prism.memory.knowledge_graph import KnowledgeGraph


@dataclass
class UncertaintyRecord:
    """A question PRISM couldn't answer confidently."""

    query: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False


@dataclass
class CorrectionRecord:
    """A fact that was corrected by the user."""

    wrong_fact: str       # What PRISM said
    correct_fact: str     # What user corrected it to
    entity: str           # Primary entity
    timestamp: datetime = field(default_factory=datetime.now)


class ActiveLearner:
    """Active learning with uncertainty tracking and correction replay.

    Capabilities:
    - Track questions PRISM couldn't answer (knowledge gaps)
    - Record user corrections (wrong → correct)
    - Adjust entity confidence based on correction history
    - Suggest most valuable questions to learn next
    - Apply confidence overrides for known-wrong areas

    Example:
        >>> learner = ActiveLearner(memory)
        >>> learner.record_uncertainty("What is quantum computing?", 0.2)
        >>> learner.record_correction("cats are reptiles", "cats are mammals", "cat")
        >>> learner.suggest_questions()
        ["What is quantum computing?"]
    """

    def __init__(self, memory: VectorMemory) -> None:
        self.memory = memory
        self._uncertainties: list[UncertaintyRecord] = []
        self._corrections: list[CorrectionRecord] = []
        self._entity_accuracy: dict[str, list[bool]] = defaultdict(list)
        self._confidence_overrides: dict[str, float] = {}

    def record_uncertainty(self, query: str, confidence: float) -> None:
        """Record a question PRISM couldn't answer confidently."""
        self._uncertainties.append(UncertaintyRecord(
            query=query, confidence=confidence,
        ))

    def record_correction(
        self,
        wrong_fact: str,
        correct_fact: str,
        entity: str,
    ) -> str:
        """Record a user correction and apply it.

        - Stores the correct fact in memory
        - Marks entity as having accuracy issues
        - Applies confidence override
        """
        entity = entity.lower()
        self._corrections.append(CorrectionRecord(
            wrong_fact=wrong_fact,
            correct_fact=correct_fact,
            entity=entity,
        ))

        # Track accuracy per entity
        self._entity_accuracy[entity].append(False)  # This was wrong

        # Apply confidence penalty for this entity's facts
        wrong_count = sum(1 for c in self._entity_accuracy[entity] if not c)
        total = len(self._entity_accuracy[entity])
        accuracy = 1.0 - (wrong_count / max(total, 1))
        self._confidence_overrides[entity] = max(0.3, accuracy)

        return (
            f"Corrected! '{wrong_fact}' → '{correct_fact}'. "
            f"Entity '{entity}' accuracy: {accuracy:.0%}"
        )

    def record_positive_feedback(self, entity: str) -> None:
        """Record that a fact about an entity was confirmed correct."""
        entity = entity.lower()
        self._entity_accuracy[entity].append(True)

        # Boost confidence if previously penalized
        correct_count = sum(1 for c in self._entity_accuracy[entity] if c)
        total = len(self._entity_accuracy[entity])
        accuracy = correct_count / max(total, 1)

        if entity in self._confidence_overrides:
            self._confidence_overrides[entity] = min(1.0, accuracy + 0.1)

    def get_confidence_override(self, entity: str) -> float | None:
        """Get confidence override for an entity.

        Returns None if no override exists.
        """
        return self._confidence_overrides.get(entity.lower())

    def apply_confidence_adjustment(
        self,
        raw_confidence: float,
        entity: str,
    ) -> float:
        """Apply accuracy-based confidence adjustment for an entity."""
        override = self.get_confidence_override(entity)
        if override is not None:
            return raw_confidence * override
        return raw_confidence

    def suggest_questions(self, top_k: int = 5) -> list[str]:
        """Suggest the most valuable questions to learn.

        Ranked by:
        1. Questions with lowest confidence (biggest gaps)
        2. Questions that haven't been resolved yet
        """
        unresolved = [
            u for u in self._uncertainties if not u.resolved
        ]
        unresolved.sort(key=lambda u: u.confidence)
        return [u.query for u in unresolved[:top_k]]

    def mark_resolved(self, query: str) -> None:
        """Mark a knowledge gap as resolved."""
        for u in self._uncertainties:
            if u.query.lower() == query.lower():
                u.resolved = True

    def get_problem_entities(self, min_corrections: int = 2) -> list[tuple[str, int]]:
        """Get entities that have been corrected multiple times."""
        entity_corrections: Counter = Counter()
        for c in self._corrections:
            entity_corrections[c.entity] += 1
        return [
            (entity, count)
            for entity, count in entity_corrections.most_common()
            if count >= min_corrections
        ]

    def get_statistics(self) -> dict:
        """Get active learning statistics."""
        return {
            "total_uncertainties": len(self._uncertainties),
            "unresolved_gaps": sum(1 for u in self._uncertainties if not u.resolved),
            "total_corrections": len(self._corrections),
            "entities_with_overrides": len(self._confidence_overrides),
            "problem_entities": self.get_problem_entities(),
        }
