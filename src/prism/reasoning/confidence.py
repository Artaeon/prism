"""Confidence Scoring — Unified confidence with propagation and calibration.

Phase 3 upgrade: Adds decay-based multi-hop scoring, evidence accumulation,
and Bayesian-inspired calibration on top of the base confidence system.

Scores responses on a 0.0-1.0 scale based on source type:
- Direct fact match: 0.95
- Semantic similarity: cosine score
- Transitive inference: product of chain scores × decay
- User profile match: 0.90
- Multi-source evidence: 1 - ∏(1-ci)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class SourceType(Enum):
    """Source of an answer."""
    DIRECT_FACT = "direct"
    SEMANTIC_SIMILARITY = "similarity"
    TRANSITIVE_INFERENCE = "transitive"
    USER_PROFILE = "profile"
    ACCUMULATED = "accumulated"
    CALIBRATED = "calibrated"
    UNKNOWN = "unknown"


@dataclass
class ScoredAnswer:
    """An answer with confidence metadata."""

    text: str
    confidence: float
    source: SourceType = SourceType.UNKNOWN
    details: list[str] = field(default_factory=list)

    @property
    def level(self) -> str:
        """Human-readable confidence level."""
        if self.confidence >= 0.8:
            return "high"
        elif self.confidence >= 0.6:
            return "medium"
        elif self.confidence >= 0.4:
            return "uncertain"
        else:
            return "low"

    def format(self) -> str:
        """Format answer with confidence indicator."""
        indicator = {
            "high": "●",
            "medium": "◐",
            "uncertain": "○",
            "low": "◌",
        }.get(self.level, "?")

        return f"{self.text} {indicator} [{self.confidence:.0%}]"


# Thresholds
SHOW_THRESHOLD = 0.4       # Show answers above this
UNCERTAIN_THRESHOLD = 0.6  # Mark as uncertain below this
CONFIDENT_THRESHOLD = 0.8  # Mark as confident above this


class ConfidenceScorer:
    """Score answers with unified confidence, decay, and calibration.

    Phase 3 upgrades:
    - Multi-hop decay: confidence drops exponentially with chain length
    - Evidence accumulation: multiple sources boost confidence
    - Bayesian calibration: adjusts raw scores based on evidence count

    Example:
        >>> scorer = ConfidenceScorer()
        >>> scored = scorer.score_direct("cats are animals", 0.95)
        >>> scored.format()  # "cats are animals ● [95%]"
        >>>
        >>> # Multi-hop with decay
        >>> chain = [0.9, 0.85, 0.8]  # 3-hop chain
        >>> hop_answer = scorer.score_multihop("cats are vertebrates", chain, ["mammal", "animal"])
        >>> hop_answer.confidence  # ~0.55 (decayed product)
        >>>
        >>> # Evidence accumulation
        >>> sources = [scorer.score_direct("X", 0.7), scorer.score_similarity("X", 0.6)]
        >>> combined = scorer.score_accumulated(sources)
        >>> combined.confidence  # ~0.88 (higher than either alone)
    """

    # Base confidence per source type
    BASE_SCORES = {
        SourceType.DIRECT_FACT: 0.95,
        SourceType.USER_PROFILE: 0.90,
        SourceType.SEMANTIC_SIMILARITY: 0.0,  # Uses raw cosine
        SourceType.TRANSITIVE_INFERENCE: 0.0,  # Uses chain product
    }

    # Decay factor per hop in multi-hop reasoning
    HOP_DECAY = 0.92  # Each hop reduces confidence by 8%

    # Bayesian prior parameters
    PRIOR_CONFIDENCE = 0.5  # Prior belief (uninformative)
    PRIOR_STRENGTH = 1.0    # How strongly prior pulls (pseudo-count)

    def score_direct(self, text: str, match_score: float = 1.0) -> ScoredAnswer:
        """Score a direct fact match."""
        conf = self.BASE_SCORES[SourceType.DIRECT_FACT] * match_score
        return ScoredAnswer(
            text=text,
            confidence=min(conf, 1.0),
            source=SourceType.DIRECT_FACT,
            details=["Direct fact from memory"],
        )

    def score_similarity(self, text: str, cosine_score: float) -> ScoredAnswer:
        """Score a semantic similarity result."""
        return ScoredAnswer(
            text=text,
            confidence=max(0.0, min(cosine_score, 1.0)),
            source=SourceType.SEMANTIC_SIMILARITY,
            details=[f"Semantic similarity: {cosine_score:.2f}"],
        )

    def score_transitive(
        self, text: str, chain_confidence: float, chain_steps: list[str],
    ) -> ScoredAnswer:
        """Score a transitive inference result (simple product)."""
        return ScoredAnswer(
            text=text,
            confidence=chain_confidence,
            source=SourceType.TRANSITIVE_INFERENCE,
            details=[f"Inferred via {len(chain_steps)} hop(s)"] + chain_steps,
        )

    def score_profile(self, text: str) -> ScoredAnswer:
        """Score a user profile match."""
        return ScoredAnswer(
            text=text,
            confidence=self.BASE_SCORES[SourceType.USER_PROFILE],
            source=SourceType.USER_PROFILE,
            details=["From your profile"],
        )

    # ─── Phase 3: Advanced Scoring ─────────────────────────────────

    def score_multihop(
        self,
        text: str,
        chain_confidences: list[float],
        intermediate_entities: list[str],
    ) -> ScoredAnswer:
        """Score a multi-hop inference with exponential decay.

        Confidence = c1 × c2 × ... × cn × decay^n

        Each hop applies both the edge confidence AND a decay factor,
        so longer chains are naturally penalized.

        Args:
            text: The inferred answer text
            chain_confidences: Confidence of each edge in the chain
            intermediate_entities: Entities along the path (for explanation)
        """
        n_hops = len(chain_confidences)
        if n_hops == 0:
            return self.score_direct(text)

        # Product of edge confidences × exponential decay
        product = 1.0
        for c in chain_confidences:
            product *= c
        decay = self.HOP_DECAY ** n_hops
        confidence = product * decay

        # Build explanation
        details = [f"Inferred via {n_hops}-hop chain (decay={decay:.2f})"]
        if intermediate_entities:
            path = " → ".join(intermediate_entities)
            details.append(f"Path: {path}")

        return ScoredAnswer(
            text=text,
            confidence=max(0.0, min(confidence, 1.0)),
            source=SourceType.TRANSITIVE_INFERENCE,
            details=details,
        )

    def score_accumulated(
        self,
        sources: list[ScoredAnswer],
    ) -> ScoredAnswer:
        """Accumulate evidence from multiple sources.

        Uses the noisy-OR model: P(true) = 1 - ∏(1 - ci)
        Multiple independent sources supporting the same answer
        increase total confidence.

        Example:
            Source A says X with 0.7 confidence
            Source B says X with 0.6 confidence
            Combined: 1 - (0.3 × 0.4) = 0.88

        Args:
            sources: Multiple scored answers for the same claim
        """
        if not sources:
            return ScoredAnswer(text="", confidence=0.0, source=SourceType.ACCUMULATED)
        if len(sources) == 1:
            return sources[0]

        # Noisy-OR accumulation
        complement_product = 1.0
        for s in sources:
            complement_product *= (1.0 - s.confidence)
        accumulated_conf = 1.0 - complement_product

        # Use the highest-scoring answer's text
        best = max(sources, key=lambda s: s.confidence)

        details = [f"Accumulated from {len(sources)} sources:"]
        for s in sources:
            details.append(f"  • {s.source.value}: {s.confidence:.2f}")

        return ScoredAnswer(
            text=best.text,
            confidence=min(accumulated_conf, 1.0),
            source=SourceType.ACCUMULATED,
            details=details,
        )

    def calibrate(
        self,
        raw_score: float,
        evidence_count: int,
        prior: float | None = None,
    ) -> float:
        """Bayesian-inspired confidence calibration.

        Adjusts raw confidence toward a prior based on evidence count.
        With little evidence, confidence is pulled toward the prior.
        With lots of evidence, the raw score dominates.

        Formula:
            calibrated = (prior × strength + raw × evidence) / (strength + evidence)

        Args:
            raw_score: The raw confidence score
            evidence_count: Number of supporting evidence pieces
            prior: Prior belief (default: PRIOR_CONFIDENCE)

        Returns:
            Calibrated confidence score
        """
        p = prior if prior is not None else self.PRIOR_CONFIDENCE
        s = self.PRIOR_STRENGTH

        calibrated = (p * s + raw_score * evidence_count) / (s + evidence_count)
        return max(0.0, min(calibrated, 1.0))

    def score_calibrated(
        self,
        text: str,
        raw_score: float,
        evidence_count: int,
        source: SourceType = SourceType.CALIBRATED,
    ) -> ScoredAnswer:
        """Score with Bayesian calibration applied."""
        calibrated = self.calibrate(raw_score, evidence_count)
        return ScoredAnswer(
            text=text,
            confidence=calibrated,
            source=source,
            details=[
                f"Raw score: {raw_score:.2f}",
                f"Evidence count: {evidence_count}",
                f"Calibrated: {calibrated:.2f}",
            ],
        )

    # ─── Display ───────────────────────────────────────────────────

    def should_show(self, answer: ScoredAnswer) -> bool:
        """Whether this answer is worth showing."""
        return answer.confidence >= SHOW_THRESHOLD

    def format_with_confidence(self, answer: ScoredAnswer) -> str:
        """Format answer with confidence and source info."""
        if answer.confidence >= CONFIDENT_THRESHOLD:
            return f"{answer.text} [{answer.confidence:.0%}]"
        elif answer.confidence >= UNCERTAIN_THRESHOLD:
            return f"{answer.text} [{answer.confidence:.0%}]"
        else:
            return f"(uncertain) {answer.text} [{answer.confidence:.0%}]"

    def uncertain_response(self) -> str:
        """Generate uncertain response."""
        return "I'm not confident enough to answer that. Could you teach me?"
