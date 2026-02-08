"""Confidence Scoring — Unified confidence for all answers.

Scores responses on a 0.0-1.0 scale based on source type:
- Direct fact match: 0.95
- Semantic similarity: cosine score
- Transitive inference: product of chain scores
- User profile match: 0.90
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceType(Enum):
    """Source of an answer."""
    DIRECT_FACT = "direct"
    SEMANTIC_SIMILARITY = "similarity"
    TRANSITIVE_INFERENCE = "transitive"
    USER_PROFILE = "profile"
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
    """Score answers with unified confidence.
    
    Example:
        >>> scorer = ConfidenceScorer()
        >>> scored = scorer.score_direct("cats are animals", 0.95)
        >>> scored.format()  # "cats are animals ● [95%]"
    """
    
    # Base confidence per source type
    BASE_SCORES = {
        SourceType.DIRECT_FACT: 0.95,
        SourceType.USER_PROFILE: 0.90,
        SourceType.SEMANTIC_SIMILARITY: 0.0,  # Uses raw cosine
        SourceType.TRANSITIVE_INFERENCE: 0.0,  # Uses chain product
    }
    
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
            confidence=cosine_score,
            source=SourceType.SEMANTIC_SIMILARITY,
            details=[f"Semantic similarity: {cosine_score:.2f}"],
        )
    
    def score_transitive(
        self, text: str, chain_confidence: float, chain_steps: list[str],
    ) -> ScoredAnswer:
        """Score a transitive inference result."""
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
