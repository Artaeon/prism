"""Response Quality Gate — Detect and reject garbage answers.

Scores response quality before returning to the user. If the response
is too thin, contains known garbage patterns, or has low confidence,
signals that a swarm fallback should be triggered instead.

Example:
    >>> gate = ResponseQualityGate()
    >>> gate.is_acceptable("Here's what dog is classified as. ◕ [70%]")
    False
    >>> gate.is_acceptable("The dog is a domesticated descendant of wolves...")
    True
"""

from __future__ import annotations

import re


class ResponseQualityGate:
    """Evaluates response quality and rejects garbage answers.
    
    Checks for:
    - Too-short responses (< 40 chars of actual content)
    - Known garbage patterns (raw similarity dumps)
    - Very thin answers (template with no substance)
    """
    
    MIN_CONTENT_LENGTH = 40
    
    # Patterns that indicate a garbage/thin response
    GARBAGE_PATTERNS = [
        r"^Similar to '.+':",
        r"^\s*•\s+\w+\s+\[[\d.]+\]\s*$",
        r"^Here's what \w+ is classified as\.",
        r"^I don't have (?:specific|any) (?:causal|records|information)",
        r"^When \w+:.*\[[\d]+%\]",
        r"^\w+ and \w+ are \d+% similar",
    ]
    
    def __init__(self) -> None:
        self._compiled = [re.compile(p, re.MULTILINE) for p in self.GARBAGE_PATTERNS]
    
    def is_acceptable(self, response: str) -> bool:
        """Check if a response is good enough to show the user.
        
        Args:
            response: The candidate response string
            
        Returns:
            True if acceptable, False if should trigger swarm fallback
        """
        if not response or not response.strip():
            return False
        
        # Strip emoji/indicators for content length check
        content = re.sub(r'[◕◐◔◌●\[\]%\d]', '', response).strip()
        if len(content) < self.MIN_CONTENT_LENGTH:
            return False
        
        # Check for garbage patterns
        for pattern in self._compiled:
            if pattern.search(response):
                return False
        
        return True
    
    def score(self, response: str) -> float:
        """Score response quality 0.0—1.0 for fine-grained control.
        
        Args:
            response: The candidate response string
            
        Returns:
            Quality score (0.0 = garbage, 1.0 = excellent)
        """
        if not response or not response.strip():
            return 0.0
        
        score = 0.5  # Base score
        
        # Length bonus
        content = re.sub(r'[◕◐◔◌●\[\]%\d]', '', response).strip()
        if len(content) > 200:
            score += 0.3
        elif len(content) > 100:
            score += 0.2
        elif len(content) > 50:
            score += 0.1
        elif len(content) < 30:
            score -= 0.3
        
        # Garbage pattern penalty
        for pattern in self._compiled:
            if pattern.search(response):
                score -= 0.4
        
        # Source citation bonus (indicates swarm-quality answer)
        if "📖" in response:
            score += 0.1
        
        return max(0.0, min(1.0, score))
