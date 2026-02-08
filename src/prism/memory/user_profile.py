"""User Profile - Track user identity, preferences, and facts.

Maintains a vector representation of the user that evolves
as we learn more about them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import numpy as np

from prism.core.vector_ops import VectorOps, HVector
from prism.core.lexicon import Lexicon


@dataclass
class Preference:
    """A single user preference."""
    
    item: str
    sentiment: str  # "positive", "negative"
    category: str = ""  # "food", "activity", etc.
    vector: HVector | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    strength: float = 1.0  # How strongly they feel


@dataclass
class UserFact:
    """A fact about the user."""
    
    predicate: str  # "is", "has", "can", "works at"
    value: str
    vector: HVector | None = None
    timestamp: datetime = field(default_factory=datetime.now)


class UserProfile:
    """Track user identity and preferences.
    
    Builds a vector representation of the user based on:
    - Their name
    - Things they like/dislike
    - Facts about them (job, location, abilities)
    
    Example:
        >>> profile = UserProfile(lexicon)
        >>> profile.learn_name("Raphael")
        >>> profile.learn_preference("like", "pizza")
        >>> profile.learn_fact("is", "a programmer")
        >>> profile.recall("food")  # → pizza
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        config: Any = None,
    ) -> None:
        """Initialize user profile."""
        self.lexicon = lexicon
        self.ops = VectorOps(lexicon.config)
        
        # User identity
        self.user_name: str | None = None
        self.user_vector: HVector | None = None
        
        # Preferences
        self.likes: list[Preference] = []
        self.dislikes: list[Preference] = []
        
        # Facts about user
        self.facts: list[UserFact] = []
        
        # Composite vector (updated as we learn)
        self._composite: HVector = self.ops.zero_vector()
    
    def learn_name(self, name: str) -> str:
        """Learn user's name.
        
        Args:
            name: User's name
            
        Returns:
            Confirmation message
        """
        self.user_name = name.strip().title()
        self.user_vector = self.lexicon.get(name.lower())
        self._update_composite()
        return f"Nice to meet you, {self.user_name}!"
    
    def learn_preference(
        self,
        sentiment: str,
        item: str,
        category: str = "",
        strength: float = 1.0,
    ) -> str:
        """Learn a user preference.
        
        Args:
            sentiment: "like" or "dislike"
            item: What they like/dislike
            category: Optional category (food, music, etc.)
            strength: How strongly they feel (0.0 to 1.0)
            
        Returns:
            Confirmation message
        """
        item = item.strip().lower()
        item_vec = self.lexicon.get(item)
        
        pref = Preference(
            item=item,
            sentiment="positive" if sentiment in ("like", "love", "enjoy") else "negative",
            category=category,
            vector=item_vec,
            strength=strength,
        )
        
        if pref.sentiment == "positive":
            self.likes.append(pref)
            # Adjust composite toward liked items
            self._composite = self._composite + (strength * 0.1 * item_vec)
        else:
            self.dislikes.append(pref)
            # Adjust composite away from disliked items
            self._composite = self._composite - (strength * 0.05 * item_vec)
        
        self._normalize_composite()
        
        sentiment_word = "like" if pref.sentiment == "positive" else "don't like"
        return f"Got it - you {sentiment_word} {item}."
    
    def learn_fact(self, predicate: str, value: str) -> str:
        """Learn a fact about the user.
        
        Args:
            predicate: "is", "has", "can", "works at", etc.
            value: The fact value
            
        Returns:
            Confirmation message
        """
        value = value.strip().lower()
        value_vec = self.lexicon.get(value)
        
        fact = UserFact(
            predicate=predicate,
            value=value,
            vector=value_vec,
        )
        self.facts.append(fact)
        
        # Adjust composite based on fact
        self._composite = self._composite + (0.05 * value_vec)
        self._normalize_composite()
        
        return f"I'll remember that you {predicate} {value}."
    
    def _update_composite(self) -> None:
        """Update composite vector from all known info."""
        if self.user_vector is not None:
            self._composite = self.user_vector.copy()
        self._normalize_composite()
    
    def _normalize_composite(self) -> None:
        """Normalize composite to unit length."""
        norm = np.linalg.norm(self._composite)
        if norm > 0:
            self._composite = self._composite / norm
    
    def recall_preferences(
        self,
        query: str | None = None,
        sentiment: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[Preference, float]]:
        """Recall user preferences.
        
        Args:
            query: Optional query to filter by similarity
            sentiment: Optional "positive" or "negative" filter
            top_k: Number of results
            
        Returns:
            List of (preference, score) tuples
        """
        prefs = []
        
        # Collect relevant preferences
        if sentiment is None or sentiment == "positive":
            prefs.extend(self.likes)
        if sentiment is None or sentiment == "negative":
            prefs.extend(self.dislikes)
        
        if not prefs:
            return []
        
        if query is None:
            # Return all, sorted by recency
            return [(p, 1.0) for p in sorted(prefs, key=lambda x: x.timestamp, reverse=True)[:top_k]]
        
        # Score by similarity to query
        query_vec = self.lexicon.get(query.lower())
        scored = []
        for pref in prefs:
            if pref.vector is not None:
                sim = self.ops.similarity(query_vec, pref.vector)
                scored.append((pref, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def recall_facts(
        self,
        predicate: str | None = None,
        query: str | None = None,
        top_k: int = 5,
    ) -> list[tuple[UserFact, float]]:
        """Recall facts about user.
        
        Args:
            predicate: Filter by predicate ("is", "has", etc.)
            query: Filter by similarity
            top_k: Number of results
            
        Returns:
            List of (fact, score) tuples
        """
        facts = self.facts
        
        if predicate:
            facts = [f for f in facts if f.predicate == predicate]
        
        if not facts:
            return []
        
        if query is None:
            return [(f, 1.0) for f in sorted(facts, key=lambda x: x.timestamp, reverse=True)[:top_k]]
        
        query_vec = self.lexicon.get(query.lower())
        scored = []
        for fact in facts:
            if fact.vector is not None:
                sim = self.ops.similarity(query_vec, fact.vector)
                scored.append((fact, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
    
    def get_summary(self) -> str:
        """Get a summary of known user info."""
        lines = []
        
        if self.user_name:
            lines.append(f"Name: {self.user_name}")
        
        if self.likes:
            like_items = [p.item for p in self.likes[:5]]
            lines.append(f"Likes: {', '.join(like_items)}")
        
        if self.dislikes:
            dislike_items = [p.item for p in self.dislikes[:5]]
            lines.append(f"Dislikes: {', '.join(dislike_items)}")
        
        if self.facts:
            for fact in self.facts[:3]:
                lines.append(f"You {fact.predicate} {fact.value}")
        
        return "\n".join(lines) if lines else "I don't know much about you yet."
    
    def is_known(self) -> bool:
        """Check if we know anything about the user."""
        return bool(self.user_name or self.likes or self.dislikes or self.facts)
    
    def clear(self) -> None:
        """Clear all user data."""
        self.user_name = None
        self.user_vector = None
        self.likes.clear()
        self.dislikes.clear()
        self.facts.clear()
        self._composite = self.ops.zero_vector()
