"""Conversation Context - Track conversation history and topic.

Maintains a sliding window of recent utterances and determines
the current conversation topic for context-aware responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import time
import numpy as np

from gunter.core.vector_ops import VectorOps, HVector
from gunter.core.lexicon import Lexicon


@dataclass
class Utterance:
    """A single utterance in conversation."""
    
    text: str
    speaker: str  # "user" or "gunter"
    timestamp: float = field(default_factory=time.time)
    vector: HVector | None = None
    
    # Extracted entities/concepts
    entities: list[str] = field(default_factory=list)


class ConversationContext:
    """Track conversation history and current topic.
    
    Maintains a sliding window of recent utterances and uses
    vector similarity to determine topic continuity.
    
    Example:
        >>> ctx = ConversationContext(lexicon)
        >>> ctx.add_utterance("Tell me about cats", "user")
        >>> ctx.add_utterance("Cats are animals that meow", "gunter")
        >>> ctx.current_topic  # "cat"
        >>> ctx.is_related("What else can they do?")  # True
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        window_size: int = 10,
    ) -> None:
        """Initialize context tracker.
        
        Args:
            lexicon: Word-to-vector lexicon
            window_size: Number of utterances to keep
        """
        self.lexicon = lexicon
        self.ops = VectorOps(lexicon.config)
        self.window_size = window_size
        
        # Conversation history
        self.history: list[Utterance] = []
        
        # Current topic
        self.current_topic: str | None = None
        self.topic_vector: HVector | None = None
        
        # Recently mentioned entities (for pronoun resolution)
        self.recent_entities: list[str] = []
    
    def add_utterance(
        self,
        text: str,
        speaker: str,
    ) -> None:
        """Add an utterance to history.
        
        Args:
            text: The utterance text
            speaker: "user" or "gunter"
        """
        # Encode utterance
        vector = self._encode_text(text)
        
        # Extract entities (simple: content words)
        entities = self._extract_entities(text)
        
        utterance = Utterance(
            text=text,
            speaker=speaker,
            vector=vector,
            entities=entities,
        )
        
        self.history.append(utterance)
        
        # Update recent entities
        self.recent_entities = entities + self.recent_entities
        self.recent_entities = self.recent_entities[:10]  # Keep 10 most recent
        
        # Trim history
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Update topic
        self._update_topic()
    
    def _encode_text(self, text: str) -> HVector:
        """Encode text as a vector."""
        words = text.lower().split()
        # Filter to meaningful words
        vectors = []
        for w in words:
            if len(w) > 2 and w not in self._skip_words():
                vectors.append(self.lexicon.get(w))
        
        if not vectors:
            return self.ops.zero_vector()
        
        # Bundle (average) word vectors
        return self.ops.bundle(vectors)
    
    def _skip_words(self) -> set[str]:
        """Words to skip when encoding."""
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "may", "might", "must",
            "and", "or", "but", "if", "then", "else", "when", "where",
            "what", "which", "who", "whom", "whose", "how", "why",
            "this", "that", "these", "those", "it", "its",
            "i", "me", "my", "we", "us", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their",
        }
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity-like words from text."""
        words = text.lower().split()
        entities = []
        
        skip = self._skip_words()
        for w in words:
            # Clean punctuation
            w = w.strip(".,!?;:'\"")
            if len(w) > 2 and w not in skip:
                entities.append(w)
        
        return entities
    
    def _update_topic(self) -> None:
        """Update current topic based on recent history."""
        if not self.history:
            return
        
        # Use last 3 utterances
        recent = self.history[-3:]
        vectors = [u.vector for u in recent if u.vector is not None]
        
        if not vectors:
            return
        
        # Topic = bundle of recent utterances
        self.topic_vector = self.ops.bundle(vectors)
        
        # Find best matching word as topic name
        nearest = self.lexicon.find_nearest(
            self.topic_vector, top_k=1, min_score=0.0
        )
        if nearest:
            self.current_topic = nearest[0][0]
    
    def resolve_pronoun(self, pronoun: str) -> str:
        """Resolve a pronoun to a recent entity.
        
        Args:
            pronoun: The pronoun ("it", "that", "they")
            
        Returns:
            The resolved entity or the original pronoun
        """
        pronoun = pronoun.lower()
        
        if pronoun in ("it", "that", "this") and self.recent_entities:
            # Return most recent non-pronoun entity
            for entity in self.recent_entities:
                if entity not in ("it", "that", "this", "they", "them"):
                    return entity
        
        if pronoun in ("they", "them") and len(self.recent_entities) > 1:
            # Could refer to multiple entities - return first
            return self.recent_entities[0]
        
        return pronoun
    
    def resolve_text(self, text: str) -> str:
        """Resolve pronouns in text.
        
        Args:
            text: Text with possible pronouns
            
        Returns:
            Text with pronouns resolved
        """
        words = text.split()
        resolved = []
        
        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"")
            if word_lower in ("it", "that", "this", "they", "them"):
                resolved_word = self.resolve_pronoun(word_lower)
                # Preserve original case if it was first word
                if word[0].isupper():
                    resolved_word = resolved_word.title()
                resolved.append(resolved_word)
            else:
                resolved.append(word)
        
        return " ".join(resolved)
    
    def is_related(self, text: str, threshold: float = 0.3) -> bool:
        """Check if text is related to current topic.
        
        Args:
            text: Text to check
            threshold: Similarity threshold
            
        Returns:
            True if related to current topic
        """
        if self.topic_vector is None:
            return False
        
        text_vec = self._encode_text(text)
        sim = self.ops.similarity(text_vec, self.topic_vector)
        return sim > threshold
    
    def get_context_summary(self) -> str:
        """Get a summary of current context."""
        lines = []
        
        if self.current_topic:
            lines.append(f"Topic: {self.current_topic}")
        
        if self.recent_entities:
            lines.append(f"Recent: {', '.join(self.recent_entities[:5])}")
        
        if self.history:
            lines.append(f"History: {len(self.history)} utterances")
        
        return "\n".join(lines) if lines else "No context yet."
    
    def get_recent_history(self, n: int = 5) -> list[tuple[str, str]]:
        """Get recent conversation history.
        
        Returns:
            List of (speaker, text) tuples
        """
        return [(u.speaker, u.text) for u in self.history[-n:]]
    
    def clear(self) -> None:
        """Clear conversation context."""
        self.history.clear()
        self.current_topic = None
        self.topic_vector = None
        self.recent_entities.clear()
