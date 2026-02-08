"""Semantic Roles - Role-based parsing and querying.

Implements semantic roles for event representation:
- AGENT: who does the action
- ACTION: what action is done
- PATIENT: what is acted upon
- TIME: when
- LOCATION: where

Example:
    "John ate pizza" → John(AGENT) + ate(ACTION) + pizza(PATIENT)
    Query: "who ate?" → John
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gunter.core import VSAConfig, DEFAULT_CONFIG
from gunter.core.vector_ops import VectorOps, HVector
from gunter.core.lexicon import Lexicon


# Semantic role labels
ROLE_AGENT = "AGENT"
ROLE_ACTION = "ACTION"
ROLE_PATIENT = "PATIENT"
ROLE_TIME = "TIME"
ROLE_LOCATION = "LOCATION"
ROLE_INSTRUMENT = "INSTRUMENT"
ROLE_CAUSE = "CAUSE"
ROLE_RESULT = "RESULT"


@dataclass
class Event:
    """A parsed event with semantic roles."""
    
    raw_text: str = ""
    agent: str = ""
    action: str = ""
    patient: str = ""
    time: str = ""
    location: str = ""
    
    # Encoded vector
    vector: HVector | None = None
    
    def summary(self) -> str:
        """Get a summary string."""
        parts = []
        if self.agent:
            parts.append(f"{self.agent}(AGENT)")
        if self.action:
            parts.append(f"{self.action}(ACTION)")
        if self.patient:
            parts.append(f"{self.patient}(PATIENT)")
        return " + ".join(parts) if parts else self.raw_text


class SemanticRoleParser:
    """Parse sentences into semantic roles.
    
    Handles simple SVO (Subject-Verb-Object) patterns.
    More complex parsing would require a proper NLP parser.
    
    Example:
        >>> parser = SemanticRoleParser(lexicon)
        >>> event = parser.parse("John ate pizza")
        >>> print(event.agent)  # "John"
        >>> print(event.action) # "ate"
        >>> print(event.patient) # "pizza"
    """
    
    # Simple SVO patterns: (pattern, agent_group, action_group, patient_group)
    PATTERNS = [
        # "X verb Y"
        (r"^(\w+)\s+(\w+)\s+(?:a |an |the )?(\w+)$", 1, 2, 3),
        # "the X verb Y"
        (r"^(?:the |a |an )?(\w+)\s+(\w+)\s+(?:a |an |the )?(\w+)$", 1, 2, 3),
        # "X verb Y yesterday/today"
        (r"^(\w+)\s+(\w+)\s+(?:a |an |the )?(\w+)\s+(?:yesterday|today|tomorrow)", 1, 2, 3),
    ]
    
    def __init__(
        self,
        lexicon: Lexicon,
        config: VSAConfig | None = None,
    ) -> None:
        """Initialize parser.
        
        Args:
            lexicon: Word-to-vector lexicon
            config: VSA configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.ops = VectorOps(self.config)
        self.lexicon = lexicon
    
    def parse(self, text: str) -> Event | None:
        """Parse a sentence into semantic roles.
        
        Args:
            text: Sentence to parse
            
        Returns:
            Parsed event or None if no match
        """
        text_clean = text.strip().lower()
        
        for pattern, ag, ac, pa in self.PATTERNS:
            match = re.match(pattern, text_clean)
            if match:
                event = Event(
                    raw_text=text,
                    agent=match.group(ag),
                    action=match.group(ac),
                    patient=match.group(pa),
                )
                event.vector = self._encode_event(event)
                return event
        
        return None
    
    def _encode_event(self, event: Event) -> HVector:
        """Encode an event as a vector.
        
        Encoding: bundle of role-bound pairs
        agent⊗AGENT + action⊗ACTION + patient⊗PATIENT
        """
        parts = []
        
        if event.agent:
            agent_vec = self.lexicon.get(event.agent)
            role_vec = self.lexicon.get(ROLE_AGENT)
            parts.append(self.ops.bind(agent_vec, role_vec))
        
        if event.action:
            action_vec = self.lexicon.get(event.action)
            role_vec = self.lexicon.get(ROLE_ACTION)
            parts.append(self.ops.bind(action_vec, role_vec))
        
        if event.patient:
            patient_vec = self.lexicon.get(event.patient)
            role_vec = self.lexicon.get(ROLE_PATIENT)
            parts.append(self.ops.bind(patient_vec, role_vec))
        
        if event.time:
            time_vec = self.lexicon.get(event.time)
            role_vec = self.lexicon.get(ROLE_TIME)
            parts.append(self.ops.bind(time_vec, role_vec))
        
        if event.location:
            loc_vec = self.lexicon.get(event.location)
            role_vec = self.lexicon.get(ROLE_LOCATION)
            parts.append(self.ops.bind(loc_vec, role_vec))
        
        return self.ops.bundle(parts) if parts else self.ops.zero_vector()
    
    def encode(self, text: str) -> HVector:
        """Parse and encode a sentence.
        
        Args:
            text: Sentence to encode
            
        Returns:
            Encoded vector
        """
        event = self.parse(text)
        if event and event.vector is not None:
            return event.vector
        
        # Fallback: just bundle word vectors
        words = text.lower().split()
        vectors = [self.lexicon.get(w) for w in words if len(w) > 2]
        return self.ops.bundle(vectors) if vectors else self.ops.zero_vector()
    
    def query_role(
        self,
        memory_vector: HVector,
        role: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Query for a specific role from memory.
        
        Args:
            memory_vector: Memory to query
            role: Role to extract (AGENT, ACTION, PATIENT)
            top_k: Number of results
            
        Returns:
            List of (word, similarity) for that role
        """
        role_vec = self.lexicon.get(role)
        result = self.ops.unbind(memory_vector, role_vec)
        
        # Exclude role names from results
        exclude = {r.lower() for r in [
            ROLE_AGENT, ROLE_ACTION, ROLE_PATIENT,
            ROLE_TIME, ROLE_LOCATION
        ]}
        
        return self.lexicon.find_nearest(result, top_k=top_k, exclude=exclude)
