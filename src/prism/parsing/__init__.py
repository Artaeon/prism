"""Enhanced Semantic Parser with relaxed patterns.

Handles structured facts, free-form SVO, and natural questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from prism.core.vector_ops import VectorOps, HVector
from prism.core.lexicon import Lexicon


@dataclass
class ParsedFact:
    """A parsed fact from text."""
    
    subject: str
    relation: str
    object: str
    raw_text: str = ""
    confidence: float = 1.0


class SemanticParser:
    """Parse text into vector representations.
    
    Handles patterns like:
    - "X is a Y" → (X, IS-A, Y)
    - "X has Y" → (X, HAS, Y)
    - "X can Y" → (X, CAN, Y)
    - "X means Y" / "X is like Y" → (X, MEANS, Y)
    - "cats meow" → (cats, DOES, meow) [SVO fallback]
    """
    
    # Pattern rules: (regex, relation, subject_group, object_group)
    PATTERNS = [
        # Synonyms: "X means Y", "X is like Y"
        (r"(?:a |an |the )?(\w+) (?:means?|is like|is similar to) (?:a |an |the )?(\w+)", "MEANS", 1, 2),
        # Type: "X is a/an Y"
        (r"(?:a |an |the )?(\w+) (?:is|are) (?:a|an) (\w+)", "IS-A", 1, 2),
        # Property: "X is Y" (adjective)
        (r"(?:a |an |the )?(\w+) (?:is|are) (\w+)", "IS", 1, 2),
        # Possession: "X has Y" / "X have Y"
        (r"(?:a |an |the )?(\w+) (?:has|have|owns?) (?:a |an |the )?(\w+)", "HAS", 1, 2),
        # Capability: "X can Y"
        (r"(?:a |an |the )?(\w+) (?:can|could|is able to) (\w+)", "CAN", 1, 2),
        # Requirement: "X needs Y"
        (r"(?:a |an |the )?(\w+) (?:needs?|requires?) (?:a |an |the )?(\w+)", "NEEDS", 1, 2),
        # Preference: "X likes Y"
        (r"(?:a |an |the )?(\w+) (?:likes?|loves?|enjoys?) (?:a |an |the )?(\w+)", "LIKES", 1, 2),
        # Consumption: "X eats Y"
        (r"(?:a |an |the )?(\w+) eats? (?:a |an |the )?(\w+)", "EATS", 1, 2),
        # Makes: "X makes Y"
        (r"(?:a |an |the )?(\w+) makes? (?:a |an |the )?(\w+)", "MAKES", 1, 2),
    ]
    
    # Stop words to skip in SVO fallback
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
        "it", "its", "this", "that", "these", "those",
    }
    
    def __init__(self, lexicon: Lexicon) -> None:
        """Initialize parser."""
        self.lexicon = lexicon
        self.ops = VectorOps(lexicon.config)
    
    def parse(self, text: str) -> list[ParsedFact]:
        """Parse text into structured facts.
        
        Tries structured patterns first, then falls back to
        simple SVO (Subject Verb Object) extraction.
        """
        facts = []
        text_lower = text.lower().strip()
        
        # Check for negation patterns FIRST (before structured patterns
        # that might match "X is not Y" as "X IS not")
        neg_fact = self._try_negation(text_lower)
        if neg_fact:
            facts.append(neg_fact)
            return facts
        
        # Try structured patterns
        for pattern, relation, subj_group, obj_group in self.PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                fact = ParsedFact(
                    subject=match.group(subj_group).strip(),
                    relation=relation,
                    object=match.group(obj_group).strip(),
                    raw_text=text,
                )
                facts.append(fact)
                return facts
        
        # Fallback: SVO extraction for simple sentences
        # "cats meow" → (cats, DOES, meow)
        # "dogs bark loudly" → (dogs, DOES, bark)
        svo_fact = self._try_svo(text_lower)
        if svo_fact:
            facts.append(svo_fact)
        
        return facts
    
    def _try_svo(self, text: str) -> ParsedFact | None:
        """Try to extract Subject-Verb-Object from simple text.
        
        Handles: "cats meow", "birds fly", "dogs bark"
        """
        words = [w.strip(".,!?;:'\"") for w in text.split()]
        # Filter out stop words and empties
        words = [w for w in words if w and w not in self.STOP_WORDS and len(w) > 1]
        
        if len(words) < 2:
            return None
        
        # Simple heuristic: first content word = subject, second = verb/action
        subject = words[0]
        action = words[1]
        
        # If there's a third word, it's the object
        if len(words) >= 3:
            obj = words[2]
            return ParsedFact(
                subject=subject,
                relation="DOES",
                object=f"{action} {obj}",
                raw_text=text,
                confidence=0.6,
            )
        
        # Two words: subject + intransitive verb
        return ParsedFact(
            subject=subject,
            relation="DOES",
            object=action,
            raw_text=text,
            confidence=0.5,
        )
    
    def encode(self, text: str) -> tuple[HVector, list[ParsedFact]]:
        """Parse and encode text into a single vector."""
        facts = self.parse(text)
        
        if not facts:
            words = text.lower().split()
            vectors = [self.lexicon.get(w) for w in words if len(w) > 2]
            if vectors:
                return self.ops.bundle(vectors), []
            return self.ops.zero_vector(), []
        
        fact_vectors = []
        for fact in facts:
            s_vec = self.lexicon.get(fact.subject)
            r_vec = self.lexicon.get(fact.relation)
            o_vec = self.lexicon.get(fact.object)
            fact_vec = self.ops.bind(self.ops.bind(s_vec, r_vec), o_vec)
            fact_vectors.append(fact_vec)
        
        result = self.ops.bundle(fact_vectors)
        return result, facts
    
    def _try_negation(self, text: str) -> ParsedFact | None:
        """Try to parse negation patterns.
        
        Handles: "dogs don't fly", "cats can't swim", 
                 "birds never sleep", "fish are not mammals"
        """
        # "X don't/doesn't Y" → (X, DOES-NOT, Y)
        match = re.search(
            r"(?:a |an |the )?(\w+) (?:don'?t|doesn'?t|do not|does not) (\w+)",
            text
        )
        if match:
            return ParsedFact(
                subject=match.group(1),
                relation="DOES-NOT",
                object=match.group(2),
                raw_text=text,
                confidence=0.9,
            )
        
        # "X can't/cannot Y" → (X, CAN-NOT, Y)
        match = re.search(
            r"(?:a |an |the )?(\w+) (?:can'?t|cannot|can not) (\w+)",
            text
        )
        if match:
            return ParsedFact(
                subject=match.group(1),
                relation="CAN-NOT",
                object=match.group(2),
                raw_text=text,
                confidence=0.9,
            )
        
        # "X is/are not Y" → (X, IS-NOT, Y)
        match = re.search(
            r"(?:a |an |the )?(\w+) (?:is|are) (?:not|never) (?:a |an )?(\w+)",
            text
        )
        if match:
            return ParsedFact(
                subject=match.group(1),
                relation="IS-NOT",
                object=match.group(2),
                raw_text=text,
                confidence=0.9,
            )
        
        # "X never Y" → (X, DOES-NOT, Y)
        match = re.search(
            r"(?:a |an |the )?(\w+) never (\w+)",
            text
        )
        if match:
            return ParsedFact(
                subject=match.group(1),
                relation="DOES-NOT",
                object=match.group(2),
                raw_text=text,
                confidence=0.85,
            )
        
        return None
    
    def encode_question(self, text: str) -> tuple[str, str, str | None]:
        """Parse a question to identify query structure.
        
        Returns (subject, relation, object_or_None).
        """
        text_lower = text.lower().strip().rstrip("?")
        
        # "What is X?" / "What is a X?" / "What are X?"
        match = re.search(r"what (?:is|are) (?:a |an |the )?(\w+)", text_lower)
        if match:
            return match.group(1), "IS-A", None
        
        # "Is X a Y?"
        match = re.search(r"(?:is|are) (?:a |an |the )?(\w+) (?:a |an )?(\w+)", text_lower)
        if match:
            return match.group(1), "IS-A", match.group(2)
        
        # "What does X have?" / "What do X have?"
        match = re.search(r"what (?:does |do )?(?:a |an |the )?(\w+) have", text_lower)
        if match:
            return match.group(1), "HAS", None
        
        # "What does X do?" / "What do X do?" / "What can X do?"
        match = re.search(r"what (?:does |do |can )?(?:a |an |the )?(\w+) do", text_lower)
        if match:
            return match.group(1), "DOES", None
        
        # "What does X eat?" / "What do X eat?"
        match = re.search(r"what (?:does |do )?(?:a |an |the )?(\w+) (\w+)", text_lower)
        if match:
            verb = match.group(2)
            if verb not in ("do", "have", "mean"):
                return match.group(1), verb.upper(), None
        
        # "Can X Y?"
        match = re.search(r"can (?:a |an |the )?(\w+) (\w+)", text_lower)
        if match:
            return match.group(1), "CAN", match.group(2)
        
        # "Does X Y?" → check fact
        match = re.search(r"(?:does|do) (?:a |an |the )?(\w+) (\w+)(?: (?:a |an |the )?(\w+))?", text_lower)
        if match:
            subj = match.group(1)
            verb = match.group(2)
            obj = match.group(3)
            return subj, verb.upper(), obj
        
        return "", "", None
