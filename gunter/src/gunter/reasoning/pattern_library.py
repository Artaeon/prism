"""Pattern Library — VSA-encoded question patterns for semantic matching.

Uses VSA bundle vectors to represent 15 question types. Each pattern
is a bundled vector of its keyword embeddings, enabling fast cosine
similarity matching against incoming questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


class PatternType(Enum):
    """The 15 core question pattern types."""
    SAMENESS = auto()
    SIMILARITY = auto()
    DIFFERENCE = auto()
    COMPARISON = auto()
    CAPABILITY = auto()
    CAUSATION = auto()
    IDENTITY = auto()
    POSSESSION = auto()
    LOCATION = auto()
    PURPOSE = auto()
    COMPOSITION = auto()
    PROPERTY = auto()
    QUANTITY = auto()
    TIME = auto()
    RELATION = auto()


@dataclass
class QuestionPattern:
    """A question pattern with its VSA vector and metadata."""
    
    name: str
    pattern_type: PatternType
    keywords: list[str]
    vsa_vector: NDArray | None = None
    # Regex triggers for exact pattern matching (fast path)
    triggers: list[str] = field(default_factory=list)


@dataclass
class PatternMatch:
    """Result of matching a question to a pattern."""
    
    pattern: QuestionPattern
    confidence: float  # 0-1 from cosine similarity
    entities: list[str] = field(default_factory=list)


# Pattern definitions: (type, name, keywords, regex triggers)
PATTERN_DEFS: list[tuple[PatternType, str, list[str], list[str]]] = [
    (PatternType.SAMENESS, "sameness",
     ["same", "equal", "identical", "equivalent", "match"],
     [r"same as", r"equal to", r"identical"]),
    
    (PatternType.SIMILARITY, "similarity",
     ["similar", "like", "resemble", "alike", "common", "share"],
     [r"similar", r"in common", r"alike", r"how (?:much )?like"]),
    
    (PatternType.DIFFERENCE, "difference",
     ["different", "unlike", "distinct", "differ", "contrast", "unique"],
     [r"differ", r"unlike", r"distinct", r"contrast"]),
    
    (PatternType.COMPARISON, "comparison",
     ["compare", "versus", "vs", "better", "worse", "bigger", "smaller"],
     [r"compare", r"vs\.?", r"versus", r"better", r"worse"]),
    
    (PatternType.CAPABILITY, "capability",
     ["can", "able", "capable", "ability", "skill", "power"],
     [r"can .+ \w+", r"able to", r"capable of"]),
    
    (PatternType.CAUSATION, "causation",
     ["why", "cause", "reason", "because", "result", "lead", "effect"],
     [r"^why", r"cause", r"reason", r"because"]),
    
    (PatternType.IDENTITY, "identity",
     ["is a", "type", "kind", "category", "class", "species", "classify"],
     [r"is (?:a|an) ", r"type of", r"kind of", r"what (?:is|are)"]),
    
    (PatternType.POSSESSION, "possession",
     ["has", "have", "own", "possess", "feature", "attribute"],
     [r"(?:does|do) .+ have", r"has .+", r"possess"]),
    
    (PatternType.LOCATION, "location",
     ["where", "location", "place", "found", "live", "habitat", "region"],
     [r"^where", r"location", r"live", r"found", r"habitat"]),
    
    (PatternType.PURPOSE, "purpose",
     ["what for", "purpose", "used for", "function", "role", "goal"],
     [r"what.+for", r"purpose", r"used for", r"function of"]),
    
    (PatternType.COMPOSITION, "composition",
     ["made of", "contains", "consists", "composed", "ingredient", "component"],
     [r"made of", r"contain", r"consist", r"composed"]),
    
    (PatternType.PROPERTY, "property",
     ["describe", "characteristics", "property", "traits", "features", "attribute"],
     [r"describe", r"characteristics", r"properties", r"tell me about", r"what is"]),
    
    (PatternType.QUANTITY, "quantity",
     ["how many", "count", "number", "amount", "total", "sum"],
     [r"how many", r"how much", r"count", r"number of", r"total"]),
    
    (PatternType.TIME, "time",
     ["when", "time", "period", "date", "era", "year", "century", "age"],
     [r"^when", r"what time", r"what year", r"how old"]),
    
    (PatternType.RELATION, "relation",
     ["related", "connection", "relationship", "link", "association", "tie"],
     [r"related", r"connection", r"relationship", r"relation between"]),
]


class PatternLibrary:
    """Library of 15 question patterns with VSA vector matching.
    
    Each pattern is encoded as a bundled VSA vector of its keyword
    embeddings. Matching uses cosine similarity between the question's
    bundled vector and each pattern vector.
    
    Example:
        >>> lib = PatternLibrary(lexicon)
        >>> match = lib.match_pattern("Are cats the same as dogs?")
        >>> match.pattern.pattern_type  # PatternType.SAMENESS
        >>> match.confidence  # 0.85
    """
    
    MATCH_THRESHOLD = 0.15  # Minimum cosine similarity
    
    def __init__(self, lexicon: Any) -> None:
        """Initialize with a Lexicon for vector operations."""
        self.lexicon = lexicon
        self.ops = lexicon.ops
        self.patterns: list[QuestionPattern] = []
        self._encode_patterns()
    
    def _encode_patterns(self) -> None:
        """Encode all 15 patterns as bundled keyword vectors."""
        for ptype, name, keywords, triggers in PATTERN_DEFS:
            # Bundle keyword vectors from the lexicon
            vecs = []
            for kw in keywords:
                vec = self.lexicon.get(kw)
                if vec is not None:
                    vecs.append(vec)
            
            vsa_vec = self.ops.bundle(vecs) if vecs else self.ops.random_vector()
            vsa_vec = self.ops.normalize(vsa_vec)
            
            self.patterns.append(QuestionPattern(
                name=name,
                pattern_type=ptype,
                keywords=keywords,
                vsa_vector=vsa_vec,
                triggers=triggers,
            ))
    
    def match_pattern(self, question: str) -> PatternMatch | None:
        """Match a question to the best matching pattern.
        
        Uses a two-phase approach:
        1. Fast path: regex trigger matching
        2. Slow path: VSA cosine similarity
        
        Args:
            question: The natural language question
            
        Returns:
            PatternMatch with best pattern + confidence, or None
        """
        import re
        q_lower = question.lower().strip().rstrip('?').strip()
        
        # Phase 1: Regex triggers (fast path, high confidence)
        for pattern in self.patterns:
            for trigger in pattern.triggers:
                if re.search(trigger, q_lower):
                    entities = self._extract_entities(q_lower)
                    return PatternMatch(
                        pattern=pattern,
                        confidence=0.9,
                        entities=entities,
                    )
        
        # Phase 2: VSA vector similarity
        # Encode the question as a bundled vector of its words
        words = q_lower.split()
        vecs = []
        for w in words:
            clean = w.strip(".,!?;:'\"")
            if len(clean) > 2:
                vec = self.lexicon.get(clean)
                if vec is not None:
                    vecs.append(vec)
        
        if not vecs:
            return None
        
        q_vec = self.ops.normalize(self.ops.bundle(vecs))
        
        # Find best matching pattern
        best_match = None
        best_score = self.MATCH_THRESHOLD
        
        for pattern in self.patterns:
            if pattern.vsa_vector is not None:
                score = self.ops.similarity(q_vec, pattern.vsa_vector)
                if score > best_score:
                    best_score = score
                    best_match = pattern
        
        if best_match is None:
            return None
        
        entities = self._extract_entities(q_lower)
        return PatternMatch(
            pattern=best_match,
            confidence=min(best_score * 1.2, 1.0),  # Scale up slightly
            entities=entities,
        )
    
    def _extract_entities(self, text: str) -> list[str]:
        """Extract key entities (nouns) from a question.
        
        Uses spaCy NER and noun chunks if available, otherwise
        falls back to simple heuristics.
        """
        import re
        
        entities = []
        
        # Use spaCy if available
        if self.lexicon._nlp is not None:
            doc = self.lexicon._nlp(text)
            # Noun chunks (may fail if parser is disabled)
            try:
                for chunk in doc.noun_chunks:
                    clean = chunk.text.lower().strip()
                    if clean not in ('what', 'which', 'who', 'how', 'where', 'when', 'why'):
                        entities.append(clean)
            except ValueError:
                pass  # Parser not available, fall through to POS
            
            # If no chunks, get nouns directly
            if not entities:
                for token in doc:
                    if token.pos_ in ('NOUN', 'PROPN') and len(token.text) > 2:
                        entities.append(token.text.lower())
        
        # Fallback: heuristic extraction
        if not entities:
            skip = {
                'what', 'which', 'who', 'how', 'where', 'when', 'why',
                'the', 'a', 'an', 'is', 'are', 'do', 'does', 'can',
                'same', 'as', 'and', 'or', 'but', 'not', 'have', 'has',
                'similar', 'different', 'like', 'between', 'compare',
            }
            words = re.findall(r'\b\w+\b', text.lower())
            entities = [w for w in words if w not in skip and len(w) > 2]
        
        # Deduplicate while preserving order
        seen = set()
        result = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                result.append(e)
        
        return result[:3]  # Max 3 entities
