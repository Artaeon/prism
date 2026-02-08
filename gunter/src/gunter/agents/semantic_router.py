"""Semantic Router — Vector-space intent classification.

Classifies user queries by comparing sentence embeddings against
pre-defined example clusters using cosine similarity. This replaces
regex-based intent detection with a flexible, paraphrase-aware system.

No LLMs, no transformers — just spaCy's 300d word vectors averaged
into sentence embeddings + cosine similarity.

Example:
    >>> router = SemanticRouter(nlp)
    >>> route = router.classify("Explain quantum physics")
    >>> route.name
    'KNOWLEDGE_QUERY'
    >>> route.confidence
    0.87
"""

from __future__ import annotations

import re
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─── Route Types ────────────────────────────────────────────────────

class RouteType(Enum):
    """Intent categories for semantic routing."""
    KNOWLEDGE_QUERY = "knowledge_query"   # "What is X?", "Explain Y", "Who is Z?"
    YES_NO = "yes_no"                     # "Can dogs fly?", "Is water wet?"
    COMPARISON = "comparison"             # "Compare X and Y", "X vs Y"
    META_CHAT = "meta_chat"               # "How are you?", "Who are you?"
    TEACH_FACT = "teach_fact"             # "Dogs are mammals", "Paris is in France"
    GREETING = "greeting"                 # "Hi", "Hello", "Hey"
    CASUAL = "casual"                     # "Interesting", "Cool", "Thanks"
    UNKNOWN = "unknown"                   # Below confidence threshold


@dataclass
class RouteResult:
    """Result of semantic routing."""
    route: RouteType
    confidence: float
    matched_example: str = ""


# ─── Route Definitions ─────────────────────────────────────────────

# Each route: list of example utterances that characterize the intent.
# At init, each example is embedded. At query time, we find the
# closest example and return its route.

ROUTE_EXAMPLES: dict[RouteType, list[str]] = {
    RouteType.KNOWLEDGE_QUERY: [
        "what is photosynthesis",
        "what is a dog",
        "what is quantum physics",
        "explain gravity",
        "explain how computers work",
        "describe the solar system",
        "tell me about black holes",
        "who is Albert Einstein",
        "who is Marie Curie",
        "who invented the telephone",
        "what does democracy mean",
        "define artificial intelligence",
        "how does photosynthesis work",
        "how do airplanes fly",
        "what are the planets in the solar system",
        "what causes earthquakes",
        "where is the Eiffel Tower",
        "when was the internet invented",
        "why is the sky blue",
        "what is the meaning of life",
        "tell me about dinosaurs",
        "what is machine learning",
        "who wrote hamlet",
        "what is evolution",
        "how do vaccines work",
    ],
    RouteType.YES_NO: [
        "can dogs fly",
        "can penguins fly",
        "is water wet",
        "do cats purr",
        "are whales mammals",
        "is the earth round",
        "does the sun revolve around the earth",
        "can humans breathe underwater",
        "is python a programming language",
        "are spiders insects",
        "do fish sleep",
        "is gold heavier than silver",
    ],
    RouteType.COMPARISON: [
        "compare cats and dogs",
        "what is the difference between cats and dogs",
        "cats vs dogs",
        "how are birds and reptiles different",
        "compare python and javascript",
        "what is better java or python",
        "difference between mass and weight",
        "compare earth and mars",
        "lions versus tigers",
    ],
    RouteType.META_CHAT: [
        "how are you",
        "how are you doing",
        "how old are you",
        "what can you do",
        "who are you",
        "what are you",
        "what is your name",
        "are you an AI",
        "are you a robot",
        "what are your capabilities",
        "how do you work",
        "do you have feelings",
        "are you smart",
        "what do you think about yourself",
        "tell me about yourself",
    ],
    RouteType.TEACH_FACT: [
        "dogs are mammals",
        "paris is the capital of france",
        "the earth orbits the sun",
        "water boils at 100 degrees",
        "cats have whiskers",
        "python was created by guido van rossum",
        "the moon is a satellite",
        "rome is in italy",
    ],
    RouteType.GREETING: [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "hi there",
        "hey there",
        "greetings",
        "howdy",
        "hallo",
        "servus",
        "moin",
    ],
    RouteType.CASUAL: [
        "interesting",
        "cool",
        "nice",
        "okay",
        "thanks",
        "thank you",
        "got it",
        "i see",
        "wow",
        "great",
        "awesome",
        "sure",
        "alright",
        "right",
        "that makes sense",
        "good to know",
    ],
}

# Minimum confidence to accept a route (below → UNKNOWN)
MIN_ROUTE_CONFIDENCE = 0.45

# Question word patterns — if matched, NEVER route to casual/greeting
_QUESTION_WORDS = re.compile(
    r'^(?:what|who|whom|whose|which|when|where|why|how|'
    r'explain|describe|define|tell me|show me)',
    re.IGNORECASE,
)
_YESNO_WORDS = re.compile(
    r'^(?:can|could|do|does|did|is|are|was|were|will|would|'
    r'has|have|had|should|shall|may|might)',
    re.IGNORECASE,
)

# Self-reference pronouns — if present, query is about the bot itself
_SELF_REFS = {'you', 'your', 'yourself', 'gunter', 'ur', "you're", "youre"}

# Words that indicate a person/entity — likely knowledge query, not meta
_ENTITY_INDICATORS = re.compile(
    r'(?:mr|mrs|ms|dr|prof|president|king|queen|emperor|saint|st)\.?\s',
    re.IGNORECASE,
)


# ─── Semantic Router ───────────────────────────────────────────────

class SemanticRouter:
    """Classifies user input by semantic similarity to example clusters.
    
    Uses spaCy's sentence embeddings (averaged word vectors) and
    cosine similarity — no LLMs, no fine-tuning needed.
    
    Example:
        >>> import spacy
        >>> nlp = spacy.load("en_core_web_md")
        >>> router = SemanticRouter(nlp)
        >>> result = router.classify("Explain quantum physics")
        >>> result.route
        RouteType.KNOWLEDGE_QUERY
    """
    
    def __init__(self, nlp: Any) -> None:
        """Initialize with a spaCy language model.
        
        Args:
            nlp: Loaded spaCy model with word vectors (e.g. en_core_web_md)
        """
        self._nlp = nlp
        self._routes: list[tuple[RouteType, str, np.ndarray]] = []
        self._build_routes()
    
    def _build_routes(self) -> None:
        """Pre-compute embeddings for all example utterances."""
        for route_type, examples in ROUTE_EXAMPLES.items():
            for example in examples:
                vec = self._embed(example)
                if vec is not None:
                    self._routes.append((route_type, example, vec))
    
    def _embed(self, text: str) -> np.ndarray | None:
        """Embed a text string using spaCy's averaged word vectors."""
        doc = self._nlp(text)
        vec = doc.vector
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return vec / norm  # L2 normalize for cosine similarity
    
    def classify(self, text: str) -> RouteResult:
        """Classify user input into a route.
        
        Uses two signals:
        1. Cosine similarity to example clusters (semantic)
        2. Question-word detection (structural override)
        
        The structural override prevents short queries like "What is a car?"
        from being misrouted to CASUAL due to averaged function words.
        
        Args:
            text: Raw user input
            
        Returns:
            RouteResult with route type, confidence, and matched example
        """
        text_clean = text.strip().rstrip("?!.")
        text_lower = text_clean.lower()
        
        # ── Structural override: question words are NEVER casual ──
        has_question_word = bool(_QUESTION_WORDS.match(text_lower))
        has_yesno_word = bool(_YESNO_WORDS.match(text_lower))
        
        # ── Self-reference detection ──
        words_set = set(text_lower.split())
        is_about_bot = bool(words_set & _SELF_REFS)
        
        # ── Entity detection (capitalized words = likely named entities) ──
        text_words = text_clean.split()
        has_entity = False
        if len(text_words) > 1:
            # Check for capitalized words (skip first word which is always caps after sentence start)
            for w in text_words[1:]:
                if w[0].isupper() and w.lower() not in _SELF_REFS and len(w) > 1:
                    has_entity = True
                    break
        # Also check for entity indicator phrases
        if _ENTITY_INDICATORS.search(text_clean):
            has_entity = True
        
        query_vec = self._embed(text_lower)
        if query_vec is None:
            if has_question_word:
                return RouteResult(RouteType.KNOWLEDGE_QUERY, 0.6, "(question word override)")
            if has_yesno_word:
                return RouteResult(RouteType.YES_NO, 0.6, "(yes/no word override)")
            return RouteResult(RouteType.UNKNOWN, 0.0)
        
        best_route = RouteType.UNKNOWN
        best_score = -1.0
        best_example = ""
        
        for route_type, example, route_vec in self._routes:
            # Cosine similarity (vectors are already L2 normalized)
            score = float(np.dot(query_vec, route_vec))
            if score > best_score:
                best_score = score
                best_route = route_type
                best_example = example
        
        # Apply confidence threshold
        if best_score < MIN_ROUTE_CONFIDENCE:
            best_route = RouteType.UNKNOWN
        
        # ── META_CHAT override: only allow if query is about the bot ──
        if best_route == RouteType.META_CHAT and not is_about_bot:
            # "who is steve jobs" matched meta_chat but isn't about the bot
            # Re-rank: find best KNOWLEDGE_QUERY match
            best_kq_score = -1.0
            best_kq_example = ""
            for route_type, example, route_vec in self._routes:
                if route_type == RouteType.KNOWLEDGE_QUERY:
                    score = float(np.dot(query_vec, route_vec))
                    if score > best_kq_score:
                        best_kq_score = score
                        best_kq_example = example
            return RouteResult(RouteType.KNOWLEDGE_QUERY, max(best_kq_score, 0.6), best_kq_example)
        
        # ── Question word overrides: can't be casual/greeting ──
        if has_question_word and best_route in (RouteType.CASUAL, RouteType.GREETING):
            best_kq_score = -1.0
            best_kq_example = ""
            for route_type, example, route_vec in self._routes:
                if route_type == RouteType.KNOWLEDGE_QUERY:
                    score = float(np.dot(query_vec, route_vec))
                    if score > best_kq_score:
                        best_kq_score = score
                        best_kq_example = example
            return RouteResult(RouteType.KNOWLEDGE_QUERY, max(best_kq_score, 0.6), best_kq_example)
        
        if has_yesno_word and best_route in (RouteType.CASUAL, RouteType.GREETING):
            return RouteResult(RouteType.YES_NO, max(best_score, 0.6), best_example)
        
        # ── Entity detection override: entities = knowledge query ──
        if has_entity and best_route in (RouteType.CASUAL, RouteType.GREETING, RouteType.UNKNOWN):
            return RouteResult(RouteType.KNOWLEDGE_QUERY, max(best_score, 0.6), best_example)
        
        return RouteResult(best_route, best_score, best_example)
    
    def classify_top_k(self, text: str, k: int = 3) -> list[RouteResult]:
        """Return top-k route matches for debugging/transparency.
        
        Args:
            text: Raw user input
            k: Number of top matches to return
            
        Returns:
            List of RouteResults sorted by confidence (descending)
        """
        query_vec = self._embed(text.lower().strip().rstrip("?!."))
        if query_vec is None:
            return [RouteResult(RouteType.UNKNOWN, 0.0)]
        
        scores: list[tuple[RouteType, float, str]] = []
        for route_type, example, route_vec in self._routes:
            score = float(np.dot(query_vec, route_vec))
            scores.append((route_type, score, example))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate by route type (keep best per type)
        seen: set[RouteType] = set()
        results: list[RouteResult] = []
        for route_type, score, example in scores:
            if route_type not in seen:
                seen.add(route_type)
                results.append(RouteResult(route_type, score, example))
                if len(results) >= k:
                    break
        
        return results
