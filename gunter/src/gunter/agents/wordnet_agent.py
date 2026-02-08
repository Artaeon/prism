"""WordNet Agent — Offline definitions and word relationships.

Uses NLTK's WordNet for instant definitions, synonyms, antonyms,
hypernyms, and word relationships. Fully offline (~1ms latency).

Example:
    >>> agent = WordNetAgent()
    >>> result = agent.lookup("photosynthesis")
    >>> print(result.definition)
    'synthesis of compounds with the aid of radiant energy...'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WordNetResult:
    """Result from a WordNet lookup."""
    word: str = ""
    definition: str = ""
    synonyms: list[str] = field(default_factory=list)
    antonyms: list[str] = field(default_factory=list)
    hypernyms: list[str] = field(default_factory=list)   # "is-a" parents
    hyponyms: list[str] = field(default_factory=list)     # "is-a" children
    examples: list[str] = field(default_factory=list)
    part_of_speech: str = ""
    found: bool = False


class WordNetAgent:
    """Agent that provides word definitions and relationships via WordNet.
    
    Features:
    - Fully offline (no network access)
    - ~1ms latency
    - Definitions, synonyms, antonyms, hypernyms, hyponyms
    - Part-of-speech information
    - Falls back gracefully if nltk/wordnet not installed
    """

    def __init__(self) -> None:
        self._wn = None
        self._available = False
        self._init_wordnet()

    def _init_wordnet(self) -> None:
        """Try to load WordNet."""
        try:
            from nltk.corpus import wordnet as wn
            # Quick availability check
            wn.synsets("test")
            self._wn = wn
            self._available = True
        except Exception as e:
            logger.debug(f"WordNet not available: {e}")
            self._available = False

    def lookup(self, word: str) -> WordNetResult:
        """Look up a word in WordNet.
        
        Args:
            word: Word to look up
            
        Returns:
            WordNetResult with definition and relationships
        """
        if not self._available or not self._wn:
            return WordNetResult(word=word)
        
        word_clean = word.lower().strip().replace(" ", "_")
        synsets = self._wn.synsets(word_clean)
        
        if not synsets:
            return WordNetResult(word=word)
        
        # Use the most common synset (first one)
        primary = synsets[0]
        
        # Get definition
        definition = primary.definition()
        
        # Get examples
        examples = primary.examples()[:3]
        
        # Get synonyms (from all synsets, deduplicated)
        synonyms: set[str] = set()
        antonyms: set[str] = set()
        
        for syn in synsets[:3]:
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    synonyms.add(name)
                # Get antonyms
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace("_", " "))
        
        # Get hypernyms (is-a parents)
        hypernyms = []
        for hyp in primary.hypernyms()[:5]:
            for lemma in hyp.lemmas()[:1]:
                hypernyms.append(lemma.name().replace("_", " "))
        
        # Get hyponyms (is-a children)
        hyponyms = []
        for hyp in primary.hyponyms()[:5]:
            for lemma in hyp.lemmas()[:1]:
                hyponyms.append(lemma.name().replace("_", " "))
        
        # Part of speech
        pos_map = {"n": "noun", "v": "verb", "a": "adjective", "r": "adverb", "s": "adjective"}
        pos = pos_map.get(primary.pos(), "")
        
        return WordNetResult(
            word=word,
            definition=definition,
            synonyms=sorted(synonyms)[:10],
            antonyms=sorted(antonyms)[:5],
            hypernyms=hypernyms,
            hyponyms=hyponyms,
            examples=examples,
            part_of_speech=pos,
            found=True,
        )

    def get_definition(self, word: str) -> str:
        """Get just the definition of a word."""
        result = self.lookup(word)
        if result.found:
            return result.definition
        return ""

    def get_synonyms(self, word: str) -> list[str]:
        """Get synonyms of a word."""
        result = self.lookup(word)
        return result.synonyms

    @property
    def is_available(self) -> bool:
        """Whether WordNet is available."""
        return self._available

    def clear_cache(self) -> None:
        """No-op for API compatibility."""
        pass
