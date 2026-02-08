"""Lexicon - Word to Vector mapping with semantic bootstrapping.

Supports two modes:
1. Pre-trained embeddings (spacy) - immediate semantic relationships
2. Random vectors with learning-based adjustment - builds semantics over time

The hybrid approach uses pre-trained embeddings when available,
and continues to adjust vectors as new relationships are learned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
import numpy as np

from prism.core import VSAConfig, DEFAULT_CONFIG
from prism.core.vector_ops import VectorOps, HVector


# Try to import spacy for pre-trained embeddings
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class Lexicon:
    """Word-to-vector mapping with semantic bootstrapping.
    
    Uses pre-trained embeddings (spacy) when available:
    - Immediate semantic relationships
    - similar("cat") → dog, kitten, pet
    - Analogies work out of the box
    
    Falls back to random vectors:
    - Builds semantics through learning
    - adjust_similarity() moves related concepts closer
    
    Example:
        >>> lex = Lexicon()  # Auto-detects spacy
        >>> cat_vec = lex.get("cat")
        >>> dog_vec = lex.get("dog")
        >>> print(lex.ops.similarity(cat_vec, dog_vec))  # 0.8+ with spacy
    """
    
    def __init__(
        self,
        config: VSAConfig | None = None,
        use_embeddings: bool = True,
    ) -> None:
        """Initialize lexicon.
        
        Args:
            config: VSA configuration
            use_embeddings: Whether to try loading pre-trained embeddings
        """
        self.config = config or DEFAULT_CONFIG
        self.ops = VectorOps(self.config)
        
        # Word -> Vector mapping
        self._vectors: dict[str, HVector] = {}
        
        # Track which vectors are from embeddings vs random
        self._from_embeddings: set[str] = set()
        
        # Spacy model for pre-trained embeddings
        self._nlp = None
        self._embedding_dim = 0
        
        if use_embeddings and SPACY_AVAILABLE:
            self._try_load_spacy()
        
        # Pre-populate with role vectors (always random)
        self._init_roles()
    
    def _try_load_spacy(self) -> None:
        """Try to load spacy model."""
        models_to_try = [
            "en_core_web_lg",  # 300d, best quality
            "en_core_web_md",  # 300d, good quality
            "en_core_web_sm",  # 96d, basic
        ]
        
        for model_name in models_to_try:
            try:
                self._nlp = spacy.load(model_name, disable=["parser", "ner", "tagger"])
                # Get embedding dimension from a test word
                test_doc = self._nlp("test")
                if len(test_doc) > 0 and test_doc[0].has_vector:
                    self._embedding_dim = len(test_doc[0].vector)
                    print(f"✓ Loaded {model_name} embeddings ({self._embedding_dim}d)")
                    return
            except OSError:
                continue
        
        print("⚠ No spacy model found. Using random vectors.")
        print("  Install with: python -m spacy download en_core_web_md")
    
    def _init_roles(self) -> None:
        """Initialize semantic role vectors (always random)."""
        roles = [
            "IS-A", "HAS", "CAN", "MEANS", "NEEDS",
            "AGENT", "ACTION", "PATIENT", "TIME", "LOCATION",
            "CAUSE", "RESULT",
        ]
        for role in roles:
            self._vectors[role] = self.ops.random_vector()
    
    def get(self, word: str) -> HVector:
        """Get vector for a word, creating if needed.
        
        Uses pre-trained embedding if available, otherwise random.
        
        Args:
            word: The word to look up
            
        Returns:
            Hyperdimensional vector for the word
        """
        # Normalize: keep uppercase for roles, lowercase for words
        key = word.strip().upper() if word.isupper() else word.strip().lower()
        
        if key in self._vectors:
            return self._vectors[key]
        
        # Try pre-trained embeddings first
        if self._nlp is not None and not key.isupper():
            vec = self._get_pretrained(key)
            if vec is not None:
                self._vectors[key] = vec
                self._from_embeddings.add(key)
                return vec
        
        # Fallback: random vector
        self._vectors[key] = self.ops.random_vector()
        return self._vectors[key]
    
    def _get_pretrained(self, word: str) -> HVector | None:
        """Get pre-trained embedding, projected to target dimension.
        
        Projects from embedding dimension (e.g., 300) to target (e.g., 10000)
        while preserving semantic relationships.
        """
        doc = self._nlp(word)
        if len(doc) == 0 or not doc[0].has_vector or doc[0].vector_norm == 0:
            return None
        
        # Get the embedding (typically 300d)
        embedding = doc[0].vector.astype(np.float64)
        
        # Project to target dimension
        # Strategy: Use embedding as seed for deterministic expansion
        target_dim = self.config.dimension
        
        if len(embedding) >= target_dim:
            # Truncate if embedding is larger (unlikely)
            return embedding[:target_dim] / np.linalg.norm(embedding[:target_dim])
        
        # Deterministic expansion: use embedding values to seed padding
        # This preserves relationships while adding dimensionality
        np.random.seed(int(np.abs(embedding[:4].sum() * 1000) % (2**31)))
        
        # Create expanded vector
        expanded = np.zeros(target_dim, dtype=np.float64)
        
        # Copy original embedding (most important part)
        expanded[:len(embedding)] = embedding
        
        # Add structured padding based on embedding statistics
        remaining = target_dim - len(embedding)
        
        # Repeat and permute the embedding to fill remaining space
        repeats = remaining // len(embedding) + 1
        padding = np.tile(embedding, repeats)[:remaining]
        
        # Add small random noise for uniqueness
        noise = np.random.randn(remaining) * 0.01
        expanded[len(embedding):] = padding * 0.1 + noise
        
        # Normalize to unit length
        norm = np.linalg.norm(expanded)
        if norm > 0:
            expanded = expanded / norm
        
        return expanded
    
    def adjust_similarity(
        self,
        word1: str,
        word2: str,
        rate: float = 0.1,
    ) -> None:
        """Adjust vectors to increase similarity between words.
        
        Used during learning: when we learn "cat IS-A animal",
        we move cat's vector closer to animal's vector.
        
        Uses linear interpolation: adjusted = (1-rate)*v1 + rate*v2
        This always increases similarity between the vectors.
        
        Args:
            word1: First word (moves toward word2)
            word2: Second word (reference)
            rate: Adjustment rate (0.0 to 1.0)
        """
        # Get current vectors (make copies to avoid reference issues)
        v1 = self.get(word1).copy()
        v2 = self.get(word2).copy()
        
        # Don't adjust role vectors
        if word1.isupper() or word2.isupper():
            return
        
        # Linear interpolation: move word1 a fraction of the way toward word2
        # adjusted = (1-rate)*v1 + rate*v2
        adjusted = (1.0 - rate) * v1 + rate * v2
        
        # Renormalize to unit length
        norm = np.linalg.norm(adjusted)
        if norm > 0:
            self._vectors[word1.lower()] = adjusted / norm
    
    def get_batch(self, words: list[str]) -> list[HVector]:
        """Get vectors for multiple words."""
        return [self.get(w) for w in words]
    
    def add(self, word: str, vector: HVector | None = None) -> HVector:
        """Add a word to the lexicon."""
        key = word.strip().lower()
        
        if vector is None:
            vector = self.ops.random_vector()
        
        self._vectors[key] = vector
        return vector
    
    def contains(self, word: str) -> bool:
        """Check if word is in lexicon."""
        key = word.strip().upper() if word.isupper() else word.strip().lower()
        return key in self._vectors
    
    def find_nearest(
        self,
        vector: HVector,
        top_k: int = 5,
        exclude: set[str] | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Find words most similar to a vector.
        
        Args:
            vector: Query vector
            top_k: Number of results
            exclude: Words to exclude from results
            min_score: Minimum similarity score to include
            
        Returns:
            List of (word, similarity) tuples, sorted by similarity
        """
        exclude = exclude or set()
        # Always exclude role vectors from general searches
        role_excludes = {
            "is-a", "has", "can", "means", "needs",
            "agent", "action", "patient", "time", "location",
            "cause", "result",
        }
        exclude = exclude | role_excludes
        
        scores = []
        for word, vec in self._vectors.items():
            if word.lower() in exclude or word in exclude:
                continue
            sim = self.ops.similarity(vector, vec)
            if sim >= min_score:
                scores.append((word, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def has_embeddings(self) -> bool:
        """Check if pre-trained embeddings are available."""
        return self._nlp is not None
    
    def __len__(self) -> int:
        """Number of words in lexicon."""
        return len(self._vectors)
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over words."""
        return iter(self._vectors)
    
    def words(self) -> list[str]:
        """Get all words in lexicon."""
        return list(self._vectors.keys())
    
    def save(self, path: str | Path) -> None:
        """Save lexicon to file."""
        path = Path(path)
        words = list(self._vectors.keys())
        vectors = np.array([self._vectors[w] for w in words])
        np.savez(path, words=np.array(words), vectors=vectors)
    
    def load(self, path: str | Path) -> None:
        """Load lexicon from file."""
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        words = data["words"]
        vectors = data["vectors"]
        self._vectors.clear()
        for word, vec in zip(words, vectors):
            self._vectors[str(word)] = vec
        self._init_roles()


def create_default_lexicon(config: VSAConfig | None = None) -> Lexicon:
    """Create a lexicon with common words pre-loaded.
    
    Args:
        config: VSA configuration
        
    Returns:
        Lexicon with starter vocabulary
    """
    lex = Lexicon(config)
    
    # Common words to pre-load
    words = [
        # Animals
        "cat", "dog", "bird", "fish", "animal", "mammal", "pet",
        "kitten", "puppy", "lion", "tiger", "elephant",
        # People
        "person", "man", "woman", "child", "king", "queen", "boy", "girl",
        # Objects
        "car", "house", "building", "city", "country",
        "food", "water", "air", "fire", "earth",
        "computer", "phone", "book", "table", "chair",
        # Nature
        "tree", "flower", "grass", "forest", "mountain",
        "sun", "moon", "star", "sky", "cloud", "rain",
        # Adjectives
        "big", "small", "tall", "short", "fast", "slow",
        "hot", "cold", "good", "bad", "new", "old", "young",
        "red", "blue", "green", "white", "black",
        # Verbs
        "run", "walk", "jump", "fly", "swim", "eat", "drink",
        "sleep", "see", "hear", "think", "know", "like", "love",
    ]
    
    for word in words:
        lex.get(word)
    
    return lex
