"""Composition - Combine and decompose concepts.

Vector composition:
- Combine concepts: red + car + fast → "a fast red car"
- Decompose: extract components from complex vectors

Implemented via bundling (superposition) and similarity matching.
"""

from __future__ import annotations

from prism.core import VSAConfig, DEFAULT_CONFIG
from prism.core.vector_ops import VectorOps, HVector
from prism.core.lexicon import Lexicon


class Composer:
    """Combine and decompose concepts in vector space.
    
    Composition creates a vector that is similar to all inputs.
    Decomposition extracts the most prominent components.
    
    Example:
        >>> comp = Composer(lexicon)
        >>> vec = comp.compose(["red", "fast", "car"])
        >>> components = comp.decompose(vec, k=5)
        >>> # Returns [("car", 0.7), ("red", 0.7), ("fast", 0.7), ...]
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        config: VSAConfig | None = None,
    ) -> None:
        """Initialize composer.
        
        Args:
            lexicon: Word-to-vector lexicon
            config: VSA configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.ops = VectorOps(self.config)
        self.lexicon = lexicon
    
    def compose(self, concepts: list[str]) -> HVector:
        """Compose multiple concepts into one vector.
        
        Uses bundling (element-wise sum) to create a vector
        that is similar to all input concepts.
        
        Args:
            concepts: List of words/concepts
            
        Returns:
            Composed vector
        """
        if not concepts:
            return self.ops.zero_vector()
        
        vectors = [self.lexicon.get(c) for c in concepts]
        return self.ops.bundle(vectors)
    
    def compose_weighted(
        self,
        concepts: list[tuple[str, float]],
    ) -> HVector:
        """Compose with weights.
        
        Args:
            concepts: List of (word, weight) tuples
            
        Returns:
            Weighted composed vector
        """
        if not concepts:
            return self.ops.zero_vector()
        
        result = self.ops.zero_vector()
        for word, weight in concepts:
            result = result + self.lexicon.get(word) * weight
        
        return result
    
    def decompose(
        self,
        vector: HVector,
        k: int = 5,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Decompose a vector into its component concepts.
        
        Finds the words most similar to the composed vector.
        
        Args:
            vector: Vector to decompose
            k: Number of components
            exclude: Words to exclude
            
        Returns:
            List of (word, similarity) components
        """
        return self.lexicon.find_nearest(
            vector,
            top_k=k,
            exclude=exclude or set(),
        )
    
    def describe(
        self,
        concepts: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Compose and describe what the composition represents.
        
        Args:
            concepts: Concepts to compose
            k: Number of description terms
            
        Returns:
            List of (word, similarity) describing the composition
        """
        composed = self.compose(concepts)
        return self.lexicon.find_nearest(
            composed,
            top_k=k + len(concepts),
            exclude=set(c.lower() for c in concepts),
        )[:k]
    
    def similarity_between(self, a: str, b: str) -> float:
        """Get similarity between two concepts.
        
        Args:
            a: First concept
            b: Second concept
            
        Returns:
            Cosine similarity
        """
        a_vec = self.lexicon.get(a)
        b_vec = self.lexicon.get(b)
        return self.ops.similarity(a_vec, b_vec)
    
    def blend(
        self,
        a: str,
        b: str,
        ratio: float = 0.5,
    ) -> HVector:
        """Blend two concepts.
        
        Args:
            a: First concept
            b: Second concept
            ratio: Blend ratio (0 = all a, 1 = all b)
            
        Returns:
            Blended vector
        """
        a_vec = self.lexicon.get(a)
        b_vec = self.lexicon.get(b)
        return a_vec * (1 - ratio) + b_vec * ratio
    
    def interpolate(
        self,
        a: str,
        b: str,
        steps: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Interpolate between two concepts.
        
        Args:
            a: Start concept
            b: End concept
            steps: Number of interpolation steps
            
        Returns:
            List of intermediate concept descriptions
        """
        results = []
        for i in range(steps):
            ratio = i / (steps - 1)
            blended = self.blend(a, b, ratio)
            nearest = self.lexicon.find_nearest(
                blended, top_k=3,
                exclude={a.lower(), b.lower()},
            )
            results.append(nearest)
        return results
