"""Core Vector Symbolic Architecture Operations.

This module implements the fundamental operations for VSA:
- random_vector: Generate random hyperdimensional vectors
- bind: Combine vectors (circular convolution in frequency domain)
- unbind: Retrieve bound component (correlation)
- similarity: Cosine similarity between vectors
- bundle: Add vectors (superposition)
- normalize: Normalize to unit length

VSA uses high-dimensional vectors (~10K) where:
- Random vectors are nearly orthogonal
- Binding is reversible: unbind(bind(A, B), B) ≈ A
- Similarity is robust to noise

References:
- Kanerva, P. (2009). Hyperdimensional computing
- Plate, T. (2003). Holographic Reduced Representations
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from gunter.core import VSAConfig, DEFAULT_CONFIG


# Type alias for hyperdimensional vectors
HVector = NDArray[np.float64]


class VectorOps:
    """Core VSA operations on hyperdimensional vectors.
    
    All operations work with numpy arrays of shape (dimension,).
    Default dimension is 10,000.
    
    Example:
        >>> ops = VectorOps()
        >>> cat = ops.random_vector()
        >>> animal = ops.random_vector()
        >>> is_a = ops.random_vector()
        >>> 
        >>> # Encode: cat IS-A animal
        >>> fact = ops.bind(ops.bind(cat, is_a), animal)
        >>> 
        >>> # Query: what IS-A animal?
        >>> query = ops.unbind(fact, ops.bind(is_a, animal))
        >>> print(ops.similarity(query, cat))  # High similarity
    """
    
    def __init__(self, config: VSAConfig | None = None) -> None:
        """Initialize vector operations.
        
        Args:
            config: VSA configuration (uses defaults if None)
        """
        self.config = config or DEFAULT_CONFIG
        self._rng = np.random.default_rng(self.config.seed)
    
    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self.config.dimension
    
    def random_vector(self) -> HVector:
        """Generate a random hyperdimensional vector.
        
        Uses bipolar encoding: values are +1 or -1.
        This ensures vectors are nearly orthogonal in high dimensions.
        
        Returns:
            Random vector of shape (dimension,)
        """
        # Bipolar: +1 or -1 with equal probability
        return self._rng.choice([-1.0, 1.0], size=self.dim).astype(np.float64)
    
    def zero_vector(self) -> HVector:
        """Create a zero vector.
        
        Returns:
            Zero vector of shape (dimension,)
        """
        return np.zeros(self.dim, dtype=np.float64)
    
    def bind(self, a: HVector, b: HVector) -> HVector:
        """Bind two vectors using circular convolution.
        
        Binding creates a new vector that represents the association
        between A and B. Properties:
        - bind(A, B) is dissimilar to both A and B
        - bind is commutative: bind(A, B) = bind(B, A)
        - bind is its own inverse: bind(bind(A, B), B) ≈ A
        
        Implemented via frequency domain (Hadamard product of FFTs).
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Bound vector
        """
        # Circular convolution via FFT
        # conv(A, B) = ifft(fft(A) * fft(B))
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    
    def unbind(self, bound: HVector, key: HVector) -> HVector:
        """Unbind a vector to retrieve the associated component.
        
        Given bind(A, B) and B, retrieves approximately A.
        Uses correlation (circular convolution with inverse).
        
        Args:
            bound: The bound vector (e.g., bind(A, B))
            key: The key to unbind with (e.g., B)
            
        Returns:
            Approximately the other component (e.g., ≈A)
        """
        # Correlation = convolution with conjugate in frequency domain
        # For real bipolar vectors, inverse ≈ reverse
        key_inv = np.roll(key[::-1], 1)
        return self.bind(bound, key_inv)
    
    def similarity(self, a: HVector, b: HVector) -> float:
        """Compute cosine similarity between two vectors.
        
        Returns value in [-1, 1]:
        - 1.0 = identical
        - 0.0 = orthogonal (typical for random vectors)
        - -1.0 = opposite
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def bundle(self, vectors: list[HVector]) -> HVector:
        """Bundle (superpose) multiple vectors.
        
        Bundling adds vectors element-wise. The result is similar
        to all input vectors. Used for creating set representations.
        
        Args:
            vectors: List of vectors to bundle
            
        Returns:
            Bundled (sum) vector
        """
        if not vectors:
            return self.zero_vector()
        
        result = np.sum(vectors, axis=0)
        return result
    
    def normalize(self, v: HVector) -> HVector:
        """Normalize vector to unit length.
        
        Args:
            v: Vector to normalize
            
        Returns:
            Unit vector (or zero if input is zero)
        """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    def threshold(self, v: HVector) -> HVector:
        """Threshold vector to bipolar (+1/-1).
        
        Cleans up noisy vectors after operations.
        
        Args:
            v: Vector to threshold
            
        Returns:
            Bipolar vector
        """
        return np.sign(v).astype(np.float64)
    
    def is_match(self, a: HVector, b: HVector) -> bool:
        """Check if two vectors are similar enough.
        
        Uses configured similarity threshold.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            True if similarity > threshold
        """
        return self.similarity(a, b) > self.config.similarity_threshold


def random_vector(dim: int = 10_000, seed: int | None = None) -> HVector:
    """Convenience function to generate a random vector.
    
    Args:
        dim: Vector dimension
        seed: Random seed
        
    Returns:
        Random bipolar vector
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=dim).astype(np.float64)


def bind(a: HVector, b: HVector) -> HVector:
    """Convenience function for binding."""
    return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


def unbind(bound: HVector, key: HVector) -> HVector:
    """Convenience function for unbinding."""
    key_inv = np.roll(key[::-1], 1)
    return bind(bound, key_inv)


def similarity(a: HVector, b: HVector) -> float:
    """Convenience function for similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def bundle(vectors: list[HVector]) -> HVector:
    """Convenience function for bundling."""
    return np.sum(vectors, axis=0) if vectors else np.zeros(len(vectors[0]))
