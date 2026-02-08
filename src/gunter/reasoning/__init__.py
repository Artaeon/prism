"""Analogy Reasoning - Vector arithmetic for analogies.

Implements analogical reasoning in vector space:
- a : b :: c : d  →  d = b - a + c

Classic example:
- king - man + woman = queen

This works because word vectors encode relationships.
"""

from __future__ import annotations

from gunter.core import VSAConfig, DEFAULT_CONFIG
from gunter.core.vector_ops import VectorOps, HVector
from gunter.core.lexicon import Lexicon


class AnalogyReasoner:
    """Analogical reasoning using vector arithmetic.
    
    The key insight: relationships between concepts are encoded
    as vector differences. So:
    
        king - man ≈ queen - woman  (the "royalty" direction)
    
    This allows solving analogies:
        king : man :: queen : ?  →  ? = man - king + queen = woman
    
    Example:
        >>> reasoner = AnalogyReasoner(lexicon)
        >>> results = reasoner.solve("king", "man", "queen")
        >>> # Returns [("woman", 0.8), ...]
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        config: VSAConfig | None = None,
    ) -> None:
        """Initialize reasoner.
        
        Args:
            lexicon: Word-to-vector lexicon
            config: VSA configuration
        """
        self.config = config or DEFAULT_CONFIG
        self.ops = VectorOps(self.config)
        self.lexicon = lexicon
    
    def solve(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Solve analogy: a is to b as c is to ?
        
        Formula: d = b - a + c
        
        Args:
            a: First term (e.g., "man")
            b: Second term (e.g., "king")
            c: Third term (e.g., "woman")
            top_k: Number of results
            
        Returns:
            List of (word, similarity) candidates for d
        """
        a_vec = self.lexicon.get(a)
        b_vec = self.lexicon.get(b)
        c_vec = self.lexicon.get(c)
        
        # d = b - a + c
        d_vec = b_vec - a_vec + c_vec
        
        # Normalize for better similarity
        d_vec = self.ops.normalize(d_vec)
        
        # Find nearest words, excluding inputs
        return self.lexicon.find_nearest(
            d_vec,
            top_k=top_k,
            exclude={a.lower(), b.lower(), c.lower()},
        )
    
    def solve_alt(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Alternative formulation: a : b :: c : d
        
        If a is to b (a → b), then c is to d (c → d)
        Relationship: d = c + (b - a)
        
        Same as solve() but clearer semantics.
        """
        return self.solve(a, b, c, top_k)
    
    def find_relationship(
        self,
        a: str,
        b: str,
    ) -> HVector:
        """Get the relationship vector from a to b.
        
        Args:
            a: Source concept
            b: Target concept
            
        Returns:
            The relationship vector (b - a)
        """
        a_vec = self.lexicon.get(a)
        b_vec = self.lexicon.get(b)
        return b_vec - a_vec
    
    def apply_relationship(
        self,
        relationship: HVector,
        source: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Apply a relationship to a new concept.
        
        Args:
            relationship: Vector representing the relationship
            source: Source concept
            top_k: Number of results
            
        Returns:
            List of (word, similarity) targets
        """
        source_vec = self.lexicon.get(source)
        target_vec = source_vec + relationship
        target_vec = self.ops.normalize(target_vec)
        
        return self.lexicon.find_nearest(
            target_vec,
            top_k=top_k,
            exclude={source.lower()},
        )
    
    def test_analogy(
        self,
        a: str,
        b: str,
        c: str,
        expected_d: str,
    ) -> tuple[bool, float, int]:
        """Test if an analogy holds.
        
        Args:
            a, b, c: Analogy terms
            expected_d: Expected answer
            
        Returns:
            Tuple of (success, score, rank)
        """
        results = self.solve(a, b, c, top_k=10)
        
        for rank, (word, score) in enumerate(results):
            if word.lower() == expected_d.lower():
                return True, score, rank + 1
        
        return False, 0.0, -1
