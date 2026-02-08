"""Configuration for Vector Symbolic Architecture."""

from dataclasses import dataclass


@dataclass
class VSAConfig:
    """Configuration for VSA operations.
    
    Attributes:
        dimension: Size of hyperdimensional vectors (default 10000)
        similarity_threshold: Minimum cosine similarity for a match
        seed: Random seed for reproducibility (None = random)
    """
    
    dimension: int = 10_000
    similarity_threshold: float = 0.2
    seed: int | None = None


# Global default config
DEFAULT_CONFIG = VSAConfig()
