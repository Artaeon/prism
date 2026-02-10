"""PRISM Benchmark Suite — Standard evaluations for paper publication.

Benchmarks:
1. Analogy: Google word analogy test set (king:queen :: man:woman)
2. Commonsense: ConceptNet-derived QA pairs
3. Reasoning: Multi-hop, causal, and temporal inference
4. Performance: Speed and memory metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    name: str
    total: int = 0
    correct: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def summary(self) -> str:
        return (
            f"{self.name}: {self.correct}/{self.total} "
            f"({self.accuracy:.1%}) in {self.duration_seconds:.2f}s"
        )
