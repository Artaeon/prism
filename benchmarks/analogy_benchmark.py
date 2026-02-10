"""Analogy Benchmark — Tests word analogy reasoning.

Tests PRISM's ability to solve word analogies like:
    king : queen :: man : woman

Uses the Google analogy test set format:
    word1 word2 word3 word4
where word1:word2 :: word3:word4.
"""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks import BenchmarkResult


# Built-in analogy test pairs (subset of Google analogy test set)
ANALOGY_PAIRS = [
    # Capitals
    ("athens", "greece", "berlin", "germany"),
    ("paris", "france", "rome", "italy"),
    ("tokyo", "japan", "london", "england"),
    ("madrid", "spain", "lisbon", "portugal"),
    ("dublin", "ireland", "oslo", "norway"),
    # Gender
    ("king", "queen", "man", "woman"),
    ("boy", "girl", "father", "mother"),
    ("brother", "sister", "uncle", "aunt"),
    ("prince", "princess", "actor", "actress"),
    ("husband", "wife", "son", "daughter"),
    # Comparative
    ("big", "bigger", "small", "smaller"),
    ("fast", "faster", "slow", "slower"),
    ("good", "better", "bad", "worse"),
    ("tall", "taller", "short", "shorter"),
    # Verb tense
    ("walk", "walked", "run", "ran"),
    ("go", "went", "see", "saw"),
    ("eat", "ate", "drink", "drank"),
    # Part-whole
    ("finger", "hand", "toe", "foot"),
    ("wheel", "car", "wing", "bird"),
    ("page", "book", "leaf", "tree"),
]


def run_analogy_benchmark(lexicon) -> BenchmarkResult:
    """Run the analogy benchmark.

    Tests: vec(word2) - vec(word1) + vec(word3) ≈ vec(word4)

    Args:
        lexicon: PRISM Lexicon with loaded embeddings
    """
    from prism.core.vector_ops import VectorOps

    result = BenchmarkResult(name="Word Analogy")
    ops = VectorOps(lexicon.config)

    start = time.perf_counter()

    for w1, w2, w3, w4 in ANALOGY_PAIRS:
        result.total += 1

        try:
            v1 = lexicon.get(w1)
            v2 = lexicon.get(w2)
            v3 = lexicon.get(w3)
            v4_expected = lexicon.get(w4)

            # Analogy: v2 - v1 + v3 should be close to v4
            predicted = v2 - v1 + v3

            # Find nearest word (excluding inputs)
            matches = lexicon.find_nearest(
                predicted, top_k=5,
                exclude={w1, w2, w3},
            )

            if matches and matches[0][0] == w4:
                result.correct += 1
            else:
                predicted_word = matches[0][0] if matches else "(none)"
                result.errors.append(
                    f"{w1}:{w2} :: {w3}:? → got '{predicted_word}', expected '{w4}'"
                )
        except Exception as e:
            result.errors.append(f"{w1}:{w2}::{w3}:? → Error: {e}")

    result.duration_seconds = time.perf_counter() - start
    return result
