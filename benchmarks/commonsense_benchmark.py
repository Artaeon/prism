"""Commonsense Reasoning Benchmark — Tests everyday knowledge.

Tests PRISM's ability to answer commonsense questions that
require understanding of everyday concepts and relationships.
"""

from __future__ import annotations

import time

from benchmarks import BenchmarkResult


# Commonsense QA pairs: (question_subject, question_relation, expected_answers)
COMMONSENSE_TESTS = [
    # IS-A (taxonomy)
    ("cat", "IS-A", ["animal", "mammal", "pet", "feline"]),
    ("dog", "IS-A", ["animal", "mammal", "pet", "canine"]),
    ("rose", "IS-A", ["flower", "plant"]),
    ("oak", "IS-A", ["tree", "plant"]),
    ("sparrow", "IS-A", ["bird", "animal"]),
    ("salmon", "IS-A", ["fish", "animal"]),
    ("python", "IS-A", ["snake", "reptile", "animal"]),
    ("apple", "IS-A", ["fruit", "food"]),
    # HAS (properties)
    ("cat", "HAS", ["whiskers", "tail", "fur", "claws", "paws"]),
    ("dog", "HAS", ["tail", "fur", "paws", "nose"]),
    ("bird", "HAS", ["wings", "feathers", "beak"]),
    ("tree", "HAS", ["leaves", "roots", "branches", "bark"]),
    ("car", "HAS", ["wheels", "engine", "doors", "seats"]),
    ("fish", "HAS", ["fins", "scales", "gills"]),
    # CAN (capabilities)
    ("bird", "CAN", ["fly", "sing", "chirp"]),
    ("fish", "CAN", ["swim"]),
    ("cat", "CAN", ["climb", "purr", "jump", "hunt"]),
    ("dog", "CAN", ["bark", "run", "swim", "fetch"]),
    # LOCATED-IN (typical locations)
    ("fish", "LOCATED-IN", ["water", "ocean", "river", "sea", "lake"]),
    ("bear", "LOCATED-IN", ["forest", "woods", "cave"]),
]


def run_commonsense_benchmark(memory) -> BenchmarkResult:
    """Run the commonsense reasoning benchmark.

    For each test, queries memory and checks if ANY expected
    answer appears in the results.

    Args:
        memory: PRISM VectorMemory with loaded knowledge
    """
    result = BenchmarkResult(name="Commonsense QA")
    start = time.perf_counter()

    for subject, relation, expected in COMMONSENSE_TESTS:
        result.total += 1

        try:
            # Query using vector operations
            matches = memory.query_object(subject, relation, top_k=10, min_score=0.0)
            predicted = {word.lower() for word, score in matches}

            # Check if any expected answer is in the results
            expected_set = {e.lower() for e in expected}
            if predicted & expected_set:
                result.correct += 1
            else:
                result.errors.append(
                    f"{subject} {relation} ? → got {list(predicted)[:3]}, "
                    f"expected one of {expected}"
                )
        except Exception as e:
            result.errors.append(f"{subject} {relation} ? → Error: {e}")

    result.duration_seconds = time.perf_counter() - start
    return result
