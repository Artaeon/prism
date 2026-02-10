"""Multi-hop Reasoning Benchmark — Tests inference chain capabilities.

Tests PRISM's ability to perform multi-hop reasoning:
- A → B → C chain inference
- Causal reasoning
- Transitive property application
"""

from __future__ import annotations

import time

from benchmarks import BenchmarkResult


# Multi-hop reasoning test cases
# Format: (setup_facts, query, expected_answer_keywords)
REASONING_TESTS = [
    # 2-hop chain: cat → mammal → animal
    {
        "setup": [
            ("cat", "IS-A", "mammal"),
            ("mammal", "IS-A", "animal"),
        ],
        "query": ("cat", "IS-A"),
        "expected": ["animal"],
        "hops": 2,
    },
    # 2-hop chain: sparrow → bird → can fly
    {
        "setup": [
            ("sparrow", "IS-A", "bird"),
            ("bird", "CAN", "fly"),
        ],
        "query": ("sparrow", "CAN"),
        "expected": ["fly"],
        "hops": 2,
    },
    # 2-hop chain: kitten → cat → has whiskers
    {
        "setup": [
            ("kitten", "IS-A", "cat"),
            ("cat", "HAS", "whiskers"),
        ],
        "query": ("kitten", "HAS"),
        "expected": ["whiskers"],
        "hops": 2,
    },
    # 3-hop chain: siamese → cat → mammal → animal
    {
        "setup": [
            ("siamese", "IS-A", "cat"),
            ("cat", "IS-A", "mammal"),
            ("mammal", "IS-A", "animal"),
        ],
        "query": ("siamese", "IS-A"),
        "expected": ["animal", "mammal"],
        "hops": 3,
    },
    # Property inheritance: puppy → dog → has tail
    {
        "setup": [
            ("puppy", "IS-A", "dog"),
            ("dog", "HAS", "tail"),
        ],
        "query": ("puppy", "HAS"),
        "expected": ["tail"],
        "hops": 2,
    },
    # Capability inheritance: eagle → bird → can fly
    {
        "setup": [
            ("eagle", "IS-A", "bird"),
            ("bird", "CAN", "fly"),
        ],
        "query": ("eagle", "CAN"),
        "expected": ["fly"],
        "hops": 2,
    },
    # Location chain: goldfish → fish → lives in water
    {
        "setup": [
            ("goldfish", "IS-A", "fish"),
            ("fish", "LOCATED-IN", "water"),
        ],
        "query": ("goldfish", "LOCATED-IN"),
        "expected": ["water"],
        "hops": 2,
    },
]


def run_reasoning_benchmark(memory, graph=None) -> BenchmarkResult:
    """Run the multi-hop reasoning benchmark.

    Sets up facts, then tests if PRISM can infer
    transitive relationships.

    Args:
        memory: PRISM VectorMemory
        graph: KnowledgeGraph (if available, tests graph traversal too)
    """
    result = BenchmarkResult(name="Multi-hop Reasoning")
    start = time.perf_counter()

    for test in REASONING_TESTS:
        result.total += 1

        try:
            # Setup facts
            for subj, rel, obj in test["setup"]:
                memory.store(subj, rel, obj, importance=1.0)

            subject, relation = test["query"]
            expected = {e.lower() for e in test["expected"]}

            # Method 1: Try vector query
            matches = memory.query_object(subject, relation, top_k=10)
            predicted = {word.lower() for word, score in matches}

            found = bool(predicted & expected)

            # Method 2: Try graph traversal if available
            if not found and graph is not None:
                for exp in expected:
                    path = graph.shortest_path(subject, exp)
                    if path is not None:
                        found = True
                        break

            if found:
                result.correct += 1
            else:
                result.errors.append(
                    f"{subject} {relation} ? ({test['hops']}-hop) → "
                    f"got {list(predicted)[:3]}, expected {test['expected']}"
                )

        except Exception as e:
            result.errors.append(
                f"{test['query']} ({test['hops']}-hop) → Error: {e}"
            )

    result.duration_seconds = time.perf_counter() - start
    return result
