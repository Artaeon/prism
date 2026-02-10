#!/usr/bin/env python3
"""PRISM Benchmark Runner — Run all benchmarks and report results.

Usage:
    python benchmarks/run_benchmarks.py --all
    python benchmarks/run_benchmarks.py --analogy
    python benchmarks/run_benchmarks.py --commonsense
    python benchmarks/run_benchmarks.py --reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prism.core import VSAConfig
from prism.core.lexicon import Lexicon
from prism.memory import VectorMemory
from prism.memory.knowledge_graph import KnowledgeGraph

from benchmarks import BenchmarkResult
from benchmarks.analogy_benchmark import run_analogy_benchmark
from benchmarks.commonsense_benchmark import run_commonsense_benchmark
from benchmarks.reasoning_benchmark import run_reasoning_benchmark


def print_result(result: BenchmarkResult, verbose: bool = False) -> None:
    """Pretty-print a benchmark result."""
    emoji = "✅" if result.accuracy >= 0.5 else "⚠️" if result.accuracy >= 0.25 else "❌"
    print(f"\n{emoji} {result.summary()}")

    if verbose and result.errors:
        print(f"   Errors ({len(result.errors)} total):")
        for err in result.errors[:10]:
            print(f"     • {err}")
        if len(result.errors) > 10:
            print(f"     ... and {len(result.errors) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="PRISM Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--analogy", action="store_true", help="Run analogy benchmark")
    parser.add_argument("--commonsense", action="store_true", help="Run commonsense benchmark")
    parser.add_argument("--reasoning", action="store_true", help="Run reasoning benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show error details")
    parser.add_argument("--dim", type=int, default=300, help="Vector dimension")
    parser.add_argument("--output", "-o", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if not any([args.all, args.analogy, args.commonsense, args.reasoning]):
        args.all = True

    print("=" * 60)
    print("PRISM Benchmark Suite")
    print("=" * 60)

    # Initialize components
    print("\nInitializing PRISM...")
    config = VSAConfig(dimension=args.dim)
    lexicon = Lexicon(config)
    memory = VectorMemory(lexicon, config)
    graph = memory.graph

    results: list[BenchmarkResult] = []

    # Run selected benchmarks
    if args.all or args.analogy:
        print("\n🔬 Running Analogy Benchmark...")
        result = run_analogy_benchmark(lexicon)
        results.append(result)
        print_result(result, args.verbose)

    if args.all or args.commonsense:
        print("\n🔬 Running Commonsense QA Benchmark...")
        result = run_commonsense_benchmark(memory)
        results.append(result)
        print_result(result, args.verbose)

    if args.all or args.reasoning:
        print("\n🔬 Running Multi-hop Reasoning Benchmark...")
        result = run_reasoning_benchmark(memory, graph)
        results.append(result)
        print_result(result, args.verbose)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_correct = sum(r.correct for r in results)
    total_tests = sum(r.total for r in results)
    total_time = sum(r.duration_seconds for r in results)

    for r in results:
        bar = "█" * int(r.accuracy * 20) + "░" * (20 - int(r.accuracy * 20))
        print(f"  {r.name:25s} {bar} {r.accuracy:6.1%} ({r.correct}/{r.total})")

    print(f"\n  {'Overall':25s} {'':20s} {total_correct / max(total_tests, 1):6.1%} ({total_correct}/{total_tests})")
    print(f"  Total time: {total_time:.2f}s")

    # Save results if requested
    if args.output:
        output = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": {"dimension": args.dim},
            "results": [
                {
                    "name": r.name,
                    "accuracy": r.accuracy,
                    "correct": r.correct,
                    "total": r.total,
                    "duration_seconds": r.duration_seconds,
                    "errors": r.errors,
                }
                for r in results
            ],
            "summary": {
                "total_correct": total_correct,
                "total_tests": total_tests,
                "overall_accuracy": total_correct / max(total_tests, 1),
                "total_time": total_time,
            },
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\n📄 Results saved to {args.output}")


if __name__ == "__main__":
    main()
