#!/usr/bin/env python3
"""Train Gunter on large-scale knowledge sources.

Usage:
    python scripts/train_knowledge.py --sources conceptnet,wordnet
    python scripts/train_knowledge.py --sources wordnet --max-facts 5000
    python scripts/train_knowledge.py --sources conceptnet,wordnet,simplewiki --output data/trained_memory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Train Gunter on large-scale knowledge sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on WordNet only (fast, no download)
  python scripts/train_knowledge.py --sources wordnet

  # Train on WordNet with fewer synsets (faster)
  python scripts/train_knowledge.py --sources wordnet --max-synsets 5000

  # Train on ConceptNet + WordNet
  python scripts/train_knowledge.py --sources conceptnet,wordnet

  # Train on all sources with fact limit
  python scripts/train_knowledge.py --sources conceptnet,wordnet,simplewiki --max-facts 100000

  # Use cached data and save to custom path
  python scripts/train_knowledge.py --sources conceptnet --cache --output data/my_memory

  # Quick sample test (no downloads)
  python scripts/train_knowledge.py --sample
        """,
    )
    
    parser.add_argument(
        '--sources',
        type=str,
        default='wordnet',
        help='Comma-separated sources: conceptnet,wordnet,simplewiki (default: wordnet)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/trained_memory',
        help='Output path for trained memory (default: data/trained_memory)',
    )
    parser.add_argument(
        '--max-facts',
        type=int,
        default=None,
        help='Maximum total facts to load (default: unlimited)',
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Use cached loader data if available',
    )
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Use sample data only (no downloads)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for memory loading (default: 1000)',
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Show statistics of existing trained memory',
    )
    parser.add_argument(
        '--max-synsets',
        type=int,
        default=10000,
        help='Max WordNet synsets to process (default: 10000). Higher = more facts but slower.',
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=1.0,
        help='Min confidence for ConceptNet facts (default: 1.0). Lower = more facts.',
    )
    
    args = parser.parse_args()
    
    # Stats only mode
    if args.stats_only:
        _show_stats(args.output)
        return
    
    sources = [s.strip() for s in args.sources.split(',')]
    valid_sources = {'conceptnet', 'wordnet', 'simplewiki'}
    for s in sources:
        if s not in valid_sources:
            print(f"Error: Unknown source '{s}'. Valid: {', '.join(valid_sources)}")
            sys.exit(1)
    
    print("=" * 60)
    print("Gunter Knowledge Training")
    print("=" * 60)
    print(f"Sources: {', '.join(sources)}")
    print(f"Max facts: {args.max_facts or 'unlimited'}")
    if 'wordnet' in sources:
        print(f"Max synsets: {args.max_synsets}")
    if 'conceptnet' in sources:
        print(f"Min confidence: {args.min_confidence}")
    print(f"Output: {args.output}")
    print(f"Cache: {'yes' if args.cache else 'no'}")
    print(f"Sample mode: {'yes' if args.sample else 'no'}")
    print()
    
    t0 = time.time()
    
    # Step 1: Load and integrate facts
    print("\n[1/3] Loading knowledge sources...")
    print("-" * 40)
    
    from gunter.data.loaders.knowledge_integrator import KnowledgeIntegrator
    
    integrator = KnowledgeIntegrator()
    
    if args.sample:
        integrator.integrate_sample()
    else:
        max_per_source = args.max_facts
        integrator.integrate_all_sources(
            sources=sources,
            max_facts_per_source=max_per_source,
            use_cache=args.cache,
            wordnet_max_synsets=args.max_synsets,
            conceptnet_min_confidence=args.min_confidence,
        )
    
    # Step 2: Load into memory
    print("\n[2/3] Loading into VSA memory...")
    print("-" * 40)
    
    from gunter.main import Gunter
    
    g = Gunter()
    
    loaded = integrator.load_into_memory(
        g.memory,
        batch_size=args.batch_size,
        max_facts=args.max_facts,
    )
    
    # Step 3: Save trained memory
    print("\n[3/3] Saving trained memory...")
    print("-" * 40)
    
    output_path = str(PROJECT_ROOT / args.output)
    g.memory.save_to_disk(output_path)
    
    # Also save knowledge base
    kb_path = str(PROJECT_ROOT / "data" / "knowledge_base.json.gz")
    integrator.save_knowledge_base(kb_path)
    
    # Statistics
    elapsed = time.time() - t0
    
    print()
    integrator.print_statistics()
    
    mem_stats = g.memory.get_statistics()
    print(f"Memory statistics:")
    print(f"  Episodes: {mem_stats['total_episodes']:,}")
    print(f"  Lexicon size: {mem_stats['lexicon_size']:,}")
    print()
    
    print("=" * 60)
    print(f"Training complete in {elapsed:.1f}s")
    print(f"  Facts loaded: {loaded:,}")
    print(f"  Memory saved to: {output_path}")
    print("=" * 60)


def _show_stats(filepath: str):
    """Show statistics of an existing trained memory."""
    from gunter.memory import VectorMemory
    
    path = str(Path(PROJECT_ROOT / filepath))
    
    try:
        memory = VectorMemory.load_from_disk(path)
    except FileNotFoundError:
        print(f"No trained memory found at {path}")
        print("Run training first: python scripts/train_knowledge.py --sources wordnet")
        sys.exit(1)
    
    stats = memory.get_statistics()
    print(f"\nMemory Statistics for: {filepath}")
    print("=" * 50)
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k, v in list(val.items())[:10]:
                print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")
        else:
            print(f"  {key}: {val:,}" if isinstance(val, int) else f"  {key}: {val}")


if __name__ == '__main__':
    main()
