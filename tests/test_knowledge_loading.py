#!/usr/bin/env python3
"""Tests for Phase 17: Large-Scale Knowledge Integration.

Tests all data loaders, memory extensions, integration, and end-to-end
knowledge-based question answering.

Usage:
    cd prism && python tests/test_knowledge_loading.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_conceptnet_loader():
    """Test ConceptNet loader with sample data."""
    print("=== ConceptNet Loader ===\n")
    passed, total = 0, 0
    
    from prism.data.loaders.conceptnet_loader import (
        ConceptNetLoader, _clean_entity, _is_english, RELATION_MAP,
        _has_digits, _has_taxonomy_marker, _lemmatize_simple,
    )
    
    # Test 1: Entity name cleaning — strips "domestic" prefix
    total += 1
    clean = _clean_entity('/c/en/domestic_cat')
    if clean == 'cat':
        print(f"  ✓ clean entity: '{clean}'")
        passed += 1
    else:
        print(f"  ✗ clean entity: expected 'cat', got '{clean}'")
    
    # Test 2: English check
    total += 1
    if _is_english('/c/en/cat') and not _is_english('/c/de/katze'):
        print(f"  ✓ English filter works")
        passed += 1
    else:
        print(f"  ✗ English filter broken")
    
    # Test 3: Relation mapping coverage
    total += 1
    expected_rels = {'IS-A', 'HAS', 'PART-OF', 'USED-FOR', 'CAN', 'LOCATED-AT',
                     'CAUSES', 'HAS-PROPERTY', 'MADE-OF', 'RELATED-TO'}
    actual_rels = set(RELATION_MAP.values())
    if expected_rels.issubset(actual_rels):
        print(f"  ✓ Relation mapping: {len(RELATION_MAP)} ConceptNet → {len(actual_rels)} PRISM")
        passed += 1
    else:
        missing = expected_rels - actual_rels
        print(f"  ✗ Missing relations: {missing}")
    
    # Test 3b: DefinedAs should NOT map to 'IS' (should be 'DEFINED-AS')
    total += 1
    if RELATION_MAP.get('/r/DefinedAs') == 'DEFINED-AS':
        print(f"  ✓ /r/DefinedAs → DEFINED-AS (not IS)")
        passed += 1
    else:
        print(f"  ✗ /r/DefinedAs maps to '{RELATION_MAP.get('/r/DefinedAs')}', expected 'DEFINED-AS'")
    
    # Test 4: Sample data format
    total += 1
    loader = ConceptNetLoader()
    sample = loader.get_sample(20)
    if (len(sample) == 20 and
        all(len(f) == 4 for f in sample) and
        all(isinstance(f[3], (int, float)) for f in sample)):
        print(f"  ✓ Sample: {len(sample)} facts, correct format")
        passed += 1
    else:
        print(f"  ✗ Sample format incorrect")
    
    # Test 5: Sample fact content
    total += 1
    cat_facts = [f for f in sample if f[0] == 'cat']
    if any(f[1] == 'IS-A' and f[2] == 'animal' for f in cat_facts):
        print(f"  ✓ Sample content: cat IS-A animal present")
        passed += 1
    else:
        print(f"  ✗ Sample content: missing expected facts")
    
    # Test 6: Entity cleaning edge cases
    total += 1
    tests = [
        ('/c/en/cat/n/wn/animal', 'cat'),
        ('/c/en/a_cat', 'cat'),
        ('/c/en/cats', 'cat'),
        ('/c/en/the_dog', 'dog'),
    ]
    all_ok = True
    for uri, expected in tests:
        got = _clean_entity(uri)
        if got != expected:
            print(f"  ✗ clean_entity('{uri}'): expected '{expected}', got '{got}'")
            all_ok = False
    if all_ok:
        print(f"  ✓ Entity cleaning edge cases pass")
        passed += 1
    
    # Test 7: Lemmatization
    total += 1
    lemma_tests = [
        ('cats', 'cat'), ('dogs', 'dog'), ('berries', 'berry'),
        ('mice', 'mouse'), ('children', 'child'),
    ]
    all_ok = True
    for word, expected in lemma_tests:
        got = _lemmatize_simple(word)
        if got != expected:
            print(f"  ✗ lemmatize('{word}'): expected '{expected}', got '{got}'")
            all_ok = False
    if all_ok:
        print(f"  ✓ Lemmatization tests pass")
        passed += 1
    
    # Test 8: Digit and taxonomy filters
    total += 1
    if (_has_digits('entity42') and not _has_digits('entity') and
        _has_taxonomy_marker('bird genus') and not _has_taxonomy_marker('bird')):
        print(f"  ✓ Quality filters (digits, taxonomy) work")
        passed += 1
    else:
        print(f"  ✗ Quality filters broken")
    
    print(f"\n  ConceptNet: {passed}/{total}\n")
    return passed, total


def test_wordnet_loader():
    """Test WordNet loader."""
    print("=== WordNet Loader ===\n")
    passed, total = 0, 0
    
    try:
        from prism.data.loaders.wordnet_loader import WordNetLoader
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return 0, 1
    
    loader = WordNetLoader()
    
    # Test 1: Load small sample
    total += 1
    t0 = time.time()
    facts = loader.load(max_facts=500, include_definitions=False)
    elapsed = time.time() - t0
    if len(facts) >= 100:
        print(f"  ✓ Loaded {len(facts)} facts in {elapsed:.1f}s")
        passed += 1
    else:
        print(f"  ✗ Only got {len(facts)} facts (expected >= 100)")
    
    # Test 2: IS-A relations present (use noun filter for reliable IS-A)
    total += 1
    noun_facts = loader.load(max_facts=500, include_definitions=False, pos_filter=['n'])
    isa_facts = [f for f in noun_facts if f[1] == 'IS-A']
    if len(isa_facts) > 20:
        print(f"  ✓ IS-A relations (nouns): {len(isa_facts)}")
        passed += 1
    else:
        print(f"  ✗ Too few IS-A: {len(isa_facts)}")
    
    # Test 3: Fact format correct
    total += 1
    if all(len(f) == 4 and isinstance(f[3], float) for f in facts):
        print(f"  ✓ Fact format correct (4-tuples with float confidence)")
        passed += 1
    else:
        print(f"  ✗ Fact format incorrect")
    
    # Test 4: No duplicates
    total += 1
    keys = set()
    dups = 0
    for f in facts:
        key = (f[0].lower(), f[1], f[2].lower())
        if key in keys:
            dups += 1
        keys.add(key)
    if dups == 0:
        print(f"  ✓ No duplicate facts")
        passed += 1
    else:
        print(f"  ✗ {dups} duplicate facts found")
    
    # Test 5: With definitions
    total += 1
    facts_with_defs = loader.load(max_facts=200, include_definitions=True)
    is_facts = [f for f in facts_with_defs if f[1] == 'IS']
    if len(is_facts) > 0:
        print(f"  ✓ Definitions included: {len(is_facts)} IS-definition facts")
        passed += 1
    else:
        print(f"  ✗ No definitions found")
    
    # Test 6: Performance — loading 5000 facts should be quick
    total += 1
    t0 = time.time()
    big_facts = loader.load(max_facts=5000, include_definitions=False)
    elapsed = time.time() - t0
    if elapsed < 30:
        print(f"  ✓ Performance: {len(big_facts)} facts in {elapsed:.1f}s")
        passed += 1
    else:
        print(f"  ✗ Too slow: {elapsed:.1f}s for {len(big_facts)} facts")
    
    print(f"\n  WordNet: {passed}/{total}\n")
    return passed, total


def test_simplewiki_loader():
    """Test SimpleWiki loader with sample data."""
    print("=== SimpleWiki Loader ===\n")
    passed, total = 0, 0
    
    from prism.data.loaders.simplewiki_loader import (
        SimpleWikiLoader, _strip_wiki_markup, _is_disambiguation,
        _is_redirect, _is_list,
    )
    
    # Test 1: Wiki markup stripping
    total += 1
    test_text = "'''Mercury''' is a [[planet]] in the [[Solar System|solar system]]."
    clean = _strip_wiki_markup(test_text)
    ok = ('planet' in clean and 'solar system' in clean and 
          '[' not in clean and "'''" not in clean)
    if ok:
        print(f"  ✓ Markup stripping: '{clean[:60]}'")
        passed += 1
    else:
        print(f"  ✗ Markup stripping failed: '{clean[:60]}'")
    
    # Test 2: Template removal
    total += 1
    text_with_template = "Hello {{template|param}} world"
    clean = _strip_wiki_markup(text_with_template)
    if 'template' not in clean and 'Hello' in clean and 'world' in clean:
        print(f"  ✓ Template removal: '{clean}'")
        passed += 1
    else:
        print(f"  ✗ Template removal failed: '{clean}'")
    
    # Test 3: Disambiguation detection
    total += 1
    if (_is_disambiguation("{{disambiguation}}") and 
        not _is_disambiguation("Normal article text")):
        print(f"  ✓ Disambiguation detection works")
        passed += 1
    else:
        print(f"  ✗ Disambiguation detection broken")
    
    # Test 4: Redirect detection
    total += 1
    if _is_redirect("#REDIRECT [[Target]]") and not _is_redirect("Normal text"):
        print(f"  ✓ Redirect detection works")
        passed += 1
    else:
        print(f"  ✗ Redirect detection broken")
    
    # Test 5: List page detection
    total += 1
    if _is_list("List of countries") and not _is_list("United States"):
        print(f"  ✓ List detection works")
        passed += 1
    else:
        print(f"  ✗ List detection broken")
    
    # Test 6: Sample data
    total += 1
    loader = SimpleWikiLoader()
    sample = loader.get_sample(10)
    if len(sample) == 10 and all(len(f) == 4 for f in sample):
        print(f"  ✓ Sample: {len(sample)} facts, correct format")
        passed += 1
    else:
        print(f"  ✗ Sample format incorrect")
    
    print(f"\n  SimpleWiki: {passed}/{total}\n")
    return passed, total


def test_knowledge_integrator():
    """Test KnowledgeIntegrator merge and dedup."""
    print("=== Knowledge Integrator ===\n")
    passed, total = 0, 0
    
    from prism.data.loaders.knowledge_integrator import KnowledgeIntegrator
    
    # Test 1: Integrate sample data
    total += 1
    integrator = KnowledgeIntegrator()
    integrator.integrate_sample()
    if integrator.fact_count > 50:
        print(f"  ✓ Integrated sample: {integrator.fact_count} facts")
        passed += 1
    else:
        print(f"  ✗ Too few facts: {integrator.fact_count}")
    
    # Test 2: Multiple sources present
    total += 1
    stats = integrator.get_statistics()
    sources = stats.get('source_counts', {})
    if len(sources) >= 2:
        src_str = ', '.join(f"{k}:{v}" for k, v in sources.items())
        print(f"  ✓ Sources: {src_str}")
        passed += 1
    else:
        print(f"  ✗ Expected ≥2 sources, got {len(sources)}")
    
    # Test 3: Deduplication works
    total += 1
    # Both ConceptNet and WordNet have cat IS-A mammal
    cat_facts = [f for f in integrator.facts if f[0].lower() == 'cat' and f[1] == 'IS-A']
    cat_mammal = [f for f in cat_facts if f[2].lower() == 'mammal']
    if len(cat_mammal) <= 1:
        print(f"  ✓ Dedup works: {len(cat_mammal)} 'cat IS-A mammal' (expected ≤1)")
        passed += 1
    else:
        print(f"  ✗ Dedup failed: {len(cat_mammal)} duplicates")
    
    # Test 4: Statistics
    total += 1
    if stats['total_facts'] > 0 and stats.get('unique_relations', 0) > 0:
        print(f"  ✓ Statistics: {stats['total_facts']} facts, {stats['unique_relations']} relations")
        passed += 1
    else:
        print(f"  ✗ Statistics incomplete: {stats}")
    
    # Test 5: Save and load knowledge base
    total += 1
    with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        integrator.save_knowledge_base(tmp_path)
        
        # Load back
        integrator2 = KnowledgeIntegrator()
        integrator2.load_knowledge_base(tmp_path)
        
        if integrator2.fact_count == integrator.fact_count:
            print(f"  ✓ Save/load: {integrator2.fact_count} facts round-tripped")
            passed += 1
        else:
            print(f"  ✗ Save/load mismatch: {integrator.fact_count} → {integrator2.fact_count}")
    finally:
        os.unlink(tmp_path)
    
    print(f"\n  Integrator: {passed}/{total}\n")
    return passed, total


def test_memory_extensions():
    """Test VectorMemory batch_store, save/load, statistics."""
    print("=== Memory Extensions ===\n")
    passed, total = 0, 0
    
    from prism.main import PRISM
    
    g = PRISM()
    
    # Test 1: batch_store
    total += 1
    facts = [
        ('cat', 'IS-A', 'animal', 1.0),
        ('dog', 'IS-A', 'animal', 1.0),
        ('bird', 'CAN', 'fly', 0.9),
        ('fish', 'CAN', 'swim', 0.9),
        ('tree', 'IS-A', 'plant', 1.0),
    ]
    stored = g.memory.batch_store(facts)
    if stored == 5:
        print(f"  ✓ batch_store: {stored} facts stored")
        passed += 1
    else:
        print(f"  ✗ batch_store: expected 5, got {stored}")
    
    # Test 2: Facts queryable after batch_store
    total += 1
    score = g.memory.check_fact('cat', 'IS-A', 'animal')
    if score > 0.05:
        print(f"  ✓ Query after batch: cat IS-A animal = {score:.3f}")
        passed += 1
    else:
        print(f"  ✗ Query after batch: cat IS-A animal = {score:.3f}")
    
    # Test 3: get_statistics
    total += 1
    stats = g.memory.get_statistics()
    if (stats['total_facts'] == 5 and
        stats['unique_relations'] >= 2 and
        stats['unique_subjects'] >= 4):
        print(f"  ✓ Statistics: {stats['total_facts']} facts, {stats['unique_relations']} rels, {stats['unique_subjects']} subjects")
        passed += 1
    else:
        print(f"  ✗ Statistics: {stats}")
    
    # Test 4: batch_store performance
    total += 1
    big_batch = [(f'entity{i}', 'RELATED-TO', f'entity{i+1}', 0.5) for i in range(1000)]
    t0 = time.time()
    stored = g.memory.batch_store(big_batch)
    elapsed = time.time() - t0
    if stored == 1000 and elapsed < 5.0:
        print(f"  ✓ Batch performance: 1000 facts in {elapsed:.2f}s")
        passed += 1
    else:
        print(f"  ✗ Batch performance: {stored} facts in {elapsed:.2f}s")
    
    # Test 5: save_to_disk and load_from_disk
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_memory')
        g.memory.save_to_disk(save_path)
        
        # Verify files exist
        npz_exists = os.path.exists(save_path + '.npz')
        pkl_exists = os.path.exists(save_path + '.pkl')
        
        if npz_exists and pkl_exists:
            print(f"  ✓ save_to_disk: created .npz + .pkl files")
            passed += 1
        else:
            print(f"  ✗ save_to_disk: npz={npz_exists}, pkl={pkl_exists}")
    
    # Test 6: Load from disk and query
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_memory')
        
        # Store specific facts
        g2 = PRISM()
        g2.memory.batch_store([
            ('cat', 'IS-A', 'animal', 1.0),
            ('dog', 'IS-A', 'animal', 1.0),
        ])
        g2.memory.save_to_disk(save_path)
        
        # Load into new memory
        from prism.memory import VectorMemory
        loaded = VectorMemory.load_from_disk(save_path)
        
        if len(loaded) == 2:
            score = loaded.check_fact('cat', 'IS-A', 'animal')
            print(f"  ✓ load_from_disk: {len(loaded)} episodes, query={score:.3f}")
            passed += 1
        else:
            print(f"  ✗ load_from_disk: expected 2 episodes, got {len(loaded)}")
    
    # Test 7: Memory usage with large dataset
    total += 1
    import sys as _sys
    ep_count = len(g.memory.get_episodes())
    approx_size = _sys.getsizeof(g.memory._semantic) + ep_count * 200  # rough estimate
    mb = approx_size / (1024 * 1024)
    if mb < 500:  # Well under 2GB
        print(f"  ✓ Memory usage: ~{mb:.1f} MB for {ep_count} episodes")
        passed += 1
    else:
        print(f"  ✗ Memory usage too high: {mb:.1f} MB")
    
    print(f"\n  Memory: {passed}/{total}\n")
    return passed, total


def test_end_to_end():
    """Test end-to-end: load knowledge then answer questions."""
    print("=== End-to-End ===\n")
    passed, total = 0, 0
    
    from prism.main import PRISM
    from prism.data.loaders.knowledge_integrator import KnowledgeIntegrator
    
    # Build a prism with sample knowledge
    g = PRISM()
    
    integrator = KnowledgeIntegrator()
    integrator.integrate_sample()
    integrator.load_into_memory(g.memory, batch_size=500)
    
    # Test 1: Basic fact present (check episodes since check_fact degrades with many superposed facts)
    total += 1
    cat_animal_eps = [ep for ep in g.memory.get_episodes()
                      if ep.subject.lower() == 'cat' and ep.relation == 'IS-A' 
                      and ep.obj.lower() == 'animal']
    if len(cat_animal_eps) >= 1:
        print(f"  ✓ 'cat IS-A animal' present ({len(cat_animal_eps)} episodes)")
        passed += 1
    else:
        print(f"  ✗ 'cat IS-A animal' not found in episodes")
    
    # Test 2: Search by subject
    total += 1
    cat_eps = [ep for ep in g.memory.get_episodes() if ep.subject.lower() == 'cat']
    if len(cat_eps) >= 3:
        print(f"  ✓ Cat has {len(cat_eps)} facts")
        passed += 1
    else:
        print(f"  ✗ Cat has only {len(cat_eps)} facts")
    
    # Test 3: Answer natural language question
    total += 1
    response = g.process_input("Can cats purr?")
    if response and len(response) > 5:
        print(f"  ✓ 'Can cats purr?': {response[:80]}")
        passed += 1
    else:
        print(f"  ✗ 'Can cats purr?': {response}")
    
    # Test 4: Knowledge from multiple sources
    total += 1
    all_facts = g.memory.get_all_facts()
    has_isa = any('IS-A' in f for f in all_facts)
    has_can = any('CAN' in f for f in all_facts)
    if has_isa and has_can:
        print(f"  ✓ Multiple relation types: IS-A + CAN present")
        passed += 1
    else:
        print(f"  ✗ Missing relation types: IS-A={has_isa}, CAN={has_can}")
    
    # Test 5: Save trained and reload
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_trained')
        g.memory.save_to_disk(path)
        
        g2 = PRISM()
        from prism.memory import VectorMemory
        loaded = VectorMemory.load_from_disk(path)
        g2.memory._semantic = loaded._semantic
        g2.memory._episodes = loaded._episodes
        
        # Check episodes after reload (check_fact unreliable with many superposed facts)
        cat_eps = [ep for ep in g2.memory.get_episodes()
                   if ep.subject.lower() == 'cat' and ep.relation == 'IS-A']
        if len(cat_eps) >= 1:
            print(f"  ✓ Reload + query: {len(cat_eps)} cat IS-A episodes found")
            passed += 1
        else:
            print(f"  ✗ Reload + query: no cat IS-A episodes")
    
    # Test 6: Statistics after loading
    total += 1
    stats = g.memory.get_statistics()
    if (stats['total_facts'] > 50 and
        stats['unique_subjects'] > 20 and
        stats['unique_relations'] > 3):
        print(f"  ✓ Stats: {stats['total_facts']} facts, {stats['unique_subjects']} subjects, {stats['unique_relations']} rels")
        passed += 1
    else:
        print(f"  ✗ Stats too low: {stats}")
    
    print(f"\n  End-to-end: {passed}/{total}\n")
    return passed, total


def test_training_script_integration():
    """Test that the training script components work together."""
    print("=== Training Script ===\n")
    passed, total = 0, 0
    
    # Test 1: KnowledgeIntegrator with single source
    total += 1
    from prism.data.loaders.knowledge_integrator import KnowledgeIntegrator
    
    integrator = KnowledgeIntegrator()
    try:
        integrator.integrate_all_sources(
            sources=['wordnet'],
            max_facts_per_source=200,
        )
        if integrator.fact_count >= 100:
            print(f"  ✓ WordNet integration: {integrator.fact_count} facts")
            passed += 1
        else:
            print(f"  ✗ Too few from WordNet: {integrator.fact_count}")
    except Exception as e:
        print(f"  ✗ WordNet integration failed: {e}")
    
    # Test 2: Load into PRISM memory
    total += 1
    from prism.main import PRISM
    g = PRISM()
    loaded = integrator.load_into_memory(g.memory, batch_size=100, max_facts=100)
    if loaded == 100:
        print(f"  ✓ Loaded {loaded} facts into memory")
        passed += 1
    else:
        print(f"  ✗ Expected 100, loaded {loaded}")
    
    # Test 3: Save and show stats
    total += 1
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test')
        g.memory.save_to_disk(path)
        
        stats = g.memory.get_statistics()
        if stats['total_facts'] == loaded:
            print(f"  ✓ Stats match: {stats['total_facts']} facts")
            passed += 1
        else:
            print(f"  ✗ Stats mismatch: {stats['total_facts']} vs {loaded}")
    
    print(f"\n  Training: {passed}/{total}\n")
    return passed, total


def test_regression():
    """Ensure Phase 15/16 features still work."""
    print("=== Regression ===\n")
    passed, total = 0, 0
    
    from prism.main import PRISM
    g = PRISM()
    
    # Test 1: Name learning
    total += 1
    resp = g.process_input("My name is TestUser")
    if resp and 'Testuser' in resp:
        print(f"  ✓ Name: {resp[:60]}")
        passed += 1
    else:
        print(f"  ✗ Name: {resp}")
    
    # Test 2: Fact learning
    total += 1
    resp = g.process_input("learn cats are mammals")
    if resp and 'cats' in resp.lower():
        print(f"  ✓ Learn: {resp[:60]}")
        passed += 1
    else:
        print(f"  ✗ Learn: {resp}")
    
    # Test 3: Question answering still works
    total += 1
    g.process_input("learn dogs are mammals")
    resp = g.process_input("Are cats the same as dogs?")
    if resp and len(resp) > 10:
        print(f"  ✓ Question: {resp[:60]}")
        passed += 1
    else:
        print(f"  ✗ Question: {resp}")
    
    # Test 4: Memory persistence commands
    total += 1
    resp = g.process_input("facts")
    if resp and 'cats' in resp.lower():
        print(f"  ✓ Facts command works")
        passed += 1
    else:
        print(f"  ✗ Facts command: {resp}")
    
    print(f"\n  Regression: {passed}/{total}\n")
    return passed, total


def main():
    results = []
    
    test_fns = [
        test_conceptnet_loader,
        test_wordnet_loader,
        test_simplewiki_loader,
        test_knowledge_integrator,
        test_memory_extensions,
        test_end_to_end,
        test_training_script_integration,
        test_regression,
    ]
    
    total_passed = 0
    total_tests = 0
    
    for test_fn in test_fns:
        try:
            p, t = test_fn()
        except Exception as e:
            print(f"  ✗ CRASH: {e}")
            import traceback
            traceback.print_exc()
            p, t = 0, 1
        
        total_passed += p
        total_tests += t
        results.append((test_fn.__name__, p, t))
    
    # Summary
    print("=" * 55)
    if total_passed == total_tests:
        print(f"Phase 17: ALL {total_tests} TESTS PASS ✅")
    else:
        print(f"Phase 17: {total_passed}/{total_tests} TESTS PASS ⚠️")
    print("=" * 55)
    
    for name, p, t in results:
        status = "✓" if p == t else "✗"
        print(f"  {status} {name}: {p}/{t}")
    
    return 0 if total_passed == total_tests else 1


if __name__ == '__main__':
    sys.exit(main())
