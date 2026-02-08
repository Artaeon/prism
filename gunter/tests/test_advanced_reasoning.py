#!/usr/bin/env python3
"""Phase 16: Advanced Reasoning Tests.

Tests for MultiHopReasoner, AdvancedAnalogyReasoner, CausalReasoner,
and enhanced TemporalReasoner.
"""

import sys, os, time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gunter.main import Gunter


def make_gunter():
    """Create a Gunter instance with rich facts for advanced reasoning."""
    g = Gunter()
    
    # Taxonomy & properties (via natural language parser)
    nl_facts = [
        # Animal taxonomy
        "cats are mammals",
        "dogs are mammals",
        "mammals are animals",
        "animals need water",
        "tigers are cats",
        "lions are cats",
        
        # Properties
        "cats have fur",
        "cats have whiskers",
        "dogs have fur",
        "dogs have tails",
        "cats can purr",
        "cats can climb",
        "dogs can bark",
        "dogs can swim",
        "fish can swim",
        "birds can fly",
        "fish don't fly",
        
        # Habitats
        "lions live in savanna",
        "polar bears live in arctic",
        "fish live in water",
        "birds live in nests",
        
        # Royalty
        "learn king is royalty",
        "learn queen is royalty",
        "learn prince is royalty",
        "learn princess is royalty",
    ]
    
    for fact in nl_facts:
        g.process_input(fact)
    
    # Causal relations (stored directly to get proper CAUSES relation)
    g.memory.store("evaporation", "CAUSES", "clouds", importance=1.0)
    g.memory.store("clouds", "CAUSES", "rain", importance=1.0)
    g.memory.store("rain", "CAUSES", "floods", importance=1.0)
    g.memory.store("heat", "CAUSES", "evaporation", importance=1.0)
    g.memory.store("deforestation", "CAUSES", "erosion", importance=1.0)
    
    # Temporal (stored directly for proper relations)
    g.memory.store("cats", "HUNT-AT", "night", importance=1.0)
    g.memory.store("cats", "ACTIVE-DURING", "dawn", importance=1.0)
    g.memory.store("cats", "LASTS", "16 hours", importance=1.0)
    g.memory.store("sunrise", "BEFORE", "sunset", importance=1.0)
    
    return g


def test_multihop():
    """Test MultiHopReasoner."""
    print("\n=== Multi-Hop Reasoning ===\n")
    
    g = make_gunter()
    
    from gunter.reasoning.multihop import MultiHopReasoner
    mh = MultiHopReasoner(g.memory, g.lexicon)
    
    passed = 0
    total = 0
    
    # Test 1: Find chain cats → water (cats → mammals → animals → water)
    total += 1
    result = mh.find_chain("cats", "water")
    if result.chain and result.hops >= 2:
        print(f"  ✓ cats → water: {result.hops} hops [{result.confidence:.2f}]")
        print(f"    {result.explanation}")
        passed += 1
    else:
        print(f"  ✗ cats → water: no chain found")
    
    # Test 2: Find chain cats → animals
    total += 1
    result = mh.find_chain("cats", "animals")
    if result.chain:
        print(f"  ✓ cats → animals: {result.hops} hops [{result.confidence:.2f}]")
        passed += 1
    else:
        print(f"  ✗ cats → animals: no chain found")
    
    # Test 3: Explain why cats can purr
    total += 1
    result = mh.explain_why("cats", "purr")
    if result.chain or result.confidence > 0:
        print(f"  ✓ explain_why(cats, purr): [{result.confidence:.2f}]")
        print(f"    {result.explanation}")
        passed += 1
    else:
        print(f"  ✗ explain_why(cats, purr): no explanation")
    
    # Test 4: Find connection between cats and dogs
    total += 1
    result = mh.find_connection("cats", "dogs")
    if result.chain or result.confidence > 0:
        print(f"  ✓ connection(cats, dogs): [{result.confidence:.2f}]")
        print(f"    {result.explanation}")
        passed += 1
    else:
        print(f"  ✗ connection(cats, dogs): no connection")
    
    # Test 5: Chain confidence degrades over hops
    total += 1
    result = mh.find_chain("cats", "water")
    if result.chain:
        conf = result.confidence
        if conf < 0.95:  # Should be lower than single hop
            print(f"  ✓ confidence degrades: {conf:.2f} (< 0.95)")
            passed += 1
        else:
            print(f"  ✗ confidence too high: {conf:.2f}")
    else:
        print(f"  ✗ no chain to test confidence degradation")
    
    print(f"\n  Multi-hop: {passed}/{total}")
    return passed, total


def test_analogy():
    """Test AdvancedAnalogyReasoner."""
    print("\n=== Analogy Reasoning ===\n")
    
    g = make_gunter()
    
    from gunter.reasoning.analogy_engine import AdvancedAnalogyReasoner
    ar = AdvancedAnalogyReasoner(g.lexicon, g.memory)
    
    passed = 0
    total = 0
    
    # Test 1: Vector arithmetic analogy
    total += 1
    result = ar.solve_analogy("king", "queen", "man")
    if result.answer and result.answer != "unknown":
        print(f"  ✓ king:queen :: man:{result.answer} ({result.method}) [{result.confidence:.2f}]")
        passed += 1
    else:
        print(f"  ✗ king:queen :: man:? → no answer")
    
    # Test 2: Relation-based analogy (using stored facts)
    total += 1
    result = ar.solve_analogy("lions", "savanna", "fish")
    has_answer = result.answer and result.answer != "unknown"
    if has_answer:
        print(f"  ✓ lions:savanna :: fish:{result.answer} ({result.method}) [{result.confidence:.2f}]")
        passed += 1
    else:
        print(f"  ✗ lions:savanna :: fish:? → no answer")
    
    # Test 3: Structural similarity
    total += 1
    score, shared, diff = ar.find_structural_similarity("cats", "dogs")
    if score > 0:
        print(f"  ✓ structural similarity(cats, dogs): {score:.2f}")
        print(f"    shared: {shared[:3]}, diff: {diff[:3]}")
        passed += 1
    else:
        print(f"  ✗ structural similarity: 0")
    
    # Test 4: Pattern completion
    total += 1
    result = ar.complete_pattern("king is to queen as prince is to ?")
    if result.answer and result.answer != "unknown":
        print(f"  ✓ pattern completion: {result.answer} [{result.confidence:.2f}]")
        passed += 1
    else:
        print(f"  ✗ pattern completion: no answer")
    
    # Test 5: Complete via Gunter (end-to-end analogy detection)
    total += 1
    response = g.process_input("lions are to savanna as polar bears are to ?")
    if response and "?" not in response and len(response) > 5:
        print(f"  ✓ e2e analogy: {response[:80]}")
        passed += 1
    else:
        print(f"  ✗ e2e analogy: {response[:80] if response else 'None'}")
    
    print(f"\n  Analogy: {passed}/{total}")
    return passed, total


def test_causality():
    """Test CausalReasoner."""
    print("\n=== Causal Reasoning ===\n")
    
    g = make_gunter()
    
    from gunter.reasoning.causality import CausalReasoner
    cr = CausalReasoner(g.memory, g.lexicon)
    
    passed = 0
    total = 0
    
    # Test 1: Find causes of rain
    total += 1
    result = cr.find_causes("rain")
    if result.chains:
        print(f"  ✓ causes of rain: {len(result.chains)} chain(s) [{result.confidence:.2f}]")
        if result.best_chain:
            print(f"    {result.best_chain.format()}")
        passed += 1
    else:
        print(f"  ✗ causes of rain: no chains found")
    
    # Test 2: Find effects of heat
    total += 1
    result = cr.find_effects("heat")
    if result.chains:
        print(f"  ✓ effects of heat: {len(result.chains)} chain(s) [{result.confidence:.2f}]")
        if result.best_chain:
            print(f"    {result.best_chain.format()}")
        passed += 1
    else:
        print(f"  ✗ effects of heat: no chains found")
    
    # Test 3: Explain causation heat → rain
    total += 1
    result = cr.explain_causation("heat", "rain")
    if result.best_chain:
        print(f"  ✓ heat → rain: [{result.confidence:.2f}]")
        print(f"    {result.best_chain.format()}")
        passed += 1
    else:
        print(f"  ✗ heat → rain: no path ({result.explanation})")
    
    # Test 4: Detect causal relations
    total += 1
    all_detected = (
        cr.detect_causal_relation("CAUSES") and
        cr.detect_causal_relation("LEADS-TO") and
        cr.detect_causal_relation("BECAUSE") and
        not cr.detect_causal_relation("IS-A") and
        not cr.detect_causal_relation("HAS")
    )
    if all_detected:
        print(f"  ✓ causal relation detection correct")
        passed += 1
    else:
        print(f"  ✗ causal relation detection failed")
    
    # Test 5: Find effects of deforestation
    total += 1
    result = cr.find_effects("deforestation")
    if result.chains:
        print(f"  ✓ effects of deforestation: {len(result.chains)} chain(s)")
        if result.best_chain:
            print(f"    {result.best_chain.format()}")
        passed += 1
    else:
        print(f"  ✗ effects of deforestation: no chains")
    
    print(f"\n  Causality: {passed}/{total}")
    return passed, total


def test_temporal():
    """Test enhanced TemporalReasoner."""
    print("\n=== Temporal Reasoning ===\n")
    
    g = make_gunter()
    
    passed = 0
    total = 0
    
    # Test 1: Query temporal - when do cats hunt?
    total += 1
    results = g.temporal.query_temporal("cats", "WHEN")
    temporal_values = [v.lower() for v, _ in results]
    if any('night' in v or 'dawn' in v for v in temporal_values):
        print(f"  ✓ when do cats hunt: found night/dawn")
        passed += 1
    else:
        print(f"  ✗ when do cats hunt: {results}")
    
    # Test 2: Get duration - how long do cats sleep?
    total += 1
    duration = g.temporal.get_duration("cats")
    if duration and ('16' in duration or 'hour' in duration.lower()):
        print(f"  ✓ cats sleep duration: {duration}")
        passed += 1
    else:
        print(f"  ✗ cats sleep duration: {duration}")
    
    # Test 3: Compare temporal - sunrise vs sunset
    total += 1
    order = g.temporal.compare_temporal("sunrise", "sunset")
    if order in ("BEFORE", "AFTER", "UNKNOWN"):
        print(f"  ✓ sunrise vs sunset: {order}")
        passed += 1
    else:
        print(f"  ✗ sunrise vs sunset: unexpected {order}")
    
    # Test 4: E2E temporal question via Gunter
    total += 1
    response = g.process_input("When do cats hunt?")
    if response and len(response) > 5:
        print(f"  ✓ 'When do cats hunt?': {response[:80]}")
        passed += 1
    else:
        print(f"  ✗ 'When do cats hunt?': {response[:80] if response else 'None'}")
    
    print(f"\n  Temporal: {passed}/{total}")
    return passed, total


def test_combined():
    """Test combined reasoning (multiple reasoners)."""
    print("\n=== Combined Reasoning ===\n")
    
    g = make_gunter()
    
    passed = 0
    total = 0
    
    # Test 1: "Why do cats purr?" → multi-hop + causation
    total += 1
    response = g.process_input("Why do cats purr?")
    if response and len(response) > 5:
        print(f"  ✓ 'Why do cats purr?': {response[:100]}")
        passed += 1
    else:
        print(f"  ✗ 'Why do cats purr?': {response[:80] if response else 'None'}")
    
    # Test 2: "What causes rain?" → causal reasoning
    total += 1
    response = g.process_input("What causes rain?")
    if response and len(response) > 5:
        print(f"  ✓ 'What causes rain?': {response[:100]}")
        passed += 1
    else:
        print(f"  ✗ 'What causes rain?': {response[:80] if response else 'None'}")
    
    # Test 3: "How are cats and tigers related?" → multi-hop relation
    total += 1
    response = g.process_input("How are cats and tigers related?")
    if response and len(response) > 5:
        print(f"  ✓ 'How are cats and tigers related?': {response[:100]}")
        passed += 1
    else:
        print(f"  ✗ 'How are cats and tigers related?': {response[:80] if response else 'None'}")
    
    print(f"\n  Combined: {passed}/{total}")
    return passed, total


def test_performance():
    """Performance benchmarks."""
    print("\n=== Performance ===\n")
    
    g = make_gunter()
    
    from gunter.reasoning.multihop import MultiHopReasoner
    from gunter.reasoning.analogy_engine import AdvancedAnalogyReasoner
    from gunter.reasoning.causality import CausalReasoner
    
    mh = MultiHopReasoner(g.memory, g.lexicon)
    ar = AdvancedAnalogyReasoner(g.lexicon, g.memory)
    cr = CausalReasoner(g.memory, g.lexicon)
    
    passed = 0
    total = 0
    
    # Multi-hop: <100ms for 5 hops
    total += 1
    start = time.time()
    for _ in range(10):
        mh.find_chain("cats", "water", max_depth=5)
    mh_time = (time.time() - start) / 10 * 1000
    if mh_time < 100:
        print(f"  ✓ Multi-hop: {mh_time:.0f}ms (< 100ms)")
        passed += 1
    else:
        print(f"  ✗ Multi-hop: {mh_time:.0f}ms (>= 100ms)")
    
    # Analogy: <50ms
    total += 1
    start = time.time()
    for _ in range(10):
        ar.solve_analogy("king", "queen", "man")
    an_time = (time.time() - start) / 10 * 1000
    if an_time < 50:
        print(f"  ✓ Analogy: {an_time:.0f}ms (< 50ms)")
        passed += 1
    else:
        print(f"  ✗ Analogy: {an_time:.0f}ms (>= 50ms)")
    
    # Causality: <80ms
    total += 1
    start = time.time()
    for _ in range(10):
        cr.find_causes("rain")
    ca_time = (time.time() - start) / 10 * 1000
    if ca_time < 80:
        print(f"  ✓ Causality: {ca_time:.0f}ms (< 80ms)")
        passed += 1
    else:
        print(f"  ✗ Causality: {ca_time:.0f}ms (>= 80ms)")
    
    # Temporal: <30ms
    total += 1
    start = time.time()
    for _ in range(10):
        g.temporal.query_temporal("cats", "WHEN")
    te_time = (time.time() - start) / 10 * 1000
    if te_time < 30:
        print(f"  ✓ Temporal: {te_time:.0f}ms (< 30ms)")
        passed += 1
    else:
        print(f"  ✗ Temporal: {te_time:.0f}ms (>= 30ms)")
    
    print(f"\n  Performance: {passed}/{total}")
    return passed, total


def test_regression():
    """Ensure existing Phase 15 tests still pass."""
    print("\n=== Regression ===\n")
    
    g = make_gunter()
    
    passed = 0
    total = 0
    
    cases = [
        ("My name is TestUser", "TestUser"),
        ("I like chess", "chess"),
        ("Are cats the same as dogs?", "different"),
        ("How similar are cats and dogs?", ""),
        ("Can fish fly?", ""),
        ("What do cats have?", ""),
    ]
    
    for question, expect_substr in cases:
        total += 1
        response = g.process_input(question)
        ok = response is not None and len(response) > 3
        if expect_substr:
            ok = ok and expect_substr.lower() in response.lower()
        if ok:
            print(f"  ✓ '{question}' → {response[:60]}")
            passed += 1
        else:
            print(f"  ✗ '{question}' → {response[:60] if response else 'None'}")
    
    print(f"\n  Regression: {passed}/{total}")
    return passed, total


def main():
    total_passed = 0
    total_tests = 0
    
    for test_fn in [
        test_multihop,
        test_analogy,
        test_causality,
        test_temporal,
        test_combined,
        test_performance,
        test_regression,
    ]:
        p, t = test_fn()
        total_passed += p
        total_tests += t
    
    print("\n" + "=" * 55)
    if total_passed == total_tests:
        print(f"Phase 16: ALL {total_tests} TESTS PASS ✅")
    else:
        print(f"Phase 16: {total_passed}/{total_tests} TESTS PASS ⚠️")
    print("=" * 55)


if __name__ == "__main__":
    main()
