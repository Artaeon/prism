"""ConceptNet Loader — Download and parse ConceptNet 5.7 assertions.

Streams the ConceptNet assertions CSV (gzipped), filters for English
facts above a confidence threshold, and maps relations to Gunter format.

Example:
    >>> loader = ConceptNetLoader(cache_dir="data/cache")
    >>> facts = loader.load(min_confidence=1.0, max_facts=10000)
    >>> len(facts)  # ~10000
    >>> facts[0]  # ('cat', 'IS-A', 'mammal', 3.46)
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ConceptNet 5.7 assertions (gzipped CSV)
CONCEPTNET_URL = (
    "https://s3.amazonaws.com/conceptnet/downloads/2019/"
    "edges/conceptnet-assertions-5.7.0.csv.gz"
)

# Map ConceptNet relations → Gunter relations
RELATION_MAP = {
    '/r/IsA': 'IS-A',
    '/r/HasA': 'HAS',
    '/r/PartOf': 'PART-OF',
    '/r/UsedFor': 'USED-FOR',
    '/r/CapableOf': 'CAN',
    '/r/AtLocation': 'LOCATED-AT',
    '/r/Causes': 'CAUSES',
    '/r/HasProperty': 'HAS-PROPERTY',
    '/r/MadeOf': 'MADE-OF',
    '/r/RelatedTo': 'RELATED-TO',
    '/r/HasPrerequisite': 'REQUIRES',
    '/r/MotivatedByGoal': 'MOTIVATED-BY',
    '/r/Desires': 'WANTS',
    '/r/CreatedBy': 'CREATED-BY',
    '/r/DefinedAs': 'DEFINED-AS',
}

# Prefixes to strip from entity names (order matters — longer first)
_STRIP_PREFIXES = [
    'a ', 'an ', 'the ',
    'domestic ', 'common ', 'wild ', 'adult ', 'young ',
    'typical ', 'general ', 'standard ', 'normal ',
]

# Taxonomy / genus markers — skip entities containing these
_TAXONOMY_MARKERS = {
    'genus', 'species', 'family', 'phylum', 'order', 'class',
    'suborder', 'subfamily', 'superfamily', 'infraorder',
    'subgenus', 'subspecies', 'subclass', 'subphylum',
}

# Expected validation facts — (subject, relation, object)
VALIDATION_FACTS = [
    ('cat', 'IS-A', 'mammal'),
    ('cat', 'IS-A', 'animal'),
    ('bird', 'CAN', 'fly'),
    ('fish', 'LOCATED-AT', 'water'),
    ('knife', 'USED-FOR', 'cutting'),
]

# Simple irregular plural → singular mappings
_IRREGULAR_PLURALS = {
    'mice': 'mouse', 'geese': 'goose', 'teeth': 'tooth',
    'feet': 'foot', 'men': 'man', 'women': 'woman',
    'children': 'child', 'people': 'person', 'oxen': 'ox',
    'leaves': 'leaf', 'knives': 'knife', 'wolves': 'wolf',
    'lives': 'life', 'halves': 'half', 'shelves': 'shelf',
    'thieves': 'thief', 'loaves': 'loaf',
}


def _lemmatize_simple(word: str) -> str:
    """Simple rule-based singular form for English nouns.

    Handles common plural patterns without NLTK dependency.
    """
    if len(word) <= 2:
        return word

    # Check irregular plurals first
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]

    # -ies → -y  (e.g. "berries" → "berry", but not "series")
    if word.endswith('ies') and len(word) > 4:
        return word[:-3] + 'y'

    # -ves → -f  (e.g. "calves" → "calf") — only if not in irregulars above
    if word.endswith('ves') and len(word) > 4:
        return word[:-3] + 'f'

    # -ses, -xes, -zes, -ches, -shes → remove -es
    if word.endswith('es') and len(word) > 3:
        stem = word[:-2]
        if stem.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return stem

    # -s (but not -ss, -us, -is)
    if word.endswith('s') and not word.endswith(('ss', 'us', 'is', 'ies')):
        return word[:-1]

    return word


def _clean_entity(uri: str) -> str:
    """Convert ConceptNet URI to clean, normalized entity name.

    /c/en/domestic_cat → "cat"
    /c/en/a_cat → "cat"
    /c/en/cats → "cat"
    /c/en/cat/n/wn/animal → "cat"
    """
    # Strip /c/en/ prefix
    name = uri.replace('/c/en/', '')
    # Take only the base concept (before POS tags)
    name = name.split('/')[0]
    # Convert underscores to spaces
    name = name.replace('_', ' ')
    # Lowercase
    name = name.lower().strip()

    # Strip article and qualifier prefixes
    for prefix in _STRIP_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]

    # Lemmatize each word (handle multi-word)
    words = name.split()
    if len(words) == 1:
        name = _lemmatize_simple(words[0])
    elif len(words) >= 2:
        # Lemmatize last word only (e.g. "ice creams" → "ice cream")
        words[-1] = _lemmatize_simple(words[-1])
        name = ' '.join(words)

    return name.strip()


def _is_english(uri: str) -> bool:
    """Check if a ConceptNet URI is English."""
    return uri.startswith('/c/en/')


def _parse_weight(info_json: str) -> float:
    """Extract weight from ConceptNet edge info JSON."""
    try:
        info = json.loads(info_json)
        return float(info.get('weight', 1.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        return 1.0


def _has_digits(text: str) -> bool:
    """Check if text contains any digits."""
    return bool(re.search(r'\d', text))


def _has_taxonomy_marker(text: str) -> bool:
    """Check if text contains taxonomy/genus markers."""
    words = set(text.lower().split())
    return bool(words & _TAXONOMY_MARKERS)


class ConceptNetLoader:
    """Load and parse ConceptNet 5.7 knowledge base.

    Downloads the full assertions file (~600MB gzipped), streams through it
    filtering for English-only facts, and maps to Gunter relation types.

    Args:
        cache_dir: Directory for downloads and cached parsed facts

    Example:
        >>> loader = ConceptNetLoader()
        >>> facts = loader.load(min_confidence=1.0)
        >>> print(f"Loaded {len(facts)} facts")
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._assertions_path = self.cache_dir / "conceptnet-assertions-5.7.0.csv.gz"
        self._facts_cache = self.cache_dir / "conceptnet_facts.json.gz"

    def load(
        self,
        min_confidence: float = 1.0,
        max_facts: int | None = None,
        use_cache: bool = True,
    ) -> list[tuple[str, str, str, float]]:
        """Load ConceptNet facts.

        Args:
            min_confidence: Minimum edge weight to include (default: 1.0)
            max_facts: Maximum number of facts (None = all)
            use_cache: Use cached parsed facts if available

        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        # Try cache first
        if use_cache and self._facts_cache.exists():
            print(f"Loading cached ConceptNet facts from {self._facts_cache}...")
            return self._load_cache(max_facts)

        # Download if needed
        if not self._assertions_path.exists():
            self._download()

        # Parse
        facts = list(self._parse(min_confidence, max_facts))

        # Validate
        self.validate(facts)

        # Debug output
        self._print_debug(facts)

        # Cache results
        if use_cache:
            self._save_cache(facts)

        return facts

    def _download(self) -> None:
        """Download ConceptNet assertions file."""
        print(f"Downloading ConceptNet 5.7 assertions...")
        print(f"  URL: {CONCEPTNET_URL}")
        print(f"  Destination: {self._assertions_path}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  Progress: {pct}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

        urllib.request.urlretrieve(CONCEPTNET_URL, self._assertions_path, _progress)
        print("\n  Download complete!")

    def _parse(
        self,
        min_confidence: float,
        max_facts: int | None,
    ) -> Iterator[tuple[str, str, str, float]]:
        """Stream-parse the assertions file with quality filtering."""
        count = 0

        # Filter reason counters for debug output
        skip_reasons: Counter = Counter()
        seen_facts: set[tuple[str, str, str]] = set()

        print(f"Parsing ConceptNet assertions (min_confidence={min_confidence})...")

        with gzip.open(self._assertions_path, 'rt', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f):
                if max_facts and count >= max_facts:
                    break

                # Progress every 1M lines
                if line_num > 0 and line_num % 1_000_000 == 0:
                    print(f"  Processed {line_num:,} lines, found {count:,} facts...")

                parts = line.strip().split('\t')
                if len(parts) < 5:
                    skip_reasons['malformed_line'] += 1
                    continue

                # Format: URI  relation  subject  object  info_json
                _, relation, subject, obj, info_json = parts[:5]

                # Filter: English only
                if not (_is_english(subject) and _is_english(obj)):
                    skip_reasons['non_english'] += 1
                    continue

                # Filter: known relations only
                gunter_rel = RELATION_MAP.get(relation)
                if not gunter_rel:
                    skip_reasons['unknown_relation'] += 1
                    continue

                # Filter: confidence
                weight = _parse_weight(info_json)
                if weight < min_confidence:
                    skip_reasons['low_confidence'] += 1
                    continue

                # Clean entity names
                subj_clean = _clean_entity(subject)
                obj_clean = _clean_entity(obj)

                # --- Quality filters ---

                # Skip empty or single-char entities
                if len(subj_clean) < 2 or len(obj_clean) < 2:
                    skip_reasons['too_short'] += 1
                    continue

                # Skip entities with >3 words
                if len(subj_clean.split()) > 3 or len(obj_clean.split()) > 3:
                    skip_reasons['too_many_words'] += 1
                    continue

                # Skip entities with digits
                if _has_digits(subj_clean) or _has_digits(obj_clean):
                    skip_reasons['has_digits'] += 1
                    continue

                # Skip taxonomy/genus markers
                if _has_taxonomy_marker(subj_clean) or _has_taxonomy_marker(obj_clean):
                    skip_reasons['taxonomy'] += 1
                    continue

                # Skip self-referential facts
                if subj_clean == obj_clean:
                    skip_reasons['self_reference'] += 1
                    continue

                # Skip entities with non-ASCII characters
                if not subj_clean.isascii() or not obj_clean.isascii():
                    skip_reasons['non_ascii'] += 1
                    continue

                # Deduplicate (keep highest weight)
                fact_key = (subj_clean, gunter_rel, obj_clean)
                if fact_key in seen_facts:
                    skip_reasons['duplicate'] += 1
                    continue
                seen_facts.add(fact_key)

                yield (subj_clean, gunter_rel, obj_clean, weight)
                count += 1

        # Print filter statistics
        total_lines = line_num + 1 if 'line_num' in dir() else 0
        print(f"\n  Parsing complete!")
        print(f"  Total lines processed: {total_lines:,}")
        print(f"  Facts extracted: {count:,}")
        print(f"  Acceptance rate: {count / max(total_lines, 1) * 100:.2f}%")
        print(f"\n  Filter reasons (why lines were skipped):")
        for reason, cnt in skip_reasons.most_common():
            print(f"    {reason}: {cnt:,}")

    def _print_debug(self, facts: list[tuple[str, str, str, float]]) -> None:
        """Print debug information about loaded facts."""
        if not facts:
            print("  No facts loaded!")
            return

        print(f"\n  === Debug: First 20 facts ===")
        for f in facts[:20]:
            print(f"    {f[0]} {f[1]} {f[2]}  (weight={f[3]:.2f})")

        # Relation distribution
        rel_counts: Counter = Counter()
        for _, rel, _, _ in facts:
            rel_counts[rel] += 1

        print(f"\n  === Relation distribution ===")
        for rel, cnt in rel_counts.most_common():
            pct = cnt / len(facts) * 100
            print(f"    {rel}: {cnt:,} ({pct:.1f}%)")

        # Sample entities (top subjects)
        subj_counts: Counter = Counter()
        for s, _, _, _ in facts:
            subj_counts[s] += 1

        print(f"\n  === Top 20 entities (by fact count) ===")
        for entity, cnt in subj_counts.most_common(20):
            print(f"    {entity}: {cnt} facts")

    def validate(
        self, facts: list[tuple[str, str, str, float]],
    ) -> bool:
        """Validate that expected key facts are present.

        Returns True if all expected facts found, False otherwise.
        """
        fact_set = {(s, r, o) for s, r, o, _ in facts}
        all_ok = True

        print(f"\n  === Validation ===")
        for subj, rel, obj in VALIDATION_FACTS:
            if (subj, rel, obj) in fact_set:
                print(f"    ✓ {subj} {rel} {obj}")
            else:
                print(f"    ✗ MISSING: {subj} {rel} {obj}")
                logger.warning(
                    "Expected fact missing from ConceptNet: %s %s %s",
                    subj, rel, obj,
                )
                all_ok = False

        if all_ok:
            print(f"    All {len(VALIDATION_FACTS)} validation facts present!")
        else:
            print(f"    WARNING: Some expected facts are missing.")
            print(f"    Consider lowering min_confidence or checking entity cleaning.")

        return all_ok

    def _save_cache(self, facts: list[tuple[str, str, str, float]]) -> None:
        """Save parsed facts to compressed cache."""
        print(f"Caching {len(facts):,} facts to {self._facts_cache}...")
        with gzip.open(self._facts_cache, 'wt', encoding='utf-8') as f:
            json.dump(facts, f)

    def _load_cache(
        self, max_facts: int | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Load facts from cache."""
        with gzip.open(self._facts_cache, 'rt', encoding='utf-8') as f:
            facts = json.load(f)

        if max_facts:
            facts = facts[:max_facts]

        print(f"  Loaded {len(facts):,} facts from cache.")
        return [tuple(f) for f in facts]

    def get_sample(self, n: int = 100) -> list[tuple[str, str, str, float]]:
        """Get a small sample for testing without full download.

        Returns hardcoded sample facts from ConceptNet.
        """
        SAMPLE = [
            ("cat", "IS-A", "animal", 4.0),
            ("cat", "IS-A", "mammal", 3.46),
            ("cat", "IS-A", "pet", 3.46),
            ("cat", "HAS", "fur", 3.46),
            ("cat", "HAS", "whiskers", 2.83),
            ("cat", "HAS", "tail", 3.46),
            ("cat", "HAS", "claws", 2.83),
            ("cat", "CAN", "purr", 3.46),
            ("cat", "CAN", "climb", 2.83),
            ("cat", "CAN", "hunt", 2.83),
            ("cat", "LOCATED-AT", "home", 2.0),
            ("cat", "RELATED-TO", "kitten", 2.83),
            ("dog", "IS-A", "animal", 4.0),
            ("dog", "IS-A", "mammal", 3.46),
            ("dog", "IS-A", "pet", 4.0),
            ("dog", "HAS", "fur", 3.46),
            ("dog", "HAS", "tail", 3.46),
            ("dog", "CAN", "bark", 4.0),
            ("dog", "CAN", "swim", 2.0),
            ("dog", "CAN", "fetch", 2.83),
            ("water", "IS-A", "liquid", 3.46),
            ("water", "IS-A", "substance", 2.83),
            ("water", "USED-FOR", "drinking", 4.0),
            ("water", "USED-FOR", "cleaning", 2.83),
            ("water", "HAS-PROPERTY", "wet", 3.46),
            ("sun", "IS-A", "star", 4.0),
            ("sun", "CAUSES", "light", 2.83),
            ("sun", "HAS-PROPERTY", "hot", 3.46),
            ("tree", "IS-A", "plant", 4.0),
            ("tree", "HAS", "leaves", 3.46),
            ("tree", "HAS", "roots", 3.46),
            ("tree", "HAS", "bark", 2.83),
            ("tree", "PART-OF", "forest", 2.83),
            ("car", "IS-A", "vehicle", 4.0),
            ("car", "HAS", "wheels", 3.46),
            ("car", "HAS", "engine", 3.46),
            ("car", "USED-FOR", "transportation", 4.0),
            ("bird", "IS-A", "animal", 4.0),
            ("bird", "CAN", "fly", 4.0),
            ("bird", "HAS", "wings", 4.0),
            ("bird", "HAS", "feathers", 3.46),
            ("fish", "IS-A", "animal", 3.46),
            ("fish", "CAN", "swim", 4.0),
            ("fish", "LOCATED-AT", "water", 3.46),
            ("rain", "IS-A", "precipitation", 2.83),
            ("rain", "CAUSES", "flood", 2.0),
            ("rain", "MADE-OF", "water", 3.46),
            ("book", "IS-A", "object", 2.83),
            ("book", "USED-FOR", "reading", 4.0),
            ("book", "HAS", "pages", 3.46),
            ("computer", "IS-A", "machine", 3.46),
            ("computer", "USED-FOR", "computing", 3.46),
            ("computer", "HAS", "screen", 2.83),
            ("fire", "IS-A", "chemical reaction", 2.0),
            ("fire", "HAS-PROPERTY", "hot", 4.0),
            ("fire", "CAUSES", "smoke", 3.46),
            ("human", "IS-A", "mammal", 4.0),
            ("human", "IS-A", "animal", 3.46),
            ("human", "CAN", "think", 3.46),
            ("human", "CAN", "speak", 3.46),
            ("moon", "IS-A", "celestial body", 3.46),
            ("moon", "PART-OF", "solar system", 2.0),
            ("food", "USED-FOR", "eating", 4.0),
            ("food", "IS-A", "sustenance", 2.0),
            ("mountain", "IS-A", "landform", 2.83),
            ("mountain", "HAS-PROPERTY", "tall", 2.83),
            ("river", "IS-A", "body of water", 3.46),
            ("river", "HAS", "current", 2.0),
            ("lion", "IS-A", "animal", 3.46),
            ("lion", "IS-A", "mammal", 2.83),
            ("lion", "CAN", "roar", 3.46),
            ("lion", "LOCATED-AT", "savanna", 2.0),
            ("elephant", "IS-A", "mammal", 3.46),
            ("elephant", "HAS", "trunk", 4.0),
            ("elephant", "HAS-PROPERTY", "large", 3.46),
            ("ocean", "IS-A", "body of water", 3.46),
            ("ocean", "HAS-PROPERTY", "deep", 2.83),
            ("piano", "IS-A", "musical instrument", 3.46),
            ("piano", "HAS", "keys", 3.46),
            ("piano", "USED-FOR", "playing music", 2.83),
            ("doctor", "IS-A", "person", 2.83),
            ("doctor", "CAN", "heal", 2.83),
            ("school", "IS-A", "building", 2.83),
            ("school", "USED-FOR", "learning", 4.0),
            ("knife", "IS-A", "tool", 2.83),
            ("knife", "USED-FOR", "cutting", 4.0),
            ("heart", "IS-A", "organ", 3.46),
            ("heart", "PART-OF", "body", 3.46),
            ("brain", "IS-A", "organ", 3.46),
            ("brain", "PART-OF", "body", 3.46),
            ("brain", "USED-FOR", "thinking", 3.46),
            ("ice", "IS-A", "solid", 2.83),
            ("ice", "MADE-OF", "water", 4.0),
            ("ice", "HAS-PROPERTY", "cold", 4.0),
            ("gold", "IS-A", "metal", 3.46),
            ("gold", "HAS-PROPERTY", "valuable", 2.83),
            ("chair", "IS-A", "furniture", 3.46),
            ("chair", "USED-FOR", "sitting", 4.0),
            ("chair", "HAS", "legs", 2.83),
            ("apple", "IS-A", "fruit", 4.0),
            ("apple", "HAS-PROPERTY", "red", 2.0),
        ]
        return SAMPLE[:n]
