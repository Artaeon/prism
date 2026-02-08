"""WordNet Loader — Extract relations from NLTK WordNet.

Extracts hypernym/hyponym/meronym/holonym/synonym relations from
the most common WordNet synsets, filtered by lemma frequency.

Defaults to 10k synsets (nouns + verbs), producing ~25k high-quality facts.

Example:
    >>> loader = WordNetLoader()
    >>> facts = loader.load(max_synsets=5000)
    >>> len(facts)  # ~12000
    >>> facts[0]  # ('cat', 'IS-A', 'feline', 0.9)
"""

from __future__ import annotations

from typing import Iterator


# Per-synset relation limits to prevent fact explosion
MAX_HYPERNYMS = 5
MAX_HYPONYMS = 3
MAX_MERONYMS = 3
MAX_HOLONYMS = 3
MAX_SUBSTANCE_MERONYMS = 2
MAX_SYNONYMS = 4
MAX_ENTAILMENTS = 3
MAX_VERB_GROUPS = 2

# Minimum lemma frequency count to include a synset
MIN_LEMMA_FREQ = 5


def _ensure_wordnet():
    """Download WordNet data if not available."""
    import nltk
    try:
        from nltk.corpus import wordnet
        # Test access
        wordnet.synsets('cat')
    except LookupError:
        print("Downloading NLTK WordNet data...")
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


class WordNetLoader:
    """Load knowledge from NLTK WordNet.
    
    Extracts hierarchical and associative relations from WordNet synsets,
    filtered by lemma frequency to focus on common/important words.
    
    Relations extracted:
    - Hypernyms → IS-A (cat IS-A feline)
    - Hyponyms → IS-A (reverse: feline has hyponym cat)
    - Part meronyms → HAS-PART (car HAS-PART wheel)
    - Member holonyms → PART-OF (tree PART-OF forest)
    - Substance meronyms → MADE-OF
    - Lemma synonyms → SIMILAR-TO
    - Definitions → IS "definition"
    - Entailments → REQUIRES (verbs)
    
    Example:
        >>> loader = WordNetLoader()
        >>> facts = loader.load(max_synsets=10000)
        >>> print(f"Loaded {len(facts)} relations")
    """
    
    def __init__(self) -> None:
        _ensure_wordnet()
        from nltk.corpus import wordnet
        self._wn = wordnet
    
    def load(
        self,
        max_facts: int | None = None,
        max_synsets: int = 10_000,
        min_freq: int = MIN_LEMMA_FREQ,
        include_definitions: bool = True,
        pos_filter: list[str] | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Load WordNet relations.
        
        Args:
            max_facts: Maximum facts to extract (None = unlimited)
            max_synsets: Maximum synsets to process (default 10000).
                         Synsets are sorted by lemma frequency so the 
                         most important ones are processed first.
            min_freq: Minimum lemma frequency count to include a synset.
                      Higher = fewer but more common words. Default 5.
            include_definitions: Include synset definitions as facts
            pos_filter: POS tags to include (default: nouns + verbs).
                        Options: 'n' (noun), 'v' (verb), 'a' (adj), 'r' (adv)
                        
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        # Default to nouns + verbs (skip adjectives/adverbs)
        if pos_filter is None:
            pos_filter = ['n', 'v']
        
        facts = list(self._extract(
            max_facts, max_synsets, min_freq,
            include_definitions, pos_filter,
        ))
        
        # Deduplicate
        seen: set[tuple[str, str, str]] = set()
        unique: list[tuple[str, str, str, float]] = []
        for f in facts:
            key = (f[0].lower(), f[1], f[2].lower())
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        print(f"  WordNet: {len(unique):,} unique facts extracted")
        return unique
    
    def _get_sorted_synsets(
        self,
        pos_filter: list[str],
        min_freq: int,
        max_synsets: int,
    ) -> list:
        """Get synsets sorted by lemma frequency (most common first).
        
        Returns at most max_synsets, filtered by POS and frequency.
        """
        candidates: list[tuple[int, object]] = []
        
        for synset in self._wn.all_synsets():
            # POS filter
            if synset.pos() not in pos_filter:
                continue
            
            # Frequency filter: use the count of the primary lemma
            primary_lemma = synset.lemmas()[0]
            freq = primary_lemma.count()
            
            if freq < min_freq:
                continue
            
            candidates.append((freq, synset))
        
        # Sort by frequency descending (most common first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take top N
        result = [synset for _, synset in candidates[:max_synsets]]
        return result
    
    def _extract(
        self,
        max_facts: int | None,
        max_synsets: int,
        min_freq: int,
        include_definitions: bool,
        pos_filter: list[str],
    ) -> Iterator[tuple[str, str, str, float]]:
        """Extract relations from filtered, sorted synsets."""
        count = 0
        skipped_errors = 0
        
        # Get sorted synsets
        synsets = self._get_sorted_synsets(pos_filter, min_freq, max_synsets)
        total = len(synsets)
        
        print(f"Extracting from {total:,} WordNet synsets "
              f"(filtered from 117k, min_freq={min_freq})...")
        
        for i, synset in enumerate(synsets):
            if max_facts and count >= max_facts:
                break
            
            if i > 0 and i % 5000 == 0:
                print(f"  Processed {i:,}/{total:,} synsets ({count:,} facts)...")
            
            try:
                name = self._synset_name(synset)
                if not name or len(name) < 2:
                    continue
                
                # --- Hypernyms → IS-A (capped) ---
                for hyper in synset.hypernyms()[:MAX_HYPERNYMS]:
                    hyper_name = self._synset_name(hyper)
                    if hyper_name:
                        yield (name, "IS-A", hyper_name, 0.9)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Hyponyms → IS-A reverse (capped) ---
                for hypo in synset.hyponyms()[:MAX_HYPONYMS]:
                    hypo_name = self._synset_name(hypo)
                    if hypo_name:
                        yield (hypo_name, "IS-A", name, 0.85)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Part meronyms → HAS-PART (capped) ---
                for mero in synset.part_meronyms()[:MAX_MERONYMS]:
                    mero_name = self._synset_name(mero)
                    if mero_name:
                        yield (name, "HAS-PART", mero_name, 0.85)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Member holonyms → PART-OF (capped) ---
                for holo in synset.member_holonyms()[:MAX_HOLONYMS]:
                    holo_name = self._synset_name(holo)
                    if holo_name:
                        yield (name, "PART-OF", holo_name, 0.85)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Substance meronyms → MADE-OF (capped) ---
                for sub_mero in synset.substance_meronyms()[:MAX_SUBSTANCE_MERONYMS]:
                    sub_name = self._synset_name(sub_mero)
                    if sub_name:
                        yield (name, "MADE-OF", sub_name, 0.8)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Synonyms (lemmas within same synset, capped) ---
                lemmas = [l.name().replace('_', ' ') for l in synset.lemmas()]
                if len(lemmas) > 1:
                    for j in range(1, min(len(lemmas), MAX_SYNONYMS + 1)):
                        if len(lemmas[j]) >= 2 and lemmas[j] != name:
                            yield (name, "SIMILAR-TO", lemmas[j], 0.95)
                            count += 1
                            if max_facts and count >= max_facts:
                                return
                
                # --- Definitions as facts ---
                if include_definitions:
                    defn = synset.definition()
                    if defn and 5 < len(defn) < 100:
                        yield (name, "IS", defn, 0.7)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Entailments (verbs, capped) ---
                for ent in synset.entailments()[:MAX_ENTAILMENTS]:
                    ent_name = self._synset_name(ent)
                    if ent_name:
                        yield (name, "REQUIRES", ent_name, 0.8)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
                
                # --- Verb groups (capped) ---
                for vg in synset.verb_groups()[:MAX_VERB_GROUPS]:
                    vg_name = self._synset_name(vg)
                    if vg_name and vg_name != name:
                        yield (name, "SIMILAR-TO", vg_name, 0.8)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
            
            except Exception as e:
                skipped_errors += 1
                if skipped_errors <= 5:
                    print(f"  ⚠ Skipped synset {i}: {e}")
                continue
        
        print(f"  Done! {count:,} facts from {total:,} synsets"
              f"{f' ({skipped_errors} errors skipped)' if skipped_errors else ''}")
    
    @staticmethod
    def _synset_name(synset) -> str:
        """Get the primary lemma name from a synset."""
        name = synset.lemmas()[0].name()
        return name.replace('_', ' ')
    
    def get_sample(self, n: int = 100) -> list[tuple[str, str, str, float]]:
        """Get a small sample for testing."""
        return self.load(max_facts=n, max_synsets=500, include_definitions=False)
