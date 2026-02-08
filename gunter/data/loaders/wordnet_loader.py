"""WordNet Loader — Extract relations from NLTK WordNet.

Extracts hypernym/hyponym/meronym/holonym/synonym relations from
all WordNet synsets and maps them to Gunter relation types.

Example:
    >>> loader = WordNetLoader()
    >>> facts = loader.load(max_facts=5000)
    >>> facts[0]  # ('cat', 'IS-A', 'feline', 0.9)
"""

from __future__ import annotations

from typing import Iterator


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
    
    Extracts hierarchical and associative relations from WordNet synsets:
    - Hypernyms → IS-A (cat IS-A feline)
    - Hyponyms → HAS-TYPE (feline HAS-TYPE cat)
    - Part meronyms → HAS-PART (car HAS-PART wheel)
    - Member holonyms → PART-OF (tree PART-OF forest)
    - Lemma synonyms → SIMILAR-TO
    - Definitions → IS "definition"
    
    Example:
        >>> loader = WordNetLoader()
        >>> facts = loader.load()
        >>> print(f"Loaded {len(facts)} relations")
    """
    
    def __init__(self) -> None:
        _ensure_wordnet()
        from nltk.corpus import wordnet
        self._wn = wordnet
    
    def load(
        self,
        max_facts: int | None = None,
        include_definitions: bool = True,
        pos_filter: list[str] | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Load WordNet relations.
        
        Args:
            max_facts: Maximum facts to extract (None = all)
            include_definitions: Include synset definitions as facts
            pos_filter: POS tags to include (None = all).
                        Options: 'n' (noun), 'v' (verb), 'a' (adj), 'r' (adv)
                        
        Returns:
            List of (subject, relation, object, confidence) tuples
        """
        facts = list(self._extract(max_facts, include_definitions, pos_filter))
        
        # Deduplicate
        seen = set()
        unique = []
        for f in facts:
            key = (f[0].lower(), f[1], f[2].lower())
            if key not in seen:
                seen.add(key)
                unique.append(f)
        
        print(f"  WordNet: {len(unique):,} unique facts extracted")
        return unique
    
    def _extract(
        self,
        max_facts: int | None,
        include_definitions: bool,
        pos_filter: list[str] | None,
    ) -> Iterator[tuple[str, str, str, float]]:
        """Extract relations from all synsets."""
        count = 0
        synsets = list(self._wn.all_synsets())
        total = len(synsets)
        
        print(f"Extracting from {total:,} WordNet synsets...")
        
        for i, synset in enumerate(synsets):
            if max_facts and count >= max_facts:
                break
            
            if i > 0 and i % 20000 == 0:
                print(f"  Processed {i:,}/{total:,} synsets ({count:,} facts)...")
            
            # POS filter
            if pos_filter and synset.pos() not in pos_filter:
                continue
            
            name = self._synset_name(synset)
            if not name or len(name) < 2:
                continue
            
            # Hypernyms → IS-A
            for hyper in synset.hypernyms():
                hyper_name = self._synset_name(hyper)
                if hyper_name:
                    yield (name, "IS-A", hyper_name, 0.9)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Hyponyms → HAS-TYPE
            for hypo in synset.hyponyms():
                hypo_name = self._synset_name(hypo)
                if hypo_name:
                    yield (hypo_name, "IS-A", name, 0.85)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Part meronyms → HAS-PART
            for mero in synset.part_meronyms():
                mero_name = self._synset_name(mero)
                if mero_name:
                    yield (name, "HAS-PART", mero_name, 0.85)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Member holonyms → PART-OF
            for holo in synset.member_holonyms():
                holo_name = self._synset_name(holo)
                if holo_name:
                    yield (name, "PART-OF", holo_name, 0.85)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Substance meronyms → MADE-OF
            for sub_mero in synset.substance_meronyms():
                sub_name = self._synset_name(sub_mero)
                if sub_name:
                    yield (name, "MADE-OF", sub_name, 0.8)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Synonyms (lemmas within same synset)
            lemmas = [l.name().replace('_', ' ') for l in synset.lemmas()]
            if len(lemmas) > 1:
                for j in range(1, len(lemmas)):
                    if len(lemmas[j]) >= 2 and lemmas[j] != name:
                        yield (name, "SIMILAR-TO", lemmas[j], 0.95)
                        count += 1
                        if max_facts and count >= max_facts:
                            return
            
            # Definitions as facts
            if include_definitions:
                defn = synset.definition()
                if defn and len(defn) > 5 and len(defn) < 100:
                    yield (name, "IS", defn, 0.7)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Entailments (for verbs)
            for ent in synset.entailments():
                ent_name = self._synset_name(ent)
                if ent_name:
                    yield (name, "REQUIRES", ent_name, 0.8)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
            
            # Verb groups (similar verbs)
            for vg in synset.verb_groups():
                vg_name = self._synset_name(vg)
                if vg_name and vg_name != name:
                    yield (name, "SIMILAR-TO", vg_name, 0.8)
                    count += 1
                    if max_facts and count >= max_facts:
                        return
        
        print(f"  Done! {count:,} facts from {total:,} synsets")
    
    @staticmethod
    def _synset_name(synset) -> str:
        """Get the primary lemma name from a synset."""
        name = synset.lemmas()[0].name()
        return name.replace('_', ' ')
    
    def get_sample(self, n: int = 100) -> list[tuple[str, str, str, float]]:
        """Get a small sample for testing."""
        return self.load(max_facts=n, include_definitions=False)
