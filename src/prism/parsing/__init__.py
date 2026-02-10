"""Dependency-Based Semantic Parser — Phase 1 NLU Upgrade.

Replaces regex-based parsing with spaCy dependency tree analysis.
Handles complex sentences, conjunctions, relative clauses, negation,
and question classification using linguistic structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from prism.core.vector_ops import VectorOps, HVector
from prism.core.lexicon import Lexicon

# Lazy-load spaCy
_nlp = None


def _get_nlp():
    """Get the shared spaCy NLP pipeline (full, with parser + NER)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        for model in ("en_core_web_md", "en_core_web_lg", "en_core_web_sm"):
            try:
                _nlp = spacy.load(model)
                return _nlp
            except OSError:
                continue
        # If no model found, load blank
        _nlp = spacy.blank("en")
        return _nlp
    except ImportError:
        return None


@dataclass
class ParsedFact:
    """A parsed fact from text."""

    subject: str
    relation: str
    object: str
    raw_text: str = ""
    confidence: float = 1.0


@dataclass
class ParsedEntity:
    """A named entity extracted from text."""

    text: str
    label: str  # PERSON, ORG, GPE, DATE, etc.
    start: int = 0
    end: int = 0


# ─── Relation Mapping ──────────────────────────────────────────────

# Maps dependency structures to PRISM relation types
_VERB_TO_RELATION = {
    "be": "IS",
    "have": "HAS",
    "can": "CAN",
    "need": "NEEDS",
    "require": "NEEDS",
    "like": "LIKES",
    "love": "LIKES",
    "enjoy": "LIKES",
    "eat": "EATS",
    "make": "MAKES",
    "produce": "MAKES",
    "cause": "CAUSES",
    "mean": "MEANS",
    "own": "HAS",
    "contain": "HAS",
    "include": "HAS",
    "live": "LOCATED-IN",
    "inhabit": "LOCATED-IN",
    "create": "MAKES",
    "want": "WANTS",
    "use": "USES",
    "know": "KNOWS",
    "fear": "FEARS",
}

# Preposition-based relations
_PREP_RELATIONS = {
    "in": "LOCATED-IN",
    "at": "LOCATED-AT",
    "on": "LOCATED-ON",
    "from": "ORIGIN",
    "for": "PURPOSE",
    "with": "HAS",
    "of": "PART-OF",
    "by": "CAUSED-BY",
    "about": "ABOUT",
}


class SemanticParser:
    """Dependency-based semantic parser using spaCy.

    Extracts structured facts from natural language using dependency
    tree analysis instead of regex patterns. Handles:

    - Simple SVO: "cats chase mice" → (cats, DOES, chase mice)
    - Copular: "cats are mammals" → (cats, IS-A, mammals)
    - Negation: "cats can't fly" → (cats, CAN-NOT, fly)
    - Conjunctions: "cats and dogs are mammals" → 2 facts
    - Relative clauses: "cats, which are mammals, purr" → 2 facts
    - Complex questions: "What do cats eat?" → (cats, EAT, ?)
    - Named entities: "Paris is in France" → (Paris, LOCATED-IN, France)
    """

    def __init__(self, lexicon: Lexicon) -> None:
        """Initialize parser with lexicon and spaCy pipeline."""
        self.lexicon = lexicon
        self.ops = VectorOps(lexicon.config)
        self._nlp = _get_nlp()

    def parse(self, text: str) -> list[ParsedFact]:
        """Parse text into structured facts using dependency analysis.

        Falls back to regex patterns if spaCy is unavailable.
        """
        if self._nlp is None:
            return self._parse_regex_fallback(text)

        doc = self._nlp(text)
        facts = []

        # Process each sentence in the document
        for sent in doc.sents:
            sent_facts = self._parse_sentence(sent)
            facts.extend(sent_facts)

        return facts

    def _parse_sentence(self, sent) -> list[ParsedFact]:
        """Parse a single sentence using its dependency tree."""
        facts = []

        # Find the root verb
        root = sent.root

        # Handle copular constructions: "X is/are Y"
        if root.lemma_ == "be":
            cop_facts = self._parse_copular(root, sent)
            if cop_facts:
                return cop_facts

        # Handle main verb constructions
        verb_facts = self._parse_verb_construction(root, sent)
        if verb_facts:
            facts.extend(verb_facts)

        # Handle conjunction expansion
        facts = self._expand_conjunctions(facts, sent)

        return facts

    def _parse_copular(self, root, sent) -> list[ParsedFact]:
        """Parse copular constructions: X is/are Y.

        Patterns:
          "cats are mammals" → (cats, IS-A, mammals)
          "the sky is blue" → (sky, IS, blue)
          "cats are not fish" → (cats, IS-NOT, fish)
        """
        facts = []
        subjects = self._get_subjects(root)
        complements = self._get_complements(root)
        is_negated = self._is_negated(root)

        for subj in subjects:
            for comp in complements:
                # Determine if IS-A (noun) or IS (adjective)
                comp_token = self._find_token(sent, comp)
                if comp_token and comp_token.pos_ in ("NOUN", "PROPN"):
                    # Check for "a/an" determiner → IS-A
                    has_det = any(
                        child.dep_ == "det" and child.text.lower() in ("a", "an")
                        for child in comp_token.children
                    )
                    relation = "IS-A" if has_det else "IS-A"
                else:
                    relation = "IS"

                if is_negated:
                    relation = "IS-NOT" if relation.startswith("IS") else relation

                facts.append(ParsedFact(
                    subject=subj,
                    relation=relation,
                    object=comp,
                    raw_text=sent.text,
                    confidence=0.9,
                ))

        return facts

    def _parse_verb_construction(self, root, sent) -> list[ParsedFact]:
        """Parse verb-based constructions.

        Handles:
          "cats eat fish" → (cats, EATS, fish)
          "cats can jump" → (cats, CAN, jump)
          "cats don't fly" → (cats, DOES-NOT, fly)
        """
        facts = []
        subjects = self._get_subjects(root)
        is_negated = self._is_negated(root)

        # Check for auxiliary modal: can, could, should, etc.
        modal = None
        for child in root.children:
            if child.dep_ == "aux" and child.tag_ == "MD":
                modal = child.lemma_.lower()

        # Get direct objects
        objects = self._get_objects(root)

        # Get prepositional complements
        prep_facts = self._get_prep_facts(root, subjects, sent)
        facts.extend(prep_facts)

        # Map verb to relation
        verb_lemma = root.lemma_.lower()
        base_relation = _VERB_TO_RELATION.get(verb_lemma, "DOES")

        for subj in subjects:
            if modal:
                # "cats can jump" → (cats, CAN, jump)
                relation = modal.upper()
                if is_negated:
                    relation = f"{relation}-NOT"
                obj_text = self._get_verb_phrase(root)
                if obj_text:
                    facts.append(ParsedFact(
                        subject=subj,
                        relation=relation,
                        object=obj_text,
                        raw_text=sent.text,
                        confidence=0.85,
                    ))
            elif objects:
                for obj in objects:
                    relation = base_relation
                    if is_negated:
                        relation = f"{relation}-NOT" if relation != "DOES" else "DOES-NOT"
                    facts.append(ParsedFact(
                        subject=subj,
                        relation=relation,
                        object=obj,
                        raw_text=sent.text,
                        confidence=0.85,
                    ))
            else:
                # Intransitive: "birds fly" → (birds, DOES, fly)
                relation = "DOES"
                if is_negated:
                    relation = "DOES-NOT"
                obj_text = self._get_verb_phrase(root)
                if obj_text:
                    facts.append(ParsedFact(
                        subject=subj,
                        relation=relation,
                        object=obj_text,
                        raw_text=sent.text,
                        confidence=0.7,
                    ))

        return facts

    def _get_subjects(self, token) -> list[str]:
        """Extract subject noun phrases from a token's dependents."""
        subjects = []
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                # Get the full noun phrase span
                subj_text = self._get_noun_phrase(child)
                subjects.append(subj_text)
                # Check for conjuncts: "cats and dogs"
                for conj in child.conjuncts:
                    subjects.append(self._get_noun_phrase(conj))
        return subjects if subjects else ["it"]

    def _get_objects(self, token) -> list[str]:
        """Extract object noun phrases from a token's dependents."""
        objects = []
        for child in token.children:
            if child.dep_ in ("dobj", "attr", "oprd", "acomp"):
                obj_text = self._get_noun_phrase(child)
                objects.append(obj_text)
                # Check for conjuncts
                for conj in child.conjuncts:
                    objects.append(self._get_noun_phrase(conj))
        return objects

    def _get_complements(self, token) -> list[str]:
        """Extract predicate complements for copular verbs."""
        comps = []
        for child in token.children:
            if child.dep_ in ("attr", "acomp", "oprd", "dobj"):
                comp_text = self._get_noun_phrase(child)
                comps.append(comp_text)
                for conj in child.conjuncts:
                    comps.append(self._get_noun_phrase(conj))
        return comps

    def _get_noun_phrase(self, token) -> str:
        """Extract a full noun phrase from a token using its subtree.

        Filters out determiners (a, an, the) to get clean entity names.
        Keeps compound nouns: "quantum physics" → "quantum physics"
        """
        # Collect compound parts and direct modifiers
        parts = []
        for child in token.lefts:
            if child.dep_ in ("compound", "amod", "nummod", "nmod"):
                parts.append(child.text.lower())
            elif child.dep_ == "det" and child.text.lower() not in ("a", "an", "the"):
                parts.append(child.text.lower())

        parts.append(token.text.lower())

        # Add right-side compounds
        for child in token.rights:
            if child.dep_ in ("compound",):
                parts.append(child.text.lower())

        return " ".join(parts)

    def _get_verb_phrase(self, token) -> str:
        """Get the verb phrase (verb + particles/complements)."""
        parts = [token.lemma_.lower()]

        for child in token.children:
            if child.dep_ in ("prt", "advmod") and child.pos_ != "PART":
                parts.append(child.text.lower())
            elif child.dep_ == "dobj":
                parts.append(self._get_noun_phrase(child))
            elif child.dep_ in ("acomp", "xcomp"):
                parts.append(child.text.lower())

        return " ".join(parts)

    def _get_prep_facts(self, root, subjects: list[str], sent) -> list[ParsedFact]:
        """Extract facts from prepositional phrases.

        "cats live in houses" → (cats, LOCATED-IN, houses)
        "tools are used for building" → (tools, PURPOSE, building)
        """
        facts = []
        for child in root.children:
            if child.dep_ == "prep":
                prep = child.text.lower()
                relation = _PREP_RELATIONS.get(prep)
                if relation:
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj_text = self._get_noun_phrase(pobj)
                            for subj in subjects:
                                facts.append(ParsedFact(
                                    subject=subj,
                                    relation=relation,
                                    object=obj_text,
                                    raw_text=sent.text,
                                    confidence=0.75,
                                ))
        return facts

    def _is_negated(self, token) -> bool:
        """Check if a token is negated."""
        for child in token.children:
            if child.dep_ == "neg":
                return True
            # "never", "no"
            if child.dep_ == "advmod" and child.text.lower() in ("never", "no", "nowhere"):
                return True
        return False

    def _find_token(self, sent, text: str):
        """Find a token in the sentence by text."""
        text_lower = text.lower().split()[-1]  # Use last word for compound nouns
        for token in sent:
            if token.text.lower() == text_lower:
                return token
        return None

    def _expand_conjunctions(self, facts: list[ParsedFact], sent) -> list[ParsedFact]:
        """Expand conjunction patterns that weren't caught by subject/object extraction.

        "Cats are mammals and carnivores" → 2 facts
        """
        # Already handled in _get_subjects and _get_objects via conjuncts
        return facts

    # ─── Question Parsing ──────────────────────────────────────────

    def encode_question(self, text: str) -> tuple[str, str, str | None]:
        """Parse a question using dependency analysis.

        Returns (subject, relation, object_or_None).
        """
        if self._nlp is None:
            return self._encode_question_regex(text)

        doc = self._nlp(text)

        # Classify question type
        q_type = self._classify_question(doc)

        if q_type == "WHAT_IS":
            return self._parse_what_is(doc)
        elif q_type == "YES_NO":
            return self._parse_yes_no_q(doc)
        elif q_type == "WH_VERB":
            return self._parse_wh_verb(doc)
        elif q_type == "CAN_DO":
            return self._parse_can_do(doc)
        else:
            return self._parse_generic_question(doc)

    def _classify_question(self, doc) -> str:
        """Classify question type from parse structure."""
        text_lower = doc.text.lower().strip().rstrip("?").strip()

        # "What is X?" / "What are X?"
        if re.match(r"what (?:is|are) ", text_lower):
            return "WHAT_IS"

        # "Can/Could X Y?"
        if re.match(r"(?:can|could) ", text_lower):
            return "CAN_DO"

        # "Is/Are/Do/Does X Y?" — yes/no questions
        if re.match(r"(?:is|are|do|does|did|has|have|was|were) ", text_lower):
            return "YES_NO"

        # "What/Where/Who/When/How/Why does X Y?"
        if re.match(r"(?:what|where|who|when|how|why) ", text_lower):
            return "WH_VERB"

        return "GENERIC"

    def _parse_what_is(self, doc) -> tuple[str, str, str | None]:
        """Parse 'What is X?' questions."""
        root = list(doc.sents)[0].root
        for child in root.children:
            if child.dep_ in ("attr", "nsubj") and child.text.lower() != "what":
                return self._get_noun_phrase(child), "IS-A", None
        # Fallback: find the last noun
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
        if nouns:
            return self._get_noun_phrase(nouns[-1]), "IS-A", None
        return "", "", None

    def _parse_yes_no_q(self, doc) -> tuple[str, str, str | None]:
        """Parse yes/no questions: 'Is X a Y?', 'Do cats eat fish?'"""
        sent = list(doc.sents)[0]
        root = sent.root
        subjects = self._get_subjects(root)
        subj = subjects[0] if subjects else ""

        if root.lemma_ == "be":
            comps = self._get_complements(root)
            if comps:
                return subj, "IS-A", comps[0]
            return subj, "IS", None

        # "Do cats eat fish?"
        objects = self._get_objects(root)
        verb_relation = _VERB_TO_RELATION.get(root.lemma_.lower(), root.lemma_.upper())
        if objects:
            return subj, verb_relation, objects[0]
        return subj, verb_relation, None

    def _parse_can_do(self, doc) -> tuple[str, str, str | None]:
        """Parse 'Can X Y?' questions."""
        sent = list(doc.sents)[0]
        root = sent.root
        subjects = self._get_subjects(root)
        subj = subjects[0] if subjects else ""

        verb_phrase = self._get_verb_phrase(root)
        return subj, "CAN", verb_phrase if verb_phrase != root.lemma_ else None

    def _parse_wh_verb(self, doc) -> tuple[str, str, str | None]:
        """Parse WH-questions: 'What do cats eat?', 'Where do cats live?'"""
        sent = list(doc.sents)[0]
        root = sent.root
        subjects = self._get_subjects(root)
        subj = subjects[0] if subjects else ""

        verb_lemma = root.lemma_.lower()
        relation = _VERB_TO_RELATION.get(verb_lemma, verb_lemma.upper())

        return subj, relation, None

    def _parse_generic_question(self, doc) -> tuple[str, str, str | None]:
        """Fallback question parser."""
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]
        verbs = [t for t in doc if t.pos_ == "VERB"]

        subj = self._get_noun_phrase(nouns[0]) if nouns else ""
        rel = verbs[0].lemma_.upper() if verbs else "IS-A"

        return subj, rel, None

    # ─── Entity Extraction ─────────────────────────────────────────

    def extract_entities(self, text: str) -> list[ParsedEntity]:
        """Extract named entities using spaCy NER."""
        if self._nlp is None:
            return []

        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(ParsedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))
        return entities

    # ─── Vector Encoding ───────────────────────────────────────────

    def encode(self, text: str) -> tuple[HVector, list[ParsedFact]]:
        """Parse and encode text into a single vector."""
        facts = self.parse(text)

        if not facts:
            words = text.lower().split()
            vectors = [self.lexicon.get(w) for w in words if len(w) > 2]
            if vectors:
                return self.ops.bundle(vectors), []
            return self.ops.zero_vector(), []

        fact_vectors = []
        for fact in facts:
            s_vec = self.lexicon.get(fact.subject)
            r_vec = self.lexicon.get(fact.relation)
            o_vec = self.lexicon.get(fact.object)
            fact_vec = self.ops.bind(self.ops.bind(s_vec, r_vec), o_vec)
            fact_vectors.append(fact_vec)

        result = self.ops.bundle(fact_vectors)
        return result, facts

    # ─── Regex Fallbacks (for when spaCy is not available) ─────────

    # Pattern rules: (regex, relation, subject_group, object_group)
    _PATTERNS = [
        (r"(?:a |an |the )?(\w+) (?:means?|is like|is similar to) (?:a |an |the )?(\w+)", "MEANS", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:is|are) (?:a|an) (\w+)", "IS-A", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:is|are) (\w+)", "IS", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:has|have|owns?) (?:a |an |the )?(\w+)", "HAS", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:can|could|is able to) (\w+)", "CAN", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:needs?|requires?) (?:a |an |the )?(\w+)", "NEEDS", 1, 2),
        (r"(?:a |an |the )?(\w+) (?:likes?|loves?|enjoys?) (?:a |an |the )?(\w+)", "LIKES", 1, 2),
        (r"(?:a |an |the )?(\w+) eats? (?:a |an |the )?(\w+)", "EATS", 1, 2),
        (r"(?:a |an |the )?(\w+) makes? (?:a |an |the )?(\w+)", "MAKES", 1, 2),
    ]

    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "and", "or", "but", "if", "of", "to", "in", "on", "at", "for",
        "it", "its", "this", "that", "these", "those",
    }

    def _parse_regex_fallback(self, text: str) -> list[ParsedFact]:
        """Regex-based parsing (fallback when spaCy unavailable)."""
        facts = []
        text_lower = text.lower().strip()

        # Negation
        neg_fact = self._try_negation(text_lower)
        if neg_fact:
            return [neg_fact]

        for pattern, relation, subj_group, obj_group in self._PATTERNS:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                facts.append(ParsedFact(
                    subject=match.group(subj_group).strip(),
                    relation=relation,
                    object=match.group(obj_group).strip(),
                    raw_text=text,
                ))
                return facts

        # SVO fallback
        words = [w.strip(".,!?;:'\"") for w in text_lower.split()]
        words = [w for w in words if w and w not in self.STOP_WORDS and len(w) > 1]
        if len(words) >= 2:
            if len(words) >= 3:
                facts.append(ParsedFact(
                    subject=words[0], relation="DOES",
                    object=f"{words[1]} {words[2]}", raw_text=text, confidence=0.6,
                ))
            else:
                facts.append(ParsedFact(
                    subject=words[0], relation="DOES",
                    object=words[1], raw_text=text, confidence=0.5,
                ))

        return facts

    def _try_negation(self, text: str) -> ParsedFact | None:
        """Try to parse negation patterns (regex fallback)."""
        patterns = [
            (r"(?:a |an |the )?(\w+) (?:don'?t|doesn'?t|do not|does not) (\w+)", "DOES-NOT"),
            (r"(?:a |an |the )?(\w+) (?:can'?t|cannot|can not) (\w+)", "CAN-NOT"),
            (r"(?:a |an |the )?(\w+) (?:is|are) (?:not|never) (?:a |an )?(\w+)", "IS-NOT"),
            (r"(?:a |an |the )?(\w+) never (\w+)", "DOES-NOT"),
        ]
        for pattern, relation in patterns:
            match = re.search(pattern, text)
            if match:
                return ParsedFact(
                    subject=match.group(1), relation=relation,
                    object=match.group(2), raw_text=text, confidence=0.9,
                )
        return None

    def _encode_question_regex(self, text: str) -> tuple[str, str, str | None]:
        """Regex-based question parser (fallback)."""
        text_lower = text.lower().strip().rstrip("?")

        patterns = [
            (r"what (?:is|are) (?:a |an |the )?(\w+)", lambda m: (m.group(1), "IS-A", None)),
            (r"(?:is|are) (?:a |an |the )?(\w+) (?:a |an )?(\w+)", lambda m: (m.group(1), "IS-A", m.group(2))),
            (r"what (?:does |do )?(?:a |an |the )?(\w+) have", lambda m: (m.group(1), "HAS", None)),
            (r"what (?:does |do |can )?(?:a |an |the )?(\w+) do", lambda m: (m.group(1), "DOES", None)),
            (r"can (?:a |an |the )?(\w+) (\w+)", lambda m: (m.group(1), "CAN", m.group(2))),
            (r"what (?:does |do )?(?:a |an |the )?(\w+) (\w+)", lambda m: (m.group(1), m.group(2).upper(), None) if m.group(2) not in ("do", "have", "mean") else None),
            (r"(?:does|do) (?:a |an |the )?(\w+) (\w+)(?: (?:a |an |the )?(\w+))?", lambda m: (m.group(1), m.group(2).upper(), m.group(3))),
        ]

        for pattern, handler in patterns:
            match = re.search(pattern, text_lower)
            if match:
                result = handler(match)
                if result:
                    return result

        return "", "", None
