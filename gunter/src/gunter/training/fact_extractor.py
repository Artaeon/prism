"""Fact Extractor — Extract structured facts using spaCy dependency parsing.

Uses spaCy's dependency parser to extract Subject-Verb-Object triples,
IS-A relations, HAS relations, and property assignments from sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import spacy


@dataclass
class ExtractedFact:
    """A fact extracted from text."""
    
    subject: str
    relation: str  # IS-A, IS, HAS, DOES, CAN, NEEDS, etc.
    object: str
    source_sentence: str = ""
    confidence: float = 0.8


class FactExtractor:
    """Extract structured facts from sentences using spaCy.
    
    Uses dependency parsing to find:
    - SVO triples (subject-verb-object)
    - "X is a Y" (IS-A classification)
    - "X is Y" (IS property)
    - "X has Y" (HAS possession)
    - "X can Y" (CAN ability)
    - "X needs Y" (NEEDS requirement)
    
    Example:
        >>> extractor = FactExtractor()
        >>> facts = extractor.extract_facts(["Cats are mammals."])
        >>> facts[0]  # ExtractedFact(subject="cats", relation="IS-A", object="mammals")
    """
    
    # Map lemmatized verbs to relation types
    VERB_TO_RELATION = {
        'be': None,  # Handled specially (IS / IS-A)
        'have': 'HAS',
        'can': 'CAN',
        'need': 'NEEDS',
        'require': 'NEEDS',
        'eat': 'EATS',
        'like': 'LIKES',
        'love': 'LIKES',
        'make': 'MAKES',
        'produce': 'MAKES',
        'contain': 'HAS',
        'include': 'HAS',
        'use': 'USES',
        'want': 'WANTS',
        'live': 'LIVES-IN',
        'belong': 'BELONGS-TO',
        'cause': 'CAUSES',
        'create': 'CREATES',
        'provide': 'PROVIDES',
        'support': 'SUPPORTS',
        'allow': 'ALLOWS',
        'prevent': 'PREVENTS',
        'serve': 'SERVES-AS',
        'grow': 'GROWS',
        'run': 'DOES',
        'play': 'DOES',
        'help': 'HELPS',
        'build': 'BUILDS',
        'form': 'FORMS',
        'cover': 'COVERS',
        'measure': 'MEASURES',
        'store': 'STORES',
        'convert': 'CONVERTS',
        'detect': 'DETECTS',
        'carry': 'CARRIES',
        'pump': 'PUMPS',
        'enable': 'ENABLES',
        'connect': 'CONNECTS',
        'process': 'PROCESSES',
        'flow': 'FLOWS-THROUGH',
        'feed': 'FEEDS',
    }
    
    # Verbs that should NEVER produce a DOES fact on their own
    VERB_BLACKLIST = {
        'do', 'does', 'did', 'be', 'is', 'are', 'was', 'were',
        'been', 'being', 'am',
        'would', 'should', 'could', 'might', 'shall', 'will',
        'get', 'got', 'getting',
        'go', 'going', 'gone', 'went',
        'say', 'said', 'call', 'called',
        'come', 'came', 'take', 'took',
        'know', 'knew', 'think', 'thought',
        'see', 'saw', 'find', 'found',
        'give', 'gave', 'tell', 'told',
        'become', 'became',
    }
    
    # Subjects to skip (relative pronouns, etc.)
    SUBJECT_BLACKLIST = {
        'that', 'which', 'who', 'whom', 'whose', 'what',
        'this', 'these', 'those', 'it', 'there', 'here',
        'one', 'some', 'many', 'most', 'all', 'both',
        'other', 'another', 'such',
    }
    
    def __init__(self, nlp: Any = None) -> None:
        """Initialize with spaCy model.
        
        Args:
            nlp: spaCy Language model. If None, loads en_core_web_md.
        """
        if nlp is None:
            self.nlp = spacy.load("en_core_web_md")
        else:
            self.nlp = nlp
    
    def extract_facts(self, sentences: list[str]) -> list[ExtractedFact]:
        """Extract facts from a list of sentences.
        
        Uses spaCy pipe for efficient batch processing.
        """
        facts = []
        
        # Process in batches for speed
        for doc in self.nlp.pipe(sentences, batch_size=50, disable=["ner"]):
            source = doc.text
            
            for sent in doc.sents:
                sent_facts = self._extract_from_sentence(sent, source)
                facts.extend(sent_facts)
        
        return facts
    
    def _extract_from_sentence(
        self, sent: Any, source: str
    ) -> list[ExtractedFact]:
        """Extract facts from a single spaCy Span (sentence)."""
        facts = []
        
        # Find the root verb(s)
        for token in sent:
            if token.dep_ == "ROOT" or token.pos_ == "VERB" or token.pos_ == "AUX":
                if token.dep_ not in ("ROOT", "relcl", "advcl", "ccomp", "xcomp"):
                    if token.dep_ != "ROOT":
                        continue
                
                extracted = self._extract_from_verb(token, source)
                if extracted:
                    facts.extend(extracted)
        
        # Also try regex-based extraction as fallback
        if not facts:
            regex_facts = self._regex_fallback(sent.text, source)
            facts.extend(regex_facts)
        
        return facts
    
    def _extract_from_verb(
        self, verb: Any, source: str
    ) -> list[ExtractedFact]:
        """Extract facts from a verb and its dependents."""
        facts = []
        
        subject = self._find_subject(verb)
        if not subject:
            return facts
        
        subj_text = self._get_compound_text(subject)
        lemma = verb.lemma_.lower()
        
        # Skip blacklisted subjects (relative pronouns, etc.)
        if subj_text.lower() in self.SUBJECT_BLACKLIST:
            return facts
        
        # Handle "be" verb separately (IS / IS-A)
        if lemma == 'be' or verb.tag_ in ('VBZ', 'VBP', 'VBD') and lemma == 'be':
            be_facts = self._handle_be_verb(verb, subj_text, source)
            facts.extend(be_facts)
            return facts
        
        # Handle modal verbs (can, could, etc.)
        if verb.pos_ == 'AUX' and lemma in ('can', 'could', 'may', 'might'):
            # Find the main verb
            for child in verb.children:
                if child.dep_ in ('xcomp', 'ccomp', 'ROOT') or child.pos_ == 'VERB':
                    obj = self._find_object(child)
                    obj_text = self._get_compound_text(obj) if obj else child.lemma_
                    facts.append(ExtractedFact(
                        subject=subj_text.lower(),
                        relation='CAN',
                        object=obj_text.lower(),
                        source_sentence=source,
                    ))
                    return facts
        
        # Map verb to relation
        relation = self.VERB_TO_RELATION.get(lemma)
        
        # If verb not in our known relations and is blacklisted, skip
        if relation is None:
            if lemma in self.VERB_BLACKLIST:
                return facts
            relation = 'DOES'
        
        # Find object
        obj = self._find_object(verb)
        if obj:
            obj_text = self._get_compound_text(obj)
            # Skip if object is too short or just a stop word  
            if len(obj_text.strip()) < 2:
                return facts
            facts.append(ExtractedFact(
                subject=subj_text.lower(),
                relation=relation,
                object=obj_text.lower(),
                source_sentence=source,
            ))
        elif lemma not in self.VERB_BLACKLIST and len(lemma) > 2:
            # Intransitive: "dogs bark" → dogs DOES bark
            facts.append(ExtractedFact(
                subject=subj_text.lower(),
                relation='DOES',
                object=lemma,
                source_sentence=source,
                confidence=0.6,
            ))
        
        return facts
    
    def _handle_be_verb(
        self, verb: Any, subject: str, source: str
    ) -> list[ExtractedFact]:
        """Handle 'be' verb: IS / IS-A / IS property."""
        facts = []
        
        for child in verb.children:
            # "X is a Y" — IS-A (attribute with determiner)
            if child.dep_ == 'attr':
                # Check if there's a determiner (a/an) → IS-A
                has_det = any(
                    c.dep_ == 'det' and c.text.lower() in ('a', 'an')
                    for c in child.children
                )
                obj_text = self._get_compound_text(child)
                
                if has_det:
                    facts.append(ExtractedFact(
                        subject=subject.lower(),
                        relation='IS-A',
                        object=obj_text.lower(),
                        source_sentence=source,
                        confidence=0.9,
                    ))
                else:
                    facts.append(ExtractedFact(
                        subject=subject.lower(),
                        relation='IS',
                        object=obj_text.lower(),
                        source_sentence=source,
                        confidence=0.85,
                    ))
            
            # "X is Y" — IS (adjective complement)
            elif child.dep_ == 'acomp':
                facts.append(ExtractedFact(
                    subject=subject.lower(),
                    relation='IS',
                    object=child.text.lower(),
                    source_sentence=source,
                    confidence=0.85,
                ))
            
            # "X is in/on/at Y" — prepositional complement
            elif child.dep_ == 'prep':
                for pobj in child.children:
                    if pobj.dep_ == 'pobj':
                        obj_text = self._get_compound_text(pobj)
                        facts.append(ExtractedFact(
                            subject=subject.lower(),
                            relation=f'IS-{child.text.upper()}',
                            object=obj_text.lower(),
                            source_sentence=source,
                            confidence=0.7,
                        ))
        
        return facts
    
    def _find_subject(self, verb: Any) -> Any | None:
        """Find the subject of a verb using dependency relations."""
        for child in verb.children:
            if child.dep_ in ('nsubj', 'nsubjpass'):
                return child
        
        # Check parent for passive constructions
        if verb.dep_ in ('xcomp', 'ccomp') and verb.head:
            return self._find_subject(verb.head)
        
        return None
    
    def _find_object(self, verb: Any) -> Any | None:
        """Find the direct object of a verb."""
        for child in verb.children:
            if child.dep_ in ('dobj', 'attr', 'oprd', 'prt'):
                return child
        
        # Check for prepositional objects
        for child in verb.children:
            if child.dep_ == 'prep':
                for grandchild in child.children:
                    if grandchild.dep_ == 'pobj':
                        return grandchild
        
        return None
    
    def _find_complement(self, verb: Any) -> Any | None:
        """Find the complement of a verb."""
        for child in verb.children:
            if child.dep_ in ('acomp', 'attr', 'oprd'):
                return child
        return None
    
    def _get_compound_text(self, token: Any) -> str:
        """Get compound noun text (e.g., 'polar bear' from 'bear')."""
        compounds = []
        for child in token.children:
            if child.dep_ in ('compound', 'amod') and child.i < token.i:
                compounds.append(child.text)
        
        compounds.append(token.text)
        return ' '.join(compounds)
    
    def _regex_fallback(self, text: str, source: str) -> list[ExtractedFact]:
        """Regex fallback for sentences spaCy couldn't parse well."""
        facts = []
        text_lower = text.lower().strip()
        
        # "X is a Y"
        match = re.search(
            r'(?:a |an |the )?(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)',
            text_lower
        )
        if match:
            facts.append(ExtractedFact(
                subject=match.group(1).strip(),
                relation='IS-A',
                object=match.group(2).strip(),
                source_sentence=source,
                confidence=0.7,
            ))
        
        # "X has Y"
        match = re.search(
            r'(?:a |an |the )?(\w+(?:\s+\w+)?)\s+(?:has|have)\s+(?:a |an |the )?(\w+(?:\s+\w+)?)',
            text_lower
        )
        if match:
            facts.append(ExtractedFact(
                subject=match.group(1).strip(),
                relation='HAS',
                object=match.group(2).strip(),
                source_sentence=source,
                confidence=0.6,
            ))
        
        return facts
