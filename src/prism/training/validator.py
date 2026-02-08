"""Fact Validator — Validate extracted facts before storing.

Checks plausibility, consistency with existing knowledge,
and quality (skips pronouns, too-generic terms, invalid verbs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from prism.training.fact_extractor import ExtractedFact


@dataclass
class ValidationResult:
    """Result of validating a batch of facts."""
    
    valid: list[ExtractedFact] = field(default_factory=list)
    invalid: list[tuple[ExtractedFact, str]] = field(default_factory=list)
    conflicts: list[tuple[ExtractedFact, str]] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.valid) + len(self.invalid) + len(self.conflicts)
    
    def summary(self) -> str:
        lines = [
            f"Validated {self.total} facts:",
            f"  ✓ Valid: {len(self.valid)}",
            f"  ✗ Invalid: {len(self.invalid)}",
            f"  ⚠ Conflicts: {len(self.conflicts)}",
        ]
        return "\n".join(lines)


# Pronouns to skip as subjects/objects
_SKIP_PRONOUNS = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
    'you', 'your', 'yours',
    'he', 'him', 'his', 'she', 'her', 'hers',
    'it', 'its', 'they', 'them', 'their', 'theirs',
    'this', 'that', 'these', 'those',
    'who', 'what', 'which', 'whom', 'whose',
    'there', 'here',
    'someone', 'something', 'anyone', 'anything',
    'everyone', 'everything', 'nobody', 'nothing',
}

# Too-generic terms that don't carry useful information
_GENERIC_TERMS = {
    'thing', 'things', 'stuff', 'way', 'ways',
    'time', 'times', 'lot', 'lots',
    'kind', 'kinds', 'type', 'types',
    'part', 'parts', 'number', 'numbers',
    'people', 'person', 'man', 'woman',
    'place', 'places', 'area', 'areas',
    'example', 'examples', 'case', 'cases',
    'fact', 'facts', 'point', 'points',
    'result', 'results', 'form', 'forms',
}

# Valid relation types from fact extraction
_VALID_RELATIONS = {
    # Core relations
    'IS', 'IS-A', 'HAS', 'CAN', 'DOES', 'NEEDS',
    # Mapped verb relations
    'EATS', 'LIKES', 'MAKES', 'USES', 'WANTS',
    'LIVES-IN', 'BELONGS-TO',
    'CAUSES', 'CREATES', 'PROVIDES', 'SUPPORTS',
    'ALLOWS', 'PREVENTS', 'HELPS',
    'SERVES-AS', 'GROWS', 'BUILDS', 'FORMS',
    'COVERS', 'MEASURES', 'STORES', 'CONVERTS',
    'DETECTS', 'CARRIES', 'PUMPS', 'ENABLES',
    'CONNECTS', 'PROCESSES', 'FLOWS-THROUGH',
    'FEEDS',
    # Negation relations
    'IS-NOT', 'DOES-NOT', 'CAN-NOT',
    # Prepositional IS
    'IS-IN', 'IS-ON', 'IS-AT', 'IS-FOR', 'IS-BY',
    'IS-FROM', 'IS-WITH', 'IS-OF', 'IS-ABOUT',
    'IS-THROUGH', 'IS-BETWEEN', 'IS-AMONG', 'IS-AROUND',
    'IS-OVER', 'IS-UNDER', 'IS-AFTER', 'IS-BEFORE',
    'IS-DURING', 'IS-WITHIN', 'IS-ACROSS', 'IS-ALONG',
    'IS-AGAINST', 'IS-INTO', 'IS-UPON', 'IS-THROUGHOUT',
}

# Objects that indicate junk extraction
_JUNK_OBJECTS = {
    'bce', 'ce', 'ad', 'bc', 'etc', 'eg', 'ie',
    'also', 'however', 'therefore', 'moreover', 'furthermore',
    'nevertheless', 'thus', 'hence', 'meanwhile',
    'although', 'though', 'despite', 'whereas',
    'often', 'sometimes', 'usually', 'generally',
    'approximately', 'roughly', 'nearly',
}

# Junk subjects
_JUNK_SUBJECTS = {
    'not', 'no', 'nor', 'neither', 'either',
    'only', 'just', 'even', 'still',
    'much', 'more', 'less', 'most', 'least',
    'rather', 'quite', 'very', 'too',
    'also', 'however', 'therefore', 'thus',
}


class FactValidator:
    """Validate facts before storing in PRISM's memory.
    
    Checks:
    1. Quality: skip pronouns, too-generic terms, very short
    2. Relation validity: only allow known relation types
    3. Semantic validity: skip junk objects/subjects
    4. Plausibility: basic common-sense checks
    5. Consistency: check for contradictions with existing knowledge
    """
    
    def __init__(self, prism: Any = None) -> None:
        self.prism = prism
    
    def validate_facts(
        self, facts: list[ExtractedFact]
    ) -> ValidationResult:
        """Validate a batch of facts."""
        result = ValidationResult()
        
        for fact in facts:
            # Quality check
            quality_ok, quality_reason = self._is_quality(fact)
            if not quality_ok:
                result.invalid.append((fact, quality_reason))
                continue
            
            # Relation check
            relation_ok, relation_reason = self._is_valid_relation(fact)
            if not relation_ok:
                result.invalid.append((fact, relation_reason))
                continue
            
            # Semantic check
            semantic_ok, semantic_reason = self._is_semantic_valid(fact)
            if not semantic_ok:
                result.invalid.append((fact, semantic_reason))
                continue
            
            # Plausibility check
            plausible, plausible_reason = self._is_plausible(fact)
            if not plausible:
                result.invalid.append((fact, plausible_reason))
                continue
            
            # Consistency check
            if self.prism is not None:
                consistent, conflict_reason = self._check_consistency(fact)
                if not consistent:
                    result.conflicts.append((fact, conflict_reason))
                    result.valid.append(fact)
                    continue
            
            result.valid.append(fact)
        
        return result
    
    def _is_quality(self, fact: Any) -> tuple[bool, str]:
        """Check if a fact meets quality standards."""
        subj = fact.subject.lower().strip()
        obj = fact.object.lower().strip()
        
        # Skip pronoun subjects/objects
        if subj in _SKIP_PRONOUNS:
            return False, f"Pronoun subject: '{subj}'"
        if obj in _SKIP_PRONOUNS:
            return False, f"Pronoun object: '{obj}'"
        
        # Skip too-short terms
        if len(subj) < 2:
            return False, f"Subject too short: '{subj}'"
        if len(obj) < 2:
            return False, f"Object too short: '{obj}'"
        
        # Skip if subject == object
        if subj == obj:
            return False, f"Tautology: '{subj}' = '{obj}'"
        
        # Skip too-generic subjects
        if subj in _GENERIC_TERMS:
            return False, f"Generic subject: '{subj}'"
        
        # Skip pure numbers
        if subj.isdigit() or obj.isdigit():
            return False, "Numeric term"
        
        return True, ""
    
    def _is_valid_relation(self, fact: Any) -> tuple[bool, str]:
        """Check if the relation type is valid."""
        relation = fact.relation.upper()
        
        if relation not in _VALID_RELATIONS:
            return False, f"Invalid relation: '{relation}'"
        
        return True, ""
    
    def _is_semantic_valid(self, fact: Any) -> tuple[bool, str]:
        """Check semantic validity — skip junk extractions."""
        subj = fact.subject.lower().strip()
        obj = fact.object.lower().strip()
        
        # Skip junk objects (dates, filler words)
        if obj in _JUNK_OBJECTS:
            return False, f"Junk object: '{obj}'"
        
        # Skip junk subjects
        if subj in _JUNK_SUBJECTS:
            return False, f"Junk subject: '{subj}'"
        
        # Object shouldn't contain only numbers/special chars
        obj_alpha = ''.join(c for c in obj if c.isalpha())
        if len(obj_alpha) < 2:
            return False, f"Non-alpha object: '{obj}'"
        
        # Subject shouldn't contain only numbers/special chars
        subj_alpha = ''.join(c for c in subj if c.isalpha())
        if len(subj_alpha) < 2:
            return False, f"Non-alpha subject: '{subj}'"
        
        # Skip very long multi-word objects (likely parsing errors)
        if len(obj.split()) > 5:
            return False, f"Object too complex: '{obj}'"
        
        # Skip very long multi-word subjects
        if len(subj.split()) > 4:
            return False, f"Subject too complex: '{subj}'"
        
        return True, ""
    
    def _is_plausible(self, fact: Any) -> tuple[bool, str]:
        """Basic plausibility checks."""
        relation = fact.relation.upper()
        obj = fact.object.lower()
        
        # Skip nonsensical IS-A relations where object is too long
        if relation == 'IS-A' and len(obj) > 30:
            return False, f"IS-A object too long: '{obj}'"
        
        # Confidence threshold
        if hasattr(fact, 'confidence') and fact.confidence < 0.3:
            return False, f"Low confidence: {fact.confidence:.2f}"
        
        return True, ""
    
    def _check_consistency(self, fact: Any) -> tuple[bool, str]:
        """Check consistency with existing knowledge."""
        if self.prism is None:
            return True, ""
        
        try:
            conflicts = self.prism.contradiction.check(
                fact.subject, fact.relation, fact.object
            )
            if conflicts:
                reasons = [c.reason for c in conflicts]
                return False, f"Conflicts: {'; '.join(reasons)}"
        except Exception:
            pass
        
        return True, ""
