"""Contradiction Detection — Detect conflicting facts.

When a new fact contradicts an existing one, detect it
using antonym wordlists and semantic opposition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gunter.memory import EpisodicMemory, VectorMemory
    from gunter.core.lexicon import Lexicon


# Common antonym pairs
ANTONYM_PAIRS: set[frozenset[str]] = {
    frozenset(pair) for pair in [
        ("big", "small"), ("large", "small"), ("big", "tiny"), ("large", "tiny"),
        ("hot", "cold"), ("warm", "cool"), ("warm", "cold"),
        ("fast", "slow"), ("quick", "slow"),
        ("tall", "short"), ("long", "short"),
        ("heavy", "light"), ("thick", "thin"),
        ("old", "new"), ("old", "young"), ("ancient", "modern"),
        ("good", "bad"), ("nice", "mean"), ("kind", "cruel"),
        ("happy", "sad"), ("glad", "upset"),
        ("friendly", "aggressive"), ("friendly", "hostile"),
        ("gentle", "rough"), ("calm", "angry"),
        ("smart", "dumb"), ("intelligent", "stupid"),
        ("rich", "poor"), ("expensive", "cheap"),
        ("clean", "dirty"), ("neat", "messy"),
        ("safe", "dangerous"), ("safe", "risky"),
        ("open", "closed"), ("open", "shut"),
        ("alive", "dead"), ("healthy", "sick"),
        ("strong", "weak"),
        ("like", "hate"), ("like", "dislike"), ("love", "hate"),
        ("true", "false"), ("right", "wrong"),
        ("possible", "impossible"),
    ]
}


@dataclass
class Contradiction:
    """A detected contradiction between two facts."""
    
    existing_fact: str
    new_fact: str
    existing_episode_id: int = 0
    reason: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class ContradictionDetector:
    """Detect contradicting facts.
    
    Checks for:
    1. Same subject+relation with antonymous objects
    2. Same subject with opposing predicates (like/hate)
    3. Semantic opposition (high negative similarity is rare in VSA,
       so we rely on antonym wordlists)
    
    Example:
        >>> detector = ContradictionDetector(memory, lexicon)
        >>> result = detector.check("cats", "IS", "friendly")
        >>> # If "cats IS aggressive" exists → Contradiction
    """
    
    def __init__(self, memory: VectorMemory, lexicon: Lexicon) -> None:
        """Initialize with memory and lexicon."""
        self.memory = memory
        self.lexicon = lexicon
    
    def check(
        self,
        subject: str,
        relation: str,
        obj: str,
    ) -> list[Contradiction]:
        """Check if a new fact contradicts existing knowledge.
        
        Args:
            subject: Subject of the new fact
            relation: Relation type
            obj: Object of the new fact
            
        Returns:
            List of all Contradictions found (empty if none)
        """
        subject_lower = subject.lower()
        obj_lower = obj.lower()
        contradictions = []
        
        for episode in self.memory.get_episodes():
            if not episode.subject:
                continue
            
            ep_subject = episode.subject.lower()
            ep_obj = episode.obj.lower()
            ep_relation = episode.relation
            
            # Same subject?
            if ep_subject != subject_lower:
                continue
            
            # Same or compatible relation?
            if not self._relations_compatible(relation, ep_relation):
                continue
            
            # Check antonym objects
            if self._are_antonyms(obj_lower, ep_obj):
                contradictions.append(Contradiction(
                    existing_fact=episode.text,
                    new_fact=f"{subject} {relation} {obj}",
                    existing_episode_id=episode.id,
                    reason=f"'{obj}' contradicts '{ep_obj}' (antonyms)",
                    confidence=0.9,
                ))
                continue
            
            # Check semantic opposition via lexicon
            if self._semantically_opposed(obj_lower, ep_obj):
                contradictions.append(Contradiction(
                    existing_fact=episode.text,
                    new_fact=f"{subject} {relation} {obj}",
                    existing_episode_id=episode.id,
                    reason=f"'{obj}' seems to oppose '{ep_obj}'",
                    confidence=0.7,
                ))
        
        return contradictions
    
    def _relations_compatible(self, r1: str, r2: str) -> bool:
        """Check if two relations can conflict."""
        r1, r2 = r1.upper(), r2.upper()
        
        if r1 == r2:
            return True
        
        # IS and IS-A are related
        compatible = {
            frozenset(("IS", "IS-A")),
            frozenset(("LIKES", "HATES")),
            frozenset(("LIKES", "DISLIKES")),
        }
        return frozenset((r1, r2)) in compatible
    
    def _are_antonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are known antonyms."""
        return frozenset((word1, word2)) in ANTONYM_PAIRS
    
    def _semantically_opposed(self, word1: str, word2: str) -> bool:
        """Check semantic opposition using vector similarity.
        
        In practice, truly opposed words can still have moderate
        similarity in embeddings. This is a supplementary check.
        """
        from gunter.core.vector_ops import VectorOps
        
        ops = VectorOps(self.lexicon.config)
        v1 = self.lexicon.get(word1)
        v2 = self.lexicon.get(word2)
        sim = ops.similarity(v1, v2)
        
        # Very low similarity + same semantic domain = possible opposition
        # (This catches things not in the antonym list)
        return sim < -0.1
    
    def get_all_conflicts(self) -> list[tuple[str, str]]:
        """Find all conflicting fact pairs in memory."""
        conflicts = []
        episodes = self.memory.get_episodes()
        
        for i, ep1 in enumerate(episodes):
            if not ep1.subject or not ep1.obj:
                continue
            for ep2 in episodes[i+1:]:
                if not ep2.subject or not ep2.obj:
                    continue
                
                if ep1.subject.lower() != ep2.subject.lower():
                    continue
                
                if not self._relations_compatible(ep1.relation, ep2.relation):
                    continue
                
                if self._are_antonyms(ep1.obj.lower(), ep2.obj.lower()):
                    conflicts.append((ep1.text, ep2.text))
        
        return conflicts
