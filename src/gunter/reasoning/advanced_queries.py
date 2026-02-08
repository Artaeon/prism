"""Advanced Query Patterns — Complex question handling.

Handles comparisons, negation, aggregation, listing,
and counting queries.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gunter.memory import VectorMemory


class AdvancedQueryHandler:
    """Handle complex query patterns.
    
    Patterns:
    - "list all X" / "show all X"
    - "how many X"
    - "compare X and Y"
    - "what is not X"
    - "everything about X"
    
    Example:
        >>> handler = AdvancedQueryHandler(memory)
        >>> handler.try_handle("list all animals")
        "Animals I know about: cat, dog, bird"
    """
    
    def __init__(self, memory: VectorMemory) -> None:
        """Initialize with memory."""
        self.memory = memory
    
    def try_handle(self, text: str) -> str | None:
        """Try to handle text as an advanced query.
        
        Returns response string if handled, None otherwise.
        """
        text_lower = text.lower().strip()
        
        # "list all X" / "show all X" / "what are all X"
        match = re.search(
            r"(?:list|show|what are) (?:all|the) (\w+)", text_lower
        )
        if match:
            return self._list_all(match.group(1))
        
        # "how many X" / "count X"
        match = re.search(r"(?:how many|count) (\w+)", text_lower)
        if match:
            return self._count(match.group(1))
        
        # "compare X and Y"
        match = re.search(r"compare (\w+) (?:and|with|to|vs) (\w+)", text_lower)
        if match:
            return self._compare(match.group(1), match.group(2))
        
        # "everything about X" / "all about X"
        match = re.search(r"(?:everything|all) about (\w+)", text_lower)
        if match:
            return self._everything_about(match.group(1))
        
        return None
    
    def _list_all(self, category: str) -> str:
        """List all facts matching a category."""
        episodes = self.memory.get_episodes()
        
        matching = []
        for ep in episodes:
            if not ep.subject:
                continue
            # Match if the object or relation contains the category
            if (category in ep.obj.lower() or 
                category in ep.relation.lower() or
                category in ep.text.lower()):
                matching.append(ep)
        
        if not matching:
            return f"I don't know any '{category}' facts yet."
        
        # Deduplicate subjects
        subjects = list(dict.fromkeys(ep.subject for ep in matching))
        
        if len(subjects) == 1:
            # Show all facts about that one subject
            lines = [f"Facts about {subjects[0]}:"]
            for ep in matching:
                lines.append(f"  • {ep.text}")
            return "\n".join(lines)
        else:
            lines = [f"Things related to '{category}' ({len(subjects)}):"]
            for s in subjects[:15]:
                lines.append(f"  • {s}")
            if len(subjects) > 15:
                lines.append(f"  ... and {len(subjects) - 15} more")
            return "\n".join(lines)
    
    def _count(self, concept: str) -> str:
        """Count facts matching a concept."""
        episodes = self.memory.get_episodes()
        
        count = sum(
            1 for ep in episodes
            if concept in ep.text.lower()
        )
        
        return f"I have {count} fact{'s' if count != 1 else ''} about '{concept}'."
    
    def _compare(self, concept1: str, concept2: str) -> str:
        """Compare two concepts side by side."""
        episodes = self.memory.get_episodes()
        
        facts1 = [ep for ep in episodes if concept1 in ep.subject.lower()]
        facts2 = [ep for ep in episodes if concept2 in ep.subject.lower()]
        
        if not facts1 and not facts2:
            return f"I don't know enough about either '{concept1}' or '{concept2}'."
        
        lines = [f"=== {concept1} vs {concept2} ==="]
        
        lines.append(f"\n{concept1.title()} ({len(facts1)} facts):")
        for ep in facts1[:5]:
            lines.append(f"  • {ep.text}")
        
        lines.append(f"\n{concept2.title()} ({len(facts2)} facts):")
        for ep in facts2[:5]:
            lines.append(f"  • {ep.text}")
        
        # Find shared relations
        rels1 = {ep.relation for ep in facts1}
        rels2 = {ep.relation for ep in facts2}
        shared = rels1 & rels2
        if shared:
            lines.append(f"\nShared relations: {', '.join(shared)}")
        
        return "\n".join(lines)
    
    def _everything_about(self, concept: str) -> str:
        """Show everything known about a concept."""
        episodes = self.memory.get_episodes()
        
        matching = [
            ep for ep in episodes
            if concept in ep.text.lower()
        ]
        
        if not matching:
            return f"I don't know anything about '{concept}'."
        
        lines = [f"Everything I know about '{concept}' ({len(matching)} facts):"]
        
        # Group by relation
        by_relation: dict[str, list[str]] = {}
        for ep in matching:
            rel = ep.relation or "MISC"
            by_relation.setdefault(rel, []).append(ep.text)
        
        for rel, facts in by_relation.items():
            lines.append(f"\n  {rel}:")
            for f in facts[:5]:
                lines.append(f"    • {f}")
        
        return "\n".join(lines)
