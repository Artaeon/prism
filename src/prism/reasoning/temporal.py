"""Temporal Reasoning — Time-aware fact management.

Adds recency weighting, temporal expression parsing,
and event ordering to queries.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import EpisodicMemory, VectorMemory


# Decay factor per hour (facts get slightly less relevant over time)
DECAY_PER_HOUR = 0.995


@dataclass
class TemporalQuery:
    """Result of parsing a temporal expression."""
    
    subject: str
    time_reference: str  # "yesterday", "last week", etc.
    start_time: datetime | None = None
    end_time: datetime | None = None


class TemporalReasoner:
    """Time-aware reasoning over facts.
    
    Features:
    - Recency weighting: recent facts score higher
    - Temporal expression parsing
    - Event ordering
    - "when" queries
    
    Example:
        >>> temporal = TemporalReasoner(memory)
        >>> temporal.when_learned("cats")
        "You learned about cats 2 hours ago"
    """
    
    def __init__(self, memory: VectorMemory) -> None:
        """Initialize with memory."""
        self.memory = memory
    
    def recency_weight(self, episode: EpisodicMemory) -> float:
        """Calculate recency weight for an episode.
        
        More recent episodes get higher weight.
        Decay: score *= 0.995 ^ hours_ago
        """
        age_seconds = (datetime.now() - episode.timestamp).total_seconds()
        age_hours = max(age_seconds / 3600, 0)
        return DECAY_PER_HOUR ** age_hours
    
    def get_latest_fact(
        self,
        subject: str,
        relation: str | None = None,
    ) -> EpisodicMemory | None:
        """Get the most recent fact about a subject.
        
        When contradicting facts exist, the most recent one wins.
        """
        subject_lower = subject.lower()
        matches = []
        
        for ep in self.memory.get_episodes():
            if not ep.subject:
                continue
            if ep.subject.lower() != subject_lower:
                continue
            if relation and ep.relation.upper() != relation.upper():
                continue
            matches.append(ep)
        
        if not matches:
            return None
        
        # Sort by timestamp, most recent first
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        return matches[0]
    
    def weighted_facts(
        self,
        subject: str,
        top_k: int = 5,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Get facts about a subject, weighted by recency.
        
        Returns:
            List of (episode, recency_weight) sorted by weight
        """
        subject_lower = subject.lower()
        results = []
        
        for ep in self.memory.get_episodes():
            if not ep.subject:
                continue
            if subject_lower in ep.subject.lower() or subject_lower in ep.text.lower():
                weight = self.recency_weight(ep)
                results.append((ep, weight))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def when_learned(self, concept: str) -> str:
        """Answer 'when did I learn about X?'"""
        concept_lower = concept.lower()
        matches = []
        
        for ep in self.memory.get_episodes():
            if concept_lower in ep.text.lower():
                matches.append(ep)
        
        if not matches:
            return f"I don't have any records about '{concept}'."
        
        # Most recent match
        matches.sort(key=lambda e: e.timestamp, reverse=True)
        latest = matches[0]
        
        age = datetime.now() - latest.timestamp
        time_str = self._format_age(age)
        
        lines = [f"I learned about '{concept}' {time_str}."]
        if len(matches) > 1:
            lines.append(f"({len(matches)} total facts about '{concept}')")
        
        return "\n".join(lines)
    
    def event_order(
        self,
        concepts: list[str],
    ) -> list[tuple[str, datetime]]:
        """Order concepts by when they were first learned.
        
        Returns:
            List of (concept, first_timestamp) ordered chronologically
        """
        first_seen: dict[str, datetime] = {}
        
        for concept in concepts:
            concept_lower = concept.lower()
            for ep in self.memory.get_episodes():
                if concept_lower in ep.text.lower():
                    if concept not in first_seen or ep.timestamp < first_seen[concept]:
                        first_seen[concept] = ep.timestamp
        
        ordered = [(c, t) for c, t in first_seen.items()]
        ordered.sort(key=lambda x: x[1])
        return ordered
    
    def parse_temporal(self, text: str) -> TemporalQuery | None:
        """Parse temporal expressions from text.
        
        Handles: "yesterday", "last week", "today", "recently",
                 "before X", "after X"
        """
        text_lower = text.lower().strip()
        now = datetime.now()
        
        # "when did I learn about X"
        match = re.search(r"when (?:did (?:i|you) )?(?:learn|hear|know) (?:about )?([\w\s]+)", text_lower)
        if match:
            return TemporalQuery(
                subject=match.group(1).strip(),
                time_reference="query",
            )
        
        # "what did I learn yesterday/today/recently"
        time_words = {
            "today": (now.replace(hour=0, minute=0, second=0), now),
            "yesterday": (
                (now - timedelta(days=1)).replace(hour=0, minute=0, second=0),
                now.replace(hour=0, minute=0, second=0),
            ),
            "recently": (now - timedelta(hours=24), now),
            "last week": (now - timedelta(weeks=1), now),
            "last hour": (now - timedelta(hours=1), now),
        }
        
        for keyword, (start, end) in time_words.items():
            if keyword in text_lower:
                subject = text_lower.replace(keyword, "").strip()
                subject = re.sub(r"^(what|when|which|facts?|things?|stuff)\s+", "", subject)
                subject = re.sub(r"\s*(did|have|has)\s+(i|you)\s+(learn|know|hear)\s*", "", subject)
                subject = subject.strip(" ?.")
                
                return TemporalQuery(
                    subject=subject or "everything",
                    time_reference=keyword,
                    start_time=start,
                    end_time=end,
                )
        
        return None
    
    def facts_in_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[EpisodicMemory]:
        """Get facts learned within a time range."""
        results = []
        
        for ep in self.memory.get_episodes():
            if start and ep.timestamp < start:
                continue
            if end and ep.timestamp > end:
                continue
            results.append(ep)
        
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results
    
    def _format_age(self, age: timedelta) -> str:
        """Format a timedelta as human-readable string."""
        seconds = age.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins} minute{'s' if mins != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
    
    # ── Phase 16: Advanced temporal reasoning ──
    
    # Temporal relation keywords
    TEMPORAL_RELATIONS = {
        'AT-TIME', 'DURING', 'BEFORE', 'AFTER', 'SINCE', 'UNTIL',
        'HUNT-AT', 'ACTIVE-DURING', 'SLEEP-AT', 'OCCURS-AT',
        'LASTS', 'DURATION', 'TAKES-TIME',
    }
    
    DURATION_RELATIONS = {'LASTS', 'DURATION', 'TAKES-TIME', 'TAKES'}
    
    def query_temporal(
        self,
        entity: str,
        temporal_type: str = "WHEN",
    ) -> list[tuple[str, float]]:
        """Query temporal information about an entity.
        
        Args:
            entity: The entity to query
            temporal_type: WHEN, DURATION, BEFORE, AFTER, DURING
            
        Returns:
            List of (temporal_value, confidence) tuples
        """
        entity_lower = entity.lower()
        results: list[tuple[str, float]] = []
        
        for ep in self.memory.get_episodes():
            if not ep.subject:
                continue
            if entity_lower not in ep.subject.lower():
                continue
            
            rel_upper = ep.relation.upper()
            
            if temporal_type == "WHEN":
                # Use word-boundary-aware checks to avoid 'AT' matching 'CAT' etc.
                has_temporal_kw = (
                    '-AT' in rel_upper or 'AT-' in rel_upper or
                    'DURING' in rel_upper or 'WHEN' in rel_upper or
                    'TIME' in rel_upper
                )
                if has_temporal_kw:
                    results.append((ep.obj, 0.85))
                elif rel_upper in self.TEMPORAL_RELATIONS:
                    results.append((ep.obj, 0.8))
            elif temporal_type == "DURATION":
                if any(kw in rel_upper for kw in ['LASTS', 'DURATION', 'TAKES']):
                    results.append((ep.obj, 0.9))
            elif temporal_type == "BEFORE":
                if 'BEFORE' in rel_upper:
                    results.append((ep.obj, 0.85))
            elif temporal_type == "AFTER":
                if 'AFTER' in rel_upper:
                    results.append((ep.obj, 0.85))
            elif temporal_type == "DURING":
                if 'DURING' in rel_upper:
                    results.append((ep.obj, 0.85))
        
        # Also check text content for temporal keywords
        if not results:
            temporal_keywords = {
                'WHEN': ['at', 'during', 'when', 'while', 'time'],
                'DURATION': ['hours', 'minutes', 'days', 'long', 'duration'],
                'BEFORE': ['before', 'prior', 'earlier'],
                'AFTER': ['after', 'following', 'later'],
                'DURING': ['during', 'while', 'throughout'],
            }
            keywords = temporal_keywords.get(temporal_type, [])
            
            for ep in self.memory.get_episodes():
                if entity_lower not in ep.text.lower():
                    continue
                text_lower = ep.text.lower()
                for kw in keywords:
                    if kw in text_lower:
                        results.append((ep.obj or ep.text, 0.5))
                        break
        
        # Deduplicate
        seen: set[str] = set()
        unique = []
        for val, conf in results:
            if val.lower() not in seen:
                seen.add(val.lower())
                unique.append((val, conf))
        
        return unique
    
    def compare_temporal(
        self,
        event1: str,
        event2: str,
    ) -> str:
        """Determine temporal ordering between two events.
        
        Returns: "BEFORE", "AFTER", "SIMULTANEOUS", or "UNKNOWN"
        """
        # Check learning order
        order = self.event_order([event1, event2])
        
        if len(order) >= 2:
            if order[0][0] == event1:
                return "BEFORE"
            elif order[0][0] == event2:
                return "AFTER"
        
        # Check explicit temporal relations
        e1_lower = event1.lower()
        e2_lower = event2.lower()
        
        for ep in self.memory.get_episodes():
            if not ep.subject or not ep.obj:
                continue
            
            subj = ep.subject.lower()
            obj = ep.obj.lower()
            rel = ep.relation.upper()
            
            if subj == e1_lower and obj == e2_lower:
                if 'BEFORE' in rel:
                    return "BEFORE"
                elif 'AFTER' in rel:
                    return "AFTER"
                elif 'DURING' in rel or 'SIMULTANEOUS' in rel:
                    return "SIMULTANEOUS"
            elif subj == e2_lower and obj == e1_lower:
                if 'BEFORE' in rel:
                    return "AFTER"
                elif 'AFTER' in rel:
                    return "BEFORE"
                elif 'DURING' in rel or 'SIMULTANEOUS' in rel:
                    return "SIMULTANEOUS"
        
        return "UNKNOWN"
    
    def get_duration(
        self,
        entity: str,
        activity: str | None = None,
    ) -> str | None:
        """Find how long an entity does an activity.
        
        Searches for DURATION, LASTS, TAKES-TIME relations.
        
        Args:
            entity: The entity
            activity: Optional activity to check duration for
            
        Returns:
            Duration string or None
        """
        entity_lower = entity.lower()
        activity_lower = activity.lower() if activity else None
        
        for ep in self.memory.get_episodes():
            if not ep.subject:
                continue
            
            subj = ep.subject.lower()
            if entity_lower not in subj:
                continue
            
            rel_upper = ep.relation.upper()
            
            # Check if this is a duration fact
            if rel_upper in self.DURATION_RELATIONS:
                if activity_lower:
                    if activity_lower in ep.text.lower():
                        return ep.obj
                else:
                    return ep.obj
            
            # Check text for duration patterns
            text = ep.text.lower()
            if activity_lower and activity_lower not in text:
                continue
            
            import re
            # Match patterns like "16 hours", "30 minutes", "2 days"
            m = re.search(r'(\d+\s*(?:hours?|minutes?|days?|seconds?|years?))', text)
            if m:
                return m.group(1)
        
        return None

