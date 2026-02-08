"""Memory Optimizer — Limits, cleanup, and export.

Prevents unbounded memory growth and provides
housekeeping tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prism.memory import VectorMemory


MAX_EPISODES = 10_000
CLEANUP_THRESHOLD = 0.8  # Start cleanup at 80% capacity


@dataclass
class MemoryStats:
    """Current memory usage statistics."""
    
    total_episodes: int
    max_episodes: int
    usage_percent: float
    oldest_timestamp: datetime | None
    newest_timestamp: datetime | None


class MemoryOptimizer:
    """Manage memory limits and cleanup.
    
    Features:
    - Enforce hard limit on stored facts
    - Cleanup old/low-importance facts
    - Export facts to JSON
    - Memory usage stats
    
    Example:
        >>> optimizer = MemoryOptimizer(memory)
        >>> optimizer.get_usage()
        MemoryStats(total=150, max=10000, usage=1.5%)
    """
    
    def __init__(self, memory: VectorMemory, max_episodes: int = MAX_EPISODES) -> None:
        """Initialize with memory and limits."""
        self.memory = memory
        self.max_episodes = max_episodes
    
    def get_usage(self) -> MemoryStats:
        """Get current memory usage."""
        episodes = self.memory.get_episodes()
        total = len(episodes)
        
        oldest = min((e.timestamp for e in episodes), default=None)
        newest = max((e.timestamp for e in episodes), default=None)
        
        return MemoryStats(
            total_episodes=total,
            max_episodes=self.max_episodes,
            usage_percent=(total / self.max_episodes * 100) if self.max_episodes else 0,
            oldest_timestamp=oldest,
            newest_timestamp=newest,
        )
    
    def needs_cleanup(self) -> bool:
        """Check if we should run cleanup."""
        usage = self.get_usage()
        return usage.usage_percent >= CLEANUP_THRESHOLD * 100
    
    def cleanup(self, keep_ratio: float = 0.7) -> int:
        """Remove lowest-importance episodes to free space.
        
        Keeps the top `keep_ratio` proportion of episodes sorted by
        a combined score of importance and recency.
        
        Returns:
            Number of episodes removed
        """
        episodes = self.memory.get_episodes()
        if not episodes:
            return 0
        
        now = datetime.now()
        keep_count = int(len(episodes) * keep_ratio)
        
        # Score each episode: importance * recency
        scored = []
        for ep in episodes:
            age_hours = max((now - ep.timestamp).total_seconds() / 3600, 0.01)
            recency = 0.995 ** age_hours
            importance = ep.importance
            score = importance * 0.6 + recency * 0.4
            scored.append((ep, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top episodes, remove the rest
        to_remove = [ep for ep, _ in scored[keep_count:]]
        removed = 0
        
        for ep in to_remove:
            if hasattr(self.memory, 'remove_episode'):
                self.memory.remove_episode(ep.id)
                removed += 1
        
        return removed
    
    def export_json(self, path: str | None = None) -> str:
        """Export all facts as JSON.
        
        Args:
            path: File path to export to. If None, returns JSON string.
            
        Returns:
            JSON string or path written to
        """
        episodes = self.memory.get_episodes()
        
        data = {
            "exported_at": datetime.now().isoformat(),
            "total_facts": len(episodes),
            "facts": [],
        }
        
        for ep in episodes:
            data["facts"].append({
                "id": ep.id,
                "text": ep.text,
                "subject": ep.subject,
                "relation": ep.relation,
                "object": ep.obj,
                "importance": ep.importance,
                "timestamp": ep.timestamp.isoformat(),
            })
        
        json_str = json.dumps(data, indent=2)
        
        if path:
            with open(path, "w") as f:
                f.write(json_str)
            return path
        
        return json_str
    
    def format_usage(self) -> str:
        """Format usage stats for display."""
        stats = self.get_usage()
        
        lines = [
            "=== Memory Usage ===",
            f"Facts: {stats.total_episodes} / {stats.max_episodes}",
            f"Usage: {stats.usage_percent:.1f}%",
        ]
        
        if stats.oldest_timestamp:
            lines.append(f"Oldest: {stats.oldest_timestamp.strftime('%Y-%m-%d %H:%M')}")
        if stats.newest_timestamp:
            lines.append(f"Newest: {stats.newest_timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        if self.needs_cleanup():
            lines.append("⚠ Cleanup recommended! Use 'cleanup' command.")
        
        return "\n".join(lines)
