"""Blackboard — Working memory for the Knowledge Swarm.

The Blackboard is a shared workspace where agents post their findings
and the reasoner reads them to synthesize an answer. One Blackboard
is created per user query and discarded after the response.

Example:
    >>> bb = Blackboard(query="What is photosynthesis?", query_type="definition")
    >>> bb.add_finding("wikipedia", Finding(text="...", confidence=0.9))
    >>> bb.add_finding("wordnet", Finding(text="...", confidence=0.8))
    >>> bb.best_findings()  # sorted by confidence
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Type of user query — determines which agents to dispatch."""
    DEFINITION = "definition"       # "what is X?"
    COMPARISON = "comparison"       # "X vs Y"
    FACTUAL = "factual"             # "who/when/where?"
    HOW_TO = "how_to"               # "how does X work?"
    YES_NO = "yes_no"               # "is X a Y?"
    LIST = "list"                   # "list all X"
    PERSONAL = "personal"           # "what's my favorite..."
    GENERAL = "general"             # catch-all


@dataclass
class Finding:
    """A single finding from an agent."""
    text: str
    confidence: float = 0.5
    source_url: str = ""
    source_name: str = ""
    facts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Blackboard:
    """Shared workspace for a single query.
    
    Agents write findings here. The reasoner reads them to
    compose a response. Created fresh for each user query.
    """
    
    # The original query
    query: str = ""
    query_type: QueryType = QueryType.GENERAL
    entities: list[str] = field(default_factory=list)
    
    # Agent findings: agent_name → list of findings
    findings: dict[str, list[Finding]] = field(default_factory=dict)
    
    # Agents that were dispatched
    agents_dispatched: list[str] = field(default_factory=list)
    agents_completed: list[str] = field(default_factory=list)
    agents_failed: list[str] = field(default_factory=list)
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    
    def add_finding(self, agent_name: str, finding: Finding) -> None:
        """Add a finding from an agent."""
        if agent_name not in self.findings:
            self.findings[agent_name] = []
        self.findings[agent_name].append(finding)
    
    def mark_completed(self, agent_name: str) -> None:
        """Mark an agent as completed."""
        if agent_name not in self.agents_completed:
            self.agents_completed.append(agent_name)
    
    def mark_failed(self, agent_name: str) -> None:
        """Mark an agent as failed."""
        if agent_name not in self.agents_failed:
            self.agents_failed.append(agent_name)
    
    def best_findings(self, top_k: int = 5) -> list[tuple[str, Finding]]:
        """Get best findings across all agents, sorted by confidence.
        
        Returns:
            List of (agent_name, finding) tuples
        """
        all_findings: list[tuple[str, Finding]] = []
        for agent_name, agent_findings in self.findings.items():
            for finding in agent_findings:
                all_findings.append((agent_name, finding))
        
        all_findings.sort(key=lambda x: x[1].confidence, reverse=True)
        return all_findings[:top_k]
    
    def get_all_facts(self) -> list[str]:
        """Get all facts from all agents."""
        facts: list[str] = []
        for agent_findings in self.findings.values():
            for finding in agent_findings:
                facts.extend(finding.facts)
        return facts
    
    def get_all_sources(self) -> list[str]:
        """Get all source URLs from findings."""
        sources: list[str] = []
        for agent_findings in self.findings.values():
            for finding in agent_findings:
                if finding.source_url:
                    sources.append(finding.source_url)
        return list(set(sources))
    
    @property
    def has_findings(self) -> bool:
        """Whether any agent returned findings."""
        return any(bool(findings) for findings in self.findings.values())
    
    @property 
    def total_findings(self) -> int:
        """Total number of findings across all agents."""
        return sum(len(f) for f in self.findings.values())
    
    def summary(self) -> str:
        """Get a summary of the blackboard state."""
        elapsed = ""
        if self.end_time:
            dt = (self.end_time - self.start_time).total_seconds()
            elapsed = f" in {dt:.1f}s"
        
        lines = [f"Blackboard: \"{self.query}\" ({self.query_type.value}){elapsed}"]
        lines.append(f"  Agents: {len(self.agents_completed)}/{len(self.agents_dispatched)} completed")
        
        if self.agents_failed:
            lines.append(f"  Failed: {', '.join(self.agents_failed)}")
        
        for agent_name, agent_findings in self.findings.items():
            count = len(agent_findings)
            best = max((f.confidence for f in agent_findings), default=0)
            lines.append(f"  [{agent_name}] {count} finding(s), best confidence: {best:.2f}")
        
        return "\n".join(lines)
