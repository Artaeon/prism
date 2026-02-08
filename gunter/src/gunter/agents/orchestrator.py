"""Swarm Orchestrator — Dispatches knowledge agents in parallel.

The orchestrator creates a Blackboard for each query, dispatches
relevant agents via ThreadPoolExecutor, and collects their findings.

Example:
    >>> swarm = SwarmOrchestrator(memory=my_memory)
    >>> blackboard = swarm.query("What is photosynthesis?", entities=["photosynthesis"])
    >>> print(blackboard.summary())
    Blackboard: "What is photosynthesis?" (definition) in 0.4s
      Agents: 4/4 completed
      [memory] 2 finding(s), best confidence: 0.90
      [wikipedia] 1 finding(s), best confidence: 0.85
      [wordnet] 1 finding(s), best confidence: 0.80
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from typing import Any, TYPE_CHECKING

from gunter.agents.blackboard import Blackboard, Finding, QueryType
from gunter.agents.wikipedia_agent import WikipediaAgent
from gunter.agents.websearch_agent import WebSearchAgent
from gunter.agents.file_agent import FileAgent
from gunter.agents.wordnet_agent import WordNetAgent
from gunter.agents.wikidata_agent import WikidataAgent
from gunter.agents.task_planner import TaskPlanner, QueryPlan
from gunter.agents.response_composer import CortexComposer

if TYPE_CHECKING:
    from gunter.memory import VectorMemory

logger = logging.getLogger(__name__)

# Default timeout for all agents (seconds)
DEFAULT_TIMEOUT = 4


class SwarmOrchestrator:
    """Orchestrates parallel knowledge agent queries.
    
    Manages a swarm of agents that each search different knowledge
    sources simultaneously using ThreadPoolExecutor. Results are
    collected into a shared Blackboard.
    
    Features:
    - Parallel agent dispatch via ThreadPoolExecutor
    - Configurable timeout (drops slow agents)
    - Selects agents based on query type
    - Memory agent always runs (instant, local)
    - Network agents optional (can be disabled for offline mode)
    """

    def __init__(
        self,
        memory: VectorMemory | None = None,
        search_dirs: list[str] | None = None,
        offline: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = 4,
        weaver: Any = None,
    ) -> None:
        """Initialize the swarm.
        
        Args:
            memory: VectorMemory instance for local memory search
            search_dirs: Directories for the file agent
            offline: If True, disable network agents
            timeout: Max time to wait for agents (seconds)
            max_workers: Max parallel threads
            weaver: Optional SemanticWeaver for natural text generation
        """
        self.memory = memory
        self.timeout = timeout
        self.max_workers = max_workers
        self.offline = offline
        
        # Initialize agents
        self.wiki_agent = WikipediaAgent()
        self.websearch_agent = WebSearchAgent()
        self.file_agent = FileAgent(search_dirs=search_dirs)
        self.wordnet_agent = WordNetAgent()
        self.wikidata_agent = WikidataAgent()
        
        # Task planning + response composition
        self.planner = TaskPlanner()
        self.composer = CortexComposer(weaver=weaver)
    
    def query_smart(self, question: str) -> tuple[Blackboard, QueryPlan]:
        """Smart query — auto-classifies intent and decomposes.
        
        Uses TaskPlanner to classify the question, extract entities,
        and decompose into sub-queries before dispatching agents.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (populated Blackboard, QueryPlan)
        """
        plan = self.planner.plan(question)
        
        # Use sub-queries as entities if they provide more coverage
        entities = plan.entities
        if plan.sub_queries:
            # Merge sub-queries into entity list (deduplicated)
            seen = set(e.lower() for e in entities)
            for sq in plan.sub_queries:
                if sq.lower() not in seen:
                    entities.append(sq)
                    seen.add(sq.lower())
        
        bb = self.query(question, entities=entities, query_type=plan.query_type)
        return bb, plan
    
    def format_smart_response(self, bb: Blackboard, plan: QueryPlan) -> str:
        """Format response using ResponseComposer and the plan.
        
        Args:
            bb: Populated blackboard
            plan: The query plan
            
        Returns:
            Formatted response string
        """
        if not bb.has_findings:
            return ""
        return self.composer.compose(bb, plan)
    
    def query(
        self,
        question: str,
        entities: list[str] | None = None,
        query_type: QueryType = QueryType.GENERAL,
    ) -> Blackboard:
        """Dispatch agents and collect findings.
        
        Args:
            question: The user's question
            entities: Key entities extracted from the question
            query_type: Type of query (affects agent selection)
            
        Returns:
            Populated Blackboard with findings from all agents
        """
        entities = entities or []
        
        # Create blackboard
        bb = Blackboard(
            query=question,
            query_type=query_type,
            entities=entities,
        )
        
        # Determine which agents to dispatch
        agent_tasks = self._select_agents(query_type, entities)
        bb.agents_dispatched = [name for name, _ in agent_tasks]
        
        if not agent_tasks:
            return bb
        
        # Dispatch all agents in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures: dict[Future, str] = {}
            
            for agent_name, agent_fn in agent_tasks:
                future = executor.submit(agent_fn, bb)
                futures[future] = agent_name
            
            # Collect results with timeout
            for future in as_completed(futures, timeout=self.timeout):
                agent_name = futures[future]
                try:
                    future.result()  # Agent writes directly to blackboard
                    bb.mark_completed(agent_name)
                except Exception as e:
                    logger.debug(f"Agent {agent_name} failed: {e}")
                    bb.mark_failed(agent_name)
        
        bb.end_time = datetime.now()
        return bb
    
    def _select_agents(
        self,
        query_type: QueryType,
        entities: list[str],
    ) -> list[tuple[str, callable]]:
        """Select which agents to dispatch based on query type.
        
        Returns:
            List of (agent_name, callable) tuples
        """
        agents: list[tuple[str, callable]] = []
        
        # Memory agent always runs (instant, local)
        if self.memory is not None:
            agents.append(("memory", self._run_memory_agent))
        
        # WordNet always runs (offline, instant)
        if self.wordnet_agent.is_available:
            agents.append(("wordnet", self._run_wordnet_agent))
        
        # File agent runs if configured
        if self.file_agent.is_configured:
            agents.append(("files", self._run_file_agent))
        
        # Network agents (skip if offline)
        if not self.offline:
            agents.append(("wikipedia", self._run_wikipedia_agent))
            
            # Web search for factual/general queries
            if query_type in {QueryType.FACTUAL, QueryType.GENERAL, QueryType.HOW_TO}:
                agents.append(("websearch", self._run_websearch_agent))
            
            # Wikidata for comparisons, factual, and definitions
            if query_type in {QueryType.COMPARISON, QueryType.FACTUAL, QueryType.DEFINITION}:
                agents.append(("wikidata", self._run_wikidata_agent))
        
        return agents
    
    # =========================================================================
    # Agent runners — each writes directly to the Blackboard
    # =========================================================================
    
    def _run_memory_agent(self, bb: Blackboard) -> None:
        """Search local memory for relevant facts."""
        if not self.memory:
            return
        
        for entity in bb.entities:
            episodes = self.memory.search_facts(entity, top_k=5)
            if episodes:
                facts = [ep.text for ep in episodes]
                bb.add_finding("memory", Finding(
                    text=f"Memory facts about '{entity}': " + "; ".join(facts[:3]),
                    confidence=0.90,
                    source_name="local memory",
                    facts=facts,
                ))
    
    def _run_wikipedia_agent(self, bb: Blackboard) -> None:
        """Fetch from Wikipedia."""
        for entity in bb.entities:
            result = self.wiki_agent.search(entity)
            if result.found:
                bb.add_finding("wikipedia", Finding(
                    text=result.summary[:500],
                    confidence=0.85,
                    source_url=result.url,
                    source_name=f"Wikipedia: {result.title}",
                    facts=result.facts,
                ))
    
    def _run_websearch_agent(self, bb: Blackboard) -> None:
        """Fetch from DuckDuckGo."""
        result = self.websearch_agent.search(bb.query)
        if result.found:
            text = result.answer or result.abstract or ""
            if text:
                bb.add_finding("websearch", Finding(
                    text=text[:500],
                    confidence=0.70,
                    source_url=result.source_url,
                    source_name=f"Web: {result.source_name}",
                    facts=result.related_topics[:3],
                ))
    
    def _run_file_agent(self, bb: Blackboard) -> None:
        """Search local files."""
        for entity in bb.entities:
            results = self.file_agent.search(entity, max_results=3)
            for r in results:
                bb.add_finding("files", Finding(
                    text=r.snippet,
                    confidence=min(r.relevance, 0.75),
                    source_name=f"File: {r.filename}",
                    metadata={"filepath": r.filepath, "line": r.line_number},
                ))
    
    def _run_wordnet_agent(self, bb: Blackboard) -> None:
        """Look up definitions in WordNet."""
        for entity in bb.entities:
            result = self.wordnet_agent.lookup(entity)
            if result.found:
                parts = [f"({result.part_of_speech}) {result.definition}"]
                if result.synonyms:
                    parts.append(f"Synonyms: {', '.join(result.synonyms[:5])}")
                if result.hypernyms:
                    parts.append(f"Type of: {', '.join(result.hypernyms[:3])}")
                
                facts = []
                if result.hypernyms:
                    facts.append(f"{entity} IS-A {result.hypernyms[0]}")
                for syn in result.synonyms[:3]:
                    facts.append(f"{entity} MEANS {syn}")
                
                bb.add_finding("wordnet", Finding(
                    text=" | ".join(parts),
                    confidence=0.80,
                    source_name="WordNet",
                    facts=facts,
                ))
    
    def _run_wikidata_agent(self, bb: Blackboard) -> None:
        """Look up structured data in Wikidata."""
        for entity in bb.entities:
            result = self.wikidata_agent.lookup(entity)
            if result.found:
                parts = []
                if result.description:
                    parts.append(result.description)
                
                facts = []
                for prop_name, prop_val in list(result.properties.items())[:5]:
                    parts.append(f"{prop_name}: {prop_val}")
                    facts.append(f"{entity} {prop_name.upper()} {prop_val}")
                
                if parts:
                    bb.add_finding("wikidata", Finding(
                        text=" | ".join(parts),
                        confidence=0.82,
                        source_url=f"https://www.wikidata.org/wiki/{result.entity_id}",
                        source_name=f"Wikidata: {result.label}",
                        facts=facts,
                    ))
    
    def format_response(self, bb: Blackboard) -> str:
        """Format blackboard findings into a readable response.
        
        Args:
            bb: Populated blackboard
            
        Returns:
            Formatted response string
        """
        if not bb.has_findings:
            return ""
        
        best = bb.best_findings(top_k=3)
        lines: list[str] = []
        
        for agent_name, finding in best:
            if finding.text:
                # Clean up the text
                text = finding.text.strip()
                if len(text) > 300:
                    text = text[:297] + "..."
                
                source_label = finding.source_name or agent_name
                lines.append(f"  {text}")
                
                if finding.source_url:
                    lines.append(f"  📖 {finding.source_url}")
        
        if not lines:
            return ""
        
        # Add sources summary
        sources = bb.get_all_sources()
        if sources:
            lines.append("")
            lines.append(f"  Sources: {', '.join(s for s in sources[:3])}")
        
        return "\n".join(lines)
