"""Task Planner — Classifies queries and decomposes into sub-tasks.

The Task Planner is the brain of the Cortex pipeline:
1. Intent Classifier — determines what type of question this is
2. Entity Extractor — pulls out key entities
3. Query Decomposer — breaks complex questions into sub-queries
4. Agent Selector — decides which agents to dispatch

Example:
    >>> planner = TaskPlanner()
    >>> plan = planner.plan("Are dolphins smarter than dogs?")
    >>> print(plan.query_type)
    QueryType.COMPARISON
    >>> print(plan.entities)
    ['dolphin', 'dog']
    >>> print(plan.sub_queries)
    ['dolphin intelligence', 'dog intelligence']
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from prism.agents.blackboard import QueryType

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """A plan for answering a user query."""
    
    original_query: str = ""
    query_type: QueryType = QueryType.GENERAL
    entities: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    comparison_axis: str = ""      # for comparison queries
    expected_answer: str = ""       # "yes/no", "entity", "description", etc.
    confidence: float = 0.0


# ─── Intent Classification Patterns ─────────────────────────────────

# Ordered by specificity — first match wins
INTENT_PATTERNS: list[tuple[str, QueryType, str]] = [
    # Comparison patterns
    (r"(?:compare|vs|versus)\s+", QueryType.COMPARISON, "description"),
    (r"(?:difference|differ)\s+between\s+", QueryType.COMPARISON, "description"),
    (r"(\w+)\s+(?:or|vs)\s+(\w+)", QueryType.COMPARISON, "description"),
    (r"(?:is|are)\s+\w+\s+(?:better|worse|faster|slower|bigger|smaller|smarter|taller|stronger)\s+than", QueryType.COMPARISON, "yes/no"),
    (r"(?:which|who)\s+is\s+(?:more|less|the most|the least)\s+\w+", QueryType.COMPARISON, "entity"),
    
    # Definition patterns
    (r"what\s+(?:is|are)\s+(?:a|an|the)?\s*\w+\??$", QueryType.DEFINITION, "description"),
    (r"define\s+", QueryType.DEFINITION, "description"),
    (r"what\s+(?:does|do)\s+\w+\s+mean", QueryType.DEFINITION, "description"),
    (r"meaning\s+of\s+", QueryType.DEFINITION, "description"),
    
    # Factual patterns
    (r"(?:who|when|where)\s+(?:is|are|was|were|did)\s+", QueryType.FACTUAL, "entity"),
    (r"how\s+(?:many|much|old|long|far|tall|big)\s+", QueryType.FACTUAL, "value"),
    (r"what\s+(?:year|date|time|day)\s+", QueryType.FACTUAL, "value"),
    
    # How-to patterns
    (r"how\s+(?:do|does|can|to|would)\s+", QueryType.HOW_TO, "description"),
    (r"explain\s+(?:how|why)\s+", QueryType.HOW_TO, "description"),
    (r"why\s+(?:do|does|is|are|did)\s+", QueryType.HOW_TO, "description"),
    
    # Yes/No patterns
    (r"(?:is|are|does|do|can|could|has|have|will|would)\s+\w+\s+", QueryType.YES_NO, "yes/no"),
    
    # List patterns
    (r"(?:list|name|give me|show me|what are)\s+(?:all|some|the)\s+", QueryType.LIST, "list"),
    
    # Personal patterns
    (r"(?:my|i|me)\s+", QueryType.PERSONAL, "description"),
    (r"(?:what|who)\s+(?:am i|is my)", QueryType.PERSONAL, "description"),
]

# Stop words to filter from entity extraction
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "if", "then", "else", "when", "where", "how",
    "what", "which", "who", "whom", "whose", "why", "that", "this",
    "these", "those", "it", "its", "they", "them", "their", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her",
    "in", "on", "at", "to", "for", "with", "from", "by", "about",
    "of", "up", "out", "off", "over", "under", "between", "through",
    "than", "more", "less", "most", "least", "very", "much", "many",
    "some", "any", "all", "each", "every", "both", "few", "no", "not",
    "me", "my", "i", "tell", "know", "think", "like", "really",
    "please", "just", "also", "still", "even", "only", "such",
    "compare", "difference", "between", "similar", "different",
    "define", "meaning", "explain", "describe",
}

# Comparison axis keywords
COMPARISON_AXES = {
    "smarter": "intelligence", "intelligent": "intelligence",
    "faster": "speed", "slower": "speed", "speed": "speed",
    "bigger": "size", "smaller": "size", "larger": "size", "size": "size",
    "taller": "height", "shorter": "height", "height": "height",
    "stronger": "strength", "weaker": "strength",
    "better": "quality", "worse": "quality",
    "older": "age", "younger": "age", "age": "age",
    "heavier": "weight", "lighter": "weight", "weight": "weight",
    "expensive": "price", "cheaper": "price", "cost": "price",
    "popular": "popularity", "famous": "fame",
}


class TaskPlanner:
    """Plans how to answer user queries.
    
    Classifies intent, extracts entities, decomposes complex queries
    into sub-queries, and selects which agents to dispatch.
    """
    
    def __init__(self) -> None:
        # Compile patterns for speed
        self._compiled = [
            (re.compile(pattern, re.IGNORECASE), qtype, expected)
            for pattern, qtype, expected in INTENT_PATTERNS
        ]
    
    def plan(self, query: str) -> QueryPlan:
        """Create a plan for answering a query.
        
        Args:
            query: The user's question
            
        Returns:
            QueryPlan with type, entities, and sub-queries
        """
        query_clean = query.strip().rstrip("?!.").strip()
        query_lower = query_clean.lower()
        
        # 1. Classify intent
        query_type, expected = self._classify_intent(query_lower)
        
        # 2. Extract entities
        entities = self._extract_entities(query_lower)
        
        # 3. Detect comparison axis
        comparison_axis = self._detect_comparison_axis(query_lower)
        
        # 4. Decompose into sub-queries
        sub_queries = self._decompose(query_lower, query_type, entities, comparison_axis)
        
        # 5. Confidence based on how well we understood
        confidence = 0.9 if entities else 0.3
        if query_type != QueryType.GENERAL:
            confidence += 0.1
        
        return QueryPlan(
            original_query=query,
            query_type=query_type,
            entities=entities,
            sub_queries=sub_queries,
            comparison_axis=comparison_axis,
            expected_answer=expected,
            confidence=min(confidence, 1.0),
        )
    
    def _classify_intent(self, query: str) -> tuple[QueryType, str]:
        """Classify the query intent using pattern matching."""
        for pattern, qtype, expected in self._compiled:
            if pattern.search(query):
                return qtype, expected
        return QueryType.GENERAL, "description"
    
    def _extract_entities(self, query: str) -> list[str]:
        """Extract key entities from the query."""
        words = query.split()
        entities: list[str] = []
        
        # Filter stop words and short words
        content_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        # Try to find multi-word entities (simple approach: consecutive content words)
        i = 0
        while i < len(content_words):
            # Check for 2-word entity
            if i + 1 < len(content_words):
                bigram = f"{content_words[i]} {content_words[i+1]}"
                # Heuristic: if consecutive words are both capitalized or both nouns
                if len(content_words[i]) > 3 and len(content_words[i+1]) > 3:
                    entities.append(bigram)
                    i += 2
                    continue
            entities.append(content_words[i])
            i += 1
        
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        
        return unique[:5]  # Max 5 entities
    
    def _detect_comparison_axis(self, query: str) -> str:
        """Detect what dimension is being compared."""
        for keyword, axis in COMPARISON_AXES.items():
            if keyword in query:
                return axis
        return ""
    
    def _decompose(
        self,
        query: str,
        query_type: QueryType,
        entities: list[str],
        comparison_axis: str,
    ) -> list[str]:
        """Decompose a query into sub-queries for agents."""
        sub_queries: list[str] = []
        
        if query_type == QueryType.COMPARISON and len(entities) >= 2:
            # For comparisons, query each entity + the comparison axis
            for entity in entities[:2]:
                sub_queries.append(entity)
                if comparison_axis:
                    sub_queries.append(f"{entity} {comparison_axis}")
        
        elif query_type == QueryType.DEFINITION and entities:
            # For definitions, query the main entity
            sub_queries.append(entities[0])
        
        elif query_type == QueryType.HOW_TO and entities:
            # For how-to, query entity + verb phrase
            sub_queries.append(entities[0])
            if len(entities) > 1:
                sub_queries.append(" ".join(entities[:2]))
        
        else:
            # Default: query each entity
            sub_queries.extend(entities[:3])
        
        return sub_queries


# ─── Response Templates ─────────────────────────────────────────────

RESPONSE_TEMPLATES = {
    QueryType.DEFINITION: (
        "{entity} — {definition}\n"
        "{facts}\n"
        "{sources}"
    ),
    QueryType.COMPARISON: (
        "Comparing {entity_a} and {entity_b}:\n"
        "{entity_a}: {facts_a}\n"
        "{entity_b}: {facts_b}\n"
        "{conclusion}\n"
        "{sources}"
    ),
    QueryType.FACTUAL: (
        "{answer}\n"
        "{sources}"
    ),
    QueryType.YES_NO: (
        "{verdict}, {explanation}\n"
        "{sources}"
    ),
    QueryType.HOW_TO: (
        "{explanation}\n"
        "{sources}"
    ),
    QueryType.LIST: (
        "{header}\n"
        "{items}\n"
        "{sources}"
    ),
    QueryType.GENERAL: (
        "{answer}\n"
        "{sources}"
    ),
}


class ResponseComposer:
    """Composes structured responses from Blackboard findings.
    
    Uses templates per query type and fills them with findings
    from the knowledge swarm agents.
    """
    
    def compose(self, plan: QueryPlan, findings: dict) -> str:
        """Compose a response from plan + findings.
        
        Args:
            plan: The query plan
            findings: Dict of agent_name → list of findings
            
        Returns:
            Formatted response string
        """
        if plan.query_type == QueryType.COMPARISON:
            return self._compose_comparison(plan, findings)
        elif plan.query_type == QueryType.DEFINITION:
            return self._compose_definition(plan, findings)
        else:
            return self._compose_general(plan, findings)
    
    def _compose_definition(self, plan: QueryPlan, findings: dict) -> str:
        """Compose a definition response."""
        entity = plan.entities[0] if plan.entities else plan.original_query
        lines = [f"About {entity}:"]
        
        # Best finding text
        all_findings = []
        for agent_name, agent_findings in findings.items():
            for f in agent_findings:
                all_findings.append((agent_name, f))
        all_findings.sort(key=lambda x: x[1].confidence, reverse=True)
        
        for agent_name, finding in all_findings[:3]:
            text = finding.text.strip()
            if len(text) > 300:
                text = text[:297] + "..."
            lines.append(f"  {text}")
        
        # Sources
        sources = list(set(f.source_url for _, f in all_findings if f.source_url))
        if sources:
            lines.append(f"\n  📖 {', '.join(sources[:2])}")
        
        return "\n".join(lines)
    
    def _compose_comparison(self, plan: QueryPlan, findings: dict) -> str:
        """Compose a comparison response."""
        if len(plan.entities) < 2:
            return self._compose_general(plan, findings)
        
        entity_a, entity_b = plan.entities[0], plan.entities[1]
        axis = plan.comparison_axis or "characteristics"
        
        lines = [f"Comparing {entity_a} and {entity_b} ({axis}):"]
        
        # Group findings by entity
        for entity in [entity_a, entity_b]:
            entity_findings = []
            for agent_name, agent_findings in findings.items():
                for f in agent_findings:
                    if entity in f.text.lower():
                        entity_findings.append(f)
            
            if entity_findings:
                best = max(entity_findings, key=lambda f: f.confidence)
                text = best.text.strip()
                if len(text) > 200:
                    text = text[:197] + "..."
                lines.append(f"  {entity}: {text}")
            else:
                lines.append(f"  {entity}: (no data found)")
        
        return "\n".join(lines)
    
    def _compose_general(self, plan: QueryPlan, findings: dict) -> str:
        """Compose a general response."""
        lines: list[str] = []
        
        all_findings = []
        for agent_name, agent_findings in findings.items():
            for f in agent_findings:
                all_findings.append((agent_name, f))
        all_findings.sort(key=lambda x: x[1].confidence, reverse=True)
        
        for agent_name, finding in all_findings[:3]:
            text = finding.text.strip()
            if len(text) > 300:
                text = text[:297] + "..."
            source = finding.source_name or agent_name
            lines.append(f"  [{source}] {text}")
        
        sources = list(set(f.source_url for _, f in all_findings if f.source_url))
        if sources:
            lines.append(f"\n  📖 {', '.join(sources[:2])}")
        
        return "\n".join(lines)
