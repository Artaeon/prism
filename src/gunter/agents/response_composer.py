"""Response Composer — Polished response generation for Cortex.

Generates structured, citation-backed responses from Blackboard findings.
Handles multiple query types with specialized templates and formatting.

Example:
    >>> composer = CortexComposer()
    >>> response = composer.compose(blackboard, plan)
    >>> print(response)
    Photosynthesis is the process by which plants convert sunlight...
      📖 Wikipedia: Photosynthesis
      🧠 Confidence: ● 92%
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gunter.agents.blackboard import Blackboard, Finding, QueryType

try:
    from gunter.agents.semantic_weaver import SemanticWeaver
except ImportError:
    SemanticWeaver = None  # type: ignore


# ─── Confidence Display ─────────────────────────────────────────────

def confidence_bar(score: float) -> str:
    """Format a confidence score as a visual indicator.
    
    Args:
        score: 0.0 to 1.0
        
    Returns:
        Formatted string like "● [92%]" or "◐ [56%]"
    """
    pct = int(score * 100)
    if score >= 0.8:
        return f"● [{pct}%]"
    elif score >= 0.5:
        return f"◐ [{pct}%]"
    elif score >= 0.3:
        return f"◔ [{pct}%]"
    else:
        return f"◌ [{pct}%]"


def format_source(finding: Finding) -> str:
    """Format a source citation."""
    if finding.source_url:
        return f"📖 {finding.source_name}: {finding.source_url}"
    elif finding.source_name:
        return f"📖 {finding.source_name}"
    return ""


# ─── CortexComposer ─────────────────────────────────────────────────

class CortexComposer:
    """Composes polished, structured responses from Blackboard findings.
    
    Features:
    - SemanticWeaver for natural text generation (if available)
    - Query-type-aware templates (definition, comparison, factual, etc.)
    - Source citations with URLs
    - Confidence indicators
    """
    
    MAX_SNIPPET_LEN = 350
    MAX_SOURCES = 3
    
    def __init__(self, weaver: Any = None) -> None:
        """Initialize composer.
        
        Args:
            weaver: Optional SemanticWeaver for natural text generation.
                    If None, falls back to template-based composition.
        """
        self._weaver = weaver
    
    def compose(self, bb: Blackboard, plan: Any = None) -> str:
        """Compose a response from blackboard findings.
        
        Uses SemanticWeaver for natural text generation when available,
        falls back to template-based composition otherwise.
        
        Args:
            bb: Populated Blackboard
            plan: Optional QueryPlan for context
            
        Returns:
            Formatted response string
        """
        if not bb.has_findings:
            return ""
        
        # Try SemanticWeaver first (natural text generation)
        if self._weaver:
            woven = self._weaver.weave(bb, plan)
            if woven:
                # Add attribution on top of woven text
                lines = [woven]
                best = bb.best_findings(top_k=3)
                
                # Add supporting facts from the blackboard
                facts = bb.get_all_facts()[:4]
                if facts:
                    for fact in facts:
                        lines.append(f"  🧩 {fact}")
                
                self._add_attribution(lines, best, [])
                return "\n".join(lines)
        
        # Fallback: template-based composition
        query_type = bb.query_type
        if plan and hasattr(plan, 'query_type'):
            query_type = plan.query_type
        
        composers = {
            QueryType.DEFINITION: self._compose_definition,
            QueryType.COMPARISON: self._compose_comparison,
            QueryType.FACTUAL: self._compose_factual,
            QueryType.YES_NO: self._compose_yes_no,
            QueryType.HOW_TO: self._compose_how_to,
            QueryType.LIST: self._compose_list,
        }
        
        composer = composers.get(query_type, self._compose_general)
        return composer(bb, plan)
    
    # ─── Type-specific composers ────────────────────────────────────
    
    def _compose_definition(self, bb: Blackboard, plan: Any) -> str:
        """Compose a definition response — prioritize clarity."""
        entity = self._primary_entity(bb, plan)
        best = bb.best_findings(top_k=3)
        
        lines: list[str] = []
        used_sources: list[str] = []
        
        # Lead with the best finding
        for agent_name, finding in best:
            text = self._clean_snippet(finding.text)
            if not text:
                continue
            
            if not lines:
                # First finding — the main answer
                lines.append(text)
            else:
                # Supporting details
                lines.append(f"  • {text}")
            
            src = format_source(finding)
            if src and src not in used_sources:
                used_sources.append(src)
        
        # Add facts as bullet points
        facts = bb.get_all_facts()[:4]
        if facts:
            lines.append("")
            for fact in facts:
                lines.append(f"  🧩 {fact}")
        
        # Attribution
        self._add_attribution(lines, best, used_sources)
        
        return "\n".join(lines)
    
    def _compose_comparison(self, bb: Blackboard, plan: Any) -> str:
        """Compose a comparison response — side-by-side."""
        entities = self._all_entities(bb, plan)
        
        if len(entities) < 2:
            # Try to extract entities from query words
            return self._compose_general(bb, plan)
        
        entity_a, entity_b = entities[0], entities[1]
        axis = ""
        if plan and hasattr(plan, 'comparison_axis'):
            axis = plan.comparison_axis
        
        header = f"Comparing {entity_a} and {entity_b}"
        if axis:
            header += f" ({axis})"
        lines = [f"{header}:"]
        lines.append("")
        
        # Group findings by entity — search all findings
        for entity in [entity_a, entity_b]:
            entity_findings = self._findings_for_entity(bb, entity)
            if entity_findings:
                best = max(entity_findings, key=lambda f: f.confidence)
                text = self._clean_snippet(best.text, max_len=250)
                lines.append(f"  ▸ {entity.title()}:")
                lines.append(f"    {text}")
            else:
                # Try using any finding even without exact entity match
                all_best = bb.best_findings(top_k=4)
                for name, f in all_best:
                    if entity.lower() in f.source_name.lower() or entity.lower() in name.lower():
                        text = self._clean_snippet(f.text, max_len=250)
                        lines.append(f"  ▸ {entity.title()}:")
                        lines.append(f"    {text}")
                        break
                else:
                    lines.append(f"  ▸ {entity.title()}: (no information found)")
        
        # Facts
        facts = bb.get_all_facts()[:4]
        if facts:
            lines.append("")
            for fact in facts:
                lines.append(f"  🧩 {fact}")
        
        # Attribution
        best = bb.best_findings(top_k=3)
        self._add_attribution(lines, best, [])
        
        return "\n".join(lines)
    
    def _compose_factual(self, bb: Blackboard, plan: Any) -> str:
        """Compose a factual response — concise answer first."""
        best = bb.best_findings(top_k=2)
        lines: list[str] = []
        
        for agent_name, finding in best:
            text = self._clean_snippet(finding.text)
            if text:
                lines.append(text)
                break  # One main answer is enough
        
        # Supporting facts
        facts = bb.get_all_facts()[:3]
        if facts:
            for fact in facts:
                lines.append(f"  🧩 {fact}")
        
        self._add_attribution(lines, best, [])
        return "\n".join(lines)
    
    def _compose_yes_no(self, bb: Blackboard, plan: Any) -> str:
        """Compose a yes/no response — synthesize verdict from evidence."""
        best = bb.best_findings(top_k=3)
        
        if not best:
            return "I'm not sure about that."
        
        # Extract the query and predicate
        query = ""
        if plan and hasattr(plan, 'original_query'):
            query = plan.original_query.lower()
        
        # Extract predicate: "Can penguins swim?" → "swim"
        import re as _re
        predicate = ""
        pred_match = _re.search(
            r'(?:can|could|do|does|is|are|will|would|has|have)\s+\w+\s+(.+)',
            query.rstrip('?').strip()
        )
        if pred_match:
            predicate = pred_match.group(1).strip()
        
        # Collect all evidence text
        all_text = " ".join(f.text.lower() for _, f in best)
        all_facts = " ".join(bb.get_all_facts()[:5]).lower()
        evidence = all_text + " " + all_facts
        
        verdict = "Based on available information"
        verdict_emoji = "🤔"
        
        # 1. Check if the predicate itself appears in evidence
        predicate_found = predicate and predicate in evidence
        
        # 2. Check for negative/positive indicators
        neg_indicators = [
            'cannot', 'can not', "can't", 'unable', 'flightless',
            'incapable', 'impossible', 'never ', 'not ', 'no ',
            'lack', 'fail', 'lost the ability', 'do not ',
        ]
        pos_indicators = [
            'can ', 'able to', 'capable', 'known for', 'adapted',
            'evolved', 'designed', 'specializ', 'excellent',
            'flipper', 'aquatic', 'marine', 'semi-aquatic',
            'streamlined', 'webbed', 'swimmer', 'swimming',
        ]
        
        neg_count = sum(1 for n in neg_indicators if n in evidence)
        pos_count = sum(1 for p in pos_indicators if p in evidence)
        
        # If predicate is found directly, that's a strong positive signal
        if predicate_found:
            pos_count += 3
        
        if neg_count > pos_count:
            verdict = "No, based on the evidence"
            verdict_emoji = "❌"
        elif pos_count > neg_count:
            verdict = "Yes, based on the evidence"
            verdict_emoji = "✅"
        
        _, top_finding = best[0]
        text = self._clean_snippet(top_finding.text, max_len=250)
        
        lines = [f"{verdict_emoji} {verdict}:"]
        lines.append(f"  {text}")
        
        # Supporting facts
        facts = bb.get_all_facts()[:3]
        if facts:
            for fact in facts:
                lines.append(f"  🧩 {fact}")
        
        self._add_attribution(lines, best, [])
        return "\n".join(lines)
    
    def _compose_how_to(self, bb: Blackboard, plan: Any) -> str:
        """Compose a how-to response — explanation style."""
        best = bb.best_findings(top_k=3)
        lines: list[str] = []
        
        for agent_name, finding in best:
            text = self._clean_snippet(finding.text)
            if text:
                lines.append(text)
                break
        
        # Additional info
        facts = bb.get_all_facts()[:3]
        if facts:
            lines.append("")
            for fact in facts:
                lines.append(f"  • {fact}")
        
        self._add_attribution(lines, best, [])
        return "\n".join(lines)
    
    def _compose_list(self, bb: Blackboard, plan: Any) -> str:
        """Compose a list response — bullet points."""
        facts = bb.get_all_facts()[:10]
        best = bb.best_findings(top_k=3)
        lines: list[str] = []
        
        if facts:
            for i, fact in enumerate(facts, 1):
                lines.append(f"  {i}. {fact}")
        else:
            for agent_name, finding in best:
                text = self._clean_snippet(finding.text)
                if text:
                    lines.append(f"  • {text}")
        
        self._add_attribution(lines, best, [])
        return "\n".join(lines)
    
    def _compose_general(self, bb: Blackboard, plan: Any) -> str:
        """Compose a general response — best findings."""
        best = bb.best_findings(top_k=3)
        lines: list[str] = []
        used_sources: list[str] = []
        
        for agent_name, finding in best:
            text = self._clean_snippet(finding.text)
            if not text:
                continue
            
            source_label = finding.source_name or agent_name
            lines.append(f"  [{source_label}] {text}")
            
            src = format_source(finding)
            if src and src not in used_sources:
                used_sources.append(src)
        
        self._add_attribution(lines, best, used_sources)
        return "\n".join(lines)
    
    # ─── Helpers ────────────────────────────────────────────────────
    
    def _clean_snippet(self, text: str, max_len: int = 0) -> str:
        """Clean and truncate a text snippet."""
        max_len = max_len or self.MAX_SNIPPET_LEN
        text = text.strip()
        if len(text) > max_len:
            text = text[:max_len - 3] + "..."
        return text
    
    def _primary_entity(self, bb: Blackboard, plan: Any) -> str:
        """Get the primary entity from plan or blackboard."""
        if plan and hasattr(plan, 'entities') and plan.entities:
            return plan.entities[0]
        if bb.entities:
            return bb.entities[0]
        return bb.query
    
    def _all_entities(self, bb: Blackboard, plan: Any) -> list[str]:
        """Get all entities."""
        if plan and hasattr(plan, 'entities') and plan.entities:
            return plan.entities
        return bb.entities
    
    def _findings_for_entity(self, bb: Blackboard, entity: str) -> list[Finding]:
        """Get all findings that mention a specific entity."""
        entity_lower = entity.lower()
        results: list[Finding] = []
        for agent_findings in bb.findings.values():
            for finding in agent_findings:
                if entity_lower in finding.text.lower():
                    results.append(finding)
        return results
    
    def _add_attribution(
        self,
        lines: list[str],
        best: list[tuple[str, Finding]],
        used_sources: list[str],
    ) -> None:
        """Add source attribution and confidence to response."""
        # Collect all sources
        all_sources = list(used_sources)
        for _, finding in best:
            src = format_source(finding)
            if src and src not in all_sources:
                all_sources.append(src)
        
        if all_sources:
            lines.append("")
            for src in all_sources[:self.MAX_SOURCES]:
                lines.append(f"  {src}")
        
        # Overall confidence
        if best:
            avg_conf = sum(f.confidence for _, f in best) / len(best)
            lines.append(f"  🧠 {confidence_bar(avg_conf)}")
