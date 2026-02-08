"""DuckDuckGo Web Search Agent — Privacy-first web search.

Uses the DuckDuckGo Instant Answer API (no API key, no tracking).
Returns instant answers, abstracts, and related topics.

Example:
    >>> agent = WebSearchAgent()
    >>> result = agent.search("Burj Khalifa height")
    >>> print(result.text)
    'The Burj Khalifa is 828 metres tall...'
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DDG_API = "https://api.duckduckgo.com/"
REQUEST_TIMEOUT = 3


@dataclass
class WebSearchResult:
    """Result from a web search."""
    abstract: str = ""
    answer: str = ""
    related_topics: list[str] = field(default_factory=list)
    source_url: str = ""
    source_name: str = ""
    found: bool = False


class WebSearchAgent:
    """Agent that searches the web via DuckDuckGo Instant Answer API.
    
    Features:
    - Zero API keys needed
    - Privacy-first (DuckDuckGo doesn't track)
    - Returns instant answers when available
    - Falls back to abstracts and related topics
    - 3-second timeout with graceful offline fallback
    """

    def __init__(self) -> None:
        self._cache: dict[str, WebSearchResult] = {}

    def search(self, query: str) -> WebSearchResult:
        """Search the web for a topic.
        
        Args:
            query: Search query
            
        Returns:
            WebSearchResult with answer/abstract
        """
        query_lower = query.lower().strip()
        
        if query_lower in self._cache:
            return self._cache[query_lower]
        
        result = self._fetch(query)
        self._cache[query_lower] = result
        return result

    def _fetch(self, query: str) -> WebSearchResult:
        """Fetch from DuckDuckGo Instant Answer API."""
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
                "t": "prism-ai",
            })
            url = f"{DDG_API}?{params}"
            
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PRISM-AI/1.0 (educational project)",
                },
            )
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            abstract = data.get("AbstractText", "")
            answer = data.get("Answer", "")
            source_url = data.get("AbstractURL", "")
            source_name = data.get("AbstractSource", "")
            
            # Extract related topics
            related = []
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    related.append(topic["Text"])
            
            found = bool(abstract or answer or related)
            
            return WebSearchResult(
                abstract=abstract,
                answer=answer,
                related_topics=related,
                source_url=source_url,
                source_name=source_name,
                found=found,
            )
            
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.debug(f"DuckDuckGo unavailable: {e}")
            return WebSearchResult()
            
        except Exception as e:
            logger.debug(f"DuckDuckGo error: {e}")
            return WebSearchResult()

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
