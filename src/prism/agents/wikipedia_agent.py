"""Wikipedia Agent — Fetches knowledge from Wikipedia on demand.

Uses the MediaWiki REST API (no API key needed) to provide live knowledge
when local memory doesn't have an answer. Falls back gracefully when offline.

Example:
    >>> agent = WikipediaAgent()
    >>> result = agent.search("Albert Einstein")
    >>> print(result.summary)
    'Albert Einstein was a German-born theoretical physicist...'
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# MediaWiki REST API base (no API key needed)
WIKI_API = "https://en.wikipedia.org/api/rest_v1"
WIKI_SEARCH_API = "https://en.wikipedia.org/w/api.php"

# Timeout for all requests (seconds)
REQUEST_TIMEOUT = 3


@dataclass
class WikiResult:
    """Result from a Wikipedia query."""
    title: str = ""
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    url: str = ""
    found: bool = False


class WikipediaAgent:
    """Agent that fetches knowledge from Wikipedia.
    
    Features:
    - Searches Wikipedia for topics
    - Extracts first paragraph summaries
    - Caches results to avoid redundant API calls
    - 3-second timeout (falls back if offline)
    - No API key required
    """

    def __init__(self) -> None:
        self._cache: dict[str, WikiResult] = {}

    def search(self, query: str) -> WikiResult:
        """Search Wikipedia for a topic.
        
        Args:
            query: Search query (e.g., "cat", "Albert Einstein")
            
        Returns:
            WikiResult with summary and URL
        """
        query_lower = query.lower().strip()
        
        # Check cache first
        if query_lower in self._cache:
            return self._cache[query_lower]
        
        result = self._fetch_summary(query)
        self._cache[query_lower] = result
        return result

    def get_facts(self, topic: str, max_facts: int = 5) -> list[str]:
        """Get structured facts about a topic.
        
        Extracts key sentences from the Wikipedia summary as facts.
        
        Args:
            topic: Topic to get facts for
            max_facts: Maximum number of facts to extract
            
        Returns:
            List of fact strings
        """
        result = self.search(topic)
        if not result.found:
            return []
        
        # Split summary into sentences and return as facts
        sentences = self._split_sentences(result.summary)
        return sentences[:max_facts]

    def _fetch_summary(self, query: str) -> WikiResult:
        """Fetch summary from Wikipedia REST API."""
        try:
            # Try direct page summary first
            title = query.replace(" ", "_")
            url = f"{WIKI_API}/page/summary/{urllib.parse.quote(title)}"
            
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PRISM-AI/1.0 (educational project)",
                    "Accept": "application/json",
                },
            )
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                if data.get("type") == "standard":
                    summary = data.get("extract", "")
                    title = data.get("title", query)
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    
                    facts = self._extract_facts(summary)
                    
                    return WikiResult(
                        title=title,
                        summary=summary,
                        facts=facts,
                        url=page_url,
                        found=True,
                    )
            
            # If direct lookup fails, try search
            return self._search_and_fetch(query)
            
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Try search API as fallback
                return self._search_and_fetch(query)
            logger.debug(f"Wikipedia HTTP error: {e.code}")
            return WikiResult()
            
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.debug(f"Wikipedia unavailable: {e}")
            return WikiResult()
            
        except Exception as e:
            logger.debug(f"Wikipedia error: {e}")
            return WikiResult()

    def _search_and_fetch(self, query: str) -> WikiResult:
        """Search Wikipedia and fetch the top result's summary."""
        try:
            params = urllib.parse.urlencode({
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
            })
            url = f"{WIKI_SEARCH_API}?{params}"
            
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PRISM-AI/1.0 (educational project)",
                },
            )
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                results = data.get("query", {}).get("search", [])
                if not results:
                    return WikiResult()
                
                # Fetch summary for top result
                top_title = results[0]["title"]
                return self._fetch_summary(top_title)
                
        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
            return WikiResult()

    def _extract_facts(self, summary: str) -> list[str]:
        """Extract individual facts from a summary."""
        sentences = self._split_sentences(summary)
        # Return non-trivial sentences as facts
        return [s for s in sentences if len(s) > 20][:5]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
