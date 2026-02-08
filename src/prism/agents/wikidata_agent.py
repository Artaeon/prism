"""Wikidata SPARQL Agent — Structured data queries.

Queries Wikidata's SPARQL endpoint for structured facts like
dates, numbers, and formal relationships.

Example:
    >>> agent = WikidataAgent()
    >>> result = agent.lookup("Albert Einstein")
    >>> print(result.properties['date of birth'])
    '1879-03-14'
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
REQUEST_TIMEOUT = 4


@dataclass
class WikidataResult:
    """Result from a Wikidata query."""
    entity_id: str = ""
    label: str = ""
    description: str = ""
    properties: dict[str, str] = field(default_factory=dict)
    found: bool = False


class WikidataAgent:
    """Agent that queries Wikidata for structured data.
    
    Features:
    - Entity search and lookup
    - Fetches key properties (dates, numbers, descriptions)
    - SPARQL queries for complex relationships
    - No API key needed
    - 4-second timeout with graceful fallback
    """

    def __init__(self) -> None:
        self._cache: dict[str, WikidataResult] = {}

    def lookup(self, entity: str) -> WikidataResult:
        """Look up an entity in Wikidata.
        
        Args:
            entity: Entity name (e.g., "Albert Einstein")
            
        Returns:
            WikidataResult with ID, description, and key properties
        """
        entity_lower = entity.lower().strip()
        
        if entity_lower in self._cache:
            return self._cache[entity_lower]
        
        result = self._search_entity(entity)
        self._cache[entity_lower] = result
        return result
    
    def _search_entity(self, query: str) -> WikidataResult:
        """Search Wikidata for an entity."""
        try:
            params = urllib.parse.urlencode({
                "action": "wbsearchentities",
                "search": query,
                "language": "en",
                "limit": 1,
                "format": "json",
            })
            url = f"{WIKIDATA_API}?{params}"
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PRISM-AI/1.0 (educational project)"},
            )
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            results = data.get("search", [])
            if not results:
                return WikidataResult()
            
            top = results[0]
            entity_id = top.get("id", "")
            label = top.get("label", "")
            description = top.get("description", "")
            
            # Fetch key properties via SPARQL
            properties = self._fetch_properties(entity_id)
            
            return WikidataResult(
                entity_id=entity_id,
                label=label,
                description=description,
                properties=properties,
                found=True,
            )
            
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            logger.debug(f"Wikidata unavailable: {e}")
            return WikidataResult()
        except Exception as e:
            logger.debug(f"Wikidata error: {e}")
            return WikidataResult()
    
    def _fetch_properties(self, entity_id: str) -> dict[str, str]:
        """Fetch key properties for an entity via SPARQL."""
        # Key properties to fetch
        property_map = {
            "P31": "instance of",
            "P279": "subclass of",
            "P569": "date of birth",
            "P570": "date of death",
            "P19": "place of birth",
            "P20": "place of death",
            "P27": "country of citizenship",
            "P106": "occupation",
            "P1082": "population",
            "P2044": "elevation",
            "P2046": "area",
            "P17": "country",
            "P36": "capital",
            "P37": "official language",
        }
        
        # Build a simple SPARQL query for common properties
        prop_ids = " ".join(f"wdt:{pid}" for pid in list(property_map)[:8])
        
        sparql = f"""
        SELECT ?prop ?propLabel ?val ?valLabel WHERE {{
          VALUES ?prop {{ {prop_ids} }}
          wd:{entity_id} ?prop ?val .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 20
        """
        
        try:
            params = urllib.parse.urlencode({
                "query": sparql.strip(),
                "format": "json",
            })
            url = f"{WIKIDATA_SPARQL}?{params}"
            
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "PRISM-AI/1.0 (educational project)",
                    "Accept": "application/sparql-results+json",
                },
            )
            
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            
            props: dict[str, str] = {}
            for binding in data.get("results", {}).get("bindings", []):
                prop_uri = binding.get("prop", {}).get("value", "")
                val_label = binding.get("valLabel", {}).get("value", "")
                
                # Extract property ID from URI
                pid = prop_uri.rsplit("/", 1)[-1] if "/" in prop_uri else prop_uri
                prop_name = property_map.get(pid, pid)
                
                if val_label and prop_name:
                    # Collect multiple values for the same property
                    if prop_name in props:
                        props[prop_name] += f", {val_label}"
                    else:
                        props[prop_name] = val_label
            
            return props
            
        except Exception as e:
            logger.debug(f"Wikidata SPARQL error: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._cache.clear()
