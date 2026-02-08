"""SimpleWiki Loader — Extract facts from Simple English Wikipedia.

Downloads and parses the SimpleWiki XML dump, extracts article text,
and uses the FactExtractor to produce structured facts.

Example:
    >>> loader = SimpleWikiLoader(cache_dir="data/cache")
    >>> facts = loader.load(max_articles=1000)
"""

from __future__ import annotations

import bz2
import gzip
import json
import os
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

SIMPLEWIKI_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)

# Pages to skip
SKIP_PREFIXES = (
    'Wikipedia:', 'Template:', 'Category:', 'File:',
    'Portal:', 'Module:', 'MediaWiki:', 'Help:',
    'Draft:', 'User:', 'Talk:', 'Special:',
)


def _strip_wiki_markup(text: str) -> str:
    """Strip MediaWiki markup from article text."""
    # Remove templates {{...}}
    depth = 0
    result = []
    i = 0
    while i < len(text):
        if i < len(text) - 1 and text[i:i+2] == '{{':
            depth += 1
            i += 2
        elif i < len(text) - 1 and text[i:i+2] == '}}':
            depth = max(0, depth - 1)
            i += 2
        elif depth == 0:
            result.append(text[i])
            i += 1
        else:
            i += 1
    text = ''.join(result)
    
    # Remove [[File:...]], [[Image:...]]
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text)
    
    # Convert [[link|display]] → display
    text = re.sub(r'\[\[[^|\]]*\|([^\]]*)\]\]', r'\1', text)
    
    # Convert [[link]] → link
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    
    # Remove external links [http://... text] → text
    text = re.sub(r'\[https?://[^\s\]]*\s*([^\]]*)\]', r'\1', text)
    
    # Remove ref tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove bold/italic markers
    text = re.sub(r"'{2,5}", '', text)
    
    # Remove section headers
    text = re.sub(r'={2,6}\s*[^=]+\s*={2,6}', '', text)
    
    # Remove list markers
    text = re.sub(r'^\s*[*#:;]+\s*', '', text, flags=re.MULTILINE)
    
    # Remove tables
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    
    # Clean up whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    
    return text.strip()


def _is_disambiguation(text: str) -> bool:
    """Check if article is a disambiguation page."""
    markers = ['{{disambiguation', '{{disambig', '{{dab}}', '{{hndis']
    text_lower = text.lower()
    return any(m in text_lower for m in markers)


def _is_redirect(text: str) -> bool:
    """Check if article is a redirect."""
    return text.strip().lower().startswith('#redirect')


def _is_list(title: str) -> bool:
    """Check if article is a list page."""
    return title.startswith('List of ')


class SimpleWikiLoader:
    """Load knowledge from Simple English Wikipedia.
    
    Downloads the XML dump, extracts articles, and uses FactExtractor
    to produce structured facts from article text.
    
    Args:
        cache_dir: Directory for downloads and cached facts
    """
    
    def __init__(self, cache_dir: str = "data/cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._dump_path = self.cache_dir / "simplewiki-latest-pages-articles.xml.bz2"
        self._facts_cache = self.cache_dir / "simplewiki_facts.json.gz"
    
    def load(
        self,
        max_articles: int = 50000,
        max_facts: int | None = None,
        use_cache: bool = True,
        batch_size: int = 100,
    ) -> list[tuple[str, str, str, float]]:
        """Load facts from SimpleWiki.
        
        Args:
            max_articles: Maximum articles to process
            max_facts: Maximum total facts
            use_cache: Use cached facts if available
            batch_size: Articles per extraction batch
            
        Returns:
            List of (subject, relation, object, confidence)
        """
        # Try cache
        if use_cache and self._facts_cache.exists():
            print(f"Loading cached SimpleWiki facts from {self._facts_cache}...")
            return self._load_cache(max_facts)
        
        # Download if needed
        if not self._dump_path.exists():
            self._download()
        
        # Parse articles and extract facts
        facts = self._extract_all(max_articles, max_facts, batch_size)
        
        # Cache
        if use_cache:
            self._save_cache(facts)
        
        return facts
    
    def _download(self) -> None:
        """Download SimpleWiki dump."""
        print(f"Downloading SimpleWiki dump...")
        print(f"  URL: {SIMPLEWIKI_URL}")
        
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded / (1024 * 1024)
                print(f"\r  Progress: {pct}% ({mb:.0f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(SIMPLEWIKI_URL, self._dump_path, _progress)
        print("\n  Download complete!")
    
    def _iter_articles(
        self, max_articles: int
    ) -> Iterator[tuple[str, str]]:
        """Iterate over (title, text) from the XML dump."""
        count = 0
        ns = '{http://www.mediawiki.org/xml/export-0.10/}'
        
        print(f"Parsing SimpleWiki XML dump...")
        
        with bz2.open(self._dump_path, 'rb') as f:
            for event, elem in ET.iterparse(f, events=('end',)):
                if event != 'end':
                    continue
                
                # Match <page> elements
                tag = elem.tag.replace(ns, '')
                if tag != 'page':
                    continue
                
                title_el = elem.find(f'{ns}title')
                text_el = elem.find(f'.//{ns}text')
                
                if title_el is None or text_el is None or text_el.text is None:
                    elem.clear()
                    continue
                
                title = title_el.text.strip()
                text = text_el.text
                
                # Skip non-article pages
                if any(title.startswith(p) for p in SKIP_PREFIXES):
                    elem.clear()
                    continue
                
                if _is_redirect(text) or _is_disambiguation(text) or _is_list(title):
                    elem.clear()
                    continue
                
                # Clean text
                clean = _strip_wiki_markup(text)
                if len(clean) < 50:
                    elem.clear()
                    continue
                
                yield title, clean
                count += 1
                
                if count >= max_articles:
                    elem.clear()
                    break
                
                # Free memory
                elem.clear()
                
                if count % 1000 == 0:
                    print(f"  Parsed {count:,} articles...")
        
        print(f"  Total articles parsed: {count:,}")
    
    def _extract_all(
        self,
        max_articles: int,
        max_facts: int | None,
        batch_size: int,
    ) -> list[tuple[str, str, str, float]]:
        """Extract facts from all articles."""
        # Lazy import to avoid loading spaCy when not needed
        from gunter.training.fact_extractor import FactExtractor
        
        extractor = FactExtractor()
        all_facts: list[tuple[str, str, str, float]] = []
        seen: set[tuple[str, str, str]] = set()
        batch: list[str] = []
        article_count = 0
        
        for title, text in self._iter_articles(max_articles):
            # Take first 3 sentences (most factual)
            sentences = re.split(r'[.!?]+', text)
            first_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
            batch.extend(first_sentences)
            article_count += 1
            
            if len(batch) >= batch_size:
                extracted = extractor.extract_facts(batch)
                for fact in extracted:
                    key = (fact.subject.lower(), fact.relation, fact.object.lower())
                    if key not in seen:
                        seen.add(key)
                        all_facts.append((
                            fact.subject, fact.relation, fact.object,
                            fact.confidence,
                        ))
                        if max_facts and len(all_facts) >= max_facts:
                            print(f"  Reached max_facts={max_facts:,}")
                            return all_facts
                batch.clear()
                
                if article_count % 500 == 0:
                    print(f"  {article_count:,} articles → {len(all_facts):,} facts")
        
        # Process remaining batch
        if batch:
            extracted = extractor.extract_facts(batch)
            for fact in extracted:
                key = (fact.subject.lower(), fact.relation, fact.object.lower())
                if key not in seen:
                    seen.add(key)
                    all_facts.append((
                        fact.subject, fact.relation, fact.object,
                        fact.confidence,
                    ))
        
        print(f"  Done! {len(all_facts):,} facts from {article_count:,} articles")
        return all_facts
    
    def _save_cache(self, facts: list[tuple[str, str, str, float]]) -> None:
        """Save facts to compressed cache."""
        print(f"Caching {len(facts):,} facts to {self._facts_cache}...")
        with gzip.open(self._facts_cache, 'wt', encoding='utf-8') as f:
            json.dump(facts, f)
    
    def _load_cache(
        self, max_facts: int | None = None,
    ) -> list[tuple[str, str, str, float]]:
        """Load from cache."""
        with gzip.open(self._facts_cache, 'rt', encoding='utf-8') as f:
            facts = json.load(f)
        if max_facts:
            facts = facts[:max_facts]
        print(f"  Loaded {len(facts):,} facts from cache.")
        return [tuple(f) for f in facts]
    
    def get_sample(self, n: int = 50) -> list[tuple[str, str, str, float]]:
        """Get sample facts without downloading."""
        SAMPLE = [
            ("Earth", "IS-A", "planet", 0.9),
            ("Earth", "HAS", "atmosphere", 0.85),
            ("Earth", "IS", "third planet from Sun", 0.8),
            ("Mars", "IS-A", "planet", 0.9),
            ("Mars", "HAS-PROPERTY", "red", 0.8),
            ("Jupiter", "IS-A", "planet", 0.9),
            ("Jupiter", "HAS-PROPERTY", "large", 0.8),
            ("Einstein", "IS-A", "physicist", 0.9),
            ("Einstein", "CREATED", "theory of relativity", 0.8),
            ("Shakespeare", "IS-A", "writer", 0.9),
            ("Shakespeare", "CREATED", "Hamlet", 0.85),
            ("oxygen", "IS-A", "element", 0.9),
            ("oxygen", "PART-OF", "air", 0.8),
            ("photosynthesis", "IS-A", "process", 0.85),
            ("photosynthesis", "REQUIRES", "sunlight", 0.8),
            ("photosynthesis", "CAUSES", "oxygen", 0.8),
            ("DNA", "IS-A", "molecule", 0.9),
            ("DNA", "LOCATED-AT", "cell", 0.8),
            ("Africa", "IS-A", "continent", 0.9),
            ("Africa", "HAS", "many countries", 0.7),
            ("Pacific Ocean", "IS-A", "ocean", 0.9),
            ("Pacific Ocean", "HAS-PROPERTY", "largest", 0.85),
            ("democracy", "IS-A", "form of government", 0.9),
            ("gravity", "IS-A", "force", 0.9),
            ("gravity", "CAUSES", "objects fall", 0.8),
            ("volcano", "IS-A", "geological feature", 0.85),
            ("volcano", "CAUSES", "eruption", 0.8),
            ("dinosaur", "IS-A", "reptile", 0.85),
            ("dinosaur", "HAS-PROPERTY", "extinct", 0.9),
            ("Python", "IS-A", "programming language", 0.9),
            ("hydrogen", "IS-A", "element", 0.9),
            ("hydrogen", "HAS-PROPERTY", "lightest", 0.85),
            ("Amazon River", "IS-A", "river", 0.9),
            ("Amazon River", "LOCATED-AT", "South America", 0.9),
            ("telescope", "IS-A", "instrument", 0.85),
            ("telescope", "USED-FOR", "observing", 0.9),
            ("vaccine", "IS-A", "medicine", 0.8),
            ("vaccine", "USED-FOR", "preventing disease", 0.85),
            ("Mozart", "IS-A", "composer", 0.9),
            ("Mozart", "CREATED", "symphonies", 0.8),
            ("Rome", "IS-A", "city", 0.9),
            ("Rome", "LOCATED-AT", "Italy", 0.9),
            ("whale", "IS-A", "mammal", 0.9),
            ("whale", "LOCATED-AT", "ocean", 0.85),
            ("coral reef", "IS-A", "ecosystem", 0.85),
            ("coral reef", "LOCATED-AT", "ocean", 0.8),
            ("hurricane", "IS-A", "storm", 0.9),
            ("hurricane", "CAUSES", "destruction", 0.8),
            ("internet", "IS-A", "network", 0.9),
            ("internet", "USED-FOR", "communication", 0.9),
        ]
        return SAMPLE[:n]
