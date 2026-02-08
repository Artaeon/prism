"""Gunter Trainer — Full training pipeline with progress tracking.

Orchestrates preprocessing → extraction → validation → storage
from files, Wikipedia, and URLs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from gunter.training.preprocessor import DataPreprocessor
from gunter.training.fact_extractor import FactExtractor, ExtractedFact
from gunter.training.validator import FactValidator, ValidationResult


@dataclass
class TrainingStats:
    """Statistics from a training run."""
    
    source: str = ""
    sentences_raw: int = 0
    sentences_clean: int = 0
    facts_extracted: int = 0
    facts_valid: int = 0
    facts_invalid: int = 0
    facts_conflicts: int = 0
    facts_stored: int = 0
    errors: int = 0
    time_seconds: float = 0.0
    
    @property
    def speed(self) -> float:
        """Facts per second."""
        if self.time_seconds > 0:
            return self.facts_stored / self.time_seconds
        return 0.0
    
    def summary(self) -> str:
        lines = [
            f"📊 Training Statistics — {self.source}",
            f"{'─' * 45}",
            f"  Sentences: {self.sentences_raw} raw → {self.sentences_clean} clean",
            f"  Facts extracted: {self.facts_extracted}",
            f"  Validation: {self.facts_valid} valid, {self.facts_invalid} invalid, {self.facts_conflicts} conflicts",
            f"  Stored: {self.facts_stored} facts",
            f"  Errors: {self.errors}",
            f"  Time: {self.time_seconds:.1f}s ({self.speed:.0f} facts/sec)",
        ]
        return "\n".join(lines)


class GunterTrainer:
    """Full training pipeline for Gunter.
    
    Orchestrates:
    1. Preprocessing (clean text, split sentences, filter)
    2. Fact extraction (spaCy dependency parsing)
    3. Validation (quality, plausibility, consistency)
    4. Storage (store in Gunter's memory with auto-save)
    
    Example:
        >>> trainer = GunterTrainer(gunter)
        >>> stats = trainer.train_from_file("knowledge.txt")
        >>> print(stats.summary())
    """
    
    AUTO_SAVE_INTERVAL = 1000  # Save every N facts
    
    def __init__(self, gunter: Any) -> None:
        """Initialize trainer with a Gunter instance."""
        self.gunter = gunter
        self.preprocessor = DataPreprocessor()
        self.extractor = FactExtractor(nlp=getattr(gunter, '_nlp', None))
        self.validator = FactValidator(gunter)
    
    def train_from_text(
        self,
        text: str,
        source: str = "text",
        show_progress: bool = True,
    ) -> TrainingStats:
        """Train from raw text."""
        stats = TrainingStats(source=source)
        start = time.time()
        
        # Preprocess
        sentences_raw = self.preprocessor.split_sentences(
            self.preprocessor.clean_text(text)
        )
        stats.sentences_raw = len(sentences_raw)
        
        sentences = self.preprocessor.filter_sentences(sentences_raw)
        stats.sentences_clean = len(sentences)
        
        if not sentences:
            stats.time_seconds = time.time() - start
            return stats
        
        # Extract & validate & store
        self._extract_validate_store(sentences, stats, show_progress)
        
        stats.time_seconds = time.time() - start
        return stats
    
    def train_from_file(
        self,
        filepath: str,
        show_progress: bool = True,
    ) -> TrainingStats:
        """Train from a text file."""
        path = Path(filepath)
        if not path.exists():
            stats = TrainingStats(source=filepath)
            stats.errors = 1
            return stats
        
        text = path.read_text(encoding='utf-8', errors='replace')
        return self.train_from_text(text, source=path.name, show_progress=show_progress)
    
    def train_from_wikipedia(
        self,
        topic: str,
        max_articles: int = 3,
        show_progress: bool = True,
    ) -> TrainingStats:
        """Train from Wikipedia articles.
        
        Args:
            topic: Wikipedia search topic
            max_articles: Maximum number of articles to process
            show_progress: Show progress bars
        """
        stats = TrainingStats(source=f"wikipedia:{topic}")
        start = time.time()
        
        try:
            import wikipediaapi
        except ImportError:
            stats.errors = 1
            stats.time_seconds = time.time() - start
            return stats
        
        wiki = wikipediaapi.Wikipedia(
            user_agent='Gunter/1.0 (training)',
            language='en',
        )
        
        # Get the main page
        page = wiki.page(topic)
        if not page.exists():
            stats.errors = 1
            stats.time_seconds = time.time() - start
            return stats
        
        # Collect text from the page and linked pages
        all_text = [page.text]
        
        if max_articles > 1:
            # Get linked pages
            links = list(page.links.keys())[:max_articles - 1]
            
            desc = f"Fetching Wikipedia articles for '{topic}'"
            iterator = tqdm(links, desc=desc, disable=not show_progress)
            
            for link_title in iterator:
                try:
                    linked_page = wiki.page(link_title)
                    if linked_page.exists() and len(linked_page.text) > 100:
                        all_text.append(linked_page.text)
                except Exception:
                    stats.errors += 1
        
        combined = "\n\n".join(all_text)
        
        # Preprocess
        sentences_raw = self.preprocessor.split_sentences(
            self.preprocessor.clean_text(combined)
        )
        stats.sentences_raw = len(sentences_raw)
        
        sentences = self.preprocessor.filter_sentences(sentences_raw)
        stats.sentences_clean = len(sentences)
        
        # Extract & validate & store
        self._extract_validate_store(sentences, stats, show_progress)
        
        stats.time_seconds = time.time() - start
        return stats
    
    def train_from_url(
        self,
        url: str,
        show_progress: bool = True,
    ) -> TrainingStats:
        """Train from a web page URL.
        
        Args:
            url: URL to fetch and extract text from
            show_progress: Show progress bars
        """
        stats = TrainingStats(source=f"url:{url}")
        start = time.time()
        
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            stats.errors = 1
            stats.time_seconds = time.time() - start
            return stats
        
        try:
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Gunter/1.0 (training bot)'
            })
            response.raise_for_status()
        except Exception as e:
            stats.errors = 1
            stats.time_seconds = time.time() - start
            return stats
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = '\n'.join(p.get_text() for p in paragraphs)
        
        # Preprocess
        sentences_raw = self.preprocessor.split_sentences(
            self.preprocessor.clean_text(text)
        )
        stats.sentences_raw = len(sentences_raw)
        
        sentences = self.preprocessor.filter_sentences(sentences_raw)
        stats.sentences_clean = len(sentences)
        
        # Extract & validate & store
        self._extract_validate_store(sentences, stats, show_progress)
        
        stats.time_seconds = time.time() - start
        return stats
    
    def _extract_validate_store(
        self,
        sentences: list[str],
        stats: TrainingStats,
        show_progress: bool,
    ) -> None:
        """Core pipeline: extract facts → validate → store."""
        
        # ── Extract facts ──
        desc = "Extracting facts"
        
        # Process in batches for the progress bar
        batch_size = 50
        all_facts: list[ExtractedFact] = []
        
        batches = [
            sentences[i:i + batch_size]
            for i in range(0, len(sentences), batch_size)
        ]
        
        iterator = tqdm(batches, desc=desc, disable=not show_progress)
        for batch in iterator:
            try:
                batch_facts = self.extractor.extract_facts(batch)
                all_facts.extend(batch_facts)
                iterator.set_postfix(facts=len(all_facts))
            except Exception:
                stats.errors += 1
        
        stats.facts_extracted = len(all_facts)
        
        # ── Validate ──
        result = self.validator.validate_facts(all_facts)
        stats.facts_valid = len(result.valid)
        stats.facts_invalid = len(result.invalid)
        stats.facts_conflicts = len(result.conflicts)
        
        # ── Store valid facts ──
        desc = "Storing facts"
        iterator = tqdm(
            result.valid,
            desc=desc,
            disable=not show_progress,
        )
        
        stored = 0
        for fact in iterator:
            try:
                self.gunter.memory.store(fact.subject, fact.relation, fact.object)
                stored += 1
                
                # Auto-save periodically
                if stored % self.AUTO_SAVE_INTERVAL == 0:
                    try:
                        self.gunter.save_memory()
                    except Exception:
                        pass
                
                iterator.set_postfix(stored=stored)
            except Exception:
                stats.errors += 1
        
        stats.facts_stored = stored
        
        # Final save
        try:
            self.gunter.save_memory()
        except Exception:
            pass
