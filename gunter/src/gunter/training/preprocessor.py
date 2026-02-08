"""Data Preprocessor — Clean and filter text for fact extraction.

Handles raw text, files, and web content, producing clean
sentence lists ready for fact extraction.
"""

from __future__ import annotations

import re
from pathlib import Path


# Skip sentences shorter or longer than these
MIN_SENTENCE_LEN = 5
MAX_SENTENCE_LEN = 200
MIN_WORD_COUNT = 3
MAX_WORD_COUNT = 40


class DataPreprocessor:
    """Clean and filter text for training.
    
    Pipeline:
    1. clean_text() — strip URLs, HTML, special chars
    2. split_sentences() — sentence boundary detection
    3. filter_sentences() — skip junk
    
    Example:
        >>> pp = DataPreprocessor()
        >>> pp.process_text("Cats are animals. See http://cat.com for more!")
        ["Cats are animals."]
    """
    
    def clean_text(self, text: str) -> str:
        """Remove URLs, HTML tags, special characters, extra whitespace."""
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove references like [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove parenthetical references
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?\'-]', ' ', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.
        
        Uses a simple regex-based approach that handles common
        abbreviations and decimal numbers.
        """
        # Split on sentence-ending punctuation followed by space + uppercase
        # or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Also split on newlines that look like sentence breaks
        result = []
        for sent in sentences:
            parts = sent.split('\n')
            for part in parts:
                part = part.strip()
                if part:
                    result.append(part)
        
        return result
    
    def filter_sentences(self, sentences: list[str]) -> list[str]:
        """Filter out unsuitable sentences.
        
        Removes:
        - Too short/long sentences
        - Questions (we want declarative facts)
        - Sentences without verbs (likely fragments)
        - Lists, headers, and metadata
        """
        filtered = []
        
        for sent in sentences:
            # Length checks
            if len(sent) < MIN_SENTENCE_LEN or len(sent) > MAX_SENTENCE_LEN:
                continue
            
            words = sent.split()
            if len(words) < MIN_WORD_COUNT or len(words) > MAX_WORD_COUNT:
                continue
            
            # Skip questions
            if sent.rstrip().endswith('?'):
                continue
            
            # Skip sentences starting with numbers (likely lists)
            if re.match(r'^\d+[.)]\s', sent):
                continue
            
            # Skip ALL CAPS (headers)
            if sent.upper() == sent and len(sent) > 10:
                continue
            
            # Skip sentences with too many numbers
            num_count = sum(1 for w in words if re.match(r'^\d+$', w))
            if num_count > len(words) * 0.4:
                continue
            
            # Basic verb check: must contain at least one common verb form
            has_verb = any(
                w.lower() in _COMMON_VERBS or w.lower().endswith(('s', 'ed', 'ing'))
                for w in words
            )
            if not has_verb and len(words) > 4:
                continue
            
            filtered.append(sent)
        
        return filtered
    
    def process_text(self, text: str) -> list[str]:
        """Full pipeline: clean → split → filter."""
        cleaned = self.clean_text(text)
        sentences = self.split_sentences(cleaned)
        return self.filter_sentences(sentences)
    
    def process_file(self, filepath: str) -> list[str]:
        """Process a text file into clean sentences."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        text = path.read_text(encoding='utf-8', errors='replace')
        return self.process_text(text)


# Common English verbs for the verb-presence check
_COMMON_VERBS = {
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did',
    'can', 'could', 'will', 'would', 'shall', 'should',
    'may', 'might', 'must',
    'need', 'needs', 'needed',
    'make', 'makes', 'made',
    'get', 'gets', 'got',
    'use', 'uses', 'used',
    'know', 'knows', 'knew',
    'think', 'thinks', 'thought',
    'take', 'takes', 'took',
    'come', 'comes', 'came',
    'go', 'goes', 'went',
    'see', 'sees', 'saw',
    'give', 'gives', 'gave',
    'find', 'finds', 'found',
    'say', 'says', 'said',
    'eat', 'eats', 'ate',
    'live', 'lives', 'lived',
    'run', 'runs', 'ran',
    'play', 'plays', 'played',
    'contain', 'contains', 'contained',
    'include', 'includes', 'included',
    'produce', 'produces', 'produced',
    'require', 'requires', 'required',
    'provide', 'provides', 'provided',
    'belong', 'belongs', 'belonged',
    'consist', 'consists', 'consisted',
    'exist', 'exists', 'existed',
    'remain', 'remains', 'remained',
    'become', 'becomes', 'became',
    'seem', 'seems', 'seemed',
    'appear', 'appears', 'appeared',
    'grow', 'grows', 'grew',
    'call', 'calls', 'called',
    'keep', 'keeps', 'kept',
    'help', 'helps', 'helped',
    'show', 'shows', 'showed',
    'move', 'moves', 'moved',
    'hold', 'holds', 'held',
    'bring', 'brings', 'brought',
    'occur', 'occurs', 'occurred',
    'serve', 'serves', 'served',
    'form', 'forms', 'formed',
    'create', 'creates', 'created',
    'build', 'builds', 'built',
    'develop', 'develops', 'developed',
    'reach', 'reaches', 'reached',
    'follow', 'follows', 'followed',
    'allow', 'allows', 'allowed',
    'begin', 'begins', 'began',
    'set', 'sets',
    'learn', 'learns', 'learned',
    'change', 'changes', 'changed',
    'lead', 'leads', 'led',
    'stand', 'stands', 'stood',
    'lose', 'loses', 'lost',
    'add', 'adds', 'added',
    'read', 'reads',
    'spend', 'spends', 'spent',
    'support', 'supports', 'supported',
    'consider', 'considers', 'considered',
    'describe', 'describes', 'described',
    'represent', 'represents', 'represented',
    'cover', 'covers', 'covered',
}
