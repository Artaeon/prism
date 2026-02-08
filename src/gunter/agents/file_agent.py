"""File Agent — Searches local documents for knowledge.

Indexes and searches user's local files (txt, md, pdf) without
any data ever leaving the machine.

Example:
    >>> agent = FileAgent(search_dirs=["~/Documents"])
    >>> results = agent.search("photosynthesis")
    >>> for r in results:
    ...     print(r.filename, r.snippet)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".log", ".json", ".py", ".yaml", ".yml"}
MAX_FILE_SIZE = 1_000_000  # 1MB max per file
MAX_RESULTS = 5


@dataclass
class FileSearchResult:
    """A search result from local files."""
    filename: str = ""
    filepath: str = ""
    snippet: str = ""
    line_number: int = 0
    relevance: float = 0.0


class FileAgent:
    """Agent that searches local documents for knowledge.
    
    Features:
    - Searches text files in configured directories
    - Keyword matching with context window
    - Privacy-first: all processing is local
    - Configurable search directories
    - File size limits to prevent hanging on large files
    """

    def __init__(self, search_dirs: list[str] | None = None) -> None:
        """Initialize with directories to search.
        
        Args:
            search_dirs: Directories to search. Defaults to none (disabled).
        """
        self._search_dirs: list[Path] = []
        if search_dirs:
            for d in search_dirs:
                path = Path(d).expanduser().resolve()
                if path.is_dir():
                    self._search_dirs.append(path)
                else:
                    logger.warning(f"FileAgent: directory not found: {d}")

    def search(self, query: str, max_results: int = MAX_RESULTS) -> list[FileSearchResult]:
        """Search local files for a query.
        
        Args:
            query: Search terms
            max_results: Maximum results to return
            
        Returns:
            List of FileSearchResult with matching snippets
        """
        if not self._search_dirs:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        results: list[FileSearchResult] = []
        
        for search_dir in self._search_dirs:
            try:
                self._search_directory(search_dir, query_lower, query_words, results, max_results)
            except Exception as e:
                logger.debug(f"FileAgent error scanning {search_dir}: {e}")
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:max_results]

    def _search_directory(
        self,
        directory: Path,
        query_lower: str,
        query_words: set[str],
        results: list[FileSearchResult],
        max_results: int,
    ) -> None:
        """Recursively search a directory."""
        try:
            for entry in directory.iterdir():
                if len(results) >= max_results:
                    return
                
                if entry.is_dir():
                    # Skip hidden directories and common non-content dirs
                    if entry.name.startswith(".") or entry.name in {
                        "node_modules", "__pycache__", ".git", "venv", ".venv",
                    }:
                        continue
                    self._search_directory(entry, query_lower, query_words, results, max_results)
                    
                elif entry.is_file() and entry.suffix in SUPPORTED_EXTENSIONS:
                    try:
                        if entry.stat().st_size > MAX_FILE_SIZE:
                            continue
                        self._search_file(entry, query_lower, query_words, results)
                    except (PermissionError, OSError):
                        continue
                        
        except PermissionError:
            pass

    def _search_file(
        self,
        filepath: Path,
        query_lower: str,
        query_words: set[str],
        results: list[FileSearchResult],
    ) -> None:
        """Search a single file for the query."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return
        
        content_lower = content.lower()
        
        # Check if any query words appear in the file
        if not any(word in content_lower for word in query_words):
            return
        
        # Find the best matching line
        lines = content.split("\n")
        best_line = 0
        best_score = 0.0
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            # Score: count how many query words appear in this line
            score = sum(1 for word in query_words if word in line_lower)
            # Bonus for exact phrase match
            if query_lower in line_lower:
                score += len(query_words)
            
            if score > best_score:
                best_score = score
                best_line = i
        
        if best_score > 0:
            # Extract context window (±2 lines)
            start = max(0, best_line - 2)
            end = min(len(lines), best_line + 3)
            snippet = "\n".join(lines[start:end]).strip()
            
            # Cap snippet length
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            
            results.append(FileSearchResult(
                filename=filepath.name,
                filepath=str(filepath),
                snippet=snippet,
                line_number=best_line + 1,
                relevance=best_score / max(len(query_words), 1),
            ))

    @property
    def is_configured(self) -> bool:
        """Whether any search directories are configured."""
        return len(self._search_dirs) > 0

    def clear_cache(self) -> None:
        """No-op for API compatibility."""
        pass
