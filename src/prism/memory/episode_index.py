"""EpisodeIndex — Fast inverted index over episodic memories.

Replaces O(n) substring scanning with O(1) exact lookup and
spaCy embedding-based fuzzy search for retrieval.

Example:
    >>> idx = EpisodeIndex()
    >>> idx.add(episode)
    >>> results = idx.search("cat", top_k=10)
    >>> results = idx.search_by_relation("cat", "IS-A", top_k=10)
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prism.memory import EpisodicMemory


class EpisodeIndex:
    """Inverted index over episodic memories for fast retrieval.

    Maintains three indices:
    - subject → [episode_ids]
    - object → [episode_ids]
    - relation → [episode_ids]

    Phase 4: Also supports vectorized batch cosine search via a
    stacked numpy matrix for O(d) queries instead of O(n×d).

    Search is exact on normalized (lowercased) fields. For fuzzy
    matching, use search_fuzzy() which leverages spaCy embeddings.
    For vector similarity, use search_by_vector() which uses numpy.
    """

    def __init__(self) -> None:
        self._episodes: dict[int, EpisodicMemory] = {}
        self._by_subject: dict[str, list[int]] = defaultdict(list)
        self._by_object: dict[str, list[int]] = defaultdict(list)
        self._by_relation: dict[str, list[int]] = defaultdict(list)
        # Combined: subject OR object
        self._by_entity: dict[str, list[int]] = defaultdict(list)
        # spaCy model for embedding similarity (lazy loaded)
        self._nlp = None

        # Phase 4: Vectorized search
        self._vector_matrix: np.ndarray | None = None
        self._vector_ids: list[int] = []
        self._index_dirty: bool = True

    def add(self, episode: EpisodicMemory) -> None:
        """Add an episode to the index."""
        eid = episode.id
        self._episodes[eid] = episode
        self._index_dirty = True  # Phase 4: mark vector index stale

        subj = episode.subject.lower().strip()
        obj = episode.obj.lower().strip()
        rel = episode.relation.upper().strip()

        if subj:
            self._by_subject[subj].append(eid)
            self._by_entity[subj].append(eid)
        if obj:
            self._by_object[obj].append(eid)
            if obj != subj:  # avoid duplicates in entity index
                self._by_entity[obj].append(eid)
        if rel:
            self._by_relation[rel].append(eid)

    def add_batch(self, episodes: list[EpisodicMemory]) -> None:
        """Add multiple episodes efficiently."""
        for ep in episodes:
            self.add(ep)

    def search(
        self,
        entity: str,
        top_k: int = 10,
        as_subject: bool = True,
        as_object: bool = True,
    ) -> list[EpisodicMemory]:
        """Search for episodes mentioning an entity.

        Args:
            entity: Entity to search for (exact match on normalized form)
            top_k: Maximum results
            as_subject: Search in subject field
            as_object: Search in object field

        Returns:
            List of matching episodes, sorted by importance (descending)
        """
        entity_lower = entity.lower().strip()
        seen: set[int] = set()
        results: list[EpisodicMemory] = []

        if as_subject:
            for eid in self._by_subject.get(entity_lower, []):
                if eid not in seen:
                    seen.add(eid)
                    results.append(self._episodes[eid])

        if as_object:
            for eid in self._by_object.get(entity_lower, []):
                if eid not in seen:
                    seen.add(eid)
                    results.append(self._episodes[eid])

        # Sort by importance (highest first)
        results.sort(key=lambda ep: ep.importance, reverse=True)
        return results[:top_k]

    def search_by_relation(
        self,
        entity: str,
        relation: str,
        top_k: int = 10,
    ) -> list[EpisodicMemory]:
        """Search for episodes with a specific entity + relation.

        E.g., search_by_relation("cat", "IS-A") → all "cat IS-A ..." episodes.

        Args:
            entity: Subject entity
            relation: Relation type (e.g., "IS-A", "CAN", "HAS")
            top_k: Maximum results

        Returns:
            Matching episodes sorted by importance
        """
        entity_lower = entity.lower().strip()
        relation_upper = relation.upper().strip()

        # Get episodes where entity is subject
        subject_eids = set(self._by_subject.get(entity_lower, []))
        # Get episodes with this relation
        relation_eids = set(self._by_relation.get(relation_upper, []))
        # Intersection
        matching_eids = subject_eids & relation_eids

        results = [self._episodes[eid] for eid in matching_eids]
        results.sort(key=lambda ep: ep.importance, reverse=True)
        return results[:top_k]

    def search_subject_facts(
        self,
        entity: str,
        top_k: int = 20,
    ) -> list[EpisodicMemory]:
        """Get all facts where entity is the subject.

        Returns structured episodes — callers can access .subject, .relation,
        .obj fields directly instead of parsing text.
        """
        entity_lower = entity.lower().strip()
        eids = self._by_subject.get(entity_lower, [])
        results = [self._episodes[eid] for eid in eids]
        results.sort(key=lambda ep: ep.importance, reverse=True)
        return results[:top_k]

    def search_fuzzy(
        self,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.6,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Fuzzy search using spaCy word embeddings.

        When exact match fails, find entities semantically similar to
        the query and return their episodes.

        Args:
            query: Search query
            top_k: Maximum results
            min_similarity: Minimum embedding cosine similarity

        Returns:
            List of (episode, similarity_score) tuples
        """
        nlp = self._get_nlp()
        if nlp is None:
            return []

        query_lower = query.lower().strip()

        # First try exact match
        exact = self.search(query_lower, top_k=top_k)
        if len(exact) >= top_k:
            return [(ep, 1.0) for ep in exact]

        # Compute query embedding
        query_doc = nlp(query_lower)
        if not query_doc.has_vector:
            return [(ep, 1.0) for ep in exact]

        # Find similar entities from index keys
        all_entities = set(self._by_subject.keys()) | set(self._by_object.keys())

        similar_entities: list[tuple[str, float]] = []
        for entity in all_entities:
            if entity == query_lower:
                continue
            entity_doc = nlp(entity)
            if entity_doc.has_vector:
                sim = query_doc.similarity(entity_doc)
                if sim >= min_similarity:
                    similar_entities.append((entity, sim))

        # Sort by similarity
        similar_entities.sort(key=lambda x: x[1], reverse=True)

        # Collect results
        seen: set[int] = {ep.id for ep in exact}
        results: list[tuple[EpisodicMemory, float]] = [(ep, 1.0) for ep in exact]

        for entity, sim in similar_entities[:20]:
            for eid in self._by_entity.get(entity, []):
                if eid not in seen and len(results) < top_k:
                    seen.add(eid)
                    results.append((self._episodes[eid], sim))

        return results[:top_k]

    def get_all_entities(self) -> set[str]:
        """Get all unique entities (subjects and objects) in the index."""
        return set(self._by_subject.keys()) | set(self._by_object.keys())

    def get_all_subjects(self) -> set[str]:
        """Get all unique subjects."""
        return set(self._by_subject.keys())

    def get_entity_count(self, entity: str) -> int:
        """Get total number of facts mentioning an entity."""
        entity_lower = entity.lower().strip()
        return len(self._by_entity.get(entity_lower, []))

    def get_relations_for(self, entity: str) -> dict[str, list[str]]:
        """Get all relations and objects for an entity (as subject).

        Returns:
            dict mapping relation → list of objects.
            E.g., {"IS-A": ["mammal", "animal"], "CAN": ["purr", "climb"]}
        """
        entity_lower = entity.lower().strip()
        props: dict[str, list[str]] = defaultdict(list)

        for eid in self._by_subject.get(entity_lower, []):
            ep = self._episodes[eid]
            props[ep.relation].append(ep.obj)

        return dict(props)

    def clear(self) -> None:
        """Clear all indices."""
        self._episodes.clear()
        self._by_subject.clear()
        self._by_object.clear()
        self._by_relation.clear()
        self._by_entity.clear()

    def __len__(self) -> int:
        return len(self._episodes)

    def _get_nlp(self):
        """Lazy-load spaCy model for embedding similarity."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_md")
            except Exception:
                self._nlp = False  # Mark as unavailable
        return self._nlp if self._nlp is not False else None

    # ─── Phase 4: Vectorized Batch Search ──────────────────────────

    def rebuild_vector_index(self) -> None:
        """Rebuild the stacked vector matrix for batch cosine search.

        Stacks all episode vectors into a single numpy matrix
        and normalizes rows for efficient dot-product similarity.
        """
        vectors = []
        ids = []
        for eid, ep in self._episodes.items():
            if ep.vector is not None:
                v = np.asarray(ep.vector, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    vectors.append(v / norm)
                    ids.append(eid)

        if vectors:
            self._vector_matrix = np.vstack(vectors)  # shape: (N, dim)
            self._vector_ids = ids
        else:
            self._vector_matrix = None
            self._vector_ids = []

        self._index_dirty = False

    def search_by_vector(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple['EpisodicMemory', float]]:
        """Vectorized batch cosine similarity search.

        Uses a single matrix multiply instead of per-episode loops.
        O(d) per query instead of O(n*d).

        Args:
            query_vec: Query vector (same dimension as episodes)
            top_k: Number of results
            min_score: Minimum cosine similarity

        Returns:
            List of (episode, score) tuples sorted by score
        """
        # Rebuild if stale
        if self._index_dirty or self._vector_matrix is None:
            self.rebuild_vector_index()

        if self._vector_matrix is None or len(self._vector_ids) == 0:
            return []

        # Normalize query
        q = np.asarray(query_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        # Single matrix multiply: all cosines at once
        scores = self._vector_matrix @ q  # shape: (N,)

        # Get top-k indices
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            eid = self._vector_ids[idx]
            if eid in self._episodes:
                results.append((self._episodes[eid], score))

        return results
