"""Knowledge Graph — Weighted graph with provenance tracking.

Phase 2: Replaces flat (S, R, O) tuples with a proper graph structure
that supports edge weights, confidence scores, source tracking,
timestamps, graph traversal, and subgraph extraction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Any

import numpy as np


@dataclass
class KGNode:
    """A node in the knowledge graph."""

    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    _edge_ids: set[int] = field(default_factory=set, repr=False)

    @property
    def degree(self) -> int:
        return len(self._edge_ids)


@dataclass
class KGEdge:
    """A weighted, typed edge in the knowledge graph."""

    id: int
    source: str
    relation: str
    target: str
    weight: float = 1.0
    confidence: float = 1.0
    source_type: str = "user"  # user, conceptnet, wordnet, wikipedia, inferred
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.source, self.relation, self.target)

    def __str__(self) -> str:
        return f"{self.source} --[{self.relation} w={self.weight:.2f}]--> {self.target}"


@dataclass
class GraphPath:
    """A path through the knowledge graph."""

    edges: list[KGEdge]
    confidence: float = 1.0

    @property
    def length(self) -> int:
        return len(self.edges)

    @property
    def entities(self) -> list[str]:
        if not self.edges:
            return []
        path = [self.edges[0].source]
        for edge in self.edges:
            path.append(edge.target)
        return path

    def __str__(self) -> str:
        if not self.edges:
            return "(empty path)"
        parts = [self.edges[0].source]
        for edge in self.edges:
            parts.append(f"--[{edge.relation}]-->")
            parts.append(edge.target)
        return f"{' '.join(parts)} [conf={self.confidence:.2f}]"


class KnowledgeGraph:
    """Weighted knowledge graph with provenance tracking.

    Supports:
    - Weighted edges with confidence scores
    - Source provenance (user, conceptnet, wikipedia, etc.)
    - Efficient neighbor lookup via adjacency lists
    - Graph traversal (BFS shortest path)
    - Subgraph extraction
    - Evidence accumulation (duplicate edges merge)
    - Relation-filtered queries

    Example:
        >>> kg = KnowledgeGraph()
        >>> kg.add_edge("cat", "IS-A", "mammal", source_type="wordnet")
        >>> kg.add_edge("cat", "HAS", "whiskers", source_type="user")
        >>> kg.get_neighbors("cat")
        [KGEdge(source='cat', relation='IS-A', target='mammal', ...),
         KGEdge(source='cat', relation='HAS', target='whiskers', ...)]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[int, KGEdge] = {}
        self._next_id: int = 0

        # Adjacency lists: node → set of edge ids
        self._outgoing: dict[str, set[int]] = defaultdict(set)
        self._incoming: dict[str, set[int]] = defaultdict(set)

        # Index: (source, relation, target) → edge id (for dedup)
        self._triple_index: dict[tuple[str, str, str], int] = {}

        # Index: relation → set of edge ids
        self._relation_index: dict[str, set[int]] = defaultdict(set)

    # ─── Core Operations ───────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        source_type: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> KGEdge:
        """Add an edge to the graph.

        If a duplicate triple exists, merges evidence (boosts confidence).
        """
        source = source.lower().strip()
        target = target.lower().strip()
        relation = relation.upper().strip()

        # Check for existing triple → merge
        triple_key = (source, relation, target)
        if triple_key in self._triple_index:
            edge_id = self._triple_index[triple_key]
            existing = self._edges[edge_id]
            # Evidence accumulation: 1 - (1-c_old)(1-c_new)
            existing.confidence = 1.0 - (1.0 - existing.confidence) * (1.0 - confidence)
            existing.weight = max(existing.weight, weight)
            if metadata:
                existing.metadata.update(metadata)
            return existing

        # Create nodes if needed
        if source not in self._nodes:
            self._nodes[source] = KGNode(name=source)
        if target not in self._nodes:
            self._nodes[target] = KGNode(name=target)

        # Create edge
        edge_id = self._next_id
        self._next_id += 1

        edge = KGEdge(
            id=edge_id,
            source=source,
            relation=relation,
            target=target,
            weight=weight,
            confidence=confidence,
            source_type=source_type,
            metadata=metadata or {},
        )

        self._edges[edge_id] = edge
        self._outgoing[source].add(edge_id)
        self._incoming[target].add(edge_id)
        self._nodes[source]._edge_ids.add(edge_id)
        self._nodes[target]._edge_ids.add(edge_id)
        self._triple_index[triple_key] = edge_id
        self._relation_index[relation].add(edge_id)

        return edge

    def remove_edge(self, edge_id: int) -> None:
        """Remove an edge by ID."""
        if edge_id not in self._edges:
            return
        edge = self._edges[edge_id]
        self._outgoing[edge.source].discard(edge_id)
        self._incoming[edge.target].discard(edge_id)
        self._nodes[edge.source]._edge_ids.discard(edge_id)
        self._nodes[edge.target]._edge_ids.discard(edge_id)
        triple_key = (edge.source, edge.relation, edge.target)
        self._triple_index.pop(triple_key, None)
        self._relation_index[edge.relation].discard(edge_id)
        del self._edges[edge_id]

    def has_edge(self, source: str, relation: str, target: str) -> bool:
        """Check if a specific triple exists."""
        return (source.lower(), relation.upper(), target.lower()) in self._triple_index

    def get_edge(self, source: str, relation: str, target: str) -> KGEdge | None:
        """Get a specific edge by triple."""
        key = (source.lower(), relation.upper(), target.lower())
        edge_id = self._triple_index.get(key)
        if edge_id is not None:
            return self._edges[edge_id]
        return None

    # ─── Queries ───────────────────────────────────────────────────

    def get_neighbors(
        self,
        node: str,
        relation: str | None = None,
        direction: str = "out",
        min_confidence: float = 0.0,
    ) -> list[KGEdge]:
        """Get edges connected to a node.

        Args:
            node: The node name
            relation: Filter by relation type (optional)
            direction: "out" (outgoing), "in" (incoming), or "both"
            min_confidence: Minimum edge confidence

        Returns:
            List of matching edges, sorted by confidence (highest first)
        """
        node = node.lower()
        edge_ids: set[int] = set()

        if direction in ("out", "both"):
            edge_ids |= self._outgoing.get(node, set())
        if direction in ("in", "both"):
            edge_ids |= self._incoming.get(node, set())

        edges = []
        for eid in edge_ids:
            edge = self._edges[eid]
            if edge.confidence < min_confidence:
                continue
            if relation and edge.relation != relation.upper():
                continue
            edges.append(edge)

        return sorted(edges, key=lambda e: e.confidence, reverse=True)

    def get_relations(self, node: str) -> dict[str, list[str]]:
        """Get all relations for a node, grouped by type.

        Returns:
            {"IS-A": ["mammal", "animal"], "CAN": ["purr", "climb"]}
        """
        node = node.lower()
        result: dict[str, list[str]] = defaultdict(list)
        for eid in self._outgoing.get(node, set()):
            edge = self._edges[eid]
            result[edge.relation].append(edge.target)
        return dict(result)

    def get_entities_with_relation(self, relation: str) -> list[tuple[str, str]]:
        """Get all (source, target) pairs for a relation type."""
        relation = relation.upper()
        pairs = []
        for eid in self._relation_index.get(relation, set()):
            edge = self._edges[eid]
            pairs.append((edge.source, edge.target))
        return pairs

    def search_by_entity(self, entity: str, top_k: int = 20) -> list[KGEdge]:
        """Find all edges mentioning an entity (as source or target)."""
        entity = entity.lower()
        edges = []
        for eid in self._outgoing.get(entity, set()):
            edges.append(self._edges[eid])
        for eid in self._incoming.get(entity, set()):
            edges.append(self._edges[eid])
        edges.sort(key=lambda e: e.confidence, reverse=True)
        return edges[:top_k]

    # ─── Graph Traversal ───────────────────────────────────────────

    def shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
        relation_filter: set[str] | None = None,
    ) -> GraphPath | None:
        """BFS shortest path between two nodes.

        Args:
            source: Start node
            target: End node
            max_depth: Maximum path length
            relation_filter: Only follow these relation types

        Returns:
            GraphPath or None if no path exists
        """
        source = source.lower()
        target = target.lower()

        if source == target:
            return GraphPath(edges=[], confidence=1.0)

        if source not in self._nodes or target not in self._nodes:
            return None

        # BFS
        from collections import deque

        queue: deque[tuple[str, list[KGEdge]]] = deque()
        queue.append((source, []))
        visited: set[str] = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            for eid in self._outgoing.get(current, set()):
                edge = self._edges[eid]

                if relation_filter and edge.relation not in relation_filter:
                    continue

                next_node = edge.target

                if next_node == target:
                    full_path = path + [edge]
                    # Confidence decays along chain
                    conf = 1.0
                    for e in full_path:
                        conf *= e.confidence * 0.95  # 5% decay per hop
                    return GraphPath(edges=full_path, confidence=conf)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [edge]))

        return None

    def all_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
    ) -> list[GraphPath]:
        """Find all paths between two nodes (up to max_depth)."""
        source = source.lower()
        target = target.lower()
        paths: list[GraphPath] = []

        def _dfs(current: str, path: list[KGEdge], visited: set[str]):
            if len(path) > max_depth:
                return
            if current == target and path:
                conf = 1.0
                for e in path:
                    conf *= e.confidence * 0.95
                paths.append(GraphPath(edges=list(path), confidence=conf))
                return

            for eid in self._outgoing.get(current, set()):
                edge = self._edges[eid]
                if edge.target not in visited:
                    visited.add(edge.target)
                    path.append(edge)
                    _dfs(edge.target, path, visited)
                    path.pop()
                    visited.discard(edge.target)

        _dfs(source, [], {source})
        return sorted(paths, key=lambda p: p.confidence, reverse=True)

    def subgraph(self, entity: str, depth: int = 2) -> KnowledgeGraph:
        """Extract a subgraph around an entity.

        Returns a new KnowledgeGraph containing all nodes and edges
        within `depth` hops of the entity.
        """
        entity = entity.lower()
        sub = KnowledgeGraph()

        if entity not in self._nodes:
            return sub

        # BFS to collect nodes within depth
        from collections import deque
        queue: deque[tuple[str, int]] = deque([(entity, 0)])
        visited: set[str] = {entity}

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue

            for eid in self._outgoing.get(current, set()):
                edge = self._edges[eid]
                sub.add_edge(
                    edge.source, edge.relation, edge.target,
                    weight=edge.weight, confidence=edge.confidence,
                    source_type=edge.source_type, metadata=dict(edge.metadata),
                )
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, d + 1))

            for eid in self._incoming.get(current, set()):
                edge = self._edges[eid]
                sub.add_edge(
                    edge.source, edge.relation, edge.target,
                    weight=edge.weight, confidence=edge.confidence,
                    source_type=edge.source_type, metadata=dict(edge.metadata),
                )
                if edge.source not in visited:
                    visited.add(edge.source)
                    queue.append((edge.source, d + 1))

        return sub

    # ─── Analytics ─────────────────────────────────────────────────

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics."""
        relation_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for edge in self._edges.values():
            relation_counts[edge.relation] = relation_counts.get(edge.relation, 0) + 1
            source_counts[edge.source_type] = source_counts.get(edge.source_type, 0) + 1

        confidences = [e.confidence for e in self._edges.values()] if self._edges else [0.0]

        return {
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "relations": relation_counts,
            "sources": source_counts,
            "avg_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
        }

    def most_connected(self, top_k: int = 10) -> list[tuple[str, int]]:
        """Get the most connected nodes."""
        counts = [(name, node.degree) for name, node in self._nodes.items()]
        counts.sort(key=lambda x: x[1], reverse=True)
        return counts[:top_k]

    # ─── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "nodes": {name: node.properties for name, node in self._nodes.items()},
            "edges": [
                {
                    "source": e.source,
                    "relation": e.relation,
                    "target": e.target,
                    "weight": e.weight,
                    "confidence": e.confidence,
                    "source_type": e.source_type,
                    "timestamp": e.timestamp.isoformat(),
                    "metadata": e.metadata,
                }
                for e in self._edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> KnowledgeGraph:
        """Deserialize from dict."""
        kg = cls()
        for edge_data in data.get("edges", []):
            ts = edge_data.get("timestamp")
            edge = kg.add_edge(
                source=edge_data["source"],
                relation=edge_data["relation"],
                target=edge_data["target"],
                weight=edge_data.get("weight", 1.0),
                confidence=edge_data.get("confidence", 1.0),
                source_type=edge_data.get("source_type", "unknown"),
                metadata=edge_data.get("metadata", {}),
            )
            if ts:
                try:
                    edge.timestamp = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass

        # Restore node properties
        for name, props in data.get("nodes", {}).items():
            if name in kg._nodes:
                kg._nodes[name].properties = props

        return kg

    # ─── Dunder Methods ────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._edges)

    def __contains__(self, entity: str) -> bool:
        return entity.lower() in self._nodes

    def __iter__(self) -> Iterator[KGEdge]:
        return iter(self._edges.values())

    def __repr__(self) -> str:
        return f"KnowledgeGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
