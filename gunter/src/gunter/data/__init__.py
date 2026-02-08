"""Data loaders for large-scale knowledge integration.

Provides loaders for ConceptNet, WordNet, and SimpleWiki,
plus a KnowledgeIntegrator to merge all sources.
"""

from gunter.data.loaders.conceptnet_loader import ConceptNetLoader
from gunter.data.loaders.wordnet_loader import WordNetLoader
from gunter.data.loaders.simplewiki_loader import SimpleWikiLoader
from gunter.data.loaders.knowledge_integrator import KnowledgeIntegrator

__all__ = [
    'ConceptNetLoader',
    'WordNetLoader',
    'SimpleWikiLoader',
    'KnowledgeIntegrator',
]
