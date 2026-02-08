"""Data loaders for large-scale knowledge integration.

Provides loaders for ConceptNet, WordNet, and SimpleWiki,
plus a KnowledgeIntegrator to merge all sources.
"""

from prism.data.loaders.conceptnet_loader import ConceptNetLoader
from prism.data.loaders.wordnet_loader import WordNetLoader
from prism.data.loaders.simplewiki_loader import SimpleWikiLoader
from prism.data.loaders.knowledge_integrator import KnowledgeIntegrator

__all__ = [
    'ConceptNetLoader',
    'WordNetLoader',
    'SimpleWikiLoader',
    'KnowledgeIntegrator',
]
