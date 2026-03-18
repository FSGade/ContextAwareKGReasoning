"""Knowledge Graph implementation with schema validation and I/O utilities."""

from __future__ import annotations

from knowledge_graph.core.graph import Entity, KnowledgeGraph
from knowledge_graph.core.schema import KnowledgeGraphSchema
from knowledge_graph.utils.stats import print_kg_stats

__all__ = [
    "Entity",
    "KnowledgeGraph",
    "KnowledgeGraphSchema",
    "print_kg_stats",
]

__version__ = "1.0.0"
