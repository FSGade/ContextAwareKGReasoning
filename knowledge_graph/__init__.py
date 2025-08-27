"""Knowledge Graph implementation with schema validation and I/O utilities."""

from __future__ import annotations

# from knowledge_graph.convert import to_hetero_torch_geometric
# from knowledge_graph.convert import to_pykeen_dataset
# from knowledge_graph.convert import to_triples_factory
from knowledge_graph.core.graph import Entity
from knowledge_graph.core.graph import KnowledgeGraph
from knowledge_graph.core.schema import KnowledgeGraphSchema
from knowledge_graph.io.exporters import GraphExporter
from knowledge_graph.io.importers import GraphImporter
from knowledge_graph.utils.stats import print_kg_stats

__all__ = [
    "KnowledgeGraphSchema",
    "KnowledgeGraph",
    "Entity",
    "GraphExporter",
    "GraphImporter",
    "print_kg_stats",
    # "to_hetero_torch_geometric",
    # "to_pykeen_dataset",
    # "to_triples_factory",
]

__version__ = "1.0.0"
