"""Conversion utilities for transforming knowledge graphs to different formats."""

from __future__ import annotations

from knowledge_graph.convert.pykeen import to_pykeen_dataset, to_triples_factory
from knowledge_graph.convert.torch_geometric import to_hetero_torch_geometric

__all__ = [
    "to_hetero_torch_geometric",
    "to_pykeen_dataset",
    "to_triples_factory",
]
