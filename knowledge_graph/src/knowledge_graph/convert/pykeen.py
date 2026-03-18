"""Conversion utilities for transforming KnowledgeGraph objects to PyKEEN datasets.

This module provides functionality to convert KnowledgeGraph objects into PyKEEN's
TriplesFactory and Dataset formats, maintaining node and edge type information.

Functions
---------
to_pykeen_dataset(kg, validation_split=0.1, test_split=0.1, random_state=None)
    Convert a KnowledgeGraph to a PyKEEN Dataset with train/validation/test splits.
to_triples_factory(kg)
    Convert a KnowledgeGraph to a PyKEEN TriplesFactory.

Examples
--------
>>> from knowledge_graph import KnowledgeGraph, KnowledgeGraphSchema
>>> from knowledge_graph.convert.pykeen import to_pykeen_dataset
>>>
>>> # Create a knowledge graph
>>> schema = KnowledgeGraphSchema(
...     node_types={'Person', 'Organization'},
...     edge_types={'WORKS_FOR', 'MANAGES'}
... )
>>> kg = KnowledgeGraph(schema=schema)
>>> kg.add_edge("John", "Person", "Acme Corp", "Organization", "WORKS_FOR")
>>>
>>> # Convert to PyKEEN dataset
>>> dataset = to_pykeen_dataset(kg)

Dependencies
-----------
- pykeen
- numpy
- torch

See Also
--------
pykeen.triples.TriplesFactory : https://pykeen.readthedocs.io/en/stable/api/pykeen.triples.TriplesFactory.html

"""

from __future__ import annotations

from typing import Any

import torch
from pykeen.datasets.base import Dataset, EagerDataset
from pykeen.triples import TriplesFactory

from knowledge_graph.core.graph import KnowledgeGraph


def to_triples_factory(kg: KnowledgeGraph) -> Any:
    """Convert a KnowledgeGraph to a PyKEEN TriplesFactory.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph to convert.

    Returns
    -------
    TriplesFactory
        A PyKEEN TriplesFactory containing the graph's triples.

    """
    import torch
    from pykeen.triples import TriplesFactory

    # Extract all unique entities and relations
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}

    # Create mappings for entities (including type in the entity name)
    entity_counter = 0
    for node in kg.nodes():
        entity_name = f"{node.name}_{node.type}"
        if entity_name not in entities:
            entities[entity_name] = entity_counter
            entity_counter += 1

    # Create mappings for relations
    relation_counter = 0
    for _, _, data in kg.edges(data=True):
        rel_type = data["type"]
        if rel_type not in relations:
            relations[rel_type] = relation_counter
            relation_counter += 1

    # Create triples
    triples = []
    for u, v, data in kg.edges(data=True):
        head = f"{u.name}_{u.type}"
        tail = f"{v.name}_{v.type}"
        relation = data["type"]

        head_idx = entities[head]
        tail_idx = entities[tail]
        relation_idx = relations[relation]

        triples.append([head_idx, relation_idx, tail_idx])

    # Convert to tensor
    triples_tensor = torch.tensor(triples, dtype=torch.long)

    # Create and return TriplesFactory
    return TriplesFactory(
        mapped_triples=triples_tensor,
        entity_to_id=entities,
        relation_to_id=relations,
    )


def to_pykeen_dataset(
    kg: KnowledgeGraph,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    random_state: int | None = None,
) -> Any:
    """Convert a KnowledgeGraph to a PyKEEN Dataset with train/validation/test splits.

    If any edge has a 'split' attribute in its data dict (under the 'split' key),
    the function will use those labels to build the splits. Recognized labels:
    - train: 'train'
    - validation: 'validation', 'valid', 'val'
    - test: 'test', 'testing'
    Unrecognized or missing labels default to 'train'.

    If no edges have a 'split' attribute, the function falls back to ratio-based
    splitting based on `validation_split` and `test_split`.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph to convert.
    validation_split : float, optional (default=0.1)
        Fraction of edges to use for validation (between 0 and 1), used only
        when no per-edge 'split' is provided.
    test_split : float, optional (default=0.1)
        Fraction of edges to use for testing (between 0 and 1), used only
        when no per-edge 'split' is provided.
    random_state : int, optional (default=None)
        Random state for reproducibility (only relevant for ratio-based split).

    Returns
    -------
    Dataset
        A PyKEEN Dataset containing the split graph.

    Raises
    ------
    ValueError
        If validation_split + test_split >= 1.0 (in ratio-based mode)

    """
    # Detect whether any edge provides a split label
    any_split_attr = False
    for _, _, data in kg.edges(data=True):
        if "split" in data:
            any_split_attr = True
            break

    if not any_split_attr:
        # Fall back to the existing ratio-based approach
        if validation_split + test_split >= 1.0:
            raise ValueError("Sum of validation and test splits must be less than 1.0")
        tf = to_triples_factory(kg)
        train_split = 1 - validation_split - test_split
        ratios = (train_split, validation_split, test_split)
        return Dataset.from_tf(tf, ratios)

    # Build a global vocabulary over all nodes/edges for consistent mapping
    entities: dict[str, int] = {}
    relations: dict[str, int] = {}

    entity_counter = 0
    for node in kg.nodes():
        entity_name = f"{node.name}_{node.type}"
        if entity_name not in entities:
            entities[entity_name] = entity_counter
            entity_counter += 1

    relation_counter = 0
    for _, _, data in kg.edges(data=True):
        rel_type = data["type"]
        if rel_type not in relations:
            relations[rel_type] = relation_counter
            relation_counter += 1

    # Partition triples by split label
    def normalize_split_label(label: str | None) -> str:
        if label is None:
            return "train"
        lab = str(label).strip().lower()
        if lab in {"train"}:
            return "train"
        if lab in {"validation", "valid", "val"}:
            return "validation"
        if lab in {"test", "testing"}:
            return "testing"
        # Default to train for unrecognized labels
        return "train"

    train_triples: list[list[int]] = []
    val_triples: list[list[int]] = []
    test_triples: list[list[int]] = []

    for u, v, data in kg.edges(data=True):
        head = f"{u.name}_{u.type}"
        tail = f"{v.name}_{v.type}"
        relation = data["type"]

        split_label = normalize_split_label(data.get("split"))

        head_idx = entities[head]
        tail_idx = entities[tail]
        relation_idx = relations[relation]
        triple = [head_idx, relation_idx, tail_idx]

        if split_label == "train":
            train_triples.append(triple)
        elif split_label == "validation":
            val_triples.append(triple)
        else:  # "testing"
            test_triples.append(triple)

    # Convert to tensors (allow empty tensors for val/test if necessary)
    train_tensor = (
        torch.tensor(train_triples, dtype=torch.long)
        if train_triples
        else torch.empty((0, 3), dtype=torch.long)
    )
    val_tensor = (
        torch.tensor(val_triples, dtype=torch.long)
        if val_triples
        else torch.empty((0, 3), dtype=torch.long)
    )
    test_tensor = (
        torch.tensor(test_triples, dtype=torch.long)
        if test_triples
        else torch.empty((0, 3), dtype=torch.long)
    )

    # Build TriplesFactory for each split with shared vocabularies
    train_tf = TriplesFactory(
        mapped_triples=train_tensor,
        entity_to_id=entities,
        relation_to_id=relations,
    )
    validation_tf = TriplesFactory(
        mapped_triples=val_tensor,
        entity_to_id=entities,
        relation_to_id=relations,
    )
    testing_tf = TriplesFactory(
        mapped_triples=test_tensor,
        entity_to_id=entities,
        relation_to_id=relations,
    )

    return EagerDataset(training=train_tf, validation=validation_tf, testing=testing_tf)
