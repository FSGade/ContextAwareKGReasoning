"""
Conversion utilities for transforming KnowledgeGraph objects to PyKEEN datasets.

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

from knowledge_graph.core.graph import KnowledgeGraph


def to_triples_factory(kg: KnowledgeGraph) -> Any:
    """
    Convert a KnowledgeGraph to a PyKEEN TriplesFactory.

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
    """
    Convert a KnowledgeGraph to a PyKEEN Dataset with train/validation/test splits.

    Parameters
    ----------
    kg : KnowledgeGraph
        The knowledge graph to convert.
    validation_split : float, optional (default=0.1)
        Fraction of edges to use for validation (between 0 and 1).
    test_split : float, optional (default=0.1)
        Fraction of edges to use for testing (between 0 and 1).
    random_state : int, optional (default=None)
        Random state for reproducibility.

    Returns
    -------
    Dataset
        A PyKEEN Dataset containing the split graph.

    Raises
    ------
    ValueError
        If validation_split + test_split >= 1.0
    """
    from pykeen.datasets.base import Dataset

    if validation_split + test_split >= 1.0:
        raise ValueError(
            "Sum of validation and test splits must be less than 1.0"
        )

    # Convert to TriplesFactory
    tf = to_triples_factory(kg)

    train_split = 1 - validation_split - test_split
    ratios = (train_split, validation_split, test_split)

    # # Get total number of triples
    # num_triples = tf.mapped_triples.shape[0]

    # # Calculate split sizes
    # test_size = int(num_triples * test_split)
    # validation_size = int(num_triples * validation_split)
    # train_size = num_triples - test_size - validation_size

    # # Create random indices for splits
    # indices = np.random.permutation(num_triples)
    # train_indices = indices[:train_size]
    # validation_indices = indices[train_size : train_size + validation_size]
    # test_indices = indices[train_size + validation_size :]

    # # Create split TriplesFactories
    # train_tf = TriplesFactory(
    #     mapped_triples=tf.mapped_triples[train_indices],
    #     entity_to_id=tf.entity_to_id,
    #     relation_to_id=tf.relation_to_id,
    # )

    # validation_tf = TriplesFactory(
    #     mapped_triples=tf.mapped_triples[validation_indices],
    #     entity_to_id=tf.entity_to_id,
    #     relation_to_id=tf.relation_to_id,
    # )

    # test_tf = TriplesFactory(
    #     mapped_triples=tf.mapped_triples[test_indices],
    #     entity_to_id=tf.entity_to_id,
    #     relation_to_id=tf.relation_to_id,
    # )

    # Create and return Dataset
    # return Dataset(
    #     training=train_tf,
    #     validation=validation_tf,
    #     testing=test_tf,
    # )
    return Dataset.from_tf(tf, ratios)


# Example usage and testing function
def example_usage():
    """Example usage of the conversion functions."""
    from knowledge_graph import KnowledgeGraph, KnowledgeGraphSchema

    # Create a schema
    schema = KnowledgeGraphSchema(
        node_types={"Person", "Organization"},
        edge_types={"WORKS_FOR", "MANAGES"},
        frozen=False,
    )

    # Create a knowledge graph
    kg = KnowledgeGraph(schema=schema)

    # Add some nodes and edges
    kg.add_typed_edge(
        "John", "Person", "Acme Corp", "Organization", "WORKS_FOR"
    )
    kg.add_typed_edge("Jane", "Person", "Acme Corp", "Organization", "MANAGES")
    kg.add_typed_edge("Bob", "Person", "Tech Inc", "Organization", "WORKS_FOR")
    kg.add_typed_edge("Alice", "Person", "Tech Inc", "Organization", "MANAGES")

    # Convert to PyKEEN dataset
    dataset = to_pykeen_dataset(kg, random_state=42)

    # Print some information about the dataset
    print("Dataset statistics:")
    print(f"Training triples: {dataset.training.num_triples}")
    print(f"Validation triples: {dataset.validation.num_triples}")
    print(f"Testing triples: {dataset.testing.num_triples}")
    print(f"\nEntities: {len(dataset.training.entity_to_id)}")
    print(f"Relations: {len(dataset.training.relation_to_id)}")

    return dataset


if __name__ == "__main__":
    example_usage()
