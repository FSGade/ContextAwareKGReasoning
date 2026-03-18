"""Filtering utilities for knowledge graphs."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from typing import Any

import networkx as nx

from knowledge_graph.core.graph import KnowledgeGraph


def sample_and_get_largest_component(
    kg: KnowledgeGraph, n_edges: float, seed: int | None = None
) -> KnowledgeGraph:
    """Takes a random sample of edges from a KnowledgeGraph, creates a subgraph, and returns its largest weakly connected component as a new KnowledgeGraph.

    Args:
        kg: Input KnowledgeGraph
        n_edges: If int, the number of edges to sample
                If float between 0 and 1, the fraction of edges to sample
        seed: Random seed for reproducibility

    Returns:
        A new KnowledgeGraph containing the largest weakly connected component

    Raises:
        ValueError: If n_edges is invalid
        TypeError: If input graph is not a KnowledgeGraph

    """
    # Verify input graph type
    if not isinstance(kg, KnowledgeGraph):
        raise TypeError("Input graph must be a NetworkX KnowledgeGraph")

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Get total number of edges (including parallel edges)
    # total_edges = kg.size()  # For KnowledgeGraph, this counts parallel edges
    total_edges = sum(kg.schema.get_edge_type_usage().values())

    # Convert fraction to number if n_edges is float
    if isinstance(n_edges, float):
        if not 0 < n_edges <= 1:
            raise ValueError("If n_edges is a float, it must be between 0 and 1")
        n_edges = int(total_edges * n_edges)

    # Validate n_edges
    if n_edges <= 0 or n_edges > total_edges:
        raise ValueError(f"n_edges must be between 1 and {total_edges}")

    # Method 1: Using reservoir sampling
    def reservoir_sample_edges():
        reservoir = []
        for i, edge in enumerate(kg.edges(keys=True, data=True)):
            if i < n_edges:
                reservoir.append(edge)
            else:
                j = random.randrange(0, i + 1)
                if j < n_edges:
                    reservoir[j] = edge
        return reservoir

    # Method 2: Using random edge generator
    def random_edge_generator():
        edges_seen = set()
        edge_count = 0

        while edge_count < n_edges:
            # Get a random node
            u = random.choice(list(kg.nodes()))
            # If node has any edges
            if kg.degree(u) > 0:
                # Get random neighbor and key
                neighbors = list(kg.adj[u].items())
                if neighbors:
                    v, edges = random.choice(neighbors)
                    k = random.choice(list(edges.keys()))
                    edge_tuple = (u, v, k)

                    if edge_tuple not in edges_seen:
                        edges_seen.add(edge_tuple)
                        edge_count += 1
                        yield (u, v, k, kg.adj[u][v][k])

    print("Running sampling...")
    sampled_edges = reservoir_sample_edges()

    # Create new KnowledgeGraph for the subgraph
    subgraph = KnowledgeGraph()

    # Add sampled edges and their nodes (with attributes) to subgraph
    nodes_to_add = set()
    for u, v, _k, _d in sampled_edges:
        nodes_to_add.add(u)
        nodes_to_add.add(v)

    # Add necessary nodes with their attributes
    for node in nodes_to_add:
        subgraph.add_node(node, **kg.nodes[node])

    # Add sampled edges to subgraph
    for u, v, k, d in sampled_edges:
        subgraph.add_edge(u, v, key=k, **d)

    # Find largest weakly connected component
    components = list(nx.weakly_connected_components(subgraph))

    if not components:  # If no components found
        return KnowledgeGraph()

    # Get the largest component
    largest_component = max(components, key=len)

    # Create new MultiDiGraph with largest component
    result = KnowledgeGraph()
    component_subgraph = subgraph.subgraph(largest_component)

    # Copy all nodes and their attributes from the component
    for node in component_subgraph.nodes():
        result.add_node(node, **component_subgraph.nodes[node])

    # Copy all edges and their attributes from the component
    for u, v, k, data in component_subgraph.edges(keys=True, data=True):
        result.add_edge(u, v, key=k, **data)

    return result


def remove_rare_relations(kg: KnowledgeGraph, n) -> KnowledgeGraph:
    """Remove edges from graph if their 'relation' attribute appears less than n times, then remove isolated nodes.

    Parameters
    ----------
    kg : KnowledgeGraph
        The input graph with 'relation' attributes on edges
    n : int
        The minimum number of occurrences required to keep edges with a specific relation

    Returns
    -------
    KnowledgeGraph
        A new graph with rare relations removed

    """
    # Create a copy of the graph to avoid modifying the original
    kg_copy = kg.copy()

    # Count occurrences of each relation
    relation_counts: Counter[Any] = Counter()
    for _, _, data in kg_copy.edges(data=True):
        if "type" in data:
            relation_counts[data["type"]] += 1

    # Find relations that appear less than n times
    rare_relations = {rel for rel, count in relation_counts.items() if count < n}

    # Remove edges with rare relations
    edges_to_remove = []
    for u, v, key, data in kg_copy.edges(keys=True, data=True):
        if "type" in data and data["type"] in rare_relations:
            edges_to_remove.append((u, v, key))

    kg_copy.remove_edges_from(edges_to_remove)

    # Remove isolated nodes
    kg_copy.remove_nodes_from(list(nx.isolates(kg_copy)))

    return kg_copy


def filter_graph(
    kg: KnowledgeGraph,
    filter_criterion: Callable[[dict], bool] | None = None,
    attr_name: str | None = None,
    attr_value: Any | None = None,
    batch_size: int = 10000,
) -> KnowledgeGraph:
    """Filter a very large KnowledgeGraph using batched processing.

    Args:
        kg: Input KnowledgeGraph
        filter_criterion: Custom filtering function that takes edge data dict
        attr_name: Attribute name to filter by (if not using custom criterion)
        attr_value: Attribute value to filter by (if not using custom criterion)
        batch_size: Number of edges to process in each batch

    Returns:
        Filtered KnowledgeGraph

    Examples:
        >>> # Filter by simple attribute
        >>> filtered = filter_graph(G, attr_name='type', attr_value='A')
        >>>
        >>> # Filter by custom criterion
        >>> filtered = filter_graph(
        ...     kg,
        ...     filter_criterion=lambda d: d.get('weight', 0) > 1.0
        ... )
        >>>
        >>> # Complex filtering
        >>> filtered = filter_graph(
        ...     kg,
        ...     filter_criterion=lambda d: (
        ...         d.get('weight', 0) > 1.0 and
        ...         d.get('type') == 'A'
        ...     )
        ... )

    """
    # Set up the filter criterion
    if filter_criterion is None and attr_name is not None:

        def filter_criterion(d):
            return d.get(attr_name) == attr_value

    if filter_criterion is None:
        raise ValueError("Must provide either filter_criterion or attr_name/attr_value")

    result = KnowledgeGraph()
    edge_buffer = []
    nodes_to_add = set()

    for u, v, _, d in kg.edges(keys=True, data=True):
        if filter_criterion(d):
            edge_buffer.append((u, v, d))
            nodes_to_add.add(u)
            nodes_to_add.add(v)

        if len(edge_buffer) >= batch_size:
            # Add batch of nodes with their attributes
            for node in nodes_to_add:
                if not result.has_node(node):
                    result.add_node(node, **kg.nodes[node])

            # Add batch of edges
            for u_, v_, d_ in edge_buffer:
                result.add_edge(u_, v_, **d_)

            edge_buffer.clear()
            nodes_to_add.clear()

    # Add remaining nodes and edges
    for node in nodes_to_add:
        if not result.has_node(node):
            result.add_node(node, **kg.nodes[node])

    for u, v, d in edge_buffer:
        result.add_edge(u, v, **d)

    return result
