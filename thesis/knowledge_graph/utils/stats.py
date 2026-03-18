"""Statistics utilities for knowledge graphs."""

from __future__ import annotations

from collections import Counter, defaultdict

from knowledge_graph.core.graph import KnowledgeGraph


def print_kg_stats(graph: KnowledgeGraph):
    """
    Print detailed statistics for the given knowledge graph.

    Parameters:
    -----------
    graph : KnowledgeGraph
        The graph to analyze
    """
    _print_basic_stats(graph)
    _print_schema_stats(graph)
    _print_degree_distribution(graph)
    _print_node_type_distribution(graph)
    _print_edge_type_distribution(graph)


def _print_basic_stats(graph: KnowledgeGraph):
    """Print basic graph statistics."""
    print("Basic Statistics:")
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    print()


def _print_schema_stats(graph: KnowledgeGraph):
    """Print schema information."""
    print("Schema Information:")
    print(
        f"Schema status: {'Frozen' if graph.schema.frozen else 'Modifiable'}"
    )
    print(f"Defined node types: {len(graph.schema.get_node_types())}")
    print(f"Defined edge types: {len(graph.schema.get_edge_types())}")
    print(f"Used edge patterns: {len(graph.schema.get_edge_type_usage())}")
    print()


def _print_degree_distribution(graph: KnowledgeGraph):
    """Print degree distribution with examples for low-degree nodes."""
    print("Degree Distribution (top 10):")

    # Initialize data structures
    degree_counts: Counter[int] = Counter()
    degree_examples = defaultdict(list)

    # Single pass through the graph
    for node, degree in graph.degree():
        degree_counts[degree] += 1
        # Only store examples for degrees that might have 1 or 2 nodes
        if degree_counts[degree] <= 2:
            degree_examples[degree].append(str(node))

    # Print results
    for degree in sorted(degree_counts)[-10:]:
        count = degree_counts[degree]
        if count in (1, 2):
            examples = " and ".join(degree_examples[degree])
            print(
                f"  Degree {degree}: {count} node{'s' if count else ''} ({examples})"
            )
        else:
            print(f"  Degree {degree}: {count} node{'s' if count else ''}")
    print()


def _print_node_type_distribution(graph: KnowledgeGraph):
    """Print distribution of node types."""
    print("Node Type Distribution:")
    node_types = Counter(node.type for node in graph.nodes())

    for node_type, count in node_types.most_common():
        print(f"  {node_type}: {count} node{'s' if count else ''}")
        # Show examples for types with few instances
        if count <= 3:
            examples = [
                node.name for node in graph.nodes() if node.type == node_type
            ]
            print(f"    Examples: {', '.join(examples)}")
    print()


def _print_edge_type_distribution(graph: KnowledgeGraph):
    """Print distribution of edge types and their usage patterns."""
    print("Edge Type Distribution:")
    edge_types = Counter(
        data.get("type") for _, _, data in graph.edges(data=True)
    )

    for edge_type, count in edge_types.most_common():
        print(f"  {edge_type}: {count} edge{'s' if count else ''}")
        if count <= 3:
            # Show examples for edge types with few instances
            examples = []
            for u, v, data in graph.edges(data=True):
                if data.get("type") == edge_type:
                    examples.append(f"{u.name} → {v.name}")
            print(f"    Examples: {', '.join(examples)}")
    print()
