"""Schema definition and validation for knowledge graphs.

This module provides the KnowledgeGraphSchema class, which defines and enforces
the structure of a knowledge graph by specifying allowed node types and edge types.
The schema tracks the actual usage patterns of edges between different node types.
"""

from __future__ import annotations

import itertools as it
from collections import defaultdict
from typing import NamedTuple

import networkx as nx


class EdgeTypeUsage(NamedTuple):
    """Records the usage of an edge type between specific node types."""

    source_type: str
    edge_type: str
    target_type: str


class KnowledgeGraphSchema:
    """A schema class for defining allowed node and edge types in a knowledge graph.

    The schema maintains sets of allowed edge types and tracks the actual usage
    patterns of nodes and edges in the associated graph through counters.

    Parameters
    ----------
    node_types : Set[str], optional
        Initial set of allowed node types
    edge_types : Set[str], optional
        Initial set of allowed edge types
    frozen : bool, default=False
        If True, the schema cannot be modified after creation

    Attributes
    ----------
    frozen : bool
        Whether the schema is frozen (immutable) or not

    """

    def __init__(
        self,
        node_types: set[str] | None = None,
        edge_types: set[str] | None = None,
        frozen: bool = False,
    ) -> None:
        """Initialise the KG schema."""
        self._frozen: bool = frozen
        self._node_type_count: defaultdict[str, int] = defaultdict(int)
        self._edge_type_usage_count: defaultdict[EdgeTypeUsage, int] = defaultdict(int)
        self._allowed_edge_types: set[str] = set()

        # Initialize with provided types
        if node_types:
            for node_type in node_types:
                self.add_node_type(node_type)
        if edge_types:
            for edge_type in edge_types:
                self.add_edge_type(edge_type)

    def __str__(self) -> str:
        """Returns a human-readable string representation of the schema."""
        lines = [
            "Knowledge Graph Schema:",
            f"  Status: {'Frozen' if self._frozen else 'Modifiable'}",
            "  Node Types:",
        ]

        # Add node types with counts
        for ntype in sorted(self._node_type_count):
            lines.append(f"    - {ntype} (count: {self._node_type_count[ntype]})")

        # Add edge types
        lines.append("  Edge Types:")
        for etype in sorted(self._allowed_edge_types):
            lines.append(f"    - {etype}")

        # Add usage patterns
        lines.append("  Used Patterns:")
        usage_by_source = self._group_usage_by_source()
        for source_type in sorted(usage_by_source):
            lines.append(f"    From {source_type}:")
            for usage, count in sorted(
                usage_by_source[source_type],
                key=lambda x: (x[0].edge_type, x[0].target_type),
            ):
                lines.append(
                    f"      - {usage.edge_type} → {usage.target_type} (count: {count})"
                )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Returns a string that could be used to reconstruct the schema."""
        return (
            f"KnowledgeGraphSchema("
            f"node_types={set(self._node_type_count.keys())!r}, "
            f"edge_types={self._allowed_edge_types!r}, "
            f"frozen={self._frozen})"
        )

    def _group_usage_by_source(
        self,
    ) -> dict[str, list[tuple[EdgeTypeUsage, int]]]:
        """Group edge type usage by source type."""
        usage_by_source: dict[str, list[tuple[EdgeTypeUsage, int]]] = defaultdict(list)
        for usage, count in self._edge_type_usage_count.items():
            usage_by_source[usage.source_type].append((usage, count))
        return usage_by_source

    @property
    def frozen(self) -> bool:
        """Whether the schema is immutable or not."""
        return self._frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        """Set the frozen state of the schema."""
        self._frozen = value

    def add_node_type(self, node_type: str) -> None:
        """Add a new node type to the schema."""
        if self._frozen:
            raise ValueError(f"Cannot add node type '{node_type}': schema is frozen")
        self._node_type_count[node_type] = 0

    def add_edge_type(self, edge_type: str) -> None:
        """Add a new edge type to the schema."""
        if self._frozen:
            raise ValueError(f"Cannot add edge type '{edge_type}': schema is frozen")
        self._allowed_edge_types.add(edge_type)

    def remove_node_type(self, node_type: str) -> None:
        """Remove a node type from the schema if it's not in use."""
        if self._frozen:
            raise ValueError(f"Cannot remove node type '{node_type}': schema is frozen")
        if self._node_type_count[node_type] > 0:
            raise ValueError(f"Cannot remove node type '{node_type}': still in use")
        del self._node_type_count[node_type]

    def remove_edge_type(self, edge_type: str) -> None:
        """Remove an edge type from the schema if it's not in use."""
        if self._frozen:
            raise ValueError(f"Cannot remove edge type '{edge_type}': schema is frozen")
        if any(
            usage.edge_type == edge_type and count > 0
            for usage, count in self._edge_type_usage_count.items()
        ):
            raise ValueError(f"Cannot remove edge type '{edge_type}': still in use")
        self._allowed_edge_types.remove(edge_type)

    def is_valid_node_type(self, node_type: str) -> bool:
        """Check if a node type is valid according to the schema."""
        return node_type in self._node_type_count

    def is_valid_edge_type(self, edge_type: str) -> bool:
        """Check if an edge type is valid according to the schema."""
        return edge_type in self._allowed_edge_types

    def register_node_usage(self, node_type: str) -> None:
        """Register the usage of a node type."""
        if not self.is_valid_node_type(node_type):
            if self._frozen:
                raise ValueError(f"Invalid node type: {node_type}")
            self.add_node_type(node_type)
        self._node_type_count[node_type] += 1

    def unregister_node_usage(self, node_type: str) -> None:
        """Unregister the usage of a node type."""
        if node_type in self._node_type_count:
            self._node_type_count[node_type] -= 1
            if self._node_type_count[node_type] == 0 and not self._frozen:
                del self._node_type_count[node_type]
                self.remove_node_type(node_type)

    def register_edge_usage(
        self, source_type: str, edge_type: str, target_type: str
    ) -> None:
        """Register the usage of an edge type between specific node types."""
        usage = EdgeTypeUsage(source_type, edge_type, target_type)
        self._edge_type_usage_count[usage] += 1

    def unregister_edge_usage(
        self, source_type: str, edge_type: str, target_type: str
    ) -> None:
        """Unregister the usage of an edge type between specific node types."""
        usage = EdgeTypeUsage(source_type, edge_type, target_type)
        if usage in self._edge_type_usage_count:
            self._edge_type_usage_count[usage] -= 1
            if self._edge_type_usage_count[usage] == 0 and not self._frozen:
                del self._edge_type_usage_count[usage]
                if not any(
                    usage.edge_type == edge_type
                    for usage, count in self._edge_type_usage_count.items()
                ):
                    self.remove_edge_type(edge_type)

    def get_node_types(self, with_count=False) -> defaultdict[str, int] | set[str]:
        """Get node types."""
        if with_count:
            return self._node_type_count
        return set(self._node_type_count.keys())

    def get_edge_types(self) -> set:
        """Get edge types."""
        return self._allowed_edge_types

    def get_edge_type_usage(self) -> defaultdict[EdgeTypeUsage, int]:
        """Get edge type usage."""
        return self._edge_type_usage_count

    def to_graph(self) -> nx.MultiDiGraph:
        """Convert the schema to a networkx MultiDiGraph showing used connections."""
        schema_graph = nx.MultiDiGraph()

        # Add nodes for all used node types
        for node_type in self._node_type_count:
            schema_graph.add_node(node_type, type="NodeType")

        # Add edges for all used patterns
        for usage in self._edge_type_usage_count:
            schema_graph.add_edge(
                usage.source_type,
                usage.target_type,
                type=usage.edge_type,
                key=usage.edge_type,
            )

        return schema_graph

    def visualize(
        self,
        output_file: str,
        figsize: tuple = (12, 8),
        node_size: int = 2000,
        font_size: int = 10,
        edge_font_size: int = 8,
        title: str = "Knowledge Graph Schema",
    ) -> None:
        """Visualize the schema and save it as a PNG file.

        Parameters
        ----------
        output_file : str
            Path where the PNG file should be saved
        figsize : tuple
            Figure size as (width, height)
        node_size : int
            Size of nodes in the visualization
        font_size : int
            Font size for node labels
        edge_font_size : int
            Font size for edge labels
        title : str
            Title for the visualization

        """
        import matplotlib.pyplot as plt

        kg = self.to_graph()
        if len(kg.nodes()) == 0:
            print(
                "Warning: No nodes to visualize. The schema has no registered usage patterns."
            )
            return

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(kg, k=1, iterations=50)

        # Draw the graph
        arrowstyle = "-|>,head_width=0.2"
        connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

        nx.draw_networkx_nodes(kg, pos, node_size=node_size)
        nx.draw_networkx_labels(
            kg,
            pos,
            labels={node: node for node in kg.nodes()},
            font_size=font_size,
        )
        nx.draw_networkx_edges(
            kg, pos, arrowstyle=arrowstyle, connectionstyle=connectionstyle
        )

        # Add edge labels
        edge_labels = {(u, v): d.get("type", "") for u, v, d in kg.edges(data=True)}
        nx.draw_networkx_edge_labels(
            kg,
            pos,
            edge_labels=edge_labels,
            font_size=edge_font_size,
            connectionstyle=connectionstyle,
        )

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        plt.close()

    # Alias for British English spelling
    visualise = visualize
