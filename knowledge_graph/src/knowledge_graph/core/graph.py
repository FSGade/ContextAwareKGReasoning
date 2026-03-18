"""Core knowledge graph implementation based on NetworkX's MultiDiGraph.

This module provides a schema-validated knowledge graph implementation,
extending NetworkX's MultiDiGraph class. It implements node and edge type validation
against a schema and maintains unique node identification through name-type pairs.
"""

from __future__ import annotations

import colorsys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, cast

import networkx as nx

from ..io.exporters import GraphExporter
from ..io.importers import GraphImporter
from .schema import KnowledgeGraphSchema

# Define the format type for import/export operations
FormatType = Literal["json", "csv", "gexf", "gml", "graphml", "pickle", "pkl"]


class Entity(NamedTuple):
    """A unique identifier for a node consisting of its name and type."""

    name: str
    type: str

    def __str__(self) -> str:
        """Show entity string representation."""
        return f"{self.name} ({self.type})"

    @staticmethod
    def from_tuple(t: tuple[str, str]) -> Entity:
        """Create an Entity from a (name, type) tuple."""
        if not (
            isinstance(t, tuple)
            and len(t) == 2
            and isinstance(t[0], str)
            and isinstance(t[1], str)
        ):
            raise ValueError(f"Expected tuple[str, str], got {type(t).__name__}: {t}")
        return Entity(name=t[0], type=t[1])


class KnowledgeGraph(nx.MultiDiGraph):
    """A knowledge graph implementation with schema validation.

    This class extends NetworkX's MultiDiGraph to provide:
    - Schema validation for node and edge types
    - Unique node identification through name-type pairs
    - Automatic schema updates for new types (when not frozen)
    """

    SUPPORTED_FORMATS: ClassVar[set[FormatType]] = {
        "json",
        "csv",
        "gexf",
        "gml",
        "graphml",
        "pickle",
        "pkl",
    }

    def __init__(self, schema: KnowledgeGraphSchema | None = None, **attr):
        """Initialize a knowledge graph with an optional schema."""
        super().__init__(**attr)
        self.schema = schema or KnowledgeGraphSchema()

    def __repr__(self):
        return f"{self.__class__!s}({self.name})"

    def _validate_entity(self, entity: Any) -> Entity:
        """Validate and convert input to Entity type."""
        if (
            isinstance(entity, tuple)
            and len(entity) == 2
            and all(isinstance(item, str) for item in entity)
        ):
            return Entity(name=entity[0], type=entity[1])
        if isinstance(entity, Entity):
            return entity
        raise TypeError(
            f"Entity must be of type Union[Entity, tuple[str, str]], "
            f"got {type(entity).__name__}: {entity}"
        )

    def add_node(self, entity: Entity | tuple[str, str], **attr) -> Entity:
        """Add a node to the graph with type validation.

        Parameters
        ----------
        entity : Union[Entity, tuple[str, str]]
            The node to add, either as an Entity or a (name, type) tuple
        **attr : dict
            Additional node attributes

        Returns
        -------
        Entity
            The added node's entity

        """
        entity = self._validate_entity(entity)

        if entity not in self:  # Only register if node doesn't exist
            self.schema.register_node_usage(entity.type)

        attr["name"] = entity.name
        attr["type"] = entity.type
        super().add_node(entity, **attr)
        return entity

    def add_nodes_from(
        self,
        nodes: Iterable[Entity | tuple[str, str] | tuple[Entity, dict[Any, Any]]],
        **attr,
    ) -> None:
        """Add multiple nodes to the graph.

        Parameters
        ----------
        nodes : iterable of Union[Entity, tuple[str, str]]
            The nodes to add
        **attr : dict
            Additional node attributes

        """
        for node in nodes:
            if isinstance(node[1], dict):
                self.add_node(node[0], **node[1])
            else:
                self.add_node(node, **attr)

    def remove_node(self, node: Entity | tuple[str, str]) -> None:
        """Remove a node and all its edges from the graph.

        Parameters
        ----------
        node : Union[Entity, tuple[str, str]]
            The node to remove, either as an Entity or a (name, type) tuple

        """
        node_entity = self._validate_entity(node)
        if node_entity not in self:
            raise nx.NetworkXError(f"Node {node_entity} not in graph")

        # Remove all connected edges first
        edges_to_remove = [
            (u, v, k, d)
            for u, v, k, d in self.edges(data=True, keys=True)
            if u == node_entity or v == node_entity
        ]
        for u, v, k, d in edges_to_remove:
            self.remove_edge(u, v, k)

        # Unregister node from schema
        self.schema.unregister_node_usage(node_entity.type)

        # Remove node from graph
        super().remove_node(node_entity)

    def add_edge(
        self,
        u: Entity | tuple[str, str],
        v: Entity | tuple[str, str],
        key: Any | None = None,
        **attr,
    ) -> tuple[Entity, Entity]:
        """Add an edge to the graph with schema validation.

        Parameters
        ----------
        u : Union[Entity, tuple[str, str]]
            The source node
        v : Union[Entity, tuple[str, str]]
            The target node
        key : Any, optional
            Edge key
        **attr : dict
            Edge attributes

        Returns
        -------
        tuple[Entity, Entity]
            The source and target entities of the added edge

        """
        if "type" not in attr:
            raise ValueError("Edge type must be specified in attributes")

        edge_type = attr["type"]
        u_entity = self._validate_entity(u)
        v_entity = self._validate_entity(v)

        # Add nodes if they don't exist
        if u_entity not in self:
            self.add_node(u_entity)
        if v_entity not in self:
            self.add_node(v_entity)

        if not self.schema.is_valid_edge_type(edge_type):
            if self.schema.frozen:
                raise ValueError(f"Invalid edge type: {edge_type}")
            self.schema.add_edge_type(edge_type)

        self.schema.register_edge_usage(u_entity.type, edge_type, v_entity.type)
        super().add_edge(u_entity, v_entity, key=key, **attr)
        return u_entity, v_entity

    def add_edges_from(self, ebunch: Iterable[tuple], **attr) -> list:
        """Add edges from an iterable of edge tuples.

        Parameters
        ----------
        ebunch : iterable of tuples
            The edges to add. Each tuple should be (u, v[, key][, dict])
        **attr : dict
            Default attributes for all edges

        Returns
        -------
        list
            List of added edge keys

        """
        added_keys = []
        for e in ebunch:
            if len(e) == 4:
                u, v, key, dd = e
            elif len(e) == 3:
                u, v, dd = e
                key = None
            elif len(e) == 2:
                u, v = e
                dd = {}
                key = None
            else:
                raise ValueError(f"Edge tuple {e} must be a 2-, 3-, or 4-tuple")

            edge_attr = attr.copy()
            if isinstance(dd, dict):
                edge_attr.update(dd)
            else:
                key = dd

            key = self.add_edge(u, v, key, **edge_attr)
            added_keys.append(key)

        return added_keys

    def remove_edge(
        self,
        u: Entity | tuple[str, str],
        v: Entity | tuple[str, str],
        key: Any | None = None,
    ) -> None:
        """Remove an edge from the graph.

        Parameters
        ----------
        u : Union[Entity, tuple[str, str]]
            Source node
        v : Union[Entity, tuple[str, str]]
            Target node
        key : Any, optional
            Edge key

        """
        u_entity = self._validate_entity(u)
        v_entity = self._validate_entity(v)

        edge_data = self.get_edge_data(u_entity, v_entity, key)
        if key is None:
            for _key, _edge_data in edge_data.items():
                edge_type = _edge_data.get("type")
                self.schema.unregister_edge_usage(
                    u_entity.type, edge_type, v_entity.type
                )
        else:
            edge_type = edge_data.get("type")
            self.schema.unregister_edge_usage(u_entity.type, edge_type, v_entity.type)

        super().remove_edge(u_entity, v_entity, key)

    def remove_edges_from(
        self,
        ebunch: Iterable[
            tuple[Entity | tuple[str, str], Entity | tuple[str, str]]
            | tuple[Entity | tuple[str, str], Entity | tuple[str, str], Any]
        ],
    ) -> None:
        """Remove edges from an iterable of edge tuples.

        Parameters
        ----------
        ebunch : iterable of tuples
            The edges to remove. Each tuple should be either:
            - (u, v) where u and v are either Entities or (name, type) tuples
            - (u, v, key) where u and v are either Entities or (name, type) tuples

        """
        for edge in ebunch:
            if len(edge) == 2:
                self.remove_edge(edge[0], edge[1])
            elif len(edge) == 3:
                self.remove_edge(edge[0], edge[1], edge[2])
            else:
                raise ValueError("Edge tuple must be 2-tuple or 3-tuple")

    def add_typed_node(self, name: str, type: str, **attr) -> Entity:
        """Convenience method to add a node using name and type."""
        return self.add_node((name, type), **attr)

    def add_typed_edge(
        self,
        head_name: str,
        head_type: str,
        tail_name: str,
        tail_type: str,
        edge_type: str,
        **attr,
    ) -> tuple[Entity, Entity]:
        """Convenience method to add an edge using names and types."""
        attr["type"] = edge_type
        return self.add_edge((head_name, head_type), (tail_name, tail_type), **attr)

    def get_nodes_by_type(self, node_type: str) -> set[Entity]:
        """Get all nodes of a specific type.

        Parameters
        ----------
        node_type : str
            The type of nodes to get

        Returns
        -------
        set[Entity]
            Set of nodes of the specified type

        """
        return {node for node in self.nodes() if node.type == node_type}

    def get_nodes_by_name(self, name: str) -> set[Entity]:
        """Get all nodes with a specific name.

        Parameters
        ----------
        name : str
            The name to search for

        Returns
        -------
        set[Entity]
            Set of nodes with the specified name

        """
        return {node for node in self.nodes() if node.name == name}

    def get_edges_by_type(self, edge_type: str) -> set[tuple[Entity, Entity]]:
        """Get all edges of a specific type.

        Parameters
        ----------
        edge_type : str
            The type of edges to get

        Returns
        -------
        set[tuple[Entity, Entity]]
            Set of (source, target) pairs for edges of the specified type

        """
        return {
            (u, v)
            for u, v, attr in self.edges(data=True)
            if attr.get("type") == edge_type
        }

    def get_node(self, name: str, type: str) -> Entity | None:
        """Get a specific node by name and type.

        Parameters
        ----------
        name : str
            Node name
        type : str
            Node type

        Returns
        -------
        Entity or None
            The node entity if it exists, None otherwise

        """
        entity = Entity(name=name, type=type)
        return entity if entity in self else None

    def visualize(
        self,
        output_file: str,
        figsize: tuple[int, int] = (12, 8),
        node_size: int = 2000,
        font_size: int = 8,
        edge_font_size: int = 8,
        title: str | None = None,
        **kwargs,
    ) -> None:
        """Visualize the knowledge graph.

        Parameters
        ----------
        output_file : str
            Path where the visualization should be saved.
            Supported formats: .png, .jpg, .pdf, .svg, .html
        figsize : tuple[int, int], default=(12, 8)
            Figure size for static visualizations
        node_size : int, default=2000
            Size of nodes in the visualization
        font_size : int, default=8
            Font size for node labels
        edge_font_size : int, default=8
            Font size for edge labels
        title : str, optional
            Title for the visualization
        **kwargs : dict
            Additional arguments for specific visualizer functions

        """
        file_path = Path(output_file)
        suffix = file_path.suffix.lower()

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".svg"}:
            self._visualize_static(
                file_path,
                figsize=figsize,
                node_size=node_size,
                font_size=font_size,
                edge_font_size=edge_font_size,
                title=title,
                **kwargs,
            )
        elif suffix == ".html":
            self._visualize_interactive(file_path, title=title, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: .png, .jpg, .pdf, .svg, .html"
            )

    def _visualize_static(
        self,
        file_path: Path,
        figsize: tuple[int, int],
        node_size: int,
        font_size: int,
        edge_font_size: int,
        title: str | None = None,
        **kwargs,
    ) -> None:
        """Create a static visualization using matplotlib."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        # Generate colors for different node types
        node_types = {node.type for node in self.nodes()}
        colors = self._generate_colors(len(node_types))
        color_map = dict(zip(sorted(node_types), colors))

        # Create node colors list
        node_colors = [color_map[node.type] for node in self.nodes()]

        # Create layout
        pos = nx.spring_layout(self, k=1, iterations=50)

        # Draw nodes
        nx.draw_networkx_nodes(self, pos, node_color=node_colors, node_size=node_size)

        # Draw node labels
        nx.draw_networkx_labels(
            self,
            pos,
            labels={node: f"{node.name}\n({node.type})" for node in self.nodes()},
            font_size=font_size,
        )

        # Draw edges
        nx.draw_networkx_edges(self, pos)

        # Draw edge labels
        edge_labels = {
            (u, v): data.get("type", "") for u, v, data in self.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self, pos, edge_labels=edge_labels, font_size=edge_font_size
        )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=node_type,
                markersize=10,
            )
            for node_type, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

    def _visualize_interactive(
        self, file_path: Path, title: str | None = None, **kwargs
    ) -> None:
        """Create an interactive visualization using pyvis."""
        from pyvis.network import Network

        # Create a copy of the graph for visualization
        mdg = nx.MultiDiGraph(self)

        # Set group attribute by node type for coloring
        group_dict = {
            node_type: i
            for i, node_type in enumerate({node.type for node in self.nodes()})
        }

        for node in mdg.nodes():
            mdg.nodes[node]["group"] = group_dict[node.type]

        # Convert nodes to string representation
        nx.relabel_nodes(mdg, lambda x: str(x), copy=False)

        # Set edge labels
        for u, v, key, data in mdg.edges(data=True, keys=True):
            mdg[u][v][key]["label"] = data["type"]

            # Avoid error when overriding "source" and "to" args in pyvis add_edge
            if "source" in mdg[u][v][key]:
                mdg[u][v][key]["_source"] = mdg[u][v][key]["source"]
                del mdg[u][v][key]["source"]
            if "tp" in mdg[u][v][key]:
                mdg[u][v][key]["_to"] = mdg[u][v][key]["to"]
                del mdg[u][v][key]["to"]

        # Create and configure network
        net = Network(
            directed=True,
            select_menu=True,
            filter_menu=True,
            cdn_resources="in_line",
        )
        net.show_buttons()
        net.from_nx(mdg, show_edge_weights=False)
        net.toggle_physics(True)

        # Save the visualization
        net.show(str(file_path), notebook=False)

    @staticmethod
    def _generate_colors(n: int) -> list:
        """Generate n visually distinct colors.

        Parameters
        ----------
        n : int
            Number of colors to generate

        Returns
        -------
        list
            List of RGB colors

        """
        return [colorsys.hsv_to_rgb(i / n, 0.7, 0.9) for i in range(n)]

    def export_graph(
        self,
        file_path: str | Path,
        file_format: FormatType | None = None,
        **kwargs,
    ) -> None:
        """Export the knowledge graph to various formats.

        Parameters
        ----------
        file_path : str
            Path where the export should be saved
        file_format : FormatType, optional
            Format to export to. If None, inferred from file extension.
            Supported formats: json, csv, gexf, gml, graphml, pickle
        **kwargs : dict
            Additional arguments for specific export functions

        """
        path = Path(file_path)

        # Determine format
        if file_format is None:
            ext = path.suffix.lower().lstrip(".")
            if ext == "gz":  # Handle compressed files
                ext = path.stem.split(".")[-1].lower()

            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format: {ext}. "
                    f"Supported formats: {sorted(self.SUPPORTED_FORMATS)}"
                )
            file_format = cast("FormatType", ext)

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Export using appropriate method
        export_methods = {
            "json": GraphExporter.export_json,
            "csv": GraphExporter.export_csv,
            "gexf": GraphExporter.export_gexf,
            "gml": GraphExporter.export_gml,
            "graphml": GraphExporter.export_graphml,
            "pickle": GraphExporter.export_pickle,
            "pkl": GraphExporter.export_pickle,  # pickle synonym
        }

        if file_format not in export_methods:
            raise ValueError(
                f"Unsupported export format: {file_format}. "
                f"Supported formats: {list(export_methods.keys())}"
            )

        export_methods[file_format](self, path, **kwargs)

    @classmethod
    def import_graph(
        cls, file_path: str, file_format: FormatType | None = None, **kwargs
    ) -> KnowledgeGraph:
        """Import a graph from various formats.

        Parameters
        ----------
        file_path : str
            Path to the input file
        file_format : FormatType, optional
            Format to import from. If None, inferred from file extension
        **kwargs : dict
            Additional arguments for specific import functions

        Returns
        -------
        KnowledgeGraph
            The imported knowledge graph

        """
        path = Path(file_path)

        # Determine format
        if file_format is None:
            ext = path.suffix.lower().lstrip(".")
            if ext == "gz":
                ext = path.stem.split(".")[-1].lower()

            if ext not in cls.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported format: {ext}. "
                    f"Supported formats: {sorted(cls.SUPPORTED_FORMATS)}"
                )
            file_format = cast("FormatType", ext)

        # Import using appropriate method
        import_methods = {
            "json": GraphImporter.import_json,
            "csv": GraphImporter.import_csv,
            "gexf": GraphImporter.import_gexf,
            "gml": GraphImporter.import_gml,
            "graphml": GraphImporter.import_graphml,
            "pickle": GraphImporter.import_pickle,
            "pkl": GraphImporter.import_pickle,  # pickle synonym
        }

        if file_format not in import_methods:
            raise ValueError(
                f"Unsupported import format: {file_format}. "
                f"Supported formats: {list(import_methods.keys())}"
            )

        return import_methods[file_format](path, cls, **kwargs)

    # Alias for British English spelling
    visualise = visualize
