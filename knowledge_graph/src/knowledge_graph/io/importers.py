"""Import operations for knowledge graphs."""

from __future__ import annotations

import gc
import json
import pickle
from pathlib import Path

import networkx as nx

from knowledge_graph.core.schema import KnowledgeGraphSchema


class GraphImporter:
    """Base class for importing graphs from various formats."""

    @staticmethod
    def import_json(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Import graph from JSON format."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Create schema from data
        schema = KnowledgeGraphSchema(
            node_types=set(data.get("schema", {}).get("node_types", [])),
            edge_types=set(data.get("schema", {}).get("edge_types", [])),
            frozen=data.get("schema", {}).get("frozen", False),
        )

        # Create new knowledge graph
        kg = kg_class(schema=schema)

        # Add nodes and edges
        for node_data in data.get("nodes", []):
            node_name = node_data.pop("name")
            node_type = node_data.pop("type")
            kg.add_typed_node(node_name, node_type, **node_data)

        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            edge_type = edge_data.pop("type", None)
            if edge_type:
                source_node = kg.get_node_by_id(source)
                target_node = kg.get_node_by_id(target)
                if source_node and target_node:
                    kg.add_typed_edge(
                        source_node.name,
                        source_node.type,
                        target_node.name,
                        target_node.type,
                        edge_type,
                        **edge_data,
                    )

        return kg

    @staticmethod
    def import_csv(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Import graph from CSV files."""
        import pandas as pd

        base_path = file_path.with_suffix("")

        # Read CSV files
        nodes_df = pd.read_csv(f"{base_path}_nodes.csv")
        edges_df = pd.read_csv(f"{base_path}_edges.csv")

        # Create schema
        node_types = set(nodes_df["type"].unique())
        edge_types = set(edges_df["type"].unique())

        schema = KnowledgeGraphSchema(
            node_types=node_types,
            edge_types=edge_types,
            frozen=kwargs.get("frozen", False),
        )

        # Create new knowledge graph
        kg = kg_class(schema=schema)

        # Add nodes and edges
        for _, row in nodes_df.iterrows():
            node_data = row.to_dict()
            node_name = node_data.pop("name")
            node_type = node_data.pop("type")
            kg.add_typed_node(node_name, node_type, **node_data)

        for _, row in edges_df.iterrows():
            edge_data = row.to_dict()
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            edge_type = edge_data.pop("type")
            source_node = kg.get_node_by_id(source)
            target_node = kg.get_node_by_id(target)
            if source_node and target_node:
                kg.add_typed_edge(
                    source_node.name,
                    source_node.type,
                    target_node.name,
                    target_node.type,
                    edge_type,
                    **edge_data,
                )

        return kg

    @staticmethod
    def import_pickle(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Load a KnowledgeGraph from a binary pickle file.

        Parameters
        ----------
        file_path : str or Path
            Path to the binary pickled graph file

        Returns
        -------
        KnowledgeGraph
            The loaded knowledge graph object

        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"No file found at {file_path}")

        if file_path.stat().st_size == 0:
            raise ValueError(f"File at {file_path} is empty")

        try:
            gc.disable()
            with open(file_path, "rb") as f:
                graph = pickle.load(f)

            if not isinstance(graph, kg_class):
                raise ValueError(
                    f"Loaded object is not a {kg_class}. Got {type(graph)} instead."
                )

            return graph
        except pickle.UnpicklingError as exc:
            raise pickle.UnpicklingError(
                f"Error unpickling file at {file_path}: {exc!s}"
            )
        except Exception as exc:
            raise Exception(f"An error occurred while loading the graph: {exc!s}")
        finally:
            gc.enable()

    @staticmethod
    def import_gexf(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Import graph from NetworkX supported formats."""
        # Import using NetworkX function
        graph = nx.read_gexf(file_path)

        # Create schema from imported graph
        node_types = {data.get("type", "Entity") for _, data in graph.nodes(data=True)}
        edge_types = {
            data.get("type", "RELATES_TO") for _, _, data in graph.edges(data=True)
        }

        schema = KnowledgeGraphSchema(
            node_types=node_types,
            edge_types=edge_types,
            frozen=kwargs.get("frozen", False),
        )

        # Create new knowledge graph
        kg = kg_class(schema=schema)

        # Copy nodes and edges
        kg.add_nodes_from(graph.nodes(data=True))
        kg.add_edges_from(graph.edges(data=True))

        return kg

    @staticmethod
    def import_gml(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Import graph from NetworkX supported formats."""
        # Import using NetworkX function
        graph = nx.read_gml(file_path)

        # Create schema from imported graph
        node_types = {data.get("type", "Entity") for _, data in graph.nodes(data=True)}
        edge_types = {
            data.get("type", "RELATES_TO") for _, _, data in graph.edges(data=True)
        }

        schema = KnowledgeGraphSchema(
            node_types=node_types,
            edge_types=edge_types,
            frozen=kwargs.get("frozen", False),
        )

        # Create new knowledge graph
        kg = kg_class(schema=schema)

        # Copy nodes and edges
        kg.add_nodes_from(graph.nodes(data=True))
        kg.add_edges_from(graph.edges(data=True))

        return kg

    @staticmethod
    def import_graphml(
        file_path: Path, kg_class: type[nx.MultiDiGraph], **kwargs
    ) -> nx.MultiDiGraph:
        """Import graph from NetworkX supported formats."""
        # Import using NetworkX function
        graph = nx.read_gexf(file_path)

        # Create schema from imported graph
        node_types = {data.get("type", "Entity") for _, data in graph.nodes(data=True)}
        edge_types = {
            data.get("type", "RELATES_TO") for _, _, data in graph.edges(data=True)
        }

        schema = KnowledgeGraphSchema(
            node_types=node_types,
            edge_types=edge_types,
            frozen=kwargs.get("frozen", False),
        )

        # Create new knowledge graph
        kg = kg_class(schema=schema)

        # Copy nodes and edges
        kg.add_nodes_from(graph.nodes(data=True))
        kg.add_edges_from(graph.edges(data=True))

        return kg
