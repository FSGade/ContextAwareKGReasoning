"""Export operations for knowledge graphs."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from knowledge_graph.core.graph import KnowledgeGraph


class GraphExporter:
    """Base class for exporting graphs to various formats."""

    @staticmethod
    def export_json(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph as JSON."""
        data = {
            "nodes": [
                {
                    "id": str(node),
                    "name": node.name,
                    "type": node.type,
                    **graph.nodes[node],
                }
                for node in graph.nodes()
            ],
            "edges": [
                {"source": str(u), "target": str(v), **data}
                for u, v, data in graph.edges(data=True)
            ],
            "schema": {
                "node_types": list(graph.schema.get_node_types()),
                "edge_types": list(graph.schema.get_edge_types()),
                "frozen": graph.schema.frozen,
            },
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def export_csv(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph as CSV files (nodes.csv and edges.csv).

        Parameters
        ----------
        graph : NetworkX graph
            The graph to export
        file_path : Path
            The base path where to save the CSV files
        **kwargs : dict
            Additional export arguments

        Returns
        -------
        None

        """
        import pandas as pd

        base_path = file_path.with_suffix("")

        nodes_data = [
            {
                "id": str(node),
                "name": node.name,
                "type": node.type,
                **graph.nodes[node],
            }
            for node in graph.nodes()
        ]
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(f"{base_path}_nodes.csv", index=False, **kwargs)

        edges_data = [
            {"source": str(u), "target": str(v), **data}
            for u, v, data in graph.edges(data=True)
        ]
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(f"{base_path}_edges.csv", index=False, **kwargs)

    @staticmethod
    def export_gml(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph in GML format.

        Parameters
        ----------
        graph : NetworkX graph
            The graph to export
        file_path : Path
            The path where to save the GML file
        **kwargs : dict
            Additional export arguments

        Returns
        -------
        None

        """
        nx.write_gml(graph, str(file_path))

    @staticmethod
    def export_gexf(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph in GEXF format.

        Parameters
        ----------
        graph : NetworkX graph
            The graph to export
        file_path : Path
            The path where to save the GEXF file
        **kwargs : dict
            Additional export arguments

        Returns
        -------
        None

        """
        nx.write_gexf(graph, str(file_path))

    @staticmethod
    def export_graphml(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph in GraphML format.

        Parameters
        ----------
        graph : NetworkX graph
            The graph to export
        file_path : Path
            The path where to save the GraphML file
        **kwargs : dict
            Additional export arguments

        Returns
        -------
        None

        """
        nx.write_graphml(graph, str(file_path))

    @staticmethod
    def export_pickle(graph: KnowledgeGraph, file_path: Path, **kwargs) -> None:
        """Export the graph in pickle format.

        Parameters
        ----------
        graph : NetworkX graph
            The graph to export
        file_path : Path
            The path where to save the pickle file
        **kwargs : dict
            Additional export arguments

        Returns
        -------
        None

        """
        file_path = Path(file_path)

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                pickle.dump(graph, f)
        except Exception as e:
            raise Exception(f"An error occurred while saving the graph: {e!s}")
