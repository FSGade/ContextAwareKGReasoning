#!/usr/bin/env python3
"""
RQ2 Preprocessing — Clean the graph for tissue-aware analysis.

Removes nodes of excluded types (DNAMutation, Species) and their edges,
then saves the cleaned graph with metadata.

Usage:
    python preprocess.py --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph

from utils import load_config


def preprocess_graph(config: dict) -> tuple:
    """Load graph, remove excluded node types, save cleaned version."""
    input_path = config['paths']['input_graph']
    output_dir = Path(config['paths']['output_dir'])
    exclude_types = set(config.get('exclude_node_types', ['DNAMutation', 'Species']))

    preprocessed_dir = output_dir / 'preprocessed'
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from: {input_path}")
    kg = KnowledgeGraph.import_graph(input_path)

    original_nodes = kg.number_of_nodes()
    original_edges = kg.number_of_edges()
    print(f"Loaded graph: {original_nodes:,} nodes, {original_edges:,} edges")

    # Count node types
    node_types = Counter(
        kg.nodes[n].get('type', 'Unknown') for n in kg.nodes()
    )
    print(f"\nNode types in graph:")
    for ntype, count in node_types.most_common(20):
        marker = " <- WILL REMOVE" if ntype in exclude_types else ""
        print(f"  {ntype}: {count:,}{marker}")

    # Remove excluded nodes (edges removed automatically by networkx)
    nodes_to_remove = [
        n for n in kg.nodes()
        if kg.nodes[n].get('type', 'Unknown') in exclude_types
    ]
    removed_by_type = Counter(
        kg.nodes[n].get('type', 'Unknown') for n in nodes_to_remove
    )

    print(f"\nRemoving {len(nodes_to_remove):,} nodes:")
    for ntype, count in removed_by_type.items():
        print(f"  {ntype}: {count:,}")

    kg.remove_nodes_from(nodes_to_remove)

    final_nodes = kg.number_of_nodes()
    final_edges = kg.number_of_edges()
    print(f"\nAfter removal: {final_nodes:,} nodes, {final_edges:,} edges")

    # Summarise detailed_tissue annotation coverage
    edges_with_tissue = sum(
        1 for _, _, d in kg.edges(data=True)
        if d.get('detailed_tissue') is not None
        and str(d['detailed_tissue']).lower() != 'not specified'
    )
    print(f"Edges with detailed_tissue: {edges_with_tissue:,} / {final_edges:,} "
          f"({100 * edges_with_tissue / final_edges:.1f}%)")

    # Save cleaned graph
    output_path = preprocessed_dir / 'cleaned_graph.pkl'
    print(f"\nSaving cleaned graph to: {output_path}")
    kg.export_graph(str(output_path))

    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'input_graph': input_path,
        'output_graph': str(output_path),
        'original_nodes': original_nodes,
        'original_edges': original_edges,
        'removed_node_types': list(exclude_types),
        'removed_nodes_by_type': dict(removed_by_type),
        'final_nodes': final_nodes,
        'final_edges': final_edges,
        'edges_with_detailed_tissue': edges_with_tissue,
    }
    metadata_path = preprocessed_dir / 'preprocess_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    print("\n✓ Preprocessing complete!")
    return kg, metadata


def main():
    parser = argparse.ArgumentParser(description='Preprocess graph for RQ2 analysis')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    preprocess_graph(config)


if __name__ == '__main__':
    main()