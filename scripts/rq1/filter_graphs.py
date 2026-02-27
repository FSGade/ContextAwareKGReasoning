#!/usr/bin/env python3
"""
RQ1 Step 1: Filter normalized graph into tissue-specific subgraphs.

This script:
1. Loads the normalized graph
2. Expands undirected/unknown direction edges bidirectionally
3. Creates 4 tissue-filtered subgraphs (baseline, adipose, nonadipose, liver)
4. Saves filtered graphs and metadata

Usage:
    python filter_graphs.py --config config.yaml
    python filter_graphs.py --input-graph /path/to/graph.pkl --output-dir /path/to/output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm

# Add parent to path for knowledge_graph import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats
from knowledge_graph.utils.filtering import filter_graph


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def analyze_direction_distribution(kg) -> dict:
    """Analyze the distribution of edge directions in the graph."""
    direction_counts = defaultdict(int)
    
    for u, v, data in kg.edges(data=True):
        direction = data.get('direction')
        if direction == '1' or direction == 1:
            direction_counts['directed'] += 1
        elif direction == '0' or direction == 0:
            direction_counts['undirected'] += 1
        else:
            direction_counts['unknown'] += 1
    
    total = sum(direction_counts.values())
    
    print("\nEdge Direction Distribution:")
    print(f"  Directed (direction=1): {direction_counts['directed']:,} ({100*direction_counts['directed']/total:.1f}%)")
    print(f"  Undirected (direction=0): {direction_counts['undirected']:,} ({100*direction_counts['undirected']/total:.1f}%)")
    print(f"  Unknown (direction=NA): {direction_counts['unknown']:,} ({100*direction_counts['unknown']/total:.1f}%)")
    
    return dict(direction_counts)


def analyze_tissue_distribution(kg) -> dict:
    """Analyze the distribution of tissue annotations in the graph."""
    tissue_counts = defaultdict(int)
    
    for u, v, data in kg.edges(data=True):
        tissue = data.get('tissue')
        if tissue is None:
            tissue_counts['<None>'] += 1
        else:
            tissue_counts[tissue] += 1
    
    total = sum(tissue_counts.values())
    
    print("\nTissue Distribution (top 15):")
    sorted_tissues = sorted(tissue_counts.items(), key=lambda x: -x[1])
    for tissue, count in sorted_tissues[:15]:
        print(f"  {tissue}: {count:,} ({100*count/total:.1f}%)")
    
    print(f"\nTotal unique tissue values: {len(tissue_counts)}")
    
    return dict(tissue_counts)


def expand_undirected_edges(kg) -> tuple:
    """
    Expand edges with direction='0', direction=0, or missing direction in both directions.
    
    After expansion:
    - Original edge A→B kept as-is
    - New edge B→A added (if not exists)
    - New edges marked with 'expanded_bidirectional' = True
    
    Returns:
        Tuple of (modified kg, number of edges added)
    """
    print("\nExpanding undirected/unknown direction edges...")
    
    edges_to_add = []
    expanded_count = 0
    
    for u, v, key, data in tqdm(kg.edges(keys=True, data=True), desc="Scanning edges"):
        direction = data.get('direction')
        
        # Check if undirected or unknown
        is_undirected = (direction == '0' or direction == 0)
        is_unknown = (direction is None or direction == '')
        
        if is_undirected or is_unknown:
            # Check if reverse edge already exists
            if not kg.has_edge(v, u):
                new_data = data.copy()
                new_data['expanded_bidirectional'] = True
                new_data['original_direction'] = str(direction) if direction is not None else 'NA'
                # Set direction to '1' for the expanded edge so PSR treats it as directed
                new_data['direction'] = '1'
                edges_to_add.append((v, u, new_data))
    
    # Add the new edges
    print(f"Adding {len(edges_to_add):,} reverse edges...")
    for u, v, data in tqdm(edges_to_add, desc="Adding edges"):
        kg.add_edge(u, v, **data)
    
    print(f"Expanded {len(edges_to_add):,} edges bidirectionally")
    print(f"Graph now has {kg.number_of_edges():,} edges")
    
    return kg, len(edges_to_add)


def filter_by_tissue(kg, context_name: str, context_config: dict) -> KnowledgeGraph:
    """
    Filter graph by tissue context.
    
    Args:
        kg: Input knowledge graph
        context_name: Name of the context (baseline, adipose, etc.)
        context_config: Configuration dict with 'filter' key
    
    Returns:
        Filtered KnowledgeGraph
    """
    print(f"\nFiltering for context: {context_name}")
    print(f"  Filter: {context_config['filter']}")
    
    # Define filter functions based on context
    if context_name == 'baseline':
        def filter_fn(data):
            tissue = data.get('tissue')
            return tissue is not None and tissue != 'Not specified'
    
    elif context_name == 'adipose':
        def filter_fn(data):
            tissue = data.get('tissue')
            return tissue == 'Adipose Tissue'
    
    elif context_name == 'nonadipose':
        def filter_fn(data):
            tissue = data.get('tissue')
            return tissue is not None and tissue != 'Not specified' and tissue != 'Adipose Tissue'
    
    elif context_name == 'liver':
        def filter_fn(data):
            tissue = data.get('tissue')
            return tissue == 'Liver'
    
    else:
        raise ValueError(f"Unknown context: {context_name}")
    
    # Use the filter_graph utility
    filtered_kg = filter_graph(kg, filter_criterion=filter_fn)
    
    print(f"  Result: {filtered_kg.number_of_nodes():,} nodes, {filtered_kg.number_of_edges():,} edges")
    
    return filtered_kg


def save_metadata(output_dir: Path, metadata: dict):
    """Save filter metadata to JSON."""
    metadata_path = output_dir / 'filter_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nSaved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Filter graph by tissue context')
    parser.add_argument('--config', type=Path, help='Path to config.yaml')
    parser.add_argument('--input-graph', type=Path, help='Path to input graph (overrides config)')
    parser.add_argument('--output-dir', type=Path, help='Path to output directory (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        input_graph = Path(args.input_graph or config['paths']['input_graph'])
        output_dir = Path(args.output_dir or config['paths']['output_dir']) / 'filtered_graphs'
    else:
        if not args.input_graph or not args.output_dir:
            parser.error("Either --config or both --input-graph and --output-dir are required")
        input_graph = args.input_graph
        output_dir = args.output_dir / 'filtered_graphs'
        config = {
            'contexts': {
                'baseline': {'filter': "tissue is not None and tissue != 'Not specified'", 'description': 'All tissue-annotated'},
                'adipose': {'filter': "tissue == 'Adipose Tissue'", 'description': 'Adipose only'},
                'nonadipose': {'filter': "tissue not in (None, 'Not specified', 'Adipose Tissue')", 'description': 'Non-adipose'},
                'liver': {'filter': "tissue == 'Liver'", 'description': 'Liver only'},
            }
        }
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("RQ1 STEP 1: TISSUE CONTEXT FILTERING")
    print("=" * 80)
    print(f"\nInput graph: {input_graph}")
    print(f"Output directory: {output_dir}")
    
    # Load graph
    print(f"\nLoading graph...")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    # Analyze distributions
    direction_dist = analyze_direction_distribution(kg)
    tissue_dist = analyze_tissue_distribution(kg)
    
    # Expand undirected edges
    kg, n_expanded = expand_undirected_edges(kg)
    
    # Initialize metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'source_graph': str(input_graph),
        'original_edges': kg.number_of_edges() - n_expanded,
        'expanded_edges': n_expanded,
        'total_edges_after_expansion': kg.number_of_edges(),
        'direction_distribution': direction_dist,
        'contexts': {}
    }
    
    # Filter for each context
    print("\n" + "=" * 80)
    print("CREATING TISSUE SUBGRAPHS")
    print("=" * 80)
    
    for context_name, context_config in config['contexts'].items():
        # Filter
        filtered_kg = filter_by_tissue(kg, context_name, context_config)
        
        # Save filtered graph
        output_path = output_dir / f'graph_{context_name}.pkl'
        print(f"  Saving to: {output_path}")
        filtered_kg.export_graph(str(output_path))
        
        # Update metadata
        metadata['contexts'][context_name] = {
            'filter': context_config['filter'],
            'description': context_config.get('description', ''),
            'n_nodes': filtered_kg.number_of_nodes(),
            'n_edges': filtered_kg.number_of_edges(),
            'output_path': str(output_path)
        }
    
    # Save metadata
    save_metadata(output_dir, metadata)
    
    # Print summary
    print("\n" + "=" * 80)
    print("FILTERING COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    for ctx_name, ctx_info in metadata['contexts'].items():
        print(f"  {ctx_name}: {ctx_info['n_nodes']:,} nodes, {ctx_info['n_edges']:,} edges")
    
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    for ctx_name in config['contexts']:
        print(f"  - graph_{ctx_name}.pkl")
    print("  - filter_metadata.json")


if __name__ == '__main__':
    main()