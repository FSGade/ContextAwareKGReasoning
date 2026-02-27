#!/usr/bin/env python3
"""
RQ1 Step 1b: Aggregate filtered graphs.

This script aggregates edges in the filtered graphs to compute evidence_score.
Must run AFTER filter_graphs.py and BEFORE run_psr.py.

Pipeline: Raw Graph → Filter → Aggregate → PSR

Usage:
    python aggregate_graphs.py --config config.yaml
    python aggregate_graphs.py --context adipose --config config.yaml
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats


# =============================================================================
# Aggregation functions (from aggregate.py)
# =============================================================================

EVID_EPS = 0.01  # Epsilon for evidence score calculation
EPS = 1e-9  # Float tolerance for tie detection


def ordered_pair(u, v):
    """Return a stable ordering for undirected edges."""
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:
        su, sv = str(u), str(v)
        return (u, v) if su <= sv else (v, u)


def aggregate_edge_probabilities(edges_data, min_prob=0.0):
    """
    Aggregate probabilities for multiple edges using PSR formula.
    
    P = 1 - ∏(1 - p_i)           (at-least-one-true)
    S = -Σ log(1 - p_i + ε)      (evidence score)
    
    Returns (probability, evidence_score, attrs) or (0, 0, None) if below threshold.
    """
    if not edges_data:
        return 0.0, 0.0, None
    
    n_edges = len(edges_data)
    
    # Collect probabilities
    probs = np.array([edge.get('probability', 0.0) for edge in edges_data], dtype=np.float64)
    
    # P = 1 - ∏(1 - p_i) computed as 1 - exp(sum(log(1-p)))
    log_complement = np.log1p(-probs)  # More stable than log(1-p)
    final_prob = -np.expm1(np.sum(log_complement))  # More stable than 1 - exp(x)
    final_prob = float(np.clip(final_prob, 0.0, 1.0))
    
    # S = Σ -log(1 - p_i + ε)
    evidence_score = float(np.sum(-np.log(1 - probs + EVID_EPS)))
    
    if final_prob < min_prob:
        return 0.0, 0.0, None
    
    # Collect metadata
    document_ids = []
    sources = set()
    for edge in edges_data:
        doc_id = edge.get('document_id')
        if doc_id:
            document_ids.append(doc_id)
        source = edge.get('source')
        if source:
            sources.add(source)
    
    first_edge = edges_data[0]
    
    attrs = {
        'type': first_edge.get('type', 'unknown'),
        'kind': first_edge.get('type', 'unknown'),
        'probability': round(float(final_prob), 6),
        'evidence_score': round(float(evidence_score), 4),
        'correlation_type': first_edge.get('correlation_type', 0),
        'direction': first_edge.get('direction', '0'),
        'is_directed': first_edge.get('direction', '0') != '0',
        'source': ', '.join(sources) if sources else 'aggregated',
        'n_supporting_edges': n_edges,
        'n_documents': len(set(document_ids)),
        'aggregated': True,
        # Preserve tissue attribute
        'tissue': first_edge.get('tissue'),
    }
    
    return final_prob, evidence_score, attrs


def aggregate_knowledge_graph(kg, min_prob=0.0):
    """
    Create aggregated knowledge graph with PSR probability aggregation.
    
    For each node pair:
    1. Group edges by (source, target, type, correlation, direction)
    2. Aggregate probabilities within each group
    3. Keep only the edge type with highest probability per node pair
    
    Returns the aggregated KnowledgeGraph.
    """
    print(f"  Input: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    agg_kg = KnowledgeGraph()
    
    # Copy nodes
    for node in tqdm(kg.nodes(), desc="  Copying nodes"):
        agg_kg.add_node(node, **kg.nodes[node].copy())
    
    # Group edges by (source, target, type, correlation, direction)
    print("  Grouping edges...")
    edge_groups = defaultdict(list)
    
    for u, v, data in tqdm(kg.edges(data=True), desc="  Collecting edges", total=kg.number_of_edges()):
        edge_type = data.get('type', 'unknown')
        correlation = data.get('correlation_type', 0)
        direction = data.get('direction', '0')
        
        # Normalize undirected edges to canonical order
        if direction == '0':
            u, v = ordered_pair(u, v)
        
        group_key = (u, v, edge_type, correlation, direction)
        edge_groups[group_key].append(data.copy())
    
    print(f"  Found {len(edge_groups):,} unique edge groups")
    
    # Aggregate by type
    print("  Aggregating by edge type...")
    aggregated_by_type = defaultdict(dict)  # (u,v) -> {edge_type: (prob, score, attrs)}
    
    for (u, v, edge_type, correlation, direction), edges_data in tqdm(edge_groups.items(), desc="  Aggregating"):
        prob, score, attrs = aggregate_edge_probabilities(edges_data, min_prob)
        
        if attrs is not None:
            node_pair = (u, v)
            aggregated_by_type[node_pair][edge_type] = (prob, score, attrs)
    
    # Select best type per node pair
    print("  Selecting best edge type per node pair...")
    n_directed = 0
    n_undirected = 0
    
    for node_pair, types_dict in tqdm(aggregated_by_type.items(), desc="  Building graph"):
        # Find type with highest probability, use evidence score as tiebreaker
        best_type = max(types_dict.items(), key=lambda x: (x[1][0], x[1][1]))
        edge_type, (prob, score, attrs) = best_type
        u, v = node_pair
        
        agg_kg.add_edge(u, v, **attrs)
        
        if attrs['is_directed']:
            n_directed += 1
        else:
            n_undirected += 1
    
    # Print results
    print(f"\n  Output: {agg_kg.number_of_nodes():,} nodes, {agg_kg.number_of_edges():,} edges")
    print(f"    Directed edges: {n_directed:,}")
    print(f"    Undirected edges: {n_undirected:,}")
    reduction = 100 * (1 - agg_kg.number_of_edges() / max(1, kg.number_of_edges()))
    print(f"    Edge reduction: {reduction:.1f}%")
    
    return agg_kg


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def aggregate_context(input_path: Path, output_path: Path, context_name: str):
    """Aggregate a single context graph."""
    print(f"\n{'='*60}")
    print(f"Aggregating: {context_name}")
    print(f"{'='*60}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load filtered graph
    print(f"\n  Loading filtered graph...")
    kg = KnowledgeGraph.import_graph(str(input_path))
    
    # Aggregate
    print(f"\n  Running aggregation...")
    agg_kg = aggregate_knowledge_graph(kg, min_prob=0.0)
    
    # Verify evidence scores exist
    sample_edge = next(iter(agg_kg.edges(data=True)), None)
    if sample_edge:
        u, v, data = sample_edge
        ev = data.get('evidence_score', 'MISSING')
        prob = data.get('probability', 'MISSING')
        print(f"\n  Sample edge: probability={prob}, evidence_score={ev}")
    
    # Save
    print(f"\n  Saving aggregated graph...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    agg_kg.export_graph(str(output_path))
    print(f"  Saved: {output_path}")
    
    return agg_kg


def main():
    parser = argparse.ArgumentParser(description='Aggregate filtered graphs')
    parser.add_argument('--config', type=Path, help='Path to config.yaml')
    parser.add_argument('--context', choices=['baseline', 'adipose', 'nonadipose', 'liver'],
                        help='Aggregate specific context only (default: all)')
    parser.add_argument('--input-dir', type=Path, help='Directory with filtered graphs')
    parser.add_argument('--output-dir', type=Path, help='Directory for aggregated graphs')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        base_output = Path(config['paths']['output_dir'])
        input_dir = args.input_dir or base_output / 'filtered_graphs'
        output_dir = args.output_dir or base_output / 'aggregated_graphs'
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("Either --config or both --input-dir and --output-dir are required")
        input_dir = args.input_dir
        output_dir = args.output_dir
    
    print("=" * 80)
    print("RQ1 STEP 1b: AGGREGATE FILTERED GRAPHS")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Determine which contexts to process
    if args.context:
        contexts = [args.context]
    else:
        contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    
    print(f"Contexts: {contexts}")
    
    # Process each context
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for ctx in contexts:
        input_path = input_dir / f'graph_{ctx}.pkl'
        output_path = output_dir / f'graph_{ctx}_aggregated.pkl'
        
        if not input_path.exists():
            print(f"\nWARNING: Filtered graph not found: {input_path}")
            continue
        
        aggregate_context(input_path, output_path, ctx)
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'contexts_processed': contexts,
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
    }
    
    metadata_path = output_dir / 'aggregate_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("AGGREGATION COMPLETE")
    print("=" * 80)
    print(f"\nAggregated graphs saved to: {output_dir}")


if __name__ == '__main__':
    main()