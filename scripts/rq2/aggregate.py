#!/usr/bin/env python3
"""
RQ2 Aggregation — Aggregate edges with tissue-specific coverage as tiebreaker.

For each tissue context (subcutaneous, visceral, white, brown):
1. Load cleaned graph
2. Compute coverage for each edge group
3. Aggregate edges using the PSR formula: P = 1 - ∏(1 - pᵢ)
4. When multiple edge types exist for the same node pair, select by
   (probability DESC, coverage DESC, evidence_score DESC)
5. Preserve list-valued attributes (topic IDs, mechanisms, pathways)
   for downstream enrichment analysis
6. Save aggregated graph

Usage:
    python aggregate.py --tissue subcutaneous --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph

from utils import load_config, ordered_pair, EPS
from tissue_mapping import matches_tissue_group


# Attributes to collect as lists from constituent edges during aggregation.
# Each key's non-None values are gathered into "{key}_list" on the aggregated edge.
COLLECT_AS_LISTS = [
    'mechanisms_topic_id',
    'pathways_topic_id',
    'mechanisms',
    'pathways',
]


def aggregate_edge_probabilities(edges_data: List[dict]) -> Tuple[float, float, dict]:
    """
    Aggregate edge probabilities using the PSR formula.

    P = 1 - ∏(1 - pᵢ)
    S = Σ -log(1 - pᵢ + ε)

    Returns (aggregated_probability, evidence_score, representative_attrs).
    """
    if not edges_data:
        return 0.0, 0.0, None

    probs = [float(e.get('probability', 0.5) or 0.5) for e in edges_data]

    agg_prob = 1.0 - np.prod([1.0 - p for p in probs])
    evidence = sum(-np.log(1.0 - p + EPS) for p in probs)

    attrs = edges_data[0].copy()
    attrs['probability'] = agg_prob
    attrs['evidence_score'] = evidence
    attrs['n_supporting_edges'] = len(edges_data)

    return agg_prob, evidence, attrs


def collect_list_attributes(edges_data: List[dict], attrs: dict,
                            tissue_name: str = None) -> None:
    """
    Collect list-valued attributes from constituent edges into the
    aggregated edge.  Modifies *attrs* in place.

    When tissue_name is provided, also splits each list into
    "{attr}_list_tissue" and "{attr}_list_other" for downstream enrichment
    (foreground/background design).
    """
    tissue_matches = None
    if tissue_name:
        tissue_matches = [
            matches_tissue_group(e.get('detailed_tissue'), tissue_name)
            for e in edges_data
        ]

    for attr in COLLECT_AS_LISTS:
        collected_all = []
        collected_tissue = []
        collected_other = []

        for i, edge in enumerate(edges_data):
            val = edge.get(attr)
            if val is None:
                continue
            vals = val if isinstance(val, list) else [val]
            collected_all.extend(vals)

            if tissue_matches is not None:
                (collected_tissue if tissue_matches[i] else collected_other).extend(vals)

        attrs[f'{attr}_list'] = collected_all
        if tissue_name:
            attrs[f'{attr}_list_tissue'] = collected_tissue
            attrs[f'{attr}_list_other'] = collected_other


def compute_coverage_for_group(edges_data: List[dict], tissue_name: str,
                               tissue_config: dict) -> Tuple[float, int]:
    """Compute fraction of edges matching the tissue group."""
    if not edges_data:
        return 0.0, 0
    count = sum(1 for e in edges_data
                if matches_tissue_group(e.get('detailed_tissue'), tissue_name))
    return count / len(edges_data), count


def aggregate_graph_for_tissue(kg: KnowledgeGraph, tissue_name: str,
                               config: dict) -> KnowledgeGraph:
    """
    Aggregate graph with tissue-specific coverage as tiebreaker.

    Groups edges by (source, target, type, correlation_type, direction),
    aggregates probabilities, then selects the best edge type per node pair.
    """
    print(f"\nAggregating graph for tissue: {tissue_name}")
    print("-" * 60)

    tissue_config = config['tissue_groups'].get(tissue_name, {})

    # Group edges
    edge_groups = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        edge_type = data.get('type', 'unknown')
        correlation = data.get('correlation_type', 0)
        direction = data.get('direction', '0')
        if direction == '0':
            u, v = ordered_pair(u, v)
        edge_groups[(u, v, edge_type, correlation, direction)].append(data.copy())

    print(f"  Edge groups: {len(edge_groups):,}")

    # Detect whether LDA topic IDs are present
    sample_edge = next(iter(next(iter(edge_groups.values()))), {}) if edge_groups else {}
    has_mech_topics = 'mechanisms_topic_id' in sample_edge
    has_path_topics = 'pathways_topic_id' in sample_edge
    print(f"  LDA topic IDs present: mechanisms={has_mech_topics}, pathways={has_path_topics}")

    # Aggregate each group
    aggregated_by_type = defaultdict(dict)
    edges_with_coverage = 0
    total_coverage = 0.0

    for idx, ((u, v, edge_type, correlation, direction), edges_data) in enumerate(edge_groups.items()):
        if (idx + 1) % 10000 == 0:
            print(f"  Processing group {idx + 1:,} / {len(edge_groups):,}")

        prob, evidence, attrs = aggregate_edge_probabilities(edges_data)
        if attrs is None:
            continue

        collect_list_attributes(edges_data, attrs, tissue_name=tissue_name)

        coverage, n_tissue_edges = compute_coverage_for_group(
            edges_data, tissue_name, tissue_config)
        attrs['coverage'] = coverage
        attrs['n_edges_tissue'] = n_tissue_edges
        attrs['n_edges_total'] = len(edges_data)
        attrs['tissue_context'] = tissue_name

        if coverage > 0:
            edges_with_coverage += 1
            total_coverage += coverage

        aggregated_by_type[(u, v)][edge_type] = (prob, coverage, evidence, attrs)

    print(f"  Node pairs with edges: {len(aggregated_by_type):,}")
    print(f"  Edge groups with tissue coverage > 0: {edges_with_coverage:,}")
    if edges_with_coverage > 0:
        print(f"  Mean coverage (where > 0): {total_coverage / edges_with_coverage:.4f}")

    # Select best type per node pair: (probability DESC, coverage DESC, evidence DESC)
    agg_kg = KnowledgeGraph()
    for node in kg.nodes():
        agg_kg.add_node(node, **kg.nodes[node])

    type_selections = defaultdict(int)
    ties_resolved_by_coverage = 0

    for node_pair, types_dict in aggregated_by_type.items():
        u, v = node_pair

        if len(types_dict) == 1:
            edge_type, (prob, cov, evidence, attrs) = list(types_dict.items())[0]
            agg_kg.add_edge(u, v, **attrs)
            type_selections[edge_type] += 1
        else:
            sorted_types = sorted(
                types_dict.items(),
                key=lambda x: (x[1][0], x[1][1], x[1][2]),
                reverse=True,
            )
            best = sorted_types[0]
            second = sorted_types[1]
            if best[1][0] == second[1][0] and best[1][1] != second[1][1]:
                ties_resolved_by_coverage += 1

            edge_type, (prob, cov, evidence, attrs) = best
            attrs['selected_from_n_types'] = len(types_dict)
            attrs['competing_types'] = list(types_dict.keys())
            agg_kg.add_edge(u, v, **attrs)
            type_selections[edge_type] += 1

    print(f"\n  Ties resolved by coverage: {ties_resolved_by_coverage:,}")
    print(f"  Final aggregated graph: "
          f"{agg_kg.number_of_nodes():,} nodes, {agg_kg.number_of_edges():,} edges")

    return agg_kg


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate graph with tissue coverage tiebreaker')
    parser.add_argument('--tissue', type=str, required=True,
                        choices=['subcutaneous', 'visceral', 'white', 'brown'])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config['paths']['output_dir'])
    preprocessed_dir = output_dir / 'preprocessed'
    aggregated_dir = output_dir / 'aggregated'
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned graph
    input_path = preprocessed_dir / 'cleaned_graph.pkl'
    if not input_path.exists():
        print(f"ERROR: Cleaned graph not found at {input_path}")
        print("Run preprocess.py first!")
        sys.exit(1)

    print(f"Loading cleaned graph from: {input_path}")
    kg = KnowledgeGraph.import_graph(str(input_path))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    start_time = datetime.now()
    agg_kg = aggregate_graph_for_tissue(kg, args.tissue, config)
    elapsed = (datetime.now() - start_time).total_seconds()

    # Save
    output_path = aggregated_dir / f'aggregated_{args.tissue}.pkl'
    print(f"\nSaving aggregated graph to: {output_path}")
    agg_kg.export_graph(str(output_path))

    # Update metadata
    metadata_path = aggregated_dir / 'aggregation_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {'aggregations': {}}

    metadata['aggregations'][args.tissue] = {
        'timestamp': datetime.now().isoformat(),
        'tissue': args.tissue,
        'input_nodes': kg.number_of_nodes(),
        'input_edges': kg.number_of_edges(),
        'output_nodes': agg_kg.number_of_nodes(),
        'output_edges': agg_kg.number_of_edges(),
        'elapsed_seconds': elapsed,
        'output_path': str(output_path),
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Aggregation complete in {elapsed:.1f}s")


if __name__ == '__main__':
    main()