#!/usr/bin/env python3
"""
RQ1 Step 2: Run PSR inference on tissue-filtered subgraphs.

This script runs either 2-hop or 3-hop PSR inference on an AGGREGATED graph
and saves results in a format suitable for cross-context comparison.

Supports optional metapath grouping for mechanistic interpretability.

Prerequisites: Run aggregate_graphs.py first to compute evidence_score.

Usage:
    python run_psr.py --context adipose --hops 2 --config config.yaml
    python run_psr.py --context liver --hops 3 --input-dir /path/to/aggregated --output-dir /path/to/results
"""

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # For metapath_utils
from knowledge_graph import KnowledgeGraph, print_kg_stats

# Import metapath utilities if available
try:
    from metapath_utils import group_paths_by_metapath, create_metapath_edge_attrs
    HAS_METAPATH_UTILS = True
    print("Metapath utilities loaded successfully")
except ImportError as e:
    HAS_METAPATH_UTILS = False
    print(f"Warning: metapath_utils not found ({e}), metapath grouping disabled")


# =============================================================================
# Helper functions (adapted from psr.py and three_hop_psr.py)
# =============================================================================

def node_name(kg, n):
    """Get a display name for node n."""
    return kg.nodes[n].get('name', str(n))


def node_type(kg, n):
    """Get a semantic type/kind for node n."""
    return kg.nodes[n].get('type', kg.nodes[n].get('kind', 'unknown'))


def get_edge_properties(kg, u, v):
    """
    Extract probability, evidence_score, sign, type, and direction for an aggregated edge.
    MultiDiGraph-safe.
    """
    data = kg.get_edge_data(u, v)
    if data is None:
        return None

    # Handle MultiDiGraph format
    if isinstance(data, dict) and any(isinstance(val, dict) for val in data.values()):
        candidates = list(data.values())
        chosen = next((d for d in candidates if d.get('aggregated') is True), None)
        if chosen is None and candidates:
            chosen = max(candidates, key=lambda d: d.get('evidence_score', 0.0), default=None)
        edge_data = chosen
    else:
        edge_data = data

    if edge_data is None:
        return None

    prob = edge_data.get('probability', 0.0)
    evidence_score = edge_data.get('evidence_score', 0.0)
    correlation_type = edge_data.get('correlation_type', 0)
    edge_type = edge_data.get('type', edge_data.get('kind', 'unknown'))
    direction = edge_data.get('direction', '1')
    is_directed = (direction != '0')

    return {
        'probability': float(prob),
        'evidence_score': float(evidence_score),
        'correlation_type': int(correlation_type),
        'edge_type': edge_type,
        'is_directed': is_directed,
        'n_supporting_edges': edge_data.get('n_supporting_edges', edge_data.get('count', 1)),
        'n_documents': edge_data.get('n_documents', 1),
    }


def has_confident_direct(kg, A, C, require_directed=True, require_known_sign=True):
    """Returns True if there exists a direct A->C edge that is 'confident'."""
    if not kg.has_edge(A, C):
        return False
    data = kg.get_edge_data(A, C)
    if data is None:
        return False

    if isinstance(data, dict) and any(isinstance(v, dict) for v in data.values()):
        records = list(data.values())
    else:
        records = [data]

    for d in records:
        dir_ok = (not require_directed) or (d.get('direction', '1') != '0')
        sign_ok = (not require_known_sign) or (d.get('correlation_type', 0) != 0)
        if dir_ok and sign_ok:
            return True
    return False


def aggregate_probabilities_psr(paths):
    """Aggregate probabilities using PSR formula: P = 1 - ∏(1 - p_i)"""
    if not paths:
        return 0.0
    probs = np.array([p['probability'] for p in paths], dtype=np.float64)
    combined = 1.0 - np.prod(1.0 - probs)
    return float(combined)


def aggregate_evidence_scores(paths):
    """Aggregate evidence scores by summing across paths."""
    return float(sum(p['evidence_score'] for p in paths))


# =============================================================================
# Two-hop inference
# =============================================================================

def compute_two_hop_inference(kg, params: dict) -> dict:
    """
    Compute two-hop inferences A -> B -> C for all paths in the graph.
    
    For Gene→Disease inference:
    - A must be Gene (source)
    - B must be Gene (intermediate) - ENFORCED
    - C must be Disease (target) - filtered later
    
    Returns dict keyed by (A, C) with inference details.
    """
    print(f"\n{'='*80}")
    print("COMPUTING TWO-HOP PSR INFERENCE")
    print(f"{'='*80}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Edges: {kg.number_of_edges():,}")
    
    # Check if we should constrain intermediate types
    require_gene_intermediates = params.get('require_gene_intermediates', True)
    print(f"  Require Gene intermediates: {require_gene_intermediates}")
    
    # Build neighbor dictionaries with edge metadata for tracking expanded edges
    print("\nBuilding neighbor index...")
    successors_dict = defaultdict(set)
    predecessors_dict = defaultdict(set)
    edge_is_expanded = {}  # Track which edges were expanded from undirected

    for u, v, data in tqdm(kg.edges(data=True), desc="Indexing edges"):
        direction = data.get('direction', '1')
        is_directed = (direction != '0')
        is_expanded = data.get('expanded_bidirectional', False)
        
        # Store expansion status for this edge
        edge_is_expanded[(u, v)] = is_expanded

        if is_directed:
            successors_dict[u].add(v)
            predecessors_dict[v].add(u)
        elif params.get('consider_undirected', False):
            successors_dict[u].add(v)
            successors_dict[v].add(u)
            predecessors_dict[u].add(v)
            predecessors_dict[v].add(u)

    # Find Gene nodes for intermediate filtering
    if require_gene_intermediates:
        gene_nodes = {n for n in kg.nodes() if node_type(kg, n) == 'Gene'}
        print(f"  Gene nodes available as intermediates: {len(gene_nodes):,}")
    else:
        gene_nodes = set(kg.nodes())  # All nodes allowed

    # Find all two-hop paths
    print("\nFinding two-hop paths...")
    two_hop_paths = defaultdict(list)
    skipped_non_gene = 0

    for Bj in tqdm(kg.nodes(), desc="Processing intermediates"):
        # Only allow Gene nodes as intermediates
        if require_gene_intermediates and Bj not in gene_nodes:
            skipped_non_gene += 1
            continue
            
        preds = predecessors_dict.get(Bj, set())
        succs = successors_dict.get(Bj, set())

        for A in preds:
            for C in succs:
                if A == C:
                    continue

                if params.get('skip_when_any_direct_exists', True):
                    if has_confident_direct(
                        kg, A, C,
                        require_directed=params.get('require_directed_for_skip', True),
                        require_known_sign=params.get('require_known_sign_for_skip', True)
                    ):
                        continue

                two_hop_paths[(A, C)].append(Bj)

    if require_gene_intermediates:
        print(f"Skipped {skipped_non_gene:,} non-Gene intermediates")
    print(f"Found {len(two_hop_paths):,} A-C pairs with two-hop paths")
    total_paths = sum(len(interms) for interms in two_hop_paths.values())
    print(f"Total two-hop paths: {total_paths:,}")

    # Compute inferred edges
    print("\nComputing inferred probabilities...")
    inferred_edges = {}
    skipped_paths = 0
    min_path_prob = params.get('min_path_probability', 0.001)
    max_intermediates = params.get('max_intermediates')
    
    # Metapath grouping settings
    use_metapath_grouping = params.get('use_metapath_grouping', False) and HAS_METAPATH_UTILS
    grouping_strategy = params.get('grouping_strategy', 'mechanistic')
    split_inconsistent = params.get('split_inconsistent_correlations', True)
    
    if use_metapath_grouping:
        print(f"  Metapath grouping: ENABLED (strategy={grouping_strategy})")
    else:
        print(f"  Metapath grouping: DISABLED")

    for (A, C), intermediates in tqdm(two_hop_paths.items(), desc="Computing inferences"):
        # Limit intermediates if configured
        if max_intermediates and len(intermediates) > max_intermediates:
            scored = []
            for Bj in intermediates:
                props_ab = get_edge_properties(kg, A, Bj)
                props_bc = get_edge_properties(kg, Bj, C)
                if props_ab and props_bc:
                    scored.append((Bj, props_ab['evidence_score'] * props_bc['evidence_score']))
            scored.sort(key=lambda x: -x[1])
            intermediates = [b for b, _ in scored[:max_intermediates]]

        # Build path information
        all_paths = []
        paths_with_expanded_edges = 0
        
        for Bj in intermediates:
            props_ab = get_edge_properties(kg, A, Bj)
            props_bc = get_edge_properties(kg, Bj, C)
            if props_ab is None or props_bc is None:
                continue

            path_prob = props_ab['probability'] * props_bc['probability']
            if path_prob < min_path_prob:
                skipped_paths += 1
                continue

            path_evidence = props_ab['evidence_score'] * props_bc['evidence_score']
            path_correlation = props_ab['correlation_type'] * props_bc['correlation_type']
            
            # Track if this path uses any expanded edges
            uses_expanded = edge_is_expanded.get((A, Bj), False) or edge_is_expanded.get((Bj, C), False)
            if uses_expanded:
                paths_with_expanded_edges += 1

            path_info = {
                'intermediate': Bj,  # Node ID for metapath utils
                'intermediate_name': node_name(kg, Bj),
                'intermediate_type': node_type(kg, Bj),
                'probability': path_prob,
                'evidence_score': path_evidence,
                'correlation': path_correlation,
                'relations': (props_ab['edge_type'], props_bc['edge_type']),
                'node_types': (node_type(kg, A), node_type(kg, Bj), node_type(kg, C)),
                'uses_expanded_edge': uses_expanded,
                'path': [A, Bj, C],  # For metapath utils
            }
            all_paths.append(path_info)

        if not all_paths:
            continue

        # Calculate fraction of paths using expanded edges
        expanded_edge_fraction = paths_with_expanded_edges / len(all_paths) if all_paths else 0.0

        if use_metapath_grouping:
            # GROUP BY METAPATH: Multiple results per (A, C)
            metapath_groups = group_paths_by_metapath(
                all_paths,
                strategy=grouping_strategy,
                split_inconsistent_correlations=split_inconsistent
            )
            
            # Create one result per metapath group
            edge_list = []
            for metapath_sig, group_info in metapath_groups.items():
                paths_in_group = group_info['paths']
                was_split = group_info['was_split']
                correlation = group_info['correlation']
                
                # Aggregate within this metapath group
                combined_prob = aggregate_probabilities_psr(paths_in_group)
                combined_evidence = aggregate_evidence_scores(paths_in_group)
                
                # Extract relation types from signature
                if len(metapath_sig) == 2:
                    sig_node_types, sig_relations = metapath_sig
                else:  # Was split, has correlation
                    sig_node_types, sig_relations, _ = metapath_sig
                
                # Create metapath name
                metapath_name = f"{sig_relations[0]}_{sig_relations[1]}"
                
                # Collect intermediate names
                intermediate_names = list(set(p['intermediate_name'] for p in paths_in_group))
                
                edge_attrs = {
                    'source_gene': node_name(kg, A),
                    'source_gene_id': str(A),
                    'source_type': node_type(kg, A),
                    'target': node_name(kg, C),
                    'target_id': str(C),
                    'target_type': node_type(kg, C),
                    'path_probability': round(float(combined_prob), 6),
                    'evidence_score': round(float(combined_evidence), 4),
                    'correlation_type': int(correlation),
                    'num_intermediates': len(paths_in_group),
                    'intermediate_genes': intermediate_names[:50],  # Limit for storage
                    'intermediate_types': [sig_node_types[1]],  # Single type per metapath
                    'relation_types': list(sig_relations),
                    'num_paths': len(paths_in_group),
                    'hop_length': 2,
                    'expanded_edge_fraction': round(expanded_edge_fraction, 3),
                    # Metapath-specific fields
                    'metapath_signature': str(metapath_sig),
                    'metapath_name': metapath_name,
                    'node_type_sequence': list(sig_node_types),
                    'relation_sequence': list(sig_relations),
                    'was_correlation_split': was_split,
                }
                edge_list.append(edge_attrs)
            
            # Store as list for this (A, C) pair
            inferred_edges[(A, C)] = edge_list
            
        else:
            # NO GROUPING: Single result per (A, C)
            combined_prob = aggregate_probabilities_psr(all_paths)
            combined_evidence = aggregate_evidence_scores(all_paths)
            
            # Weighted correlation
            if combined_evidence > 0:
                weighted_corr = sum(p['correlation'] * p['evidence_score'] for p in all_paths) / combined_evidence
                if weighted_corr > 0.5:
                    combined_correlation = 1
                elif weighted_corr < -0.5:
                    combined_correlation = -1
                else:
                    combined_correlation = 0
            else:
                combined_correlation = 0

            # Collect unique relation types and intermediates
            relation_types = list(set(r for p in all_paths for r in p['relations']))
            intermediate_names = [p['intermediate_name'] for p in all_paths]
            intermediate_types = list(set(p['intermediate_type'] for p in all_paths))

            inferred_edges[(A, C)] = {
                'source_gene': node_name(kg, A),
                'source_gene_id': str(A),
                'source_type': node_type(kg, A),
                'target': node_name(kg, C),
                'target_id': str(C),
                'target_type': node_type(kg, C),
                'path_probability': round(float(combined_prob), 6),
                'evidence_score': round(float(combined_evidence), 4),
                'correlation_type': int(combined_correlation),
                'num_intermediates': len(all_paths),
                'intermediate_genes': intermediate_names,
                'intermediate_types': intermediate_types,
                'relation_types': relation_types,
                'num_paths': len(all_paths),
                'hop_length': 2,
                'expanded_edge_fraction': round(expanded_edge_fraction, 3),
            }

    print(f"\nInferred {len(inferred_edges):,} indirect associations")
    print(f"Skipped {skipped_paths:,} low-probability paths")
    
    if use_metapath_grouping:
        # Count total metapath-level results
        total_metapath_results = sum(
            len(v) if isinstance(v, list) else 1 
            for v in inferred_edges.values()
        )
        print(f"Total metapath-grouped results: {total_metapath_results:,}")

    return inferred_edges


# =============================================================================
# Three-hop inference
# =============================================================================

def compute_three_hop_inference(kg, params: dict) -> dict:
    """
    Compute three-hop inferences A -> B -> C -> D for all paths in the graph.
    
    For Gene→Gene→Gene→Disease inference:
    - A must be Gene (source)
    - B must be Gene (intermediate 1) - ENFORCED
    - C must be Gene (intermediate 2) - ENFORCED
    - D must be Disease (target) - filtered later
    
    Returns dict keyed by (A, D) with inference details.
    """
    print(f"\n{'='*80}")
    print("COMPUTING THREE-HOP PSR INFERENCE")
    print(f"{'='*80}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Edges: {kg.number_of_edges():,}")

    # Check if we should constrain intermediate types
    require_gene_intermediates = params.get('require_gene_intermediates', True)
    print(f"  Require Gene intermediates: {require_gene_intermediates}")

    # Build neighbor dictionaries with edge metadata for tracking expanded edges
    print("\nBuilding neighbor index...")
    successors_dict = defaultdict(set)
    predecessors_dict = defaultdict(set)
    edge_is_expanded = {}  # Track which edges were expanded from undirected

    for u, v, data in tqdm(kg.edges(data=True), desc="Indexing edges"):
        direction = data.get('direction', '1')
        is_directed = (direction != '0')
        is_expanded = data.get('expanded_bidirectional', False)
        
        # Store expansion status for this edge
        edge_is_expanded[(u, v)] = is_expanded

        if is_directed:
            successors_dict[u].add(v)
            predecessors_dict[v].add(u)
        elif params.get('consider_undirected', False):
            successors_dict[u].add(v)
            successors_dict[v].add(u)
            predecessors_dict[u].add(v)
            predecessors_dict[v].add(u)

    # Find Gene nodes for intermediate filtering
    if require_gene_intermediates:
        gene_nodes = {n for n in kg.nodes() if node_type(kg, n) == 'Gene'}
        print(f"  Gene nodes available as intermediates: {len(gene_nodes):,}")
    else:
        gene_nodes = set(kg.nodes())  # All nodes allowed

    # Find all three-hop paths
    print("\nFinding three-hop paths...")
    
    # Only consider B nodes that are Genes (if required) and have both predecessors and successors
    if require_gene_intermediates:
        active_B_nodes = [B for B in gene_nodes if predecessors_dict.get(B) and successors_dict.get(B)]
    else:
        active_B_nodes = [B for B in kg.nodes() if predecessors_dict.get(B) and successors_dict.get(B)]
    
    print(f"Processing {len(active_B_nodes):,} B nodes with both predecessors and successors")

    three_hop_paths = defaultdict(list)
    skipped_non_gene_C = 0

    for B in tqdm(active_B_nodes, desc="Processing B nodes"):
        preds_B = predecessors_dict.get(B, set())
        succs_B = successors_dict.get(B, set())

        for C in succs_B:
            if B == C:
                continue
            
            # Only allow Gene nodes as intermediate C
            if require_gene_intermediates and C not in gene_nodes:
                skipped_non_gene_C += 1
                continue

            succs_C = successors_dict.get(C, set())
            if not succs_C:
                continue

            for A in preds_B:
                if A == C or A == B:
                    continue

                for D in succs_C:
                    if D == A or D == B or D == C:
                        continue

                    if params.get('skip_when_any_direct_exists', True):
                        if has_confident_direct(
                            kg, A, D,
                            require_directed=params.get('require_directed_for_skip', True),
                            require_known_sign=params.get('require_known_sign_for_skip', True)
                        ):
                            continue

                    three_hop_paths[(A, D)].append((B, C))

    if require_gene_intermediates:
        print(f"Skipped {skipped_non_gene_C:,} paths with non-Gene C intermediates")
    print(f"Found {len(three_hop_paths):,} A-D pairs with three-hop paths")
    total_paths = sum(len(interms) for interms in three_hop_paths.values())
    print(f"Total three-hop paths: {total_paths:,}")

    # Compute inferred edges
    print("\nComputing inferred probabilities...")
    inferred_edges = {}
    skipped_paths = 0
    min_path_prob = params.get('min_path_probability', 0.001)
    max_intermediates = params.get('max_intermediates')
    
    # Metapath grouping settings
    use_metapath_grouping = params.get('use_metapath_grouping', False) and HAS_METAPATH_UTILS
    grouping_strategy = params.get('grouping_strategy', 'mechanistic')
    split_inconsistent = params.get('split_inconsistent_correlations', True)
    
    if use_metapath_grouping:
        print(f"  Metapath grouping: ENABLED (strategy={grouping_strategy})")
    else:
        print(f"  Metapath grouping: DISABLED")

    for (A, D), intermediate_pairs in tqdm(three_hop_paths.items(), desc="Computing inferences"):
        # Limit intermediate pairs if configured
        if max_intermediates and len(intermediate_pairs) > max_intermediates:
            scored = []
            for (B, C) in intermediate_pairs:
                props_ab = get_edge_properties(kg, A, B)
                props_bc = get_edge_properties(kg, B, C)
                props_cd = get_edge_properties(kg, C, D)
                if props_ab and props_bc and props_cd:
                    total_evidence = props_ab['evidence_score'] * props_bc['evidence_score'] * props_cd['evidence_score']
                    scored.append(((B, C), total_evidence))
            scored.sort(key=lambda x: -x[1])
            intermediate_pairs = [pair for pair, _ in scored[:max_intermediates]]

        # Build path information
        all_paths = []
        paths_with_expanded_edges = 0
        
        for (B, C) in intermediate_pairs:
            props_ab = get_edge_properties(kg, A, B)
            props_bc = get_edge_properties(kg, B, C)
            props_cd = get_edge_properties(kg, C, D)

            if props_ab is None or props_bc is None or props_cd is None:
                continue

            path_prob = props_ab['probability'] * props_bc['probability'] * props_cd['probability']
            if path_prob < min_path_prob:
                skipped_paths += 1
                continue

            path_evidence = props_ab['evidence_score'] * props_bc['evidence_score'] * props_cd['evidence_score']
            path_correlation = props_ab['correlation_type'] * props_bc['correlation_type'] * props_cd['correlation_type']

            # Track if this path uses any expanded edges
            uses_expanded = (
                edge_is_expanded.get((A, B), False) or 
                edge_is_expanded.get((B, C), False) or 
                edge_is_expanded.get((C, D), False)
            )
            if uses_expanded:
                paths_with_expanded_edges += 1

            path_info = {
                'intermediate_B': B,  # Node ID
                'intermediate_C': C,  # Node ID
                'intermediate_B_name': node_name(kg, B),
                'intermediate_C_name': node_name(kg, C),
                'intermediate_B_type': node_type(kg, B),
                'intermediate_C_type': node_type(kg, C),
                'probability': path_prob,
                'evidence_score': path_evidence,
                'correlation': path_correlation,
                'relations': (props_ab['edge_type'], props_bc['edge_type'], props_cd['edge_type']),
                'node_types': (node_type(kg, A), node_type(kg, B), node_type(kg, C), node_type(kg, D)),
                'uses_expanded_edge': uses_expanded,
                'path': [A, B, C, D],  # For metapath utils
            }
            all_paths.append(path_info)

        if not all_paths:
            continue

        # Calculate fraction of paths using expanded edges
        expanded_edge_fraction = paths_with_expanded_edges / len(all_paths) if all_paths else 0.0

        if use_metapath_grouping:
            # GROUP BY METAPATH: Multiple results per (A, D)
            metapath_groups = group_paths_by_metapath(
                all_paths,
                strategy=grouping_strategy,
                split_inconsistent_correlations=split_inconsistent
            )
            
            # Create one result per metapath group
            edge_list = []
            for metapath_sig, group_info in metapath_groups.items():
                paths_in_group = group_info['paths']
                was_split = group_info['was_split']
                correlation = group_info['correlation']
                
                # Aggregate within this metapath group
                combined_prob = aggregate_probabilities_psr(paths_in_group)
                combined_evidence = aggregate_evidence_scores(paths_in_group)
                
                # Extract relation types from signature
                if len(metapath_sig) == 2:
                    sig_node_types, sig_relations = metapath_sig
                else:  # Was split, has correlation
                    sig_node_types, sig_relations, _ = metapath_sig
                
                # Create metapath name (3 relations for 3-hop)
                metapath_name = f"{sig_relations[0]}_{sig_relations[1]}_{sig_relations[2]}"
                
                # Collect intermediate names
                intermediate_B_names = list(set(p['intermediate_B_name'] for p in paths_in_group))
                intermediate_C_names = list(set(p['intermediate_C_name'] for p in paths_in_group))
                
                edge_attrs = {
                    'source_gene': node_name(kg, A),
                    'source_gene_id': str(A),
                    'source_type': node_type(kg, A),
                    'target': node_name(kg, D),
                    'target_id': str(D),
                    'target_type': node_type(kg, D),
                    'path_probability': round(float(combined_prob), 6),
                    'evidence_score': round(float(combined_evidence), 4),
                    'correlation_type': int(correlation),
                    'num_intermediates': len(paths_in_group),
                    'intermediate_genes': (intermediate_B_names + intermediate_C_names)[:50],
                    'intermediate_B_genes': intermediate_B_names[:25],
                    'intermediate_C_genes': intermediate_C_names[:25],
                    'intermediate_types': [sig_node_types[1], sig_node_types[2]],
                    'relation_types': list(sig_relations),
                    'num_paths': len(paths_in_group),
                    'hop_length': 3,
                    'expanded_edge_fraction': round(expanded_edge_fraction, 3),
                    # Metapath-specific fields
                    'metapath_signature': str(metapath_sig),
                    'metapath_name': metapath_name,
                    'node_type_sequence': list(sig_node_types),
                    'relation_sequence': list(sig_relations),
                    'was_correlation_split': was_split,
                }
                edge_list.append(edge_attrs)
            
            # Store as list for this (A, D) pair
            inferred_edges[(A, D)] = edge_list
            
        else:
            # NO GROUPING: Single result per (A, D)
            combined_prob = aggregate_probabilities_psr(all_paths)
            combined_evidence = aggregate_evidence_scores(all_paths)

            # Weighted correlation
            if combined_evidence > 0:
                weighted_corr = sum(p['correlation'] * p['evidence_score'] for p in all_paths) / combined_evidence
                if weighted_corr > 0.5:
                    combined_correlation = 1
                elif weighted_corr < -0.5:
                    combined_correlation = -1
                else:
                    combined_correlation = 0
            else:
                combined_correlation = 0

            # Collect unique relation types and intermediates
            relation_types = list(set(r for p in all_paths for r in p['relations']))
            intermediate_B_names = list(set(p['intermediate_B_name'] for p in all_paths))
            intermediate_C_names = list(set(p['intermediate_C_name'] for p in all_paths))
            intermediate_types = list(set(p['intermediate_B_type'] for p in all_paths) | 
                                       set(p['intermediate_C_type'] for p in all_paths))

            inferred_edges[(A, D)] = {
                'source_gene': node_name(kg, A),
                'source_gene_id': str(A),
                'source_type': node_type(kg, A),
                'target': node_name(kg, D),
                'target_id': str(D),
                'target_type': node_type(kg, D),
                'path_probability': round(float(combined_prob), 6),
                'evidence_score': round(float(combined_evidence), 4),
                'correlation_type': int(combined_correlation),
                'num_intermediates': len(all_paths),
                'intermediate_genes': intermediate_B_names + intermediate_C_names,
                'intermediate_B_genes': intermediate_B_names,
                'intermediate_C_genes': intermediate_C_names,
                'intermediate_types': intermediate_types,
                'relation_types': relation_types,
                'num_paths': len(all_paths),
                'hop_length': 3,
                'expanded_edge_fraction': round(expanded_edge_fraction, 3),
            }

    print(f"\nInferred {len(inferred_edges):,} indirect associations")
    print(f"Skipped {skipped_paths:,} low-probability paths")
    
    if use_metapath_grouping:
        # Count total metapath-level results
        total_metapath_results = sum(
            len(v) if isinstance(v, list) else 1 
            for v in inferred_edges.values()
        )
        print(f"Total metapath-grouped results: {total_metapath_results:,}")

    return inferred_edges


# =============================================================================
# Filter to Gene -> Disease results
# =============================================================================

def filter_gene_disease_results(inferred_edges: dict, kg) -> list:
    """
    Filter inferred edges to only Gene -> Disease pairs.
    
    Handles both formats:
    - Without metapath grouping: inferred_edges[(A,C)] = dict
    - With metapath grouping: inferred_edges[(A,C)] = list of dicts
    
    Returns list of result dictionaries.
    """
    results = []
    
    for (A, target), data in inferred_edges.items():
        # Handle both formats: single dict or list of dicts
        if isinstance(data, list):
            # Metapath grouping: multiple results per (A, C)
            for edge_attrs in data:
                source_type = edge_attrs.get('source_type', node_type(kg, A))
                target_type = edge_attrs.get('target_type', '')
                
                if source_type == 'Gene' and target_type == 'Disease':
                    results.append(edge_attrs)
        else:
            # No metapath grouping: single result per (A, C)
            source_type = data.get('source_type', node_type(kg, A))
            target_type = data.get('target_type', '')
            
            if source_type == 'Gene' and target_type == 'Disease':
                results.append(data)
    
    print(f"Filtered to {len(results):,} Gene -> Disease results")
    return results


# =============================================================================
# Save results
# =============================================================================

def save_results(results: list, context: str, hops: int, output_dir: Path, params: dict):
    """Save results to Parquet and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{context}_{hops}hop"
    
    # Check if metapath grouping was used
    use_metapath = params.get('use_metapath_grouping', False) and len(results) > 0 and 'metapath_name' in results[0]
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by tiebreakers: probability desc, evidence desc, num_intermediates desc, gene name asc
        # For metapath: also sort by metapath_name for consistency
        sort_cols = ['path_probability', 'evidence_score', 'num_intermediates', 'source_gene']
        sort_asc = [False, False, False, True]
        
        if use_metapath and 'metapath_name' in df.columns:
            sort_cols.append('metapath_name')
            sort_asc.append(True)
        
        df = df.sort_values(by=sort_cols, ascending=sort_asc)
        
        # Add rank column
        df['rank'] = range(1, len(df) + 1)
    
    # Save as Parquet (efficient for large data)
    parquet_path = output_dir / f"{prefix}_results.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved {len(df):,} results to: {parquet_path}")
    
    # Save top results as JSON (for quick inspection)
    json_path = output_dir / f"{prefix}_top1000.json"
    top_results = df.head(1000).to_dict(orient='records')
    
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'context': context,
                'hops': hops,
                'total_results': len(df),
                'created': datetime.now().isoformat(),
                'params': params,
                'metapath_grouping': use_metapath,
            },
            'results': top_results
        }, f, indent=2, default=str)
    print(f"Saved top 1000 results to: {json_path}")
    
    # Save summary statistics
    stats_path = output_dir / f"{prefix}_stats.json"
    
    # Compute expanded edge fraction stats if column exists
    expanded_stats = {}
    if len(df) > 0 and 'expanded_edge_fraction' in df.columns:
        expanded_stats = {
            'min': float(df['expanded_edge_fraction'].min()),
            'max': float(df['expanded_edge_fraction'].max()),
            'mean': float(df['expanded_edge_fraction'].mean()),
            'median': float(df['expanded_edge_fraction'].median()),
            'fraction_with_any_expanded': float((df['expanded_edge_fraction'] > 0).mean()),
            'fraction_all_expanded': float((df['expanded_edge_fraction'] == 1.0).mean()),
        }
    
    # Metapath-specific stats
    metapath_stats = {}
    if use_metapath and len(df) > 0:
        metapath_stats = {
            'unique_metapaths': df['metapath_name'].nunique() if 'metapath_name' in df.columns else 0,
            'metapath_counts': df['metapath_name'].value_counts().to_dict() if 'metapath_name' in df.columns else {},
            'unique_gene_disease_pairs': df.groupby(['source_gene', 'target']).ngroups,
        }
    
    stats = {
        'context': context,
        'hops': hops,
        'total_results': len(df),
        'metapath_grouping': use_metapath,
        'unique_genes': df['source_gene'].nunique() if len(df) > 0 else 0,
        'unique_diseases': df['target'].nunique() if len(df) > 0 else 0,
        'probability_stats': {
            'min': float(df['path_probability'].min()) if len(df) > 0 else 0,
            'max': float(df['path_probability'].max()) if len(df) > 0 else 0,
            'mean': float(df['path_probability'].mean()) if len(df) > 0 else 0,
            'median': float(df['path_probability'].median()) if len(df) > 0 else 0,
        },
        'evidence_stats': {
            'min': float(df['evidence_score'].min()) if len(df) > 0 else 0,
            'max': float(df['evidence_score'].max()) if len(df) > 0 else 0,
            'mean': float(df['evidence_score'].mean()) if len(df) > 0 else 0,
        },
        'intermediates_stats': {
            'min': int(df['num_intermediates'].min()) if len(df) > 0 else 0,
            'max': int(df['num_intermediates'].max()) if len(df) > 0 else 0,
            'mean': float(df['num_intermediates'].mean()) if len(df) > 0 else 0,
        },
        'num_paths_stats': {
            'min': int(df['num_paths'].min()) if len(df) > 0 and 'num_paths' in df.columns else 0,
            'max': int(df['num_paths'].max()) if len(df) > 0 and 'num_paths' in df.columns else 0,
            'mean': float(df['num_paths'].mean()) if len(df) > 0 and 'num_paths' in df.columns else 0,
        },
        'expanded_edge_stats': expanded_stats,
        'metapath_stats': metapath_stats,
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")
    
    # Print expanded edge warning if significant
    if expanded_stats and expanded_stats.get('fraction_with_any_expanded', 0) > 0.5:
        print(f"\n  WARNING: {expanded_stats['fraction_with_any_expanded']*100:.1f}% of results use expanded (bidirectional) edges")
        print(f"    Consider running sensitivity analysis without edge expansion")
    
    # Print metapath summary
    if use_metapath and metapath_stats:
        print(f"\nMetapath grouping summary:")
        print(f"  Unique metapaths: {metapath_stats['unique_metapaths']}")
        print(f"  Unique (gene, disease) pairs: {metapath_stats['unique_gene_disease_pairs']}")
        print(f"  Top metapaths:")
        for mp, count in sorted(metapath_stats['metapath_counts'].items(), key=lambda x: -x[1])[:5]:
            print(f"    {mp}: {count:,}")
    
    return df


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Run PSR inference on aggregated tissue graph')
    parser.add_argument('--context', required=True, choices=['baseline', 'adipose', 'nonadipose', 'liver'],
                        help='Tissue context to analyze')
    parser.add_argument('--hops', required=True, type=int, choices=[2, 3],
                        help='Number of hops (2 or 3)')
    parser.add_argument('--config', type=Path, help='Path to config.yaml')
    parser.add_argument('--input-dir', type=Path, help='Directory with aggregated graphs (overrides config)')
    parser.add_argument('--output-dir', type=Path, help='Directory for results (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        # Read from aggregated_graphs (not filtered_graphs)
        input_dir = Path(args.input_dir or config['paths']['output_dir']) / 'aggregated_graphs'
        output_dir = Path(args.output_dir or config['paths']['output_dir']) / 'psr_results'
        params = config.get('psr_params', {})
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("Either --config or both --input-dir and --output-dir are required")
        input_dir = args.input_dir
        output_dir = args.output_dir
        params = {
            'max_intermediates': None,
            'min_path_probability': 0.001,
            'consider_undirected': False,
            'skip_when_any_direct_exists': True,
            'require_directed_for_skip': False,
            'require_known_sign_for_skip': False,
        }
    
    # Paths - read aggregated graph
    input_graph = input_dir / f'graph_{args.context}_aggregated.pkl'
    
    print("=" * 80)
    print(f"RQ1 STEP 2: PSR INFERENCE ({args.hops}-HOP)")
    print("=" * 80)
    print(f"\nContext: {args.context}")
    print(f"Hops: {args.hops}")
    print(f"Input graph: {input_graph}")
    print(f"Output directory: {output_dir}")
    
    # Load aggregated graph
    print(f"\nLoading aggregated graph...")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    # Verify graph is aggregated (has evidence_score)
    sample_edge = next(iter(kg.edges(data=True)), None)
    if sample_edge:
        u, v, data = sample_edge
        ev_score = data.get('evidence_score', None)
        if ev_score is None:
            print(f"\nERROR: Graph does not have evidence_score!")
            print(f"  Make sure to run aggregate_graphs.py first.")
            print(f"  Edge attributes: {list(data.keys())}")
            sys.exit(1)
        print(f"  Sample edge: probability={data.get('probability')}, evidence_score={ev_score}")
    
    # Run inference
    if args.hops == 2:
        inferred_edges = compute_two_hop_inference(kg, params)
    else:
        inferred_edges = compute_three_hop_inference(kg, params)
    
    if not inferred_edges:
        print("\nNo inferences found!")
        return
    
    # Filter to Gene -> Disease
    print("\nFiltering to Gene -> Disease pairs...")
    gene_disease_results = filter_gene_disease_results(inferred_edges, kg)
    
    # Add context information
    for result in gene_disease_results:
        result['tissue_context'] = args.context
    
    # Save results
    print("\nSaving results...")
    df = save_results(gene_disease_results, args.context, args.hops, output_dir, params)
    
    # Print summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(f"\nResults summary:")
    print(f"  Total Gene -> Disease pairs: {len(df):,}")
    if len(df) > 0:
        print(f"  Unique genes: {df['source_gene'].nunique():,}")
        print(f"  Unique diseases: {df['target'].nunique():,}")
        print(f"  Probability range: [{df['path_probability'].min():.4f}, {df['path_probability'].max():.4f}]")
        print(f"\nTop 10 genes by probability:")
        for i, row in df.head(10).iterrows():
            print(f"    {row['rank']:3d}. {row['source_gene']} -> {row['target']}: p={row['path_probability']:.4f}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)