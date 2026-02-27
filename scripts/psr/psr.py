"""
Perform two-hop PSR inference on an aggregated subgraph (MultiDiGraph-safe).
Infers indirect associations A -> Bj -> C following the PSR algorithm.
Edges are already aggregated before subsetting.
"""

import sys
import json
import traceback
from pathlib import Path
from collections import defaultdict
from contextlib import redirect_stdout
from tqdm import tqdm
import numpy as np
from metapath_utils import (
    group_paths_by_metapath, 
    create_metapath_edge_attrs,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats


# -------------------------------
# Helpers for node & edge access
# -------------------------------

def node_name(kg, n):
    """Get a display name for node n."""
    return kg.nodes[n].get('name', str(n))

def node_type(kg, n):
    """Get a semantic type/kind for node n."""
    return kg.nodes[n].get('type', kg.nodes[n].get('kind', 'unknown'))


def get_edge_properties(kg, u, v):
    """
    Extract probability, evidence_score, sign, type, and direction for an aggregated edge.
    MultiDiGraph-safe: if get_edge_data returns {key -> attrs}, prefer aggregated=True;
    otherwise take the one with highest evidence_score.
    """
    data = kg.get_edge_data(u, v)
    if data is None:
        return None

    # Inline unwrap for MultiDiGraph
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

    # Use canonical keys produced by aggregate.py / dataset_ikraph.py
    prob = edge_data.get('probability', 0.0)                     # <- default to 0.0
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
    """
    Returns True if there exists a direct A->C edge that is 'confident':
      - If require_directed: direction != '0'
      - If require_known_sign: correlation_type != 0
    MultiDiGraph-safe.
    """
    if not kg.has_edge(A, C):
        return False
    data = kg.get_edge_data(A, C)
    if data is None:
        return False

    # Normalize to iterable of attribute dicts
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

# =============================================================================
# PSR-specific aggregation functions
# =============================================================================

def aggregate_probabilities_psr(paths):
    """
    Aggregate probabilities using PSR formula: P = 1 - ∏(1 - p_i)
    
    This combines multiple paths within a metapath group.
    
    Args:
        paths: List of path dicts, each with 'probability' key
               (Each path probability is already calculated as product of edge probabilities)
    
    Returns:
        Combined probability as float
    """
    if not paths:
        return 0.0
    
    probs = np.array([p['probability'] for p in paths], dtype=np.float64)
    combined = 1.0 - np.prod(1.0 - probs)
    return float(combined)


def aggregate_evidence_scores(paths):
    """
    Aggregate evidence scores by summing across paths.
    
    Note: Each path's evidence_score is already calculated multiplicatively
    (product of edge evidence scores along that path). This function sums
    those path-level evidence scores across all paths in the metapath group.
    
    Args:
        paths: List of path dicts, each with 'evidence_score' key
               (Each already calculated as: edge1_evidence × edge2_evidence)
    
    Returns:
        Sum of path evidence scores
    """
    return float(sum(p['evidence_score'] for p in paths))

# -------------------------------
# Core algorithm
# -------------------------------

def compute_two_hop_inference(
    kg,
    max_intermediates=None,
    min_path_probability=0.01,
    consider_undirected=False,
    skip_when_any_direct_exists=True,
    require_directed_for_skip=True,
    require_known_sign_for_skip=True,
    grouping_strategy='mechanistic',
    split_inconsistent_correlations=True,
):
    """
    Compute two-hop inferences A -> Bj -> C for all paths in the graph.

    Args:
        kg: Knowledge graph (MultiDiGraph-compatible)
        max_intermediates: Max # of Bj to consider per (A,C); if None, unlimited
        min_path_probability: Minimum P(A->Bj)*P(Bj->C) to accept a path
        consider_undirected: If True, traverse edges with direction == '0' both ways
        skip_when_any_direct_exists: If True, skip inferring A->C if a direct edge
                                     already exists (subject to require_* flags)
        require_directed_for_skip: If True, only skip when the direct edge is directed
        require_known_sign_for_skip: If True, only skip when direct edge has known sign

    Returns:
        inferred_edges: Dict keyed by (A, C) with inference details.
    """
    print(f"\n{'='*80}")
    print("COMPUTING TWO-HOP PSR INFERENCE")
    print(f"{'='*80}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Direct edges: {kg.number_of_edges():,}")
    print(f"  Max intermediates per pair: {max_intermediates or 'unlimited'}")
    print(f"  Min path probability: {min_path_probability}")
    print(f"  Consider undirected edges: {consider_undirected}")
    print(f"  Skip when any direct exists: {skip_when_any_direct_exists} "
          f"(require_directed={require_directed_for_skip}, require_known_sign={require_known_sign_for_skip})")

    # Build neighbor dictionaries for fast lookup
    print("\nBuilding neighbor index...")
    successors_dict = defaultdict(set)
    predecessors_dict = defaultdict(set)

    for u, v, data in tqdm(kg.edges(data=True), desc="Indexing edges"):
        # For MultiDiGraph, data is a flat attr dict for each (u, v, key)
        direction = data.get('direction', '1')
        is_directed = (direction != '0')

        if is_directed:
            successors_dict[u].add(v)
            predecessors_dict[v].add(u)
        else:
            if consider_undirected:
                successors_dict[u].add(v)
                successors_dict[v].add(u)
                predecessors_dict[u].add(v)
                predecessors_dict[v].add(u)

    # Find all two-hop paths structurally
    print("\nFinding two-hop paths...")
    two_hop_paths = defaultdict(list)

    for Bj in tqdm(kg.nodes(), desc="Processing intermediates"):
        preds = predecessors_dict.get(Bj, set())
        succs = successors_dict.get(Bj, set())

        for A in preds:
            for C in succs:
                if A == C:
                    continue

                if skip_when_any_direct_exists:
                    if has_confident_direct(
                        kg, A, C,
                        require_directed=require_directed_for_skip,
                        require_known_sign=require_known_sign_for_skip
                    ):
                        continue

                two_hop_paths[(A, C)].append(Bj)

    print(f"Found {len(two_hop_paths):,} A-C pairs with two-hop paths")
    total_paths = sum(len(interms) for interms in two_hop_paths.values())
    print(f"Total two-hop paths: {total_paths:,}")

    # Compute inferred probabilities and scores
    print("\nComputing inferred probabilities and scores...")
    inferred_edges = {}  # Now keyed by (A, C, metapath_sig) if grouping, else (A, C)
    skipped_paths = 0

    for (A, C), intermediates in tqdm(two_hop_paths.items(), desc="Computing inferences"):
        # If limiting intermediates, rank by product of evidence scores
        if max_intermediates and len(intermediates) > max_intermediates:
            scored = []
            for Bj in intermediates:
                props_ab = get_edge_properties(kg, A, Bj)
                props_bc = get_edge_properties(kg, Bj, C)
                if props_ab and props_bc:
                    scored.append((Bj, props_ab['evidence_score'] * props_bc['evidence_score']))
            scored.sort(key=lambda x: -x[1])
            intermediates = [b for b, _ in scored[:max_intermediates]]

        # Build structured path information for each intermediate
        all_paths = []
        
        for Bj in intermediates:
            props_ab = get_edge_properties(kg, A, Bj)
            props_bc = get_edge_properties(kg, Bj, C)
            if props_ab is None or props_bc is None:
                continue

            path_prob = props_ab['probability'] * props_bc['probability']

            # Filter low-probability paths
            if path_prob < min_path_probability:
                skipped_paths += 1
                continue

            path_evidence = props_ab['evidence_score'] * props_bc['evidence_score']
            path_correlation = props_ab['correlation_type'] * props_bc['correlation_type']

            # Create structured path info
            path_info = {
                'path': [A, Bj, C],
                'node_types': (node_type(kg, A), node_type(kg, Bj), node_type(kg, C)),
                'relations': (props_ab['edge_type'], props_bc['edge_type']),
                'probability': path_prob,
                'correlation': path_correlation,
                'evidence_score': path_evidence,
                'intermediate': Bj,
                'props_ab': props_ab,
                'props_bc': props_bc,
            }
            
            all_paths.append(path_info)

        if not all_paths:
            continue

        if grouping_strategy is None:
            # Flat aggregation: combine all paths into a single inferred edge
            probs_array = np.array([p['probability'] for p in all_paths], dtype=np.float64)
            combined_prob = 1.0 - np.prod(1.0 - probs_array)
            combined_evidence = float(sum(p['evidence_score'] for p in all_paths))
            
            # Calculate weighted correlation
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
            
            # Create single edge for this (A, C) pair
            inferred_edges[(A, C)] = {
                'source': node_name(kg, A),
                'source_type': node_type(kg, A),
                'target': node_name(kg, C),
                'target_type': node_type(kg, C),
                'probability': round(float(combined_prob), 6),
                'evidence_score': round(float(combined_evidence), 4),
                'correlation_type': int(combined_correlation),
                'num_paths': len(all_paths),
                'num_intermediates_tested': len(intermediates),
            }
        
        else:
            # Metapath grouping: one inferred edge per metapath signature
            metapath_groups = group_paths_by_metapath(
                all_paths, 
                strategy=grouping_strategy,
                split_inconsistent_correlations=split_inconsistent_correlations
            )
            
            # Create list of edge dicts, one per metapath
            edge_list = []
            
            for metapath_sig, group_info in metapath_groups.items():
                paths_in_group = group_info['paths']
                was_split = group_info['was_split']
                correlation = group_info['correlation']
                
                # Aggregate probabilities for this metapath
                combined_prob = aggregate_probabilities_psr(paths_in_group)
                combined_evidence = aggregate_evidence_scores(paths_in_group)
                
                # Create edge attributes
                edge_attrs = create_metapath_edge_attrs(
                    source=A,
                    target=C,
                    kg=kg,
                    metapath_sig=metapath_sig,
                    paths=paths_in_group,
                    combined_prob=combined_prob,
                    combined_evidence=combined_evidence,
                    correlation=correlation,
                    path_length=2,
                    was_split=was_split,
                    grouping_strategy=grouping_strategy
                )
                
                # Add source/target names for JSON output
                edge_attrs['source'] = node_name(kg, A)
                edge_attrs['source_type'] = node_type(kg, A)
                edge_attrs['target'] = node_name(kg, C)
                edge_attrs['target_type'] = node_type(kg, C)
                edge_attrs['num_intermediates_tested'] = len(intermediates)
                
                edge_list.append(edge_attrs)
            
            # Store list - key is just (A, C), no metapath_sig
            inferred_edges[(A, C)] = edge_list

    print(f"\nInferred {len(inferred_edges):,} indirect associations")
    print(f"Skipped {skipped_paths:,} low-probability paths")

    if inferred_edges:
        # Flatten: handle both dict and list formats
        all_edges = []
        for edge_data in inferred_edges.values():
            if isinstance(edge_data, list):
                all_edges.extend(edge_data)
            else:
                all_edges.append(edge_data)
        
        probs = [e['probability'] for e in all_edges]
        scores = [e['evidence_score'] for e in all_edges]
        num_paths = [e['num_paths'] for e in all_edges]

        print(f"\nInferred probability distribution:")
        print(f"  Min: {min(probs):.6f}, Max: {max(probs):.6f}, Mean: {np.mean(probs):.6f}")
        print(f"  Median: {np.median(probs):.6f}, Std: {np.std(probs):.6f}")

        print(f"\nInferred evidence score distribution:")
        print(f"  Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}, Std: {np.std(scores):.4f}")

        print(f"\nPaths per inference:")
        print(f"  Min: {min(num_paths)}, Max: {max(num_paths)}, Mean: {np.mean(num_paths):.1f}")

        # Count correlation types across the flattened edge list (handles both
        # legacy single-dict and metapath list formats)
        corr_counts = defaultdict(int)
        for e in all_edges:
            # Some generators use 'correlation_type', others may use 'correlation'
            corr_type = e.get('correlation_type', e.get('correlation', 0))
            try:
                corr_counts[int(corr_type)] += 1
            except Exception:
                # Fallback: count as unknown (0)
                corr_counts[0] += 1
        print(f"\nCorrelation type distribution:")
        total = len(inferred_edges)
        print(f"  Positive (+1): {corr_counts[1]:,} ({100*corr_counts[1]/total:.1f}%)")
        print(f"  Negative (-1): {corr_counts[-1]:,} ({100*corr_counts[-1]/total:.1f}%)")
        print(f"  Unknown (0): {corr_counts[0]:,} ({100*corr_counts[0]/total:.1f}%)")

    return inferred_edges


def create_inferred_graph(kg, inferred_edges, min_probability=0.1, min_evidence=1.0):
    """
    Create a new graph with both direct and inferred edges.
    
    Handles both formats:
    - Legacy: inferred_edges[(A,C)] = {single edge dict}
    - Metapath: inferred_edges[(A,C)] = [list of edge dicts]
    
    MultiDiGraph will automatically create parallel edges for multiple metapaths.
    """
    print(f"\n{'='*80}")
    print("CREATING GRAPH WITH INFERRED EDGES")
    print(f"{'='*80}")
    print(f"  Min probability: {min_probability}")
    print(f"  Min evidence score: {min_evidence}")

    inferred_kg = kg.__class__(schema=kg.schema)

    # Copy all nodes
    for node in tqdm(kg.nodes(), desc="Adding nodes"):
        inferred_kg.add_node(node, **kg.nodes[node])

    # Copy all direct edges
    direct_count = 0
    for u, v, data in tqdm(kg.edges(data=True), desc="Adding direct edges"):
        inferred_kg.add_edge(u, v, **data)
        direct_count += 1

    # Add inferred edges
    added_count = 0
    filtered_count = 0

    for (A, C), edge_data in tqdm(inferred_edges.items(), desc="Adding inferred edges"):
        # Detect format: dict (legacy) or list (metapath)
        if isinstance(edge_data, dict):
            # Legacy format: single edge
            edge_list = [edge_data]
        elif isinstance(edge_data, list):
            # Metapath format: multiple edges
            edge_list = edge_data
        else:
            print(f"WARNING: Unexpected edge data type for ({A}, {C}): {type(edge_data)}")
            continue
        
        # Add each edge (MultiDiGraph creates parallel edges automatically)
        for data in edge_list:
            if data['probability'] >= min_probability and data['evidence_score'] >= min_evidence:
                # Remove metadata that shouldn't be edge attributes
                edge_attrs = {
                    k: v for k, v in data.items() 
                    if k not in ['source', 'target', 'source_type', 'target_type']
                }
                
                # Add edge - MultiDiGraph will create parallel edge if needed
                inferred_kg.add_edge(A, C, **edge_attrs)
                added_count += 1
            else:
                filtered_count += 1

    print(f"\nAdded {added_count:,} inferred edges")
    print(f"Filtered out {filtered_count:,} low-confidence inferences")
    print(f"\nFinal graph composition:")
    print(f"  Direct edges: {direct_count:,}")
    print(f"  Inferred edges: {added_count:,}")
    print(f"  Total edges: {inferred_kg.number_of_edges():,}")
    print(f"  Total nodes: {inferred_kg.number_of_nodes():,}")

    return inferred_kg


def save_outputs(inferred_kg, inferred_edges, input_path, output_dir, params):
    """Save outputs with consistent, truthful metadata."""
    input_stem = input_path.stem
    # Add metapath/grouping info to output filenames for clarity
    grouping = params.get('grouping_strategy')
    split_flag = params.get('split_inconsistent_correlations', False)
    grouping_part = grouping if grouping else "no_grouping"
    split_part = "split" if split_flag else "nosplit"
    # sanitize stem (avoid spaces or problematic chars in filenames)
    safe_stem = input_stem.replace(" ", "_")
    input_set = f"{safe_stem}_metapath_{grouping_part}_{split_part}"

    print(f"\nSaving inferred edges...")
    output_json = output_dir / f"{input_set}_inferred_edges.json"

    ## Sort by probability and save top N
    # Flatten and sort all edges (handle both single dict and list formats)
    all_edges_with_keys = []
    for key, edge_data in inferred_edges.items():
        if isinstance(edge_data, list):
            # Multiple metapaths: add each edge dict separately
            for edge_dict in edge_data:
                all_edges_with_keys.append((key, edge_dict))
        else:
            # Single edge (legacy)
            all_edges_with_keys.append((key, edge_data))

    # Sort by probability (descending). Use .get to avoid KeyError if missing.
    all_edges_with_keys.sort(key=lambda x: -float(x[1].get('probability', 0.0)))

    # Take top N and extract just the edge dicts
    max_to_save = min(5000, len(all_edges_with_keys))
    inferred_list = [edge_dict for _, edge_dict in all_edges_with_keys[:max_to_save]]

    with open(output_json, 'w') as f:
        json.dump({
            'metadata': {
                'source_graph': str(input_path),
                'num_total_inferred': len(inferred_edges),
                'num_saved': len(inferred_list),
                **params,  # record exactly what was used
            },
            'inferred_edges': inferred_list
        }, f, indent=2)
    print(f"Saved top {len(inferred_list):,} inferences to JSON")

    # Save combined graph
    output_pkl = output_dir / f"{input_set}_with_inferred.pkl"
    print(f"\nSaving combined graph...")
    inferred_kg.export_graph(output_pkl)

    # Save schema and stats
    info_dir = output_dir.parent.parent / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    schema_file = info_dir / f"{input_set}_with_inferred_schema.txt"
    with open(schema_file, 'w') as f:
        f.write(str(inferred_kg.schema))

    stats_file = info_dir / f"{input_set}_with_inferred_stats.txt"
    with open(stats_file, 'w') as f:
        with redirect_stdout(f):
            print_kg_stats(inferred_kg)

    # Visualize if small
    if inferred_kg.number_of_nodes() < 1000:
        print(f"\nCreating visualizations...")
        viz_dir = output_dir.parent.parent / "results" / "graph_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        output_html = viz_dir / f"{input_set}_with_inferred.html"
        inferred_kg.visualize(
            str(output_html),
            title=f"{input_set.replace('_', ' ').title()} + Inferred"
        )

        output_png = viz_dir / f"{input_set}_with_inferred.png"
        inferred_kg.visualize(
            str(output_png),
            figsize=(20, 15),
            node_size=500,
            font_size=5
        )

        
        print(f"Saved visualizations")
    else:
        print(f"\nSkipping visualization (graph has {inferred_kg.number_of_nodes():,} nodes)")

    return output_json, output_pkl


def main():
    # -----------------------------
    # Configuration (edit here)
    # -----------------------------
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")

    # Choose graph
    input_dir = base_path / "graphs/subsets"
    input_graph = input_dir / "prototype_8_12_aggregated.pkl"
    # For full subgraph:
    # input_dir = base_path / "graphs/subsets/ikraph"
    # input_graph = input_dir / "adipose_inflammation_snn_expanded.pkl"

    output_dir = input_dir / "inferred_metapath_mechanistic/"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters (kept in one dict and threaded everywhere)
    params = dict(
        max_intermediates=None,        # e.g., 100
        min_path_probability=0.001,     # filter weak paths
        consider_undirected=False,     # treat undirected edges as bidirectional
        skip_when_any_direct_exists=True,  # skip if confident direct A->C exists
        require_directed_for_skip=False, # only skip if direct edge is directed, so directed = '0' won't block
        require_known_sign_for_skip=False,

        # Metapath grouping parameters
        grouping_strategy='mechanistic',  # Options: 'mechanistic', 'semantic', or None
        split_inconsistent_correlations=True,  # For mechanistic: split if correlations disagree

        # thresholds for adding inferred edges into final graph
        min_inference_probability=0.0,
        min_inference_evidence=0.0,
    )

    print("=" * 80)
    print("TWO-HOP PSR INFERENCE")
    print("Edges are pre-aggregated from literature mentions")
    print("=" * 80)
    print(f"Input: {input_graph.name}")
    print(f"Output directory: {output_dir}")

    print(f"\nLoading: {input_graph}")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    # Sanity: check aggregated flag presence in some edges
    sample_edges = list(kg.edges(data=True))[:5]
    has_aggregated = any('aggregated' in data for _, _, data in sample_edges)
    if not has_aggregated:
        print("\nWARNING: Graph edges may not be aggregated. Run aggregate.py first for best results.")

    # Run inference
    inferred_edges = compute_two_hop_inference(
        kg,
        max_intermediates=params['max_intermediates'],
        min_path_probability=params['min_path_probability'],
        consider_undirected=params['consider_undirected'],
        skip_when_any_direct_exists=params['skip_when_any_direct_exists'],
        require_directed_for_skip=params['require_directed_for_skip'],
        require_known_sign_for_skip=params['require_known_sign_for_skip'],
        grouping_strategy=params.get('grouping_strategy', None),
        split_inconsistent_correlations=params.get('split_inconsistent_correlations', True),
    )

    if not inferred_edges:
        print("\nNo inferences found. This could mean:")
        print("  - All possible indirect paths already have confident direct edges")
        print("  - No valid two-hop paths with probability >= min_path_probability")
        print("  - Edge probabilities missing/low and filtered out")
        return

    inferred_kg = create_inferred_graph(
        kg,
        inferred_edges,
        min_probability=params['min_inference_probability'],
        min_evidence=params['min_inference_evidence'],
    )

    output_json, output_pkl = save_outputs(
        inferred_kg,
        inferred_edges,
        input_graph,
        output_dir,
        params
    )

    # Summary
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)

    print(f"\nOriginal graph:")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Edges: {kg.number_of_edges():,}")

    print(f"\nInference results:")
    print(f"  Potential inferences found: {len(inferred_edges):,}")
    # Count inferences above threshold
    count_above_threshold = 0
    for edge_data in inferred_edges.values():
        if isinstance(edge_data, list):
            for e in edge_data:
                if e['probability'] >= params['min_inference_probability'] and e['evidence_score'] >= params['min_inference_evidence']:
                    count_above_threshold += 1
        else:
            if edge_data['probability'] >= params['min_inference_probability'] and edge_data['evidence_score'] >= params['min_inference_evidence']:
                count_above_threshold += 1

    print(f"  Inferences above threshold: {count_above_threshold:,}")

    print(f"\nCombined graph:")
    print(f"  Nodes: {inferred_kg.number_of_nodes():,}")
    print(f"  Total edges: {inferred_kg.number_of_edges():,}")

    edge_types = defaultdict(int)
    for _, _, data in inferred_kg.edges(data=True):
        edge_types[data.get('type', 'unknown')] += 1

    print(f"\nEdge type breakdown:")
    for etype, count in sorted(edge_types.items()):
        print(f"  {etype}: {count:,}")

    print(f"\nOutput files:")
    print(f"  - {Path(output_json).name} (inferred edges JSON)")
    print(f"  - {Path(output_pkl).name} (combined graph)")

    if inferred_kg.number_of_nodes() < 1000:
        print(f"  - Visualizations in graphs/visualizations/")
    print(f"  - Schema and stats in graphs/info/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)