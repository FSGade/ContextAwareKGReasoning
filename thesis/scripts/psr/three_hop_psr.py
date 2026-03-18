"""
Perform three-hop PSR inference on an aggregated subgraph (MultiDiGraph-safe).
Infers indirect associations A -> B -> C -> D following the PSR algorithm.
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
    MultiDiGraph-safe: if get_edge_data returns {key -> attrs}, prefer aggregated=True.
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

def has_confident_direct(kg, A, D, require_directed=True, require_known_sign=True):
    """
    Returns True if there exists a direct A->D edge that is 'confident'.
    """
    if not kg.has_edge(A, D):
        return False
    data = kg.get_edge_data(A, D)
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

# =============================================================================
# PSR-specific aggregation functions
# =============================================================================

def aggregate_probabilities_psr(paths):
    """
    Aggregate probabilities using PSR formula: P = 1 - ∏(1 - p_i)
    
    For 3-hop: combines path probabilities where each is already p_AB × p_BC × p_CD
    """
    if not paths:
        return 0.0
    
    probs = np.array([p['probability'] for p in paths], dtype=np.float64)
    combined = 1.0 - np.prod(1.0 - probs)
    return float(combined)


def aggregate_evidence_scores(paths):
    """
    Aggregate evidence scores by summing across paths.
    
    For 3-hop: each path evidence = evidence_AB × evidence_BC × evidence_CD
    Then sum across all paths in the metapath group.
    """
    return float(sum(p['evidence_score'] for p in paths))

# -------------------------------
# Core algorithm: 3-hop inference
# -------------------------------

def compute_three_hop_inference(
    kg,
    max_intermediates=None,
    min_path_probability=0.001,
    consider_undirected=False,
    skip_when_any_direct_exists=True,
    require_directed_for_skip=True,
    require_known_sign_for_skip=True,
    grouping_strategy='mechanistic',
    split_inconsistent_correlations=True,
):
    """
    Compute three-hop inferences A -> B -> C -> D for all paths in the graph.

    Args:
        kg: Knowledge graph (MultiDiGraph-compatible)
        max_intermediates: Max # of (B,C) pairs to consider per (A,D); if None, unlimited
        min_path_probability: Minimum P(A->B)*P(B->C)*P(C->D) to accept a path
        consider_undirected: If True, traverse edges with direction == '0' both ways
        skip_when_any_direct_exists: If True, skip inferring A->D if a direct edge
                                     already exists (subject to require_* flags)
        require_directed_for_skip: If True, only skip when the direct edge is directed
        require_known_sign_for_skip: If True, only skip when direct edge has known sign

    Returns:
        inferred_edges: Dict keyed by (A, D) with inference details.
    """
    print(f"\n{'='*80}")
    print("COMPUTING THREE-HOP PSR INFERENCE")
    print(f"{'='*80}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Direct edges: {kg.number_of_edges():,}")
    print(f"  Max intermediate pairs per (A,D): {max_intermediates or 'unlimited'}")
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

    # Find all three-hop paths structurally
    print("\nFinding three-hop paths...")
    
    # Progress estimate: count B nodes that have both predecessors and successors
    active_B_nodes = [
        B for B in kg.nodes()
        if predecessors_dict.get(B) and successors_dict.get(B)
    ]
    print(f"Processing {len(active_B_nodes):,} B nodes with both predecessors and successors")
    
    three_hop_paths = defaultdict(list)

    for B in tqdm(active_B_nodes, desc="Processing B nodes"):
        preds_B = predecessors_dict.get(B, set())
        succs_B = successors_dict.get(B, set())

        for C in succs_B:
            if B == C:
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

                    if skip_when_any_direct_exists:
                        if has_confident_direct(
                            kg, A, D,
                            require_directed=require_directed_for_skip,
                            require_known_sign=require_known_sign_for_skip
                        ):
                            continue

                    # Record this three-hop path: A -> B -> C -> D
                    three_hop_paths[(A, D)].append((B, C))

    print(f"Found {len(three_hop_paths):,} A-D pairs with three-hop paths")
    total_paths = sum(len(interms) for interms in three_hop_paths.values())
    print(f"Total three-hop paths: {total_paths:,}")

    # Compute inferred edges using PSR equations
    print("\nComputing inferred probabilities and scores...")
    inferred_edges = {}  # Keyed by (A, D, metapath_sig) if grouping, else (A, D)
    skipped_paths = 0

    for (A, D), intermediate_pairs in tqdm(three_hop_paths.items(), desc="Computing 3-hop inferences"):
        
        # =====================================================================
        # Optional: Limit number of intermediate pairs to consider
        # =====================================================================
        if max_intermediates and len(intermediate_pairs) > max_intermediates:
            # Rank intermediate pairs by product of all three edge evidence scores
            scored = []
            for (B, C) in intermediate_pairs:
                props_ab = get_edge_properties(kg, A, B)
                props_bc = get_edge_properties(kg, B, C)
                props_cd = get_edge_properties(kg, C, D)
                if props_ab and props_bc and props_cd:
                    total_evidence = (props_ab['evidence_score'] * 
                                    props_bc['evidence_score'] * 
                                    props_cd['evidence_score'])
                    scored.append(((B, C), total_evidence))
            
            # Keep only top N by evidence
            scored.sort(key=lambda x: -x[1])
            intermediate_pairs = [pair for pair, _ in scored[:max_intermediates]]

        # =====================================================================
        # Build structured path information for all paths A→B→C→D
        # =====================================================================
        all_paths = []
        
        for (B, C) in intermediate_pairs:
            # Get properties for all three edges
            props_ab = get_edge_properties(kg, A, B)
            props_bc = get_edge_properties(kg, B, C)
            props_cd = get_edge_properties(kg, C, D)
            
            # Skip if any edge is missing
            if props_ab is None or props_bc is None or props_cd is None:
                continue

            # Calculate path-level metrics (multiplicative across edges)
            path_prob = (props_ab['probability'] * 
                        props_bc['probability'] * 
                        props_cd['probability'])

            # Filter out very low-probability paths
            if path_prob < min_path_probability:
                skipped_paths += 1
                continue

            path_evidence = (props_ab['evidence_score'] * 
                           props_bc['evidence_score'] * 
                           props_cd['evidence_score'])
            
            path_correlation = (props_ab['correlation_type'] * 
                              props_bc['correlation_type'] * 
                              props_cd['correlation_type'])

            # Create structured path info dictionary
            path_info = {
                # The complete 4-node path
                'path': [A, B, C, D],
                
                # Node types at each position (4 types for 3-hop)
                'node_types': (
                    node_type(kg, A), 
                    node_type(kg, B), 
                    node_type(kg, C), 
                    node_type(kg, D)
                ),
                
                # Relation types for the 3 edges
                'relations': (
                    props_ab['edge_type'], 
                    props_bc['edge_type'], 
                    props_cd['edge_type']
                ),
                
                # Aggregated metrics for this path
                'probability': path_prob,
                'correlation': path_correlation,
                'evidence_score': path_evidence,
                
                # Store edge properties for reference
                'props_ab': props_ab,
                'props_bc': props_bc,
                'props_cd': props_cd,
            }
            
            all_paths.append(path_info)

        # Skip if no valid paths found
        if not all_paths:
            continue

        # =====================================================================
        # BRANCHING: Legacy behavior vs Metapath grouping
        # =====================================================================
        
        if grouping_strategy is None:
            # -----------------------------------------------------------------
            # Flat aggregation: combine all paths into a single inferred edge
            # Creates single edge (A, D) with all paths combined
            # -----------------------------------------------------------------
            
            probs_array = np.array([p['probability'] for p in all_paths], dtype=np.float64)
            combined_prob = 1.0 - np.prod(1.0 - probs_array)
            combined_evidence = float(sum(p['evidence_score'] for p in all_paths))
            
            # Calculate weighted correlation
            if combined_evidence > 0:
                weighted_corr = sum(
                    p['correlation'] * p['evidence_score'] 
                    for p in all_paths
                ) / combined_evidence
                
                if weighted_corr > 0.5:
                    combined_correlation = 1
                elif weighted_corr < -0.5:
                    combined_correlation = -1
                else:
                    combined_correlation = 0
            else:
                combined_correlation = 0

            # Create single aggregated edge
            inferred_edges[(A, D)] = {
                'source': node_name(kg, A),
                'source_type': node_type(kg, A),
                'target': node_name(kg, D),
                'target_type': node_type(kg, D),
                'probability': round(float(combined_prob), 6),
                'evidence_score': round(float(combined_evidence), 4),
                'correlation_type': int(combined_correlation),
                'num_paths': len(all_paths),
                'num_intermediate_pairs_tested': len(intermediate_pairs),
                'kind': 'inferred_3_hop',
                'aggregated': True,  # All paths aggregated together
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
                
                # Aggregate within this metapath using PSR formulas
                combined_prob = aggregate_probabilities_psr(paths_in_group)
                combined_evidence = aggregate_evidence_scores(paths_in_group)
                
                # Create comprehensive edge attributes
                edge_attrs = create_metapath_edge_attrs(
                    source=A,
                    target=D,
                    kg=kg,
                    metapath_sig=metapath_sig,
                    paths=paths_in_group,
                    combined_prob=combined_prob,
                    combined_evidence=combined_evidence,
                    correlation=correlation,
                    path_length=3,  # ← 3 for three-hop
                    was_split=was_split,
                    grouping_strategy=grouping_strategy
                )
                
                # Add additional metadata for JSON output
                edge_attrs['source'] = node_name(kg, A)
                edge_attrs['source_type'] = node_type(kg, A)
                edge_attrs['target'] = node_name(kg, D)
                edge_attrs['target_type'] = node_type(kg, D)
                edge_attrs['num_intermediate_pairs_tested'] = len(intermediate_pairs)
                
                edge_list.append(edge_attrs)
            
            # Store list
            inferred_edges[(A, D)] = edge_list

    # =========================================================================
    # Statistics and reporting
    # =========================================================================
    
    print(f"\nInferred {len(inferred_edges):,} indirect 3-hop associations")
    print(f"Skipped {skipped_paths:,} low-probability paths")
    
    if grouping_strategy is not None:
        # Additional stats for metapath grouping
        unique_pairs = len(inferred_edges)  # Already unique (A, D) pairs
        
        # Count total metapaths across all pairs
        total_metapaths = sum(
            len(edge_data) if isinstance(edge_data, list) else 1
            for edge_data in inferred_edges.values()
        )
        
        print(f"  Unique (source, target) pairs: {unique_pairs:,}")
        print(f"  Total metapaths: {total_metapaths:,}")
        print(f"  Average metapaths per pair: {total_metapaths/unique_pairs:.1f}")

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
        print(f"  Median: {np.median(num_paths):.0f}")

    return inferred_edges


# -------------------------------
# Create combined graph with inferred edges
# -------------------------------

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


# -------------------------------
# Save outputs (JSON + PKL + stats)
# -------------------------------

def save_outputs(inferred_kg, inferred_edges, input_path, output_dir, params):
    """Save outputs with metadata."""
    input_stem = input_path.stem

    print(f"\nSaving inferred edges...")
    output_json = output_dir / f"{input_stem}_three_hop_inferred_edges.json"

    # Sort by probability and save top N
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
                **params,
            },
            'inferred_edges': inferred_list
        }, f, indent=2)
    print(f"Saved top {len(inferred_list):,} inferences to JSON")

    # Save combined graph
    output_pkl = output_dir / f"{input_stem}_three_hop_with_inferred.pkl"
    print(f"\nSaving combined graph...")
    inferred_kg.export_graph(output_pkl)

    # Save schema and stats
    info_dir = output_dir.parent.parent / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    schema_file = info_dir / f"{input_stem}_three_hop_with_inferred_schema.txt"
    with open(schema_file, 'w') as f:
        f.write(str(inferred_kg.schema))

    stats_file = info_dir / f"{input_stem}_three_hop_with_inferred_stats.txt"
    with open(stats_file, 'w') as f:
        with redirect_stdout(f):
            print_kg_stats(inferred_kg)

    # Visualize if small
    if inferred_kg.number_of_nodes() < 500:
        print(f"\nCreating visualizations...")
        viz_dir = output_dir.parent.parent / "results" / "graph_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        output_html = viz_dir / f"{input_stem}_three_hop_with_inferred.html"
        inferred_kg.visualize(
            str(output_html),
            title=f"{input_stem.replace('_', ' ').title()} + Three-Hop Inferred"
        )
        print(f"Saved visualization")
    else:
        print(f"\nSkipping visualization (graph has {inferred_kg.number_of_nodes():,} nodes)")

    return output_json, output_pkl


# -------------------------------
# Comparison utility
# -------------------------------

def compare_two_hop_and_three_hop(two_hop_graph_path, three_hop_graph_path):
    """
    Load both graphs and compare 2-hop vs 3-hop inferred edges.
    
    Args:
        two_hop_graph_path: Path to graph with 2-hop inferences
        three_hop_graph_path: Path to graph with 3-hop inferences
    """
    print(f"\n{'='*80}")
    print("COMPARING 2-HOP vs 3-HOP INFERENCES")
    print(f"{'='*80}")
    
    # Load both graphs
    print(f"\nLoading 2-hop graph: {two_hop_graph_path}")
    kg_2hop = KnowledgeGraph.import_graph(str(two_hop_graph_path))
    
    print(f"Loading 3-hop graph: {three_hop_graph_path}")
    kg_3hop = KnowledgeGraph.import_graph(str(three_hop_graph_path))
    
    # Extract inferred edges from each
    inferred_2hop = set()
    for u, v, data in kg_2hop.edges(data=True):
        if data.get('type') == 'inferred_two_hop' or data.get('inferred'):
            inferred_2hop.add((
                kg_2hop.nodes[u].get('name', str(u)),
                kg_2hop.nodes[v].get('name', str(v))
            ))
    
    inferred_3hop = set()
    for u, v, data in kg_3hop.edges(data=True):
        if data.get('type') == 'inferred_three_hop' or data.get('inferred'):
            inferred_3hop.add((
                kg_3hop.nodes[u].get('name', str(u)),
                kg_3hop.nodes[v].get('name', str(v))
            ))
    
    # Comparison
    overlap = inferred_2hop & inferred_3hop
    only_2hop = inferred_2hop - inferred_3hop
    only_3hop = inferred_3hop - inferred_2hop
    
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"\n2-hop inferences: {len(inferred_2hop):,}")
    print(f"3-hop inferences: {len(inferred_3hop):,}")
    print(f"\nOverlap: {len(overlap):,} edges")
    if len(inferred_2hop) > 0:
        print(f"  ({len(overlap)/len(inferred_2hop)*100:.1f}% of 2-hop inferences)")
    
    print(f"\nOnly in 2-hop: {len(only_2hop):,} edges")
    if only_2hop:
        print(f"  Sample: {list(only_2hop)[:3]}")
    
    print(f"\nOnly in 3-hop: {len(only_3hop):,} edges")
    if len(inferred_2hop) > 0:
        print(f"  (+{len(only_3hop)/len(inferred_2hop)*100:.1f}% new inferences)")
    if only_3hop:
        print(f"  Sample: {list(only_3hop)[:3]}")
    
    # Node type breakdown for new 3-hop inferences
    if only_3hop and kg_3hop:
        print(f"\nNew 3-hop inferences by source-target type:")
        type_counts = defaultdict(int)
        for source_name, target_name in only_3hop:
            # Find nodes by name
            source_node = None
            target_node = None
            for n, data in kg_3hop.nodes(data=True):
                if data.get('name') == source_name:
                    source_node = n
                if data.get('name') == target_name:
                    target_node = n
            
            if source_node and target_node:
                src_type = node_type(kg_3hop, source_node)
                tgt_type = node_type(kg_3hop, target_node)
                type_counts[(src_type, tgt_type)] += 1
        
        for (src_type, tgt_type), count in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {src_type} → {tgt_type}: {count:,}")
    
    return {
        'two_hop': len(inferred_2hop),
        'three_hop': len(inferred_3hop),
        'overlap': len(overlap),
        'only_2hop': len(only_2hop),
        'only_3hop': len(only_3hop),
    }


# -------------------------------
# Main
# -------------------------------

def main():
    """Example usage of three-hop PSR inference."""
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")

    # Input graph
    input_dir = base_path / "graphs/subsets"
    input_graph = input_dir / "prototype_8_12_aggregated.pkl"

    output_dir = input_dir / "inferred_metapath_mechanistic/"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    params = dict(
        max_intermediates=None,        # No limit on intermediate pairs
        min_path_probability=0.001,    # Filter weak paths
        consider_undirected=False,     # Only directed edges
        skip_when_any_direct_exists=True,
        require_directed_for_skip=False, 
        require_known_sign_for_skip=False,

        # Metapath grouping parameters
        grouping_strategy='mechanistic',  # Options: 'mechanistic', 'semantic', or None
        split_inconsistent_correlations=True,  # For mechanistic: split if correlations disagree

        # Thresholds for adding to final graph
        min_inference_probability=0.001,
        min_inference_evidence=0.0,
    )

    print("=" * 80)
    print("THREE-HOP PSR INFERENCE")
    print("Extends two-hop to three-hop: A -> B -> C -> D")
    print("=" * 80)
    print(f"Input: {input_graph.name}")
    print(f"Output directory: {output_dir}")

    print(f"\nLoading: {input_graph}")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    # Run inference
    inferred_edges = compute_three_hop_inference(
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
        print("\nNo inferences found.")
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
    print(f"\nOutput files:")
    print(f"  - {Path(output_json).name}")
    print(f"  - {Path(output_pkl).name}")
    
    # Optional: Compare with 2-hop if available
    two_hop_graph = input_dir / "inferred" / f"{input_graph.stem}_with_inferred.pkl"
    if two_hop_graph.exists():
        print(f"\n2-hop graph found, running comparison...")
        compare_two_hop_and_three_hop(two_hop_graph, output_pkl)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)