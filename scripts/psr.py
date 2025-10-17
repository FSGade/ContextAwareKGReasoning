""" Perform two-hop PSR inference on aggregated subgraph.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats


def get_edge_properties(kg, u, v):
    """Extract probability, evidence_score, and sign from aggregated edge."""
    edge_data = kg.get_edge_data(u, v)
    if edge_data is None:
        return None
    
    # Get aggregated probability
    prob = edge_data.get('probability', 0.0)
    if prob <= 0.0:
        return None
    
    # Get evidence score from aggregation
    evidence_score = edge_data.get('evidence_score', 0.0)
    # Get correlation type for sign
    # correlation_type: 1 = positive, -1 = negative, 0 = unknown
    correlation_type = edge_data.get('correlation_type', 0)
    
    # Get edge type and direction
    edge_type = edge_data.get('type', edge_data.get('kind', 'unknown'))
    direction = edge_data.get('direction', '1')
    is_directed = (direction != '0')
    
    return {
        'probability': prob,
        'evidence_score': evidence_score,
        'correlation_type': correlation_type,
        'edge_type': edge_type,
        'is_directed': is_directed,
        'n_supporting_edges': edge_data.get('n_supporting_edges', 1),
        'n_documents': edge_data.get('n_documents', 1)
    }


def compute_two_hop_inference(kg, max_intermediates=None, min_path_probability=0.01,
                               consider_undirected=True):
    """
    Compute two-hop inferences A -> Bj -> C for all paths in the graph.
    Follows the PSR algorithm from the iKraph paper.
    
    Args:
        kg: Knowledge graph with aggregated edges
        max_intermediates: Maximum number of intermediate nodes to consider per pair
        min_path_probability: Minimum probability threshold for individual paths
        consider_undirected: If True, treat undirected edges as bidirectional
    
    Returns:
        inferred_edges: Dict mapping (A, C) to inference results
    """
    print(f"\n{'='*80}")
    print("COMPUTING TWO-HOP PSR INFERENCE")
    print(f"{'='*80}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Direct edges: {kg.number_of_edges():,}")
    print(f"  Max intermediates per pair: {max_intermediates or 'unlimited'}")
    print(f"  Min path probability: {min_path_probability}")
    print(f"  Consider undirected edges: {consider_undirected}")
    
    # Build neighbor dictionaries for fast lookup
    print("\nBuilding neighbor index...")
    successors_dict = defaultdict(set)
    predecessors_dict = defaultdict(set)
    
    for u, v, data in tqdm(kg.edges(data=True), desc="Indexing edges"):
        direction = data.get('direction', '1')
        is_directed = (direction != '0')
        
        if is_directed:
            successors_dict[u].add(v)
            predecessors_dict[v].add(u)
        else:
            # Undirected edges can be traversed both ways
            if consider_undirected:
                successors_dict[u].add(v)
                successors_dict[v].add(u)
                predecessors_dict[u].add(v)
                predecessors_dict[v].add(u)
    
    # Find all two-hop paths
    print("\nFinding two-hop paths...")
    two_hop_paths = defaultdict(list)
    
    for Bj in tqdm(kg.nodes(), desc="Processing intermediate nodes"):
        # Get predecessors (A -> Bj) and successors (Bj -> C)
        predecessors = predecessors_dict.get(Bj, set())
        successors = successors_dict.get(Bj, set())
        
        # For each A -> Bj -> C path
        for A in predecessors:
            for C in successors:
                # Skip if A == C (to avoid self-loops)
                if A == C:
                    continue
                
                # Skip if direct edge A -> C already exists
                if kg.has_edge(A, C):
                    continue
                
                # Store the intermediate for this A-C pair
                two_hop_paths[(A, C)].append(Bj)
    
    print(f"Found {len(two_hop_paths):,} A-C pairs with two-hop paths")
    total_paths = sum(len(intermediates) for intermediates in two_hop_paths.values())
    print(f"Total two-hop paths: {total_paths:,}")
    
    # Compute inferred edges using PSR equations
    print("\nComputing inferred probabilities and scores...")
    inferred_edges = {}
    skipped_paths = 0
    
    for (A, C), intermediates in tqdm(two_hop_paths.items(), desc="Computing inferences"):
        # Limit number of intermediates if specified
        if max_intermediates and len(intermediates) > max_intermediates:
            # Score each path and take top-k by evidence score
            intermediate_scores = []
            for Bj in intermediates:
                props_ab = get_edge_properties(kg, A, Bj)
                props_bc = get_edge_properties(kg, Bj, C)
                if props_ab and props_bc:
                    # Score by product of evidence scores
                    path_score = props_ab['evidence_score'] * props_bc['evidence_score']
                    intermediate_scores.append((Bj, path_score))

            # Sort by score and take top-k
            intermediate_scores.sort(key=lambda x: -x[1])
            intermediates = [b for b, _ in intermediate_scores[:max_intermediates]]
        
        # Compute probability, score, and correlation for each path
        path_probabilities = []
        path_evidence_scores = []
        path_correlations = []
        path_details = []
        
        for Bj in intermediates:
            # Get edge properties for both edges
            props_ab = get_edge_properties(kg, A, Bj)
            props_bc = get_edge_properties(kg, Bj, C)
            
            if props_ab is None or props_bc is None:
                continue
            
            # Two-hop probability: P_{A,Bj,C} = P_{A,Bj} × P_{Bj,C} (Eq. 2 in iKraph paper)
            path_prob = props_ab['probability'] * props_bc['probability']
            
            # Skip low-probability paths
            if path_prob < min_path_probability:
                skipped_paths += 1
                continue
            
            # Two-hop evidence score: S_{A,Bj,C} = S_{A,Bj} × S_{Bj,C}
            path_evidence = props_ab['evidence_score'] * props_bc['evidence_score']
            
            # Correlation propagation: multiply correlation types
            # +1 × +1 = +1 (activation → activation = activation)
            # +1 × -1 = -1 (activation → inhibition = inhibition)
            # -1 × -1 = +1 (inhibition → inhibition = activation)
            # 0 × anything = 0 (unknown)
            path_correlation = props_ab['correlation_type'] * props_bc['correlation_type']
            
            path_probabilities.append(path_prob)
            path_evidence_scores.append(path_evidence)
            path_correlations.append(path_correlation)

            # Store path details for interpretability
            path_details.append({
                'intermediate': Bj.name,
                'intermediate_type': Bj.type,
                'probability': round(path_prob, 6),
                'evidence_score': round(path_evidence, 4),
                'correlation_type': path_correlation,
                'edge_ab_type': props_ab['edge_type'],
                'edge_bc_type': props_bc['edge_type'],
                'n_supporting_ab': props_ab['n_supporting_edges'],
                'n_supporting_bc': props_bc['n_supporting_edges']
            })
        
        # Skip if no valid paths
        if not path_probabilities:
            continue
        
        # Aggregate across all intermediates (Eq. 3 in iKraph paper)
        # P_{A,·,C} = 1 - ∏(1 - P_{A,Bj,C})
        probs_array = np.array(path_probabilities, dtype=np.float64)
        combined_prob = 1.0 - np.prod(1.0 - probs_array)
        
        # S_{A,·,C} = Σ S_{A,Bj,C}
        combined_evidence = sum(path_evidence_scores)
        
        # For correlation type, use weighted average by evidence score
        if sum(path_evidence_scores) > 0:
            weighted_correlation = sum(
                c * s for c, s in zip(path_correlations, path_evidence_scores)
            ) / sum(path_evidence_scores)
            
            # Round to nearest integer correlation type
            if weighted_correlation > 0.5:
                combined_correlation = 1
            elif weighted_correlation < -0.5:
                combined_correlation = -1
            else:
                combined_correlation = 0
        else:
            combined_correlation = 0
        
        # Store inferred edge
        inferred_edges[(A, C)] = {
            'source': A.name,
            'source_type': A.type,
            'target': C.name,
            'target_type': C.type,
            'probability': round(float(combined_prob), 6),
            'evidence_score': round(float(combined_evidence), 4),
            'correlation_type': combined_correlation,
            'num_paths': len(path_probabilities),
            'num_intermediates_tested': len(intermediates),
            # Keep top 50 paths sorted by evidence score for interpretability
            'paths': sorted(path_details, key=lambda x: -x['evidence_score'])[:50]
        }
    
    print(f"\nInferred {len(inferred_edges):,} indirect associations")
    print(f"Skipped {skipped_paths:,} low-probability paths")
    
    # Statistics
    if inferred_edges:
        probs = [e['probability'] for e in inferred_edges.values()]
        scores = [e['evidence_score'] for e in inferred_edges.values()]
        num_paths = [e['num_paths'] for e in inferred_edges.values()]
        
        print(f"\nInferred probability distribution:")
        print(f"  Min: {min(probs):.6f}, Max: {max(probs):.6f}, Mean: {np.mean(probs):.6f}")
        print(f"  Median: {np.median(probs):.6f}, Std: {np.std(probs):.6f}")
        
        print(f"\nInferred evidence score distribution:")
        print(f"  Min: {min(scores):.4f}, Max: {max(scores):.4f}, Mean: {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}, Std: {np.std(scores):.4f}")
        
        print(f"\nPaths per inference:")
        print(f"  Min: {min(num_paths)}, Max: {max(num_paths)}, Mean: {np.mean(num_paths):.1f}")
        
        # Correlation distribution
        corr_counts = defaultdict(int)
        for e in inferred_edges.values():
            corr_counts[e['correlation_type']] += 1
        print(f"\nCorrelation type distribution:")
        print(f"  Positive (+1): {corr_counts[1]:,} ({100*corr_counts[1]/len(inferred_edges):.1f}%)")
        print(f"  Negative (-1): {corr_counts[-1]:,} ({100*corr_counts[-1]/len(inferred_edges):.1f}%)")
        print(f"  Unknown (0): {corr_counts[0]:,} ({100*corr_counts[0]/len(inferred_edges):.1f}%)")
    
    return inferred_edges


def create_inferred_graph(kg, inferred_edges, min_probability=0.1, min_evidence=1.0):
    """
    Create a new graph with both direct and inferred edges.
    
    Args:
        kg: Original knowledge graph
        inferred_edges: Dictionary of inferred edges from compute_two_hop_inference
        min_probability: Minimum probability threshold for including inferred edges
        min_evidence: Minimum evidence score threshold for including inferred edges
    
    Returns:
        KnowledgeGraph with both direct and inferred edges
    """
    print(f"\n{'='*80}")
    print("CREATING GRAPH WITH INFERRED EDGES")
    print(f"{'='*80}")
    print(f"  Min probability: {min_probability}")
    print(f"  Min evidence score: {min_evidence}")
    
    # Create new graph with same schema
    inferred_kg = kg.__class__(schema=kg.schema)
    
    # Add all nodes
    for node in tqdm(kg.nodes(), desc="Adding nodes"):
        inferred_kg.add_node(node, **kg.nodes[node])
    
    # Add all direct edges (these are already aggregated)
    direct_count = 0
    for u, v, data in tqdm(kg.edges(data=True), desc="Adding direct edges"):
        inferred_kg.add_edge(u, v, **data)
        direct_count += 1
    
    # Add inferred edges above thresholds
    added_count = 0
    filtered_count = 0
    
    for (A, C), data in tqdm(inferred_edges.items(), desc="Adding inferred edges"):
        if data['probability'] >= min_probability and data['evidence_score'] >= min_evidence:
            inferred_kg.add_edge(
                A, C,
                type='inferred_two_hop',
                kind='inferred_two_hop',
                probability=data['probability'],
                evidence_score=data['evidence_score'],
                correlation_type=data['correlation_type'],
                direction='1',  # Inferred edges are directed
                num_paths=data['num_paths'],
                num_intermediates=data['num_intermediates_tested'],
                aggregated=False,
                inferred=True,
                source='PSR_inference'
            )
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


def save_outputs(inferred_kg, inferred_edges, input_path, output_dir):
    """Save all outputs with consistent naming."""
    
    input_stem = input_path.stem
    
    # Save inferred edges as JSON (top 5000 by probability)
    print(f"\nSaving inferred edges...")
    output_json = output_dir / f"{input_stem}_inferred_edges.json"
    
    # Sort by probability and take top entries
    max_to_save = min(5000, len(inferred_edges))
    inferred_list = []
    for (A, C), data in sorted(inferred_edges.items(), 
                                key=lambda x: -x[1]['probability'])[:max_to_save]:
        inferred_list.append(data)
    
    with open(output_json, 'w') as f:
        json.dump({
            'metadata': {
                'source_graph': str(input_path),
                'num_total_inferred': len(inferred_edges),
                'num_saved': len(inferred_list),
                'max_intermediates': 100,
                'min_path_probability': 0.01
            },
            'inferred_edges': inferred_list
        }, f, indent=2)
    print(f"Saved top {len(inferred_list):,} inferences to JSON")
    
    # Save combined graph
    output_pkl = output_dir / f"{input_stem}_with_inferred.pkl"
    print(f"\nSaving combined graph...")
    inferred_kg.export_graph(output_pkl)
    
    # Save graph info and stats
    info_dir = output_dir.parent.parent / "graphs" / "info"
    info_dir.mkdir(parents=True, exist_ok=True)
    
    # Save schema
    schema_file = info_dir / f"{input_stem}_with_inferred_schema.txt"
    with open(schema_file, 'w') as f:
        f.write(str(inferred_kg.schema))
    
    # Save stats
    stats_file = info_dir / f"{input_stem}_with_inferred_stats.txt"
    with open(stats_file, 'w') as f:
        with redirect_stdout(f):
            print_kg_stats(inferred_kg)
    
    # Create visualization if graph is small enough
    if inferred_kg.number_of_nodes() < 1000:
        print(f"\nCreating visualizations...")
        viz_dir = output_dir.parent.parent / "graphs" / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML visualization
        output_html = viz_dir / f"{input_stem}_with_inferred.html"
        inferred_kg.visualize(
            str(output_html),
            title=f"{input_stem.replace('_', ' ').title()} + Inferred"
        )
        
        # PNG visualization
        output_png = viz_dir / f"{input_stem}_with_inferred.png"
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

    # Configuration
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    
    # ===================================================================
    # CONFIGURE INPUT/OUTPUT
    # Change these paths to switch between prototype and full graph
    # ===================================================================
    
    # For prototype (small graph):
    input_dir = base_path / "graphs/subsets"
    input_graph = input_dir / "prototype_8_12_aggregated.pkl"
    
    # For full subgraph (uncomment to use):
    # input_dir = base_path / "graphs/subsets/ikraph"
    # input_graph = input_dir / "adipose_inflammation_snn_expanded.pkl"
    
    output_dir = input_dir / "inferred"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # PARAMETERS
    # ===================================================================
    
    # PSR parameters
    max_intermediates = 100      # Top 100 intermediates per A-C pair
    min_path_probability = 0.01  # Filter very weak paths
    consider_undirected = True    # Treat undirected edges as bidirectional
    
    # Filtering parameters for final graph
    min_inference_probability = 0.1  # Only high-confidence inferences
    min_inference_evidence = 2.0     # Require substantial evidence
    
    # ===================================================================
    # RUN INFERENCE
    # ===================================================================
    
    print("="*80)
    print("TWO-HOP PSR INFERENCE")
    print("Edges are pre-aggregated from literature mentions")
    print("="*80)
    print(f"Input: {input_graph.name}")
    print(f"Output directory: {output_dir}")
    
    # Load graph
    print(f"\nLoading: {input_graph}")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    # Check if edges are aggregated
    sample_edges = list(kg.edges(data=True))[:5]
    has_aggregated = any('aggregated' in data for _, _, data in sample_edges)
    if not has_aggregated:
        print("\nWARNING: Graph edges may not be aggregated. Run aggregate.py first for best results.")
    
    # Compute two-hop inferences
    inferred_edges = compute_two_hop_inference(
        kg,
        max_intermediates=max_intermediates,
        min_path_probability=min_path_probability,
        consider_undirected=consider_undirected
    )
    
    if not inferred_edges:
        print("\nNo inferences found. This could mean:")
        print("  - All possible indirect paths already have direct edges")
        print("  - No valid two-hop paths exist in the graph")
        print("  - Path probabilities are all below threshold")
        return
    
    # Create combined graph with inferred edges
    inferred_kg = create_inferred_graph(
        kg, 
        inferred_edges,
        min_probability=min_inference_probability,
        min_evidence=min_inference_evidence
    )
    
    # Save all outputs
    output_json, output_pkl = save_outputs(
        inferred_kg, 
        inferred_edges, 
        input_graph, 
        output_dir
    )
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    
    print(f"\nOriginal graph:")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Edges: {kg.number_of_edges():,}")
    
    print(f"\nInference results:")
    print(f"  Potential inferences found: {len(inferred_edges):,}")
    print(f"  Inferences above threshold: {sum(1 for e in inferred_edges.values() if e['probability'] >= min_inference_probability and e['evidence_score'] >= min_inference_evidence):,}")
    
    print(f"\nCombined graph:")
    print(f"  Nodes: {inferred_kg.number_of_nodes():,}")
    print(f"  Total edges: {inferred_kg.number_of_edges():,}")
    
    # Count edge types
    edge_types = defaultdict(int)
    for _, _, data in inferred_kg.edges(data=True):
        edge_types[data.get('type', 'unknown')] += 1
    
    print(f"\nEdge type breakdown:")
    for etype, count in sorted(edge_types.items()):
        print(f"  {etype}: {count:,}")
    
    print(f"\nOutput files:")
    print(f"  - {output_json.name} (inferred edges JSON)")
    print(f"  - {output_pkl.name} (combined graph)")
    
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