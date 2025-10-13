"""
Aggregate edges in the knowledge graph using PSR algorithm.
Groups edges by (source, target, edge_type) and aggregates probabilities.
Keeps only the highest probability edge type between node pairs.
"""

import sys
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout
from collections import defaultdict
from tqdm import tqdm
from knowledge_graph import KnowledgeGraph, print_kg_stats
import traceback


def aggregate_edge_probabilities(edges_data, min_prob=0.0):
    """
    Aggregate probabilities for multiple edges using PSR formula.
    
    P_A→B = 1 - ∏(1 - p_i)
    
    Evidence score: S = Σ -log(1 - p_i + ε)
    
    Returns (probability, evidence_score, attrs) or (0, 0, None) if below threshold.
    """
    if not edges_data:
        return 0, 0, None
    
    n_edges = len(edges_data)
    
    # Pre-allocate array and fill in one pass
    probs = np.empty(n_edges, dtype=np.float32)
    document_ids = []
    sources = set()
    
    # Single pass through edges_data
    for i, edge in enumerate(edges_data):
        probs[i] = edge['probability']
        doc_id = edge.get('document_id')
        if doc_id:
            document_ids.append(doc_id)
        source = edge.get('source')
        if source:
            sources.add(source)
    
    # probs = np.array([edge['probability'] for edge in edges_data])
    
    # P = 1 - ∏(1 - p_i)
    # final_prob = 1 - np.prod(1 - probs)

    # Vectorized probability calculation
    # P = 1 - ∏(1 - p_i) computed as 1 - exp(sum(log(1-p)))
    log_complement = np.log1p(-probs)  # More stable than log(1-p)
    final_prob = -np.expm1(np.sum(log_complement))  # More stable than 1 - exp(x)
    
    # S = Σ -log(1 - p_i + ε)
    evidence_score = np.sum(-np.log(1 - probs + 0.01))
    
    if final_prob < min_prob:
        return 0, 0, None
    
    # document_ids = [e.get('document_id') for e in edges_data if e.get('document_id')]
    # sources = list(set(e.get('source') for e in edges_data if e.get('source')))
    
    first_edge = edges_data[0]
    
    attrs = {
        'type': first_edge['type'],
        'kind': first_edge['type'],
        'probability': round(float(final_prob), 4),
        'evidence_score': round(float(evidence_score), 4),
        'correlation_type': first_edge.get('correlation_type', 0),
        'direction': first_edge.get('direction', '0'),
        'is_directed': first_edge.get('direction', '0') != '0',
        'source': ', '.join(sources) if sources else 'aggregated',
        'n_supporting_edges': len(edges_data),
        'n_documents': len(document_ids),
        'aggregated': True
    }
    
    return final_prob, evidence_score, attrs


def aggregate_knowledge_graph(kg, min_prob=0.0, output_path=None, conflicts_log=None):
    """
    Create aggregated knowledge graph with PSR probability aggregation.
    
    For each node pair:
    1. Aggregate edges separately by type (Association, Bind, etc.)
    2. Keep only the edge type with highest probability (use evidence score as tiebreaker)
    3. Log conflicts where same type has different correlation_type or direction

    Parameters:
    ----------
    kg : KnowledgeGraph
        Input knowledge graph to aggregate.
    min_prob : float
        Minimum probability threshold to keep an edge.
    output_path : str or Path, optional
        Path to save the aggregated graph.
    conflicts_log : str or Path, optional
        Path to save conflicts log as JSON.

    Returns:
    -------
    KnowledgeGraph
        The aggregated knowledge graph.
    conflicts : list of dict
        List of conflicts detected during aggregation.
    """
    print(f"Input: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    agg_kg = KnowledgeGraph(schema=kg.schema)
    
    # Copy nodes
    for node in tqdm(kg.nodes(), desc="Copying nodes"):
        agg_kg.add_node(node, **kg.nodes[node].copy())
    
    # Group edges by (source, target, type, correlation, direction)
    print("Grouping edges...")
    edge_groups = defaultdict(list)
    
    for u, v, key, data in tqdm(kg.edges(keys=True, data=True), 
                                 desc="Collecting", total=kg.number_of_edges()):
        edge_type = data.get('type', 'unknown')
        correlation = data.get('correlation_type', 0)
        direction = data.get('direction', '0')  # Undirected by default
        
        # Normalize undirected edges
        if direction == '0' and u > v:
            u, v = v, u
        
        group_key = (u, v, edge_type, correlation, direction)
        edge_groups[group_key].append(data.copy())
    
    print(f"Found {len(edge_groups):,} unique groups")
    
    # Aggregate by type and detect conflicts
    print("Aggregating by edge type...")
    conflicts = []
    aggregated_by_type = defaultdict(dict)  # (u,v) -> {edge_type: (prob, score, attrs, u, v)}
    
    for (u, v, edge_type, correlation, direction), edges_data in tqdm(edge_groups.items(), desc="Aggregating"):
        # Check for conflicts within this group
        # correlations = set(e.get('correlation_type', 0) for e in edges_data)
        # directions = set(e.get('direction', '0') for e in edges_data)
        
        # if len(correlations) > 1 or len(directions) > 1:
        #     conflicts.append({
        #         'source': u.name,
        #         'target': v.name,
        #         'edge_type': edge_type,
        #         'correlations': list(correlations),
        #         'directions': list(directions),
        #         'n_edges': len(edges_data)
        #     })
        if len(edges_data) > 1:
            first_corr = edges_data[0].get('correlation_type', 0)
            first_dir = edges_data[0].get('direction', '0')
            has_conflict = False
        
            for edge in edges_data[1:]:
                if (edge.get('correlation_type', 0) != first_corr or 
                    edge.get('direction', '0') != first_dir):
                    has_conflict = True
                    break
            
            if has_conflict:
                correlations = set(e.get('correlation_type', 0) for e in edges_data)
                directions = set(e.get('direction', '0') for e in edges_data)
                conflicts.append({
                    'source': u.name,
                    'target': v.name,
                    'edge_type': edge_type,
                    'correlations': list(correlations),
                    'directions': list(directions),
                    'n_edges': len(edges_data)
                })

        # Aggregate this type
        prob, score, attrs = aggregate_edge_probabilities(edges_data, min_prob)
        
        if attrs is not None:
            node_pair = (u, v)
            if node_pair not in aggregated_by_type:
                aggregated_by_type[node_pair] = {}
            #aggregated_by_type[node_pair][edge_type] = (prob, score, attrs, u, v)
            aggregated_by_type[node_pair][edge_type] = (prob, score, attrs)  # Remove u, v
    
    # Select best type per node pair
    print("Selecting highest probability edge type per node pair...")
    n_directed = 0
    n_undirected = 0
    n_node_pairs = len(aggregated_by_type)
    n_tied = 0
    
    for node_pair, types_dict in tqdm(aggregated_by_type.items()):
        # Find type with highest probability, use evidence score as tiebreaker
        best_type = max(types_dict.items(), 
                       key=lambda x: (x[1][0], x[1][1]))  # (probability, evidence_score)
        
        #edge_type, (prob, score, attrs, u, v) = best_type
        edge_type, (prob, score, attrs) = best_type  # Remove u, v
        u, v = node_pair  

        # Check if there was a tie
        max_prob = prob
        #tied_types = [t for t, (p, s, a, _, _) in types_dict.items() if p == max_prob]
        tied_types = [t for t, (p, s, a) in types_dict.items() if p == max_prob]  # Remove _, _
        if len(tied_types) > 1:
            n_tied += 1
        
        agg_kg.add_edge(u, v, **attrs)
        
        if attrs['is_directed']:
            n_directed += 1
        else:
            n_undirected += 1
    
    # Print results
    print(f"\nOutput: {agg_kg.number_of_nodes():,} nodes, {agg_kg.number_of_edges():,} edges")
    print(f"  Node pairs: {n_node_pairs:,}")
    print(f"  Directed edges: {n_directed:,}")
    print(f"  Undirected edges: {n_undirected:,}")
    print(f"  Reduction: {100 * (1 - agg_kg.number_of_edges() / kg.number_of_edges()):.1f}%")
    
    if n_tied > 0:
        print(f"  Ties resolved by evidence score: {n_tied:,}")
    
    if conflicts:
        print(f"\nConflicts detected: {len(conflicts):,} groups with mixed correlation/direction")
    

    # Free memory before saving
    print("Freeing memory before save...")
    del edge_groups
    del aggregated_by_type
    import gc
    gc.collect()

    # Save graph
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving: {output_path}")
        agg_kg.export_graph(output_path)
    
    # Save conflicts log
    if conflicts and conflicts_log:
        import json
        conflicts_log = Path(conflicts_log)
        with open(conflicts_log, 'w') as f:
            json.dump(conflicts, f, indent=2)
        print(f"Conflicts log: {conflicts_log}")
    
    return agg_kg, conflicts


def main():
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    input_graph = base_path / "graphs/prototypes/prototype_8seeds_12nodes.pkl"
    output_graph = base_path / "graphs/subsets/prototype_8_12_aggregated.pkl"
    output_info = base_path / "graphs/info"
    output_viz = base_path / "graphs/visualizations"
    conflicts_log = output_info / "prototype_8_12_aggregation_conflicts.json"
    
    min_probability = 0.001
    
    print("="*80)
    print("PSR AGGREGATION")
    print("="*80)
    print(f"Min probability: {min_probability}\n")
    
    print("Loading graph...")
    kg = KnowledgeGraph.import_graph(input_graph)
    
    agg_kg, conflicts = aggregate_knowledge_graph(
        kg, 
        min_prob=min_probability,
        output_path=output_graph,
        conflicts_log=conflicts_log
    )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

    # Print schema and stats of aggregated graph to files
    print(f"\nSaving schema and stats of aggregated graph...")
    with open(output_info / "prototype_8_12_aggregated_schema.txt", "w") as f:
        f.write(str(agg_kg.schema))

    with open(output_info / "prototype_8_12_aggregated_stats.txt", "w") as f:
        with redirect_stdout(f):
            print_kg_stats(agg_kg)

    if conflicts:
        print(f"\n Review conflicts: {conflicts_log}")

    # Visualize graph if n nodes < 1000
    if agg_kg.number_of_nodes() < 1000:
        agg_kg.visualize(
            str(output_viz / "prototype_8_12_aggregated_graph.html"), title = "Aggregated KG")

        agg_kg.visualize(
            str(output_viz / "prototype_8_12_aggregated_graph.png"),
            figsize=(20, 15),
            node_size=500,
            font_size=5
            )
        


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)