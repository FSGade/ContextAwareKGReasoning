""" Extract small prototype subgraph by connecting nodes via their IDs.
Creates toy examples with maximum number of nodes for algorithm prototyping.
"""
import sys
import json
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats

def find_node_by_id(kg, node_id):
    """Find node by exact ID match."""
    for node in kg.nodes():
        nid = getattr(node, 'id', kg.nodes[node].get('id'))
        if nid == node_id:
            return node
    return None

def get_seeds(kg, node_ids):
    """Get nodes from list of IDs."""
    seeds = []
    for node_id in node_ids:
        node = find_node_by_id(kg, node_id)
        if node:
            seeds.append(node)
        else:
            print(f"Warning: Node ID not found: {node_id}")
    return seeds


def extract_connecting_subgraph(kg, seeds, max_nodes=100):
    """Extract subgraph by finding shortest paths between all seed pairs."""

    path_nodes = set(seeds)
    
    # Find shortest paths between all seed pairs
    all_paths = []
    for i, source in enumerate(seeds):
        for target in seeds[i+1:]:
            try:
                path = nx.bidirectional_shortest_path(kg, source, target)
                all_paths.append((len(path), path))
            except nx.NetworkXNoPath:
                print(f"Warning: No path between {source.name} and {target.name}")
    
    # Sort by length and add paths until max_nodes
    all_paths.sort(key=lambda x: x[0])
    for length, path in all_paths:
        if len(path_nodes) >= max_nodes:
            break
        new_nodes = set(path) - path_nodes
        if len(path_nodes) + len(new_nodes) <= max_nodes:
            path_nodes.update(path)
    
    return path_nodes


def create_subgraph(kg, nodes):
    """Extract subgraph containing specified nodes and edges between them."""
    subgraph = kg.__class__(schema=kg.schema)
    
    for node in nodes:
        if node in kg:
            subgraph.add_node(node, **kg.nodes[node])
    
    for u, v, data in kg.edges(data=True):
        if u in nodes and v in nodes:
            subgraph.add_edge(u, v, **data)
    
    return subgraph


def main():
    # Configuration
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    input_graph = base_path / "graphs/subsets/ikraph_pubmed_human.pkl"
    output_dir = base_path / "graphs/prototypes"
    output_info = base_path / "graphs/info"
    output_viz = base_path / "graphs/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_info.mkdir(parents=True, exist_ok=True)
    output_viz.mkdir(parents=True, exist_ok=True)

    # ===================================================================
    # DEFINE NODE IDs TO CONNECT
    # ===================================================================
    
    seed_ids = [
        # "2521",
        "4412283",
        "10113144",
        "9882832", 
        "8805123",
        "11860572",
        "9655996",
        "8642672",
        "6559185",
        
    ]
    
    max_nodes = 30
    
    # ===================================================================
    # EXTRACT SUBGRAPH
    # ===================================================================
    
    print("Loading graph...")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges\n")
    
    print("Finding seed nodes...")
    seeds = get_seeds(kg, seed_ids)
    
    if len(seeds) < 2:
        print(f"Error: Need at least 2 valid seeds (found {len(seeds)})")
        return
    
    print(f"Found {len(seeds)} seeds:")
    for s in seeds:
        sid = getattr(s, 'id', kg.nodes[s].get('id'))
        print(f"  {sid}: {s.name} ({s.type})")
    
    print(f"\nExtracting connecting paths (max {max_nodes} nodes)...")
    nodes = extract_connecting_subgraph(kg, seeds, max_nodes)
    
    print(f"Creating subgraph with {len(nodes)} nodes...")
    prototype = create_subgraph(kg, nodes)
    
    # Save outputs
    output_name = f"prototype_{len(seeds)}seeds_{len(nodes)}nodes"
    output_pkl = output_dir / f"{output_name}.pkl"
    output_html = output_viz / f"{output_name}.html"
    output_png = output_viz / f"{output_name}.png"
    
    print(f"\nSaving to {output_dir}...")
    prototype.export_graph(output_pkl)

    print(f"\nSaving prototype schema to {output_info / 'prototype_schema.txt'}...")
    with open(output_info / "prototype_schema.txt", "w") as f:
        f.write(str(prototype.schema))

    print(f"\nSaving prototype stats to {output_info / 'prototype_stats.txt'}...")
    with open(output_info / "prototype_stats.txt", "w") as f:
        with redirect_stdout(f):
            print_kg_stats(prototype)
    print("Done!")

    print(f"\nVisualizing prototype...")
    prototype.visualize(str(output_html), title=f"Prototype ({len(seeds)} seeds)")

    prototype.visualize(
    str(output_png),
    figsize=(20, 15),
    node_size=500,
    font_size=5
    )
    
    print(f"\nComplete!")
    print(f"  Nodes: {prototype.number_of_nodes()}")
    print(f"  Edges: {prototype.number_of_edges()}")
    print(f"  Saved: {output_pkl.name}")


if __name__ == "__main__":
    main()