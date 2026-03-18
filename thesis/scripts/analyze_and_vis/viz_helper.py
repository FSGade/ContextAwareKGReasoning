"""
Static graph visualization with separate styling for direct, 2-hop, and 3-hop edges.
"""

import colorsys
import sys
import traceback
from math import inf
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats


def _edge_width_from_prob(p: float) -> float:
    # keep edges readable: 0.5..6.0 range
    return max(0.5, min(6.0, 0.5 + 5.5 * float(p or 0.0)))

def visualize_static(
    kg,
    output_path,
    title=None,
    figsize=(20, 15),
    node_size=1000,
    font_size=12,
    edge_font_size=9,
    include_undirected_edges: bool = True,
    direct_edge_width: float = 1.0,   # fixed width for non-inferred edges
    layout_k: float = 2.0,
    layout_iter: int = 50,
    layout_seed: int = 42
):
    """
    Draw the graph with:
    - Direct edges in gray (fixed width)
    - 2-hop inferred edges in red (width ∝ probability)
    - 3-hop inferred edges in green (width ∝ probability)
    - Edge labels rotated to align along edges
    """
    G = kg  # MultiDiGraph
    pos = nx.spring_layout(G, k=layout_k, iterations=layout_iter, seed=layout_seed)

    # Node colors by type (simple deterministic palette)
    node_types = {n.type for n in G.nodes()}
    def _hsv(i, n):
        return colorsys.hsv_to_rgb(i / max(1, n), 0.7, 0.9)
    color_map = {t: _hsv(i, len(node_types)) for i, t in enumerate(sorted(node_types))}
    node_colors = [color_map[n.type] for n in G.nodes()]

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: f"{n.name}\n({n.type})" for n in G.nodes()},
        font_size=font_size
    )

    # helper to optionally exclude undirected edges (marked with direction == '0')
    def _edge_included(d):
        if not include_undirected_edges and str(d.get("direction", "0")) == '0':
            return False
        return True

    # Split edges by type: direct, 2-hop inferred, 3-hop inferred
    direct_edges = []
    two_hop_edges = []
    three_hop_edges = []
    
    for u, v, k, d in G.edges(keys=True, data=True):
        if not _edge_included(d):
            continue
        
        if not d.get("inferred", False):
            direct_edges.append((u, v, k, d))
        elif d.get("type") == "inferred_three_hop":
            three_hop_edges.append((u, v, k, d))
        else:  # inferred_two_hop or other inferred
            two_hop_edges.append((u, v, k, d))

    # Draw edges
    if direct_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, k) for u, v, k, _ in direct_edges],
            edge_color="0.6",
            width=direct_edge_width,   # FIXED width for direct edges
            arrows=True
        )

    if two_hop_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, k) for u, v, k, _ in two_hop_edges],
            edge_color="tomato",
            width=[_edge_width_from_prob(d.get("probability", 0.0)) for _, _, _, d in two_hop_edges],
            arrows=True
        )
    
    if three_hop_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, k) for u, v, k, _ in three_hop_edges],
            edge_color="limegreen",
            width=[_edge_width_from_prob(d.get("probability", 0.0)) for _, _, _, d in three_hop_edges],
            arrows=True
        )

    # Edge labels with: type | p | corr
    # Draw labels rotated along the edge direction
    edge_labels = {}
    for u, v, k, d in G.edges(keys=True, data=True):
        if not _edge_included(d):
            continue
        etype = d.get("type", "")
        p = d.get("probability", None)
        corr = d.get("correlation_type", None)
        bits = [etype] if etype else []
        if p is not None:
            bits.append(f"p={float(p):.2f}")
        if corr is not None:
            sym = {1: "+", -1: "−"}.get(int(corr), "0")
            bits.append(f"corr={sym}")
        if bits:
            edge_labels[(u, v, k)] = " | ".join(bits)

    # Draw edge labels with rotation aligned to edges
    if edge_labels:
        ax = plt.gca()
        for (u, v, k), label in edge_labels.items():
            # Get edge positions
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Calculate angle for rotation
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Flip text if it would be upside down
            if angle > 90:
                angle -= 180
            elif angle < -90:
                angle += 180
            
            # Position label at midpoint of edge
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
            
            # Add text with rotation
            ax.text(
                label_x, label_y, label,
                fontsize=edge_font_size,
                rotation=angle,
                rotation_mode='anchor',
                ha='center',
                va='bottom',  # Position text slightly above the edge
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
            )

    # Legend
    node_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[t], label=t, markersize=10)
        for t in sorted(node_types)
    ]
    edge_handles = [
        mlines.Line2D([0], [0], color="0.6", lw=direct_edge_width, label="direct (fixed width)"),
        mlines.Line2D([0], [0], color="tomato", lw=2, label="2-hop inferred (width ∝ p)"),
        mlines.Line2D([0], [0], color="limegreen", lw=2, label="3-hop inferred (width ∝ p)"),
    ]
    plt.legend(handles=node_handles + edge_handles, loc="upper left", bbox_to_anchor=(1, 1))

    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=1200)
    plt.close()

def main():
    base_path = Path("/home/projects2/ContextAwareKGReasoning/")

    # Choose graph
    input_dir = base_path / "data/graphs"
    prot_graph = input_dir / "prototypes/prototype_8seeds_12nodes.pkl"
    aggr_graph = input_dir / "subsets/prototype_8_12_aggregated.pkl"
    two_hop_inferred = input_dir / "subsets/inferred_metapath_mechanistic/prototype_8_12_aggregated_metapath_mechanistic_split_with_inferred.pkl"
    three_hop_inferred = input_dir / "subsets/inferred_metapath_mechanistic/prototype_8_12_aggregated_three_hop_with_inferred.pkl"
    combined_inferred = input_dir / "subsets/inferred/prototype_8_12_aggregated_combined_inferred.pkl"

    # Output
    output_dir = base_path / "results/graph_viz/prototype_mechanistic/"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graphs
    print("Loading graphs...")
    graphs = {}
    
    if prot_graph.exists():
        graphs["Prototype Graph"] = KnowledgeGraph.import_graph(str(prot_graph))
        print(f"Loaded prototype graph")
    
    if aggr_graph.exists():
        graphs["Aggregated Graph"] = KnowledgeGraph.import_graph(str(aggr_graph))
        print(f"Loaded aggregated graph")
    
    if two_hop_inferred.exists():
        graphs["2-Hop Inferred"] = KnowledgeGraph.import_graph(str(two_hop_inferred))
        print(f"Loaded 2-hop inferred graph")
    
    if three_hop_inferred.exists():
        graphs["3-Hop Inferred"] = KnowledgeGraph.import_graph(str(three_hop_inferred))
        print(f"Loaded 3-hop inferred graph")
    
    if combined_inferred.exists():
        graphs["Combined Inferred (2-hop + 3-hop)"] = KnowledgeGraph.import_graph(str(combined_inferred))
        print(f"Loaded combined inferred graph")
    
    if not graphs:
        print("ERROR: No graphs found!")
        return

    # Visualize each graph
    print(f"\nGenerating visualizations...")
    for title, kg in graphs.items():
        output_path = output_dir / f"{title.replace(' ', '_').lower().replace('(', '').replace(')', '')}.png"

        # Print edge statistics
        undirected_edges = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if str(d.get("direction", "0")) == '0'
        ]
        directed_edges = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if str(d.get("direction", "0")) != '0'
        ]
        two_hop_inf = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if d.get("type") == "inferred_two_hop"
        ]
        three_hop_inf = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if d.get("type") == "inferred_three_hop"
        ]
        
        print(f"\n{title}:")
        print(f"  Total edges: {kg.number_of_edges():,}")
        print(f"  Directed: {len(directed_edges):,}")
        print(f"  Undirected: {len(undirected_edges):,}")
        print(f"  2-hop inferred: {len(two_hop_inf):,}")
        print(f"  3-hop inferred: {len(three_hop_inf):,}")

        visualize_static(
            kg=kg,
            output_path=output_path,
            title=title,
            include_undirected_edges=False
        )
        print(f"  Saved: {output_path.name}")
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)