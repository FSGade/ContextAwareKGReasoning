from math import inf
import matplotlib.pyplot as plt
import networkx as nx
import sys
import traceback
from pathlib import Path
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
    Draw the graph so that ONLY inferred edges are thickness-scaled by probability.
    Direct (non-inferred) edges use a fixed width.
    """
    G = kg  # MultiDiGraph
    pos = nx.spring_layout(G, k=layout_k, iterations=layout_iter, seed=layout_seed)

    # Node colors by type (simple deterministic palette)
    node_types = {n.type for n in G.nodes()}
    def _hsv(i, n):
        import colorsys
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

    # Split edges by inferred flag
    direct_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if not d.get("inferred", False) and _edge_included(d)
    ]
    inferred_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if d.get("inferred", False) and _edge_included(d)
    ]

    # Draw edges
    if direct_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, k) for u, v, k, _ in direct_edges],
            edge_color="0.6",
            width=direct_edge_width,   # FIXED width for direct edges
            arrows=True
        )

    if inferred_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v, k) for u, v, k, _ in inferred_edges],
            edge_color="tomato",
            width=[_edge_width_from_prob(d.get("probability", 0.0)) for _, _, _, d in inferred_edges],
            arrows=True
        )

    # Edge labels with: type | p | corr
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

    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels,
            font_size=edge_font_size, label_pos=0.5, rotate=False
        )

    # Legend
    import matplotlib.lines as mlines
    node_handles = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[t], label=t, markersize=10)
        for t in sorted(node_types)
    ]
    edge_handles = [
        mlines.Line2D([0], [0], color="0.6", lw=direct_edge_width, label="direct (fixed width)"),
        mlines.Line2D([0], [0], color="tomato", lw=2, label="inferred (width ∝ p)"),
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
    inferred_graph = input_dir / "subsets/inferred/prototype_8_12_aggregated_with_inferred.pkl"
    

    # Output
    output_dir = base_path / "results/graph_viz/without_undir_2/"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prot = KnowledgeGraph.import_graph(str(prot_graph))
    aggr = KnowledgeGraph.import_graph(str(aggr_graph))
    inferred = KnowledgeGraph.import_graph(str(inferred_graph))

    graphs = {
        "Prototype Graph": prot,
        "Aggregated Graph": aggr,
        "Inferred Graph": inferred,
    }   

    for title, kg in graphs.items():
        output_path = output_dir / f"{title.replace(' ', '_').lower()}.png"

        # Print number of undirected edges
        undirected_edges = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if str(d.get("direction", "0")) == '0'
        ]
        print(f"{title}: Number of undirected edges = {len(undirected_edges)}")

        directed_edges = [
            (u, v, k, d)
            for u, v, k, d in kg.edges(keys=True, data=True)
            if str(d.get("direction", "0")) != '0'
        ]
        print(f"{title}: Number of directed edges = {len(directed_edges)}")

        visualize_static(
            kg=kg,
            output_path=output_path,
            title=title,
            include_undirected_edges=False
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
