#!/usr/bin/env python3
"""
Generate publication-ready visualizations of knowledge graph statistics.
Works with both raw and aggregated knowledge graphs.

Usage:
    python visualize_kg_stats.py --input /path/to/graph.pkl --output /path/to/output_dir/
"""

import argparse
import warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph

# ============================================================================
# Publication-ready style settings
# ============================================================================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
})

# Color palette (colorblind-friendly)
COLORS = {
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'tertiary': '#029E73',
    'quaternary': '#CC78BC',
    'quinary': '#CA9161',
    'directed': '#D55E00',
    'undirected': '#56B4E9',
}

# ============================================================================
# Helper functions
# ============================================================================

def safe_get_edge_attr(data, attr, default=None):
    """Safely get edge attribute, return default if missing."""
    return data.get(attr, default)


def collect_node_types(kg):
    """Collect node type counts."""
    node_types = Counter()
    for node in kg.nodes():
        node_type = node.type if hasattr(node, 'type') else kg.nodes[node].get('type', 'Unknown')
        node_types[node_type] += 1
    return node_types


def collect_edge_types(kg):
    """Collect edge type counts."""
    edge_types = Counter()
    for u, v, key, data in kg.edges(keys=True, data=True):
        edge_type = safe_get_edge_attr(data, 'type', 'Unknown')
        edge_types[edge_type] += 1
    return edge_types


def collect_degrees(kg):
    """Collect degree information by node type."""
    degrees_by_type = defaultdict(list)
    for node in kg.nodes():
        node_type = node.type if hasattr(node, 'type') else kg.nodes[node].get('type', 'Unknown')
        degree = kg.degree(node)
        degrees_by_type[node_type].append(degree)
    
    # Overall degrees
    all_degrees = [kg.degree(node) for node in kg.nodes()]
    
    return all_degrees, degrees_by_type


def collect_parallel_edges(kg):
    """
    Collect parallel edge counts.
    For raw graphs: count actual parallel edges between node pairs.
    For aggregated graphs: use n_supporting_edges if available.
    """
    parallel_counts = []
    
    # Check if this is an aggregated graph
    is_aggregated = False
    for u, v, key, data in kg.edges(keys=True, data=True):
        if 'n_supporting_edges' in data:
            is_aggregated = True
            break
    
    if is_aggregated:
        # Use n_supporting_edges field
        for u, v, key, data in kg.edges(keys=True, data=True):
            n_edges = safe_get_edge_attr(data, 'n_supporting_edges', 1)
            parallel_counts.append(n_edges)
    else:
        # Count actual parallel edges
        edge_pair_counts = defaultdict(int)
        for u, v, key, data in kg.edges(keys=True, data=True):
            direction = safe_get_edge_attr(data, 'direction', '0')
            edge_type = safe_get_edge_attr(data, 'type', 'Unknown')
            
            # Normalize undirected edges to canonical order
            if direction == '0':
                pair = tuple(sorted([id(u), id(v)])) + (edge_type,)
            else:
                pair = (id(u), id(v), edge_type)
            
            edge_pair_counts[pair] += 1
        
        parallel_counts = list(edge_pair_counts.values())
    
    return parallel_counts


def collect_probabilities(kg):
    """Collect edge probabilities."""
    probabilities = []
    for u, v, key, data in kg.edges(keys=True, data=True):
        prob = safe_get_edge_attr(data, 'probability')
        if prob is not None:
            probabilities.append(prob)
    return probabilities


def collect_evidence_scores(kg):
    """Collect evidence scores (only for aggregated graphs)."""
    evidence_scores = []
    for u, v, key, data in kg.edges(keys=True, data=True):
        score = safe_get_edge_attr(data, 'evidence_score')
        if score is not None:
            evidence_scores.append(score)
    return evidence_scores


def collect_directed_undirected_by_type(kg):
    """Collect directed/undirected counts by edge type."""
    edge_direction_counts = defaultdict(lambda: {'directed': 0, 'undirected': 0})
    
    for u, v, key, data in kg.edges(keys=True, data=True):
        edge_type = safe_get_edge_attr(data, 'type', 'Unknown')
        direction = safe_get_edge_attr(data, 'direction', '0')
        
        if direction == '0':
            edge_direction_counts[edge_type]['undirected'] += 1
        else:
            edge_direction_counts[edge_type]['directed'] += 1
    
    return edge_direction_counts


def collect_novelty(kg):
    """Collect novelty counts."""
    novelty_counts = {'novel': 0, 'background': 0}
    
    for u, v, key, data in kg.edges(keys=True, data=True):
        is_novel = safe_get_edge_attr(data, 'novelty', False)
        if is_novel:
            novelty_counts['novel'] += 1
        else:
            novelty_counts['background'] += 1
    
    return novelty_counts


def collect_supporting_documents(kg):
    """Collect number of supporting documents per edge."""
    doc_counts = []
    
    for u, v, key, data in kg.edges(keys=True, data=True):
        n_docs = safe_get_edge_attr(data, 'n_documents')
        if n_docs is not None:
            doc_counts.append(n_docs)
    
    return doc_counts


def collect_prob_vs_evidence(kg):
    """Collect probability and evidence score pairs."""
    pairs = []
    
    for u, v, key, data in kg.edges(keys=True, data=True):
        prob = safe_get_edge_attr(data, 'probability')
        evidence = safe_get_edge_attr(data, 'evidence_score')
        
        if prob is not None and evidence is not None:
            pairs.append((prob, evidence))
    
    return pairs


# ============================================================================
# Visualization functions
# ============================================================================

def plot_node_type_counts(node_types, output_path):
    """Plot node type distribution."""
    if not node_types:
        print("  No node type data to plot")
        return
    
    # Sort by count descending
    types, counts = zip(*sorted(node_types.items(), key=lambda x: x[1], reverse=True))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(types, counts, color=COLORS['primary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count:,}', va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Node Type')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Total node types: {len(node_types)}, Total nodes: {sum(counts):,}")


def plot_edge_type_frequency(edge_types, output_path):
    """Plot edge type distribution."""
    if not edge_types:
        print("  No edge type data to plot")
        return
    
    # Sort by count descending
    types, counts = zip(*sorted(edge_types.items(), key=lambda x: x[1], reverse=True))
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(types) * 0.4)))
    bars = ax.barh(types, counts, color=COLORS['secondary'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count:,}', va='center', ha='left', fontsize=9)
    
    ax.set_xlabel('Number of Edges')
    ax.set_ylabel('Edge Type')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Total edge types: {len(edge_types)}, Total edges: {sum(counts):,}")


def plot_degree_distribution(degrees, output_path):
    """Plot degree distribution (log-log)."""
    if not degrees:
        print("  No degree data to plot")
        return
    
    degrees = np.array(degrees)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create histogram
    counts, bins = np.histogram(degrees, bins=50)
    bins_center = (bins[:-1] + bins[1:]) / 2
    
    # Remove zero counts for log-log plot
    mask = counts > 0
    bins_center = bins_center[mask]
    counts = counts[mask]
    
    ax.loglog(bins_center, counts, 'o', color=COLORS['primary'], markersize=6, alpha=0.7)
    
    ax.set_xlabel('Degree (k)')
    ax.set_ylabel('Frequency P(k)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Degree stats: min={degrees.min()}, median={np.median(degrees):.1f}, " +
          f"max={degrees.max()}, mean={degrees.mean():.1f}")


def plot_parallel_edges(parallel_counts, output_path):
    """Plot distribution of parallel edges."""
    if not parallel_counts:
        print("  No parallel edge data to plot")
        return
    
    parallel_counts = np.array(parallel_counts)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use log scale for x-axis if range is large
    max_count = parallel_counts.max()
    if max_count > 100:
        bins = np.logspace(0, np.log10(max_count), 50)
        ax.set_xscale('log')
    else:
        bins = 50
    
    ax.hist(parallel_counts, bins=bins, color=COLORS['tertiary'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Edges Between Node Pair')
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Parallel edges stats: min={parallel_counts.min()}, " +
          f"median={np.median(parallel_counts):.1f}, max={parallel_counts.max()}")


def plot_probability_distribution(probabilities, output_path):
    """Plot edge probability distribution."""
    if not probabilities:
        print("  No probability data to plot")
        return
    
    probabilities = np.array(probabilities)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.hist(probabilities, bins=50, color=COLORS['quaternary'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Edge Probability')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Probability stats: min={probabilities.min():.3f}, " +
          f"median={np.median(probabilities):.3f}, max={probabilities.max():.3f}, " +
          f"mean={probabilities.mean():.3f}")


def plot_directed_undirected_by_type(edge_direction_counts, output_path):
    """Plot directed vs undirected ratio by edge type."""
    if not edge_direction_counts:
        print("  No direction data to plot")
        return
    
    # Sort by total count descending
    edge_types = sorted(edge_direction_counts.keys(), 
                       key=lambda x: sum(edge_direction_counts[x].values()), 
                       reverse=True)
    
    directed = [edge_direction_counts[et]['directed'] for et in edge_types]
    undirected = [edge_direction_counts[et]['undirected'] for et in edge_types]
    totals = [d + u for d, u in zip(directed, undirected)]
    
    # Calculate percentages
    directed_pct = [100 * d / t if t > 0 else 0 for d, t in zip(directed, totals)]
    undirected_pct = [100 * u / t if t > 0 else 0 for u, t in zip(undirected, totals)]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(edge_types) * 0.4)))
    
    y_pos = np.arange(len(edge_types))
    
    # Stacked horizontal bar
    ax.barh(y_pos, directed_pct, color=COLORS['directed'], alpha=0.8, 
            label='Directed', edgecolor='black', linewidth=0.5)
    ax.barh(y_pos, undirected_pct, left=directed_pct, color=COLORS['undirected'], 
            alpha=0.8, label='Undirected', edgecolor='black', linewidth=0.5)
    
    # Add percentage labels
    for i, (d_pct, u_pct, total) in enumerate(zip(directed_pct, undirected_pct, totals)):
        if d_pct > 5:
            ax.text(d_pct / 2, i, f'{d_pct:.0f}%', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        if u_pct > 5:
            ax.text(d_pct + u_pct / 2, i, f'{u_pct:.0f}%', ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Add total count on right
        ax.text(102, i, f'n={total:,}', va='center', ha='left', fontsize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(edge_types)
    ax.set_xlabel('Percentage (%)')
    ax.set_xlim(0, 115)
    ax.set_ylabel('Edge Type')
    ax.legend(loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Edge types analyzed: {len(edge_types)}")


def plot_hub_degree_by_type(degrees_by_type, output_path):
    """Plot boxplot of degree distribution by node type."""
    if not degrees_by_type:
        print("  No degree by type data to plot")
        return
    
    # Sort by median degree
    sorted_types = sorted(degrees_by_type.items(), 
                         key=lambda x: np.median(x[1]), 
                         reverse=True)
    
    types, degree_lists = zip(*sorted_types)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(types) * 0.5)))
    
    # Create boxplot
    bp = ax.boxplot(degree_lists, vert=False, patch_artist=True,
                    boxprops=dict(facecolor=COLORS['primary'], alpha=0.6),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1))
    
    ax.set_yticklabels(types)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Node Type')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Log scale if range is large
    max_degree = max(max(degrees) for degrees in degree_lists)
    if max_degree > 1000:
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    for node_type, degrees in sorted_types[:5]:
        print(f"    {node_type}: median={np.median(degrees):.1f}, max={max(degrees)}")


def plot_novelty_distribution(novelty_counts, output_path):
    """Plot novelty distribution (pie or bar)."""
    if not novelty_counts or sum(novelty_counts.values()) == 0:
        print("  No novelty data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(novelty_counts.keys())
    counts = list(novelty_counts.values())
    total = sum(counts)
    
    # Bar chart
    bars = ax.bar(labels, counts, color=[COLORS['tertiary'], COLORS['primary']], 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width() / 2, height,
               f'{count:,}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Edges')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Novel: {novelty_counts['novel']:,} ({100*novelty_counts['novel']/total:.1f}%), " +
          f"Background: {novelty_counts['background']:,} ({100*novelty_counts['background']/total:.1f}%)")


def plot_evidence_score_distribution(evidence_scores, output_path):
    """Plot evidence score distribution."""
    if not evidence_scores:
        print("  No evidence score data to plot (likely a raw graph)")
        return
    
    evidence_scores = np.array(evidence_scores)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use log scale for x if range is large
    max_score = evidence_scores.max()
    if max_score > 100:
        bins = np.logspace(np.log10(evidence_scores.min() + 0.001), 
                          np.log10(max_score), 50)
        ax.set_xscale('log')
    else:
        bins = 50
    
    ax.hist(evidence_scores, bins=bins, color=COLORS['quinary'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Evidence Score')
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Evidence score stats: min={evidence_scores.min():.2f}, " +
          f"median={np.median(evidence_scores):.2f}, max={evidence_scores.max():.2f}")


def plot_prob_vs_evidence(pairs, output_path):
    """Plot probability vs evidence score scatter plot."""
    if not pairs:
        print("  No probability-evidence pairs to plot (likely a raw graph)")
        return
    
    probs, evidences = zip(*pairs)
    probs = np.array(probs)
    evidences = np.array(evidences)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Hexbin for density
    hb = ax.hexbin(probs, evidences, gridsize=50, cmap='Blues', 
                   mincnt=1, linewidths=0.2, alpha=0.8)
    
    ax.set_xlabel('Probability')
    ax.set_ylabel('Evidence Score')
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Correlation (Spearman): {np.corrcoef(probs, evidences)[0, 1]:.3f}")


def plot_supporting_documents(doc_counts, output_path):
    """Plot distribution of supporting documents per edge."""
    if not doc_counts:
        print("  No supporting document data to plot")
        return
    
    doc_counts = np.array(doc_counts)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use log scale for x if range is large
    max_docs = doc_counts.max()
    if max_docs > 50:
        bins = np.logspace(0, np.log10(max_docs), 50)
        ax.set_xscale('log')
    else:
        bins = np.arange(0, max_docs + 2) - 0.5
    
    ax.hist(doc_counts, bins=bins, color=COLORS['secondary'], 
            alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Number of Supporting Documents')
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")
    print(f"  Supporting documents stats: min={doc_counts.min()}, " +
          f"median={np.median(doc_counts):.1f}, max={doc_counts.max()}")


# ============================================================================
# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-ready KG statistics visualizations'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input graph (.pkl)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Setup
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}\n")
    
    # Load graph
    print("Loading graph...")
    kg = KnowledgeGraph.import_graph(str(input_path))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges\n")
    
    # Collect data
    print("Collecting statistics...")
    node_types = collect_node_types(kg)
    edge_types = collect_edge_types(kg)
    all_degrees, degrees_by_type = collect_degrees(kg)
    parallel_counts = collect_parallel_edges(kg)
    probabilities = collect_probabilities(kg)
    evidence_scores = collect_evidence_scores(kg)
    edge_direction_counts = collect_directed_undirected_by_type(kg)
    novelty_counts = collect_novelty(kg)
    doc_counts = collect_supporting_documents(kg)
    prob_evidence_pairs = collect_prob_vs_evidence(kg)
    print("Done.\n")
    
    # Generate plots
    print("Generating visualizations...")
    
    print("\n1. Node Type Counts")
    plot_node_type_counts(node_types, output_dir / "node_type_counts.png")
    
    print("\n2. Edge Type Frequency")
    plot_edge_type_frequency(edge_types, output_dir / "edge_type_frequency.png")
    
    print("\n3. Degree Distribution")
    plot_degree_distribution(all_degrees, output_dir / "degree_distribution.png")
    
    print("\n4. Parallel Edges Distribution")
    plot_parallel_edges(parallel_counts, output_dir / "parallel_edges_distribution.png")
    
    print("\n5. Probability Distribution")
    plot_probability_distribution(probabilities, output_dir / "probability_distribution.png")
    
    print("\n6. Directed/Undirected by Edge Type")
    plot_directed_undirected_by_type(edge_direction_counts, 
                                     output_dir / "directed_undirected_by_type.png")
    
    print("\n7. Hub Degree by Node Type")
    plot_hub_degree_by_type(degrees_by_type, output_dir / "hub_degree_by_type.png")
    
    print("\n8. Novelty Distribution")
    plot_novelty_distribution(novelty_counts, output_dir / "novelty_distribution.png")
    
    print("\n9. Evidence Score Distribution")
    plot_evidence_score_distribution(evidence_scores, 
                                     output_dir / "evidence_score_distribution.png")
    
    print("\n10. Probability vs Evidence Score")
    plot_prob_vs_evidence(prob_evidence_pairs, 
                         output_dir / "probability_vs_evidence.png")
    
    print("\n11. Supporting Documents Distribution")
    plot_supporting_documents(doc_counts, 
                             output_dir / "supporting_documents_distribution.png")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"All visualizations saved to: {output_dir}\n")


if __name__ == "__main__":
    main()