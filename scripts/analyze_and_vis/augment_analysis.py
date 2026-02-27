#!/usr/bin/env python3
"""
Post-hoc analysis of augmented graphs.

For each augmentation strategy:
- Load augmented_graph.pkl
- Build augmented_abstracts_cache.db (if missing)
- Extract BOTH strict and augmented edges (total and unique), nodes, probabilities, PMIDs

Then:
- Plot 3-way Venn diagrams (unique edges, nodes, PMIDs) for both subsets
- Plot probability distributions (KDE + histogram)
- Summary table with total vs unique edge counts for both subsets

This script does NOT modify graphs.
It only derives caches and analysis artefacts.
"""
import sys
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph
from pubmed.pubmed_cache import PubMedBatchCache


# =============================================================================
# Data container
# =============================================================================

@dataclass
class EdgeSubsetStats:
    """Statistics for a subset of edges (strict or augmented)."""
    total_edges: int = 0              # Raw count (with duplicates)
    unique_edges: set = field(default_factory=set)  # Unique (u, v, type) tuples
    edge_instances: set = field(default_factory=set)  # All edges: (u, v, type, doc_id)
    nodes: set = field(default_factory=set)         # Unique nodes
    pmids: set = field(default_factory=set)         # Unique PMIDs
    probabilities: List[float] = field(default_factory=list)  # All probabilities
    
    @property
    def n_unique_edges(self) -> int:
        return len(self.unique_edges)
    
    @property
    def n_edge_instances(self) -> int:
        return len(self.edge_instances)
    
    @property
    def n_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def n_pmids(self) -> int:
        return len(self.pmids)
    
    @property
    def mean_prob(self) -> float:
        if not self.probabilities:
            return 0.0
        return sum(self.probabilities) / len(self.probabilities)
    
    @property
    def edge_multiplicity(self) -> float:
        """Average edges per unique (u, v, type) pair."""
        if self.n_unique_edges == 0:
            return 0.0
        return self.total_edges / self.n_unique_edges


@dataclass
class GraphStats:
    """Statistics for an entire augmented graph (strict + augmented subsets)."""
    strategy: str
    strict: EdgeSubsetStats = field(default_factory=EdgeSubsetStats)
    augmented: EdgeSubsetStats = field(default_factory=EdgeSubsetStats)
    
    @property
    def combined_total_edges(self) -> int:
        return self.strict.total_edges + self.augmented.total_edges
    
    @property
    def combined_unique_edges(self) -> set:
        return self.strict.unique_edges | self.augmented.unique_edges
    
    @property
    def combined_nodes(self) -> set:
        return self.strict.nodes | self.augmented.nodes
    
    @property
    def combined_pmids(self) -> set:
        return self.strict.pmids | self.augmented.pmids


# =============================================================================
# Helpers
# =============================================================================

def collect_pmids_from_graph(kg: KnowledgeGraph) -> set:
    pmids = set()
    for _, _, _, d in kg.edges(keys=True, data=True):
        doc_id = d.get("document_id", "")
        if doc_id:
            pmids.add(doc_id.split(".")[0])
    return pmids


def cache_augmented_graph_pmids(
    graph_path: Path,
    cache_path: Path,
    email: str,
):
    if cache_path.exists():
        print(f"[cache] Exists → {cache_path.name}, skipping")
        return

    print(f"[cache] Creating {cache_path.name}")

    kg = KnowledgeGraph.import_graph(str(graph_path))
    pmids = collect_pmids_from_graph(kg)

    print(f"[cache] {len(pmids)} unique PMIDs")

    if not pmids:
        print("[cache] No PMIDs found, skipping")
        return

    cache = PubMedBatchCache(db_path=str(cache_path), email=email)
    cache.fetch_batch(list(pmids), batch_size=200, rate_limiting=0.4)
    cache.close()

    print("[cache] Done")


def load_graph_stats(path: Path, strategy: str) -> GraphStats:
    """Load graph and extract statistics for both strict and augmented subsets."""
    kg = KnowledgeGraph.import_graph(str(path))

    stats = GraphStats(strategy=strategy)

    for u, v, k, d in kg.edges(keys=True, data=True):
        source_subset = d.get("source_subset", "unknown")
        edge_type = d.get("type", "unknown")
        prob = d.get("probability", 0.0)
        doc_id = d.get("document_id", "")
        pmid = doc_id.split(".")[0] if doc_id else None
        
        # Determine which subset this edge belongs to
        if source_subset == "strict":
            subset = stats.strict
        elif source_subset == "augmented":
            subset = stats.augmented
        else:
            # Skip edges without source_subset label
            continue
        
        # Count every edge (total)
        subset.total_edges += 1
        
        # Track unique (u, v, type) combinations
        subset.unique_edges.add((u, v, edge_type))
        
        # Track all edge instances (u, v, type, doc_id) for Venn totals
        subset.edge_instances.add((u, v, edge_type, doc_id))
        
        # Nodes
        subset.nodes.add(u)
        subset.nodes.add(v)

        # Probabilities
        subset.probabilities.append(prob)

        # PMIDs
        if pmid:
            subset.pmids.add(pmid)

    return stats


def save_venn_diagram(sets: dict, labels: list, title: str, out_path: Path):
    plt.figure(figsize=(6, 6))
    venn3([sets[l] for l in labels], set_labels=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def print_summary_table(stats_dict: Dict[str, GraphStats]):
    """Print a formatted summary table for both strict and augmented subsets."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE")
    print("=" * 110)
    
    header = f"{'Strategy':<12} {'Subset':<10} {'Total':>10} {'Unique':>10} {'Mult':>8} {'Nodes':>8} {'PMIDs':>8} {'Mean P':>8}"
    print(header)
    print("-" * 110)
    
    for strat, gs in stats_dict.items():
        # Strict row
        print(
            f"{strat:<12} "
            f"{'strict':<10} "
            f"{gs.strict.total_edges:>10,} "
            f"{gs.strict.n_unique_edges:>10,} "
            f"{gs.strict.edge_multiplicity:>8.2f} "
            f"{gs.strict.n_nodes:>8,} "
            f"{gs.strict.n_pmids:>8,} "
            f"{gs.strict.mean_prob:>8.3f}"
        )
        # Augmented row
        print(
            f"{'':<12} "
            f"{'augmented':<10} "
            f"{gs.augmented.total_edges:>10,} "
            f"{gs.augmented.n_unique_edges:>10,} "
            f"{gs.augmented.edge_multiplicity:>8.2f} "
            f"{gs.augmented.n_nodes:>8,} "
            f"{gs.augmented.n_pmids:>8,} "
            f"{gs.augmented.mean_prob:>8.3f}"
        )
        # Combined row
        print(
            f"{'':<12} "
            f"{'COMBINED':<10} "
            f"{gs.combined_total_edges:>10,} "
            f"{len(gs.combined_unique_edges):>10,} "
            f"{gs.combined_total_edges / max(1, len(gs.combined_unique_edges)):>8.2f} "
            f"{len(gs.combined_nodes):>8,} "
            f"{len(gs.combined_pmids):>8,} "
            f"{'---':>8}"
        )
        print("-" * 110)
    
    print("Mult = Edge multiplicity (total / unique)")
    print()


# =============================================================================
# Main analysis
# =============================================================================

def main():
    # -----------------------------
    # EDIT THESE
    # -----------------------------
    base_dir = Path.cwd()
    email = "s233139@dtu.dk"
    strategies = ["greedy", "random", "weighted"]

    stats_dict: Dict[str, GraphStats] = {}

    print("=" * 60)
    print("Loading graphs + building caches")
    print("=" * 60)

    for strat in strategies:
        strat_dir = base_dir / strat
        graph_path = strat_dir / "augmented_graph.pkl"
        cache_path = strat_dir / "augmented_abstracts_cache.db"

        print(f"\n--- {strat.upper()} ---")

        # Build cache for ALL PMIDs in graph (both strict and augmented)
        cache_augmented_graph_pmids(
            graph_path=graph_path,
            cache_path=cache_path,
            email=email,
        )

        # Load stats (collects both strict and augmented)
        gs = load_graph_stats(graph_path, strat)
        stats_dict[strat] = gs

        # Log all stats
        print(f"STRICT:")
        print(
            f"  Total: {gs.strict.total_edges:,} | "
            f"Unique: {gs.strict.n_unique_edges:,} | "
            f"Mult: {gs.strict.edge_multiplicity:.2f} | "
            f"Nodes: {gs.strict.n_nodes:,} | "
            f"PMIDs: {gs.strict.n_pmids:,}"
        )
        print(f"AUGMENTED:")
        print(
            f"  Total: {gs.augmented.total_edges:,} | "
            f"Instances: {gs.augmented.n_edge_instances:,} | "
            f"Unique: {gs.augmented.n_unique_edges:,} | "
            f"Nodes: {gs.augmented.n_nodes:,} | "
            f"PMIDs: {gs.augmented.n_pmids:,}"
        )
        # Sanity check: edge_instances should equal total_edges
        if gs.augmented.n_edge_instances != gs.augmented.total_edges:
            print(f"  ⚠️  edge_instances ({gs.augmented.n_edge_instances}) != total_edges ({gs.augmented.total_edges})")
        print(f"COMBINED:")
        print(
            f"  Total: {gs.combined_total_edges:,} | "
            f"Unique: {len(gs.combined_unique_edges):,} | "
            f"Nodes: {len(gs.combined_nodes):,} | "
            f"PMIDs: {len(gs.combined_pmids):,}"
        )

    # --------------------------------------------------
    # Summary table
    # --------------------------------------------------
    print_summary_table(stats_dict)

    # --------------------------------------------------
    # Pairwise overlap - AUGMENTED only (using edge_instances)
    # --------------------------------------------------
    print("=" * 60)
    print("Pairwise overlap - AUGMENTED subset (all edge instances)")
    print("=" * 60)

    for a, b in combinations(strategies, 2):
        edge_overlap = len(stats_dict[a].augmented.edge_instances & stats_dict[b].augmented.edge_instances)
        node_overlap = len(stats_dict[a].augmented.nodes & stats_dict[b].augmented.nodes)
        pmid_overlap = len(stats_dict[a].augmented.pmids & stats_dict[b].augmented.pmids)
        
        print(
            f"{a} ∩ {b} | "
            f"edges: {edge_overlap:,}, "
            f"nodes: {node_overlap:,}, "
            f"PMIDs: {pmid_overlap:,}"
        )

    # --------------------------------------------------
    # Venn diagrams - AUGMENTED only (using edge_instances for totals)
    # --------------------------------------------------
    print("\nSaving Venn diagrams (augmented edges)...")

    save_venn_diagram(
        sets={s: stats_dict[s].augmented.edge_instances for s in strategies},
        labels=strategies,
        title="Augmented: All edges (u, v, type, doc_id)",
        out_path=base_dir / "venn_edges.png",
    )

    save_venn_diagram(
        sets={s: stats_dict[s].augmented.nodes for s in strategies},
        labels=strategies,
        title="Augmented: Nodes",
        out_path=base_dir / "venn_nodes.png",
    )

    save_venn_diagram(
        sets={s: stats_dict[s].augmented.pmids for s in strategies},
        labels=strategies,
        title="Augmented: PMIDs",
        out_path=base_dir / "venn_pmids.png",
    )

    # --------------------------------------------------
    # Bar chart: Total vs Unique edges (AUGMENTED)
    # --------------------------------------------------
    print("Saving bar charts...")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(strategies))
    width = 0.35

    total_vals = [stats_dict[s].augmented.total_edges for s in strategies]
    unique_vals = [stats_dict[s].augmented.n_unique_edges for s in strategies]

    bars1 = ax.bar([i - width/2 for i in x], total_vals, width, label='Total edges', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], unique_vals, width, label='Unique (u,v,type)', color='coral')

    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Edge count', fontsize=12)
    ax.set_title('Augmented Edges: Total vs Unique by Strategy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in strategies], fontsize=11)
    ax.legend(fontsize=10)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:,}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(base_dir / "bar_total_vs_unique.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Bar chart: Edge multiplicity (AUGMENTED)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    mult_vals = [stats_dict[s].augmented.edge_multiplicity for s in strategies]
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    
    bars = ax.bar([s.upper() for s in strategies], mult_vals, color=colors)
    
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Multiplicity (total / unique)', fontsize=12)
    ax.set_title('Augmented Edge Multiplicity by Strategy', fontsize=14)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No duplicates')

    for bar, val in zip(bars, mult_vals):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(base_dir / "bar_multiplicity.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Probability distributions - AUGMENTED
    # --------------------------------------------------
    print("Saving probability plots...")

    # KDE
    plt.figure(figsize=(8, 5))
    for strat in strategies:
        if stats_dict[strat].augmented.probabilities:
            sns.kdeplot(
                stats_dict[strat].augmented.probabilities,
                label=f"{strat.upper()} (n={stats_dict[strat].augmented.total_edges:,})",
                bw_adjust=0.5,
                clip=(0, 1),
            )
    plt.xlabel("Edge probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Augmented Edge Probability Distribution", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(base_dir / "probability_kde.png", dpi=300)
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    for strat in strategies:
        if stats_dict[strat].augmented.probabilities:
            plt.hist(
                stats_dict[strat].augmented.probabilities,
                bins=50, range=(0, 1), density=True, alpha=0.4,
                label=f"{strat.upper()} (μ={stats_dict[strat].augmented.mean_prob:.3f})",
            )
    plt.xlabel("Edge probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Augmented Edge Probability Histogram", fontsize=14)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(base_dir / "probability_hist.png", dpi=300)
    plt.close()

    # Box plot
    plt.figure(figsize=(8, 5))
    colors = ['#2ecc71', '#3498db', '#9b59b6']
    data = [stats_dict[s].augmented.probabilities for s in strategies]
    bp = plt.boxplot(data, labels=[s.upper() for s in strategies], patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.xlabel("Strategy", fontsize=12)
    plt.ylabel("Edge probability", fontsize=12)
    plt.title("Augmented Edge Probability by Strategy", fontsize=14)
    plt.tight_layout()
    plt.savefig(base_dir / "probability_boxplot.png", dpi=300)
    plt.close()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("Saved artefacts:")
    print("=" * 60)
    print("Venn diagrams:")
    print(" - venn_edges.png, venn_nodes.png, venn_pmids.png")
    print("Bar charts:")
    print(" - bar_total_vs_unique.png")
    print(" - bar_multiplicity.png")
    print("Probability plots:")
    print(" - probability_kde.png, probability_hist.png, probability_boxplot.png")
    print("Caches:")
    print(" - augmented_abstracts_cache.db (per strategy, contains ALL graph PMIDs)")


if __name__ == "__main__":
    main()