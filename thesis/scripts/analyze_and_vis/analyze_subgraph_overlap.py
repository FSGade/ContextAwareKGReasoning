#!/usr/bin/env python3
"""
Analyze overlap in nodes and edges between different subgraph configurations.

Enhancements:
- Added grouped 3-way Venn diagrams:
    1) Per filter strictness (compare article sets)
    2) Per article set (compare strictness)
- Added detailed progress prints.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3, venn3_circles, venn2
import numpy as np
from upsetplot import from_memberships, UpSet
import json
import re
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Base path (set at module level for use by all functions)
base_results_dir = Path("/home/projects2/ContextAwareKGReasoning/results/search_subset_2")


# ------------------------------------------------------------
# LOAD SUBGRAPH DATA
# ------------------------------------------------------------
def load_subgraph_data(graph_name: str, graph_dir: Path):
    print(f"\n[LOAD] Loading subgraph data for graph: {graph_name}")

    node_list_dir = graph_dir / "node_lists"
    edge_list_dir = graph_dir / "edge_lists"
    stats_dir = graph_dir / "stats"

    if not node_list_dir.exists() or not edge_list_dir.exists():
        print("  [WARN] Node or edge list directory missing. Skipping.")
        return {}

    subgraphs = {}
    node_files = list(node_list_dir.glob("*_nodes.json"))

    if not node_files:
        print("  [WARN] No node list files found.")
        return {}

    for node_file in node_files:
        config_name = node_file.stem.replace("_nodes", "")
        edge_file = edge_list_dir / f"{config_name}_edges.json"

        if not edge_file.exists():
            print(f"  [WARN] Missing edge file for {config_name}")
            continue

        print(f"  [LOAD] Reading config: {config_name}")

        try:
            # Load nodes
            with open(node_file, "r") as f:
                node_data = json.load(f)

            nodes = set()
            for n in node_data:
                if isinstance(n, dict):
                    nodes.add((n["name"], n["type"]))
                else:
                    nodes.add(n)

            # Load edges
            with open(edge_file, "r") as f:
                edge_data = json.load(f)

            edges = set()
            for e in edge_data:
                if len(e) >= 3:
                    if isinstance(e[0], list):
                        edges.add((tuple(e[0]), tuple(e[1]), e[2]))
                    else:
                        edges.add((e[0], e[1], e[2]))
                else:
                    if isinstance(e[0], list):
                        edges.add((tuple(e[0]), tuple(e[1]), 0))
                    else:
                        edges.add((e[0], e[1], 0))

            # Node/edge types (optional)
            node_types = {}
            edge_types = {}

            stats_file = stats_dir / f"{config_name}_stats.txt"
            if stats_file.exists():
                try:
                    with open(stats_file, "r") as f:
                        content = f.read()

                    node_section = re.search(
                        r"Node Type Distribution:(.*?)(?=Edge Type Distribution:|$)",
                        content,
                        re.DOTALL,
                    )
                    if node_section:
                        for line in node_section.group(1).strip().split("\n"):
                            m = re.search(r"(.+?):\s*(\d+)\s+nodes", line.strip())
                            if m:
                                node_types[m.group(1)] = int(m.group(2))

                    edge_section = re.search(
                        r"Edge Type Distribution:(.*?)$", content, re.DOTALL
                    )
                    if edge_section:
                        for line in edge_section.group(1).strip().split("\n"):
                            m = re.search(r"(.+?):\s*(\d+)\s+edges", line.strip())
                            if m:
                                edge_types[m.group(1)] = int(m.group(2))

                except Exception as e:
                    print(f"    [WARN] Could not parse stats: {e}")

            subgraphs[config_name] = {
                "nodes": nodes,
                "edges": edges,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "node_types": node_types,
                "edge_types": edge_types,
            }

            print(f"    ✓ Loaded {len(nodes):,} nodes, {len(edges):,} edges")

        except Exception as e:
            print(f"    [ERROR] Failed loading {config_name}: {e}")
            traceback.print_exc()

    return subgraphs


# ------------------------------------------------------------
# JACCARD
# ------------------------------------------------------------
def calculate_jaccard(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def analyze_pairwise_overlap(subgraphs, output_dir, data_type="nodes"):
    print(f"\n[ANALYZE] Pairwise {data_type.upper()} Jaccard Similarity")

    configs = list(subgraphs.keys())
    n = len(configs)
    if n < 2:
        print("  [WARN] Not enough configs for Jaccard.")
        return

    jmat = pd.DataFrame(index=configs, columns=configs, dtype=float)

    stats = []
    for i, c1 in enumerate(configs):
        for j, c2 in enumerate(configs):
            s1 = subgraphs[c1][data_type]
            s2 = subgraphs[c2][data_type]
            jacc = calculate_jaccard(s1, s2)
            jmat.loc[c1, c2] = jacc

            if i < j:
                stats.append(
                    {
                        "config1": c1,
                        "config2": c2,
                        "size1": len(s1),
                        "size2": len(s2),
                        "intersection": len(s1 & s2),
                        "union": len(s1 | s2),
                        "unique_to_1": len(s1 - s2),
                        "unique_to_2": len(s2 - s1),
                        "jaccard": jacc,
                    }
                )

    jmat.to_csv(output_dir / f"jaccard_{data_type}.csv")
    print(f"  ✓ Saved jaccard_{data_type}.csv")

    if stats:
        df = pd.DataFrame(stats)
        df.to_csv(output_dir / f"pairwise_overlap_{data_type}.csv", index=False)
        print(f"  ✓ Saved pairwise_overlap_{data_type}.csv")

    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        jmat.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        linewidths=0.5,
    )
    plt.title(f"{data_type.capitalize()} Jaccard Similarity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"heatmap_{data_type}_jaccard.png", dpi=300)
    plt.close()
    print(f"  ✓ Saved heatmap_{data_type}_jaccard.png")


# ------------------------------------------------------------
# BASIC VENN DIAGRAMS (when n=2 or n=3)
# ------------------------------------------------------------
def create_venn_diagrams(subgraphs, output_dir, data_type="nodes"):
    print(f"\n[PLOT] Basic Venn Diagrams for {data_type.upper()}")

    configs = list(subgraphs.keys())
    n = len(configs)

    if n < 2:
        print("  [WARN] Need ≥2 configs for Venn.")
        return
    if n > 3:
        print("  [INFO] >3 configs — skipping basic Venns.")
        return

    sets = [subgraphs[c][data_type] for c in configs]
    labels = configs

    if n == 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        venn2(sets, set_labels=labels, ax=ax)
        plt.title(f"2-Way Venn ({data_type})")
        plt.tight_layout()
        fname = output_dir / f"venn_{data_type}_2way.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"  ✓ Saved {fname.name}")

    if n == 3:
        fig, ax = plt.subplots(figsize=(10, 9))
        v = venn3(sets, set_labels=labels, ax=ax)
        venn3_circles(sets, ax=ax)
        plt.title(f"3-Way Venn ({data_type})")
        plt.tight_layout()
        fname = output_dir / f"venn_{data_type}_3way.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"  ✓ Saved {fname.name}")


# ------------------------------------------------------------
# GROUPED VENN DIAGRAMS (NEW)
# ------------------------------------------------------------
def create_grouped_venn_diagrams(subgraphs, output_dir):
    print("\n[PLOT] Grouped Venn Diagrams (Article Sets × Strictness)")

    venn_dir = output_dir / "venn_grouped"
    venn_dir.mkdir(exist_ok=True)

    configs = list(subgraphs.keys())

    article_sets = ["all_articles", "no_reviews", "reviews_only"]
    filters = ["1hop", "permissive", "strict"]

    def find_config(article, flt):
        for c in configs:
            if article in c and c.endswith(f"_{flt}"):
                return c
        return None

    # 1️⃣ Per filter strictness: compare article sets
    print("\n[GROUP 1] Per filter strictness → Compare article sets")

    for flt in filters:
        cfgs = [
            find_config("all_articles", flt),
            find_config("no_reviews", flt),
            find_config("reviews_only", flt),
        ]

        if any(c is None for c in cfgs):
            print(f"  [SKIP] Missing configs for strictness={flt}")
            continue

        labels = ["all_articles", "no_reviews", "reviews_only"]

        for dtype in ["nodes", "edges"]:
            sets = [subgraphs[c][dtype] for c in cfgs]
            fig, ax = plt.subplots(figsize=(12, 10))
            v = venn3(sets, set_labels=labels, ax=ax)
            venn3_circles(sets, ax=ax)

            # Percent labels
            all_items = set.union(*sets)
            total = len(all_items)
            for sec in ["100","010","001","110","101","011","111"]:
                lab = v.get_label_by_id(sec)
                if lab and lab.get_text():
                    num = int(lab.get_text())
                    pct = 100 * num / total if total else 0
                    lab.set_text(f"{num:,}\n({pct:.1f}%)")

            plt.title(f"{dtype.capitalize()} • Article Sets @ {flt}")
            plt.tight_layout()
            fname = venn_dir / f"venn_articles_{flt}_{dtype}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"    ✓ Saved {fname.name}")

    # 2️⃣ Per article set: compare strictness
    print("\n[GROUP 2] Per article set → Compare strictness")

    for art in article_sets:
        cfgs = [
            find_config(art, "1hop"),
            find_config(art, "permissive"),
            find_config(art, "strict"),
        ]

        if any(c is None for c in cfgs):
            print(f"  [SKIP] Missing configs for article_set={art}")
            continue

        labels = ["1hop", "permissive", "strict"]

        for dtype in ["nodes", "edges"]:
            sets = [subgraphs[c][dtype] for c in cfgs]

            fig, ax = plt.subplots(figsize=(12, 10))
            v = venn3(sets, set_labels=labels, ax=ax)
            venn3_circles(sets, ax=ax)

            all_items = set.union(*sets)
            total = len(all_items)
            for sec in ["100","010","001","110","101","011","111"]:
                lab = v.get_label_by_id(sec)
                if lab and lab.get_text():
                    num = int(lab.get_text())
                    pct = 100 * num / total if total else 0
                    lab.set_text(f"{num:,}\n({pct:.1f}%)")

            plt.title(f"{dtype.capitalize()} • Strictness @ {art}")
            plt.tight_layout()
            fname = venn_dir / f"venn_filters_{art}_{dtype}.png"
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"    ✓ Saved {fname.name}")

    print("\n  ✓ Completed grouped Venn diagrams.\n")


# ------------------------------------------------------------
# UPSET PLOT
# ------------------------------------------------------------
def create_upset_plot(subgraphs, output_dir, data_type="nodes"):
    print(f"\n[PLOT] UpSet Plot ({data_type.upper()})")

    configs = list(subgraphs.keys())
    if len(configs) < 2:
        print("  [WARN] Need ≥2 configs for UpSet.")
        return

    all_items = set()
    for c in configs:
        all_items |= subgraphs[c][data_type]

    if not all_items:
        print("  [WARN] No items to plot.")
        return

    memberships = []
    for item in all_items:
        memberships.append({c: item in subgraphs[c][data_type] for c in configs})

    upset_data = from_memberships(memberships)

    fig = plt.figure(figsize=(14, 8))
    upset = UpSet(upset_data, subset_size="count", show_counts=True)
    upset.plot(fig=fig)

    plt.suptitle(f"{data_type.capitalize()} Overlaps (UpSet)")
    plt.tight_layout()
    fname = output_dir / f"upset_{data_type}.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  ✓ Saved {fname.name}")


# ------------------------------------------------------------
# TYPE DISTRIBUTION
# ------------------------------------------------------------
def analyze_type_distributions(subgraphs, output_dir, data_type="node_types"):
    print(f"\n[ANALYZE] {data_type.upper()} Distribution")

    configs = list(subgraphs.keys())
    all_types = sorted(set().union(*[subgraphs[c][data_type].keys() for c in configs]))

    rows = []
    for c in configs:
        tdict = subgraphs[c][data_type]
        total = sum(tdict.values())
        for t in all_types:
            count = tdict.get(t, 0)
            pct = 100 * count / total if total else 0
            rows.append(
                {
                    "configuration": c,
                    "type": t,
                    "count": count,
                    "percentage": pct,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"{data_type}_distribution.csv", index=False)
    print(f"  ✓ Saved {data_type}_distribution.csv")

    pivot = df.pivot(index="configuration", columns="type", values="percentage").fillna(0)
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    plt.xticks(rotation=45)
    plt.title(f"{data_type.replace('_',' ').title()} Distribution")
    plt.tight_layout()
    fname = output_dir / f"{data_type}_distribution.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"  ✓ Saved {fname.name}")


# ------------------------------------------------------------
# SUMMARY REPORT
# ------------------------------------------------------------
def create_summary_report(graph_name, subgraphs, output_dir):
    print("\n[REPORT] Creating summary markdown")

    report = []
    report.append(f"# Subgraph Overlap Analysis Report\n")
    report.append(f"## Graph: {graph_name}\n\n")

    report.append("### Configuration Summary\n\n")
    report.append("| Configuration | Nodes | Edges |\n")
    report.append("|--------------|-------|-------|\n")
    for c, d in sorted(subgraphs.items()):
        report.append(f"| {c} | {d['n_nodes']:,} | {d['n_edges']:,} |\n")

    report.append("\n### Key Findings\n\n")

    configs = list(subgraphs.keys())
    if len(configs) >= 2:
        all_nodes = set.union(*[subgraphs[c]["nodes"] for c in configs])
        core_nodes = set.intersection(*[subgraphs[c]["nodes"] for c in configs])
        all_edges = set.union(*[subgraphs[c]["edges"] for c in configs])
        core_edges = set.intersection(*[subgraphs[c]["edges"] for c in configs])

        report.append(f"- Total unique nodes: {len(all_nodes):,}\n")
        report.append(f"- Core nodes present in all configs: {len(core_nodes):,}\n")
        report.append(f"- Total unique edges: {len(all_edges):,}\n")
        report.append(f"- Core edges present in all configs: {len(core_edges):,}\n")

    report_path = output_dir / "OVERLAP_ANALYSIS_SUMMARY.md"
    with open(report_path, "w") as f:
        f.writelines(report)

    print(f"  ✓ Saved OVERLAP_ANALYSIS_SUMMARY.md")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print("SUBGRAPH OVERLAP ANALYSIS")
    print("=" * 80)

    print("\n[MAIN] Scanning for graph directories…")

    graph_dirs = []
    for d in base_results_dir.iterdir():
        if not d.is_dir():
            continue
        if (d / "node_lists").exists() and (d / "edge_lists").exists():
            graph_dirs.append(d)

    if not graph_dirs:
        print("[ERROR] No graph directories found.")
        return

    print(f"[MAIN] Found {len(graph_dirs)} graphs:")
    for d in graph_dirs:
        print(f"   - {d.name}")

    for graph_dir in graph_dirs:
        graph_name = graph_dir.name

        print("\n" + "=" * 80)
        print(f"[GRAPH] Processing graph: {graph_name}")
        print("=" * 80)

        node_list_dir = graph_dir / "node_lists"
        edge_list_dir = graph_dir / "edge_lists"

        if not node_list_dir.exists() or not edge_list_dir.exists():
            print("[WARN] Missing node/edge list folders — skipping.")
            continue

        output_dir = base_results_dir / "overlap_analysis_2" / graph_name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OUT] Output dir: {output_dir}")

        subgraphs = load_subgraph_data(graph_name, graph_dir)
        if not subgraphs:
            print("[WARN] No subgraphs loaded — skipping graph.")
            continue

        print(f"[MAIN] Loaded {len(subgraphs)} configurations.\n")

        analyze_pairwise_overlap(subgraphs, output_dir, "nodes")
        analyze_pairwise_overlap(subgraphs, output_dir, "edges")

        create_venn_diagrams(subgraphs, output_dir, "nodes")
        create_venn_diagrams(subgraphs, output_dir, "edges")

        create_grouped_venn_diagrams(subgraphs, output_dir)

        create_upset_plot(subgraphs, output_dir, "nodes")
        create_upset_plot(subgraphs, output_dir, "edges")

        analyze_type_distributions(subgraphs, output_dir, "node_types")
        analyze_type_distributions(subgraphs, output_dir, "edge_types")

        create_summary_report(graph_name, subgraphs, output_dir)

        print(f"\n[GRAPH] Completed: {graph_name}")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()