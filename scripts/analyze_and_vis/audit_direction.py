#!/usr/bin/env python3
"""
Audit direction integrity in KnowledgeGraph objects.

- Counts raw direction values exactly as stored
- Detects unexpected / unknown directions
- Breaks down unknowns by source and edge type
- Cleans memory explicitly between graphs

Intended for large graphs (ikraph_full, pubmed_human).
"""

from pathlib import Path
from collections import Counter, defaultdict
import gc
import sys

# Make sure KnowledgeGraph is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


# ============================================================================
# Core audit function
# ============================================================================

def audit_directions(kg, graph_name="GRAPH"):
    print("=" * 80)
    print(f"DIRECTION AUDIT: {graph_name}")
    print("=" * 80)

    raw_dir_counts = Counter()
    source_dir_counts = defaultdict(Counter)
    type_dir_counts = defaultdict(Counter)

    for _, _, _, data in kg.edges(keys=True, data=True):
        direction = data.get("direction", None)
        source = data.get("source", "UNKNOWN_SOURCE")
        edge_type = data.get("type", "UNKNOWN_TYPE")

        raw_dir_counts[direction] += 1
        source_dir_counts[source][direction] += 1
        type_dir_counts[edge_type][direction] += 1

    total = sum(raw_dir_counts.values())

    # ------------------------------------------------------------------------
    # 1. Raw direction values
    # ------------------------------------------------------------------------
    print("\nRaw direction values (no normalisation):")
    for d, c in raw_dir_counts.items():
        print(f"  {repr(d):>6}: {c:12d} ({c/total:.2%})")

    # ------------------------------------------------------------------------
    # 2. Normalisation check
    # ------------------------------------------------------------------------
    expected = {"0", "1"}
    observed = set(raw_dir_counts.keys())
    unexpected = observed - expected

    if unexpected:
        print("\n⚠️  UNEXPECTED DIRECTION VALUES FOUND:")
        for d in unexpected:
            print(f"  - {repr(d)}")
    else:
        print("\n✅ Direction values fully normalised (only '0' and '1').")

    # ------------------------------------------------------------------------
    # 3. Breakdown by source (unexpected only)
    # ------------------------------------------------------------------------
    if unexpected:
        print("\nBreakdown of unexpected directions by SOURCE:")
        for src, cnts in source_dir_counts.items():
            bad = {d: c for d, c in cnts.items() if d in unexpected}
            if bad:
                print(f"  {src}: {bad}")

    # ------------------------------------------------------------------------
    # 4. Breakdown by edge type (top offenders)
    # ------------------------------------------------------------------------
    if unexpected:
        print("\nTop edge types containing unexpected directions:")
        offenders = []
        for et, cnts in type_dir_counts.items():
            bad_count = sum(c for d, c in cnts.items() if d in unexpected)
            if bad_count > 0:
                offenders.append((et, bad_count))

        for et, c in sorted(offenders, key=lambda x: -x[1])[:10]:
            print(f"  {et:30s}: {c:12d}")

    # ------------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------------
    print("\nSummary:")
    print(f"  Total edges        : {total:,}")
    print(f"  Direction clean    : {not unexpected}")
    print("=" * 80)
    print()


# ============================================================================
# Wrapper to load → audit → free memory
# ============================================================================

def audit_graph(path, name):
    print(f"\nLoading graph: {name}")
    print(f"Path: {path}")

    kg = KnowledgeGraph.import_graph(path)
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    audit_directions(kg, graph_name=name)

    # Explicit cleanup
    del kg
    gc.collect()


# ============================================================================
# Main
# ============================================================================

def main():
    graphs = {
        "IKRAPH FULL": "/home/projects2/ContextAwareKGReasoning/data/graphs/ikraph.pkl",
        "IKRAPH PUBMED HUMAN": (
            "/home/projects2/ContextAwareKGReasoning/data/graphs/subsets/"
            "ikraph_pubmed_human.pkl"
        ),
    }

    for name, path in graphs.items():
        audit_graph(path, name)


if __name__ == "__main__":
    main()
