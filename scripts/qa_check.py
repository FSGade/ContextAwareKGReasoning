#!/usr/bin/env python3
"""
Check a knowledge graph for BioRED-annotations, and correlation 
or direction annotation conflicts. Writes JSON reports.

Edit the constants in MAIN SETTINGS as needed.
"""

from os import mkdir
from pathlib import Path
from collections import defaultdict
import json
import sys
from tqdm import tqdm

from knowledge_graph import KnowledgeGraph


# Helpers & BioRED tables
def _norm(s): return str(s).strip() if s is not None else ""
def ordered_pair(u, v):
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:  # fallback for non-orderable node keys
        su, sv = str(u), str(v)
        return (u, v) if su <= sv else (v, u)

# Polarity of relation labels
BIORED_POLARITY = {
    # Gene–Gene
    "Upregulation": "pos",
    "Downregulation": "neg",
    "Regulation": "neutral",
    "Positive_Correlation": "pos",
    "Negative_Correlation": "neg",
    "Bind": "neutral",
    "Modification": "neutral",
    "Association": "neutral",

    # Chemical–Gene (direction-specific names + others)
    "Exhibition": "pos",         # C -> G
    "Response": "pos",           # G -> C (inverse name)
    "Suppression": "neg",        # C -> G
    "Resistance": "neg",         # G -> C (inverse name)
    "Receptor": "neutral",
    "Chem_Motification": "neutral",  # keep typo form if present
    "Chem_Modification": "neutral",
    # "Association": "neutral",   # already listed

    # Chemical–Variant
    "Resistance": "neg",
    "Response": "pos",

    # Gene–Disease
    # "Positive_Correlation": "pos",  # already listed
    # "Negative_Correlation": "neg",  # already listed
    # "Association": "neutral",       # already listed

    # Variant–Disease
    "Cause": "neutral",

    # Chemical–Disease
    "Treatment": "neg",     # decreases disease
    "Induce": "pos",        # increases disease

    # Chemical–Chemical
    "Cotreatment": "neutral",
    "Inhibition": "neg",
    "Increase": "pos",
    "Drug_Interaction": "neutral",
    "Comparison": "neutral",
}

BIORED_DIRECTIONAL_INVERSES = {
    # Chemical–Gene vs Gene–Chemical
    "Exhibition": "Response",
    "Response": "Exhibition",
    "Suppression": "Resistance",
    "Resistance": "Suppression",
    
}

ANTAGONISTIC_TYPE_PAIRS = {
    frozenset({"Upregulation", "Downregulation"}),                  # Gene–Gene (dir)
    frozenset({"Positive_Correlation", "Negative_Correlation"}),    # (non-dir)
    frozenset({"Increase", "Inhibition"}),                          # Chemical–Chemical
    frozenset({"Exhibition", "Suppression"}),                       # C->G
    frozenset({"Response", "Resistance"}),                          # G->C
    frozenset({"Induce", "Treatment"}),                             # Chemical–Disease
}

# =========================
# Core functions
# =========================
def build_edge_groups(kg):
    """Group raw edges by (u, v, type, correlation, direction)."""
    edge_groups = defaultdict(list)
    for u, v, key, data in tqdm(kg.edges(keys=True, data=True),
                                 desc="Collecting", total=kg.number_of_edges()):
        edge_type = data.get('type', 'unknown')
        correlation = data.get('correlation_type', 0)
        direction = data.get('direction', '0')  # '0' means undirected

        # Canonical order for undirected edges
        if str(direction) == '0':
            u, v = ordered_pair(u, v)

        edge_groups[(u, v, edge_type, correlation, direction)].append(data.copy())
    return edge_groups

def detect_cross_group_conflicts(edge_groups):
    """
    Detect cases where the SAME (u,v,edge_type) appears with different
    correlation_type and/or direction across groups.
    """
    per_pair_type_variants = defaultdict(lambda: defaultdict(set))  # (u,v) -> edge_type -> {(corr, dir)}
    for (u, v, edge_type, correlation, direction), _edges in edge_groups.items():
        per_pair_type_variants[(u, v)][edge_type].add((correlation, direction))

    conflicts = []
    for (u, v), type_map in per_pair_type_variants.items():
        for edge_type, variants in type_map.items():
            if len(variants) > 1:
                correlations = sorted({c for c, d in variants})
                directions   = sorted({d for c, d in variants})
                conflicts.append({
                    'source': getattr(u, 'name', str(u)),
                    'target': getattr(v, 'name', str(v)),
                    'edge_type': edge_type,
                    'correlations': correlations,
                    'directions': directions,
                    'n_variants': len(variants)
                })
    return conflicts

def run_biored_qa(edge_groups):
    """
    Compute four QA lists:
      - two_way_same_type: same relation name appears in both directions
      - two_way_inverse_type: BioRED inverse names across directions
      - contradictory_types: antagonistic name pairs on the same direction
      - polarity_conflict: both 'pos' and 'neg' polarity on the same direction (or undirected pair)
    """
    dir_types = defaultdict(set)   # (u,v) -> set(type names)
    dir_pols  = defaultdict(set)   # (u,v) -> set of {'pos','neg','neutral'}

    # Build direction-aware indexes from groups
    for (u, v, edge_type, correlation, direction), _edges in edge_groups.items():
        t = _norm(edge_type)
        pol = BIORED_POLARITY.get(t, "neutral")
        if str(direction) == '0':
            U, V = ordered_pair(u, v)
            dir_types[(U, V)].add(t)
            dir_pols[(U, V)].add(pol)
        else:
            dir_types[(u, v)].add(t)
            dir_pols[(u, v)].add(pol)

    # Collectors
    two_way_same_type = []
    two_way_inverse_type = []
    contradictory_types = []
    polarity_conflict = []

    # Two-way checks (need both directions)
    seen_pairs = set()
    for (u, v) in list(dir_types.keys()):
        base = ordered_pair(u, v)
        if base in seen_pairs:
            continue
        seen_pairs.add(base)

        uv = (u, v)
        vu = (v, u)
        if uv in dir_types and vu in dir_types:
            A = dir_types[uv]
            B = dir_types[vu]

            # same name both directions
            for t in sorted(A & B):
                two_way_same_type.append({
                    "source": getattr(u, "name", str(u)),
                    "target": getattr(v, "name", str(v)),
                    "edge_type": t,
                })

            # inverse names both directions (BioRED)
            for t in A:
                inv = BIORED_DIRECTIONAL_INVERSES.get(t)
                if inv and inv in B and inv != t:
                    two_way_inverse_type.append({
                        "source": getattr(u, "name", str(u)),
                        "target": getattr(v, "name", str(v)),
                        "forward_type": t,
                        "reverse_type": inv,
                    })

    # Same-direction contradictions & polarity conflicts
    def explicit_antagonists(ts):
        found = []
        S = {_norm(x) for x in ts}
        for pair in ANTAGONISTIC_TYPE_PAIRS:
            if pair.issubset(S):
                a, b = tuple(sorted(pair))
                found.append((a, b))
        return found

    for arc, types_set in dir_types.items():
        u, v = arc
        # antagonistic pairs on same arc
        for a, b in explicit_antagonists(types_set):
            contradictory_types.append({
                "source": getattr(u, "name", str(u)),
                "target": getattr(v, "name", str(v)),
                "type_a": a,
                "type_b": b,
            })
        # polarity conflict on same arc (or undirected canonical pair)
        pols = dir_pols[arc]
        if "pos" in pols and "neg" in pols:
            polarity_conflict.append({
                "source": getattr(u, "name", str(u)),
                "target": getattr(v, "name", str(v)),
                "polarities": sorted(pols),
            })

    return {
        "two_way_same_type": two_way_same_type,
        "two_way_inverse_type": two_way_inverse_type,
        "contradictory_types": contradictory_types,
        "polarity_conflict": polarity_conflict,
    }

def save_json_suite(base_no_ext: Path, **named_lists):
    base_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for name, payload in named_lists.items():
        p = base_no_ext.with_name(base_no_ext.name + f"_{name}.json")
        with open(p, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {p}")

# =========================
# Main
# =========================
def main():
    print("="*80)
    print("BioRED QA (exploration only)")
    print("="*80)

    base = Path("/home/projects2/ContextAwareKGReasoning/data")
    input_graph = base / "graphs/ikraph.pkl"
    qa_base = base / "graphs/info/ikraph_qa"   # we will write *_*.json using this as base
    

    print("Loading graph...")
    kg = KnowledgeGraph.import_graph(input_graph)
    print(f"Graph: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    edge_groups = build_edge_groups(kg)
    print(f"Grouped edges: {len(edge_groups):,} groups")

    print("Detecting cross-group conflicts...")
    conflicts = detect_cross_group_conflicts(edge_groups)
    print(f"Cross-group conflicts (mixed correlation/direction for same (u,v,edge_type)): {len(conflicts):,}")

           
    print("Running BioRED QA checks...")
    qa = run_biored_qa(edge_groups)

    # Print a short summary
    print("\nQA summary:")
    for k, v in qa.items():
        print(f"  {k}: {len(v):,}")

    # Save detailed JSONs
    # Save JSON reports
    print("\nSaving QA reports...")
    save_json_suite(
        qa_base,
        cross_group_conflicts=conflicts,
        two_way_same_type=qa["two_way_same_type"],
        two_way_inverse_type=qa["two_way_inverse_type"],
        contradictory_types=qa["contradictory_types"],
        polarity_conflict=qa["polarity_conflict"]
    )

    print("\nDONE (exploration)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
