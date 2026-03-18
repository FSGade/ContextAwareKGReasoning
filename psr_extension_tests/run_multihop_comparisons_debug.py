#!/usr/bin/env python3
"""
Run multi-hop (exact-length) method comparisons and write Parquet files
for interactive reporting.

Methods:
  - bdd_exact        (requires: pip install dd)
  - monte_carlo
  - hierarchical     (PSR-style approximation; 2-hop or 3-hop hierarchical)
  - path_noisy_or    (OR over full paths; your current-style approximation)

Outputs one parquet per (hops, consider_undirected):
  results_{hops}hop_undirected{0|1}.parquet

Each parquet contains rows:
  (source, target, method, probability, hops, consider_undirected, ... extra columns)

Usage:
  python run_multihop_comparisons.py \
    --input-graph /path/to/graph.pkl \
    --output-dir /path/to/out \
    --max-pairs 2000 \
    --n-samples 2000 \
    --max-paths-per-query 20000

Notes:
  - query discovery is structural and stops at --max-pairs
  - if you set --min-path-probability > 0, BDD/path_noisy_or/hierarchical can return 0 for some queries
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from knowledge_graph import KnowledgeGraph  # your class

from psr_multihop_methods import (
    build_support_graph,
    enumerate_simple_node_paths,
    two_hop_probability_hierarchical,
    three_hop_probability_hierarchical,
    k_hop_probability_path_noisy_or,
    estimate_k_hop_probability_monte_carlo,
    exact_k_hop_probability_bdd,
)


# -----------------------------
# Node helpers
# -----------------------------
def node_name(kg, n):
    return kg.nodes[n].get("name", str(n))

def node_type(kg, n):
    return kg.nodes[n].get("type", kg.nodes[n].get("kind", "unknown"))


# -----------------------------
# Query discovery with tqdm
# -----------------------------
def discover_query_pairs_exact_k(
    G,
    k: int,
    max_pairs: int,
    max_sources: int | None = None,
    seed: int = 0,
):
    """
    Find (A,Z) pairs with >=1 SIMPLE path of exact length k in SupportGraph G.
    Stops after max_pairs (unique pairs).

    This is intentionally probability-agnostic (structure only).
    """
    nodes = list(set(list(G.out.keys()) + list(G.inn.keys())))
    rng = np.random.default_rng(seed)
    rng.shuffle(nodes)
    if max_sources is not None:
        nodes = nodes[:max_sources]

    pairs_set = set()
    pairs = []

    pbar = tqdm(nodes, desc=f"Discovering {k}-hop query pairs", unit="src")
    for A in pbar:
        visited = {A}

        def dfs(u, depth: int):
            nonlocal pairs, pairs_set
            if len(pairs) >= max_pairs:
                return
            if depth == k:
                key = (A, u)
                if key not in pairs_set:
                    pairs_set.add(key)
                    pairs.append(key)
                return
            for v, _vids, _pav in G.out.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                dfs(v, depth + 1)
                visited.remove(v)

        dfs(A, 0)
        pbar.set_postfix({"pairs": len(pairs)})
        if len(pairs) >= max_pairs:
            break

    return pairs


# -----------------------------
# Helper: results dict -> DataFrame (as requested)
# -----------------------------
def results_dict_to_df(results: dict, method: str, hops: int, consider_undirected: bool, kg) -> pd.DataFrame:
    rows = []
    for (A, Z), info in results.items():
        row = {
            "source": str(A),
            "target": str(Z),
            "source_name": node_name(kg, A),
            "target_name": node_name(kg, Z),
            "source_type": node_type(kg, A),
            "target_type": node_type(kg, Z),
            "method": str(method),
            "hops": int(hops),
            "consider_undirected": bool(consider_undirected),
        }
        # merge info
        if isinstance(info, dict):
            row.update(info)
        # ensure probability
        row["probability"] = float(row.get("probability", 0.0))

        # flatten Monte Carlo CI if present
        if "ci_wilson" in row and row["ci_wilson"] is not None:
            try:
                lo, hi = row["ci_wilson"]
                row["ci_low"] = float(lo) if lo is not None else None
                row["ci_high"] = float(hi) if hi is not None else None
            except Exception:
                row["ci_low"], row["ci_high"] = None, None

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Method runners with tqdm
# -----------------------------

import time, json
from collections import defaultdict

def _choose2(n: int) -> int:
    return n * (n - 1) // 2

def explain_core_heavy_hitters(node_paths, topk=5):
    from collections import defaultdict
    B_to_C = defaultdict(set)
    C_to_B = defaultdict(set)

    for A, B, C, D in node_paths:
        B_to_C[B].add(C)
        C_to_B[C].add(B)

    topB = sorted(((B, len(Cs)) for B, Cs in B_to_C.items()), key=lambda x: -x[1])[:topk]
    topC = sorted(((C, len(Bs)) for C, Bs in C_to_B.items()), key=lambda x: -x[1])[:topk]

    # overlap between the worst Bs (how much they share Cs)
    overlaps = []
    Bs = [b for b, _ in topB]
    for i in range(len(Bs)):
        for j in range(i+1, len(Bs)):
            Si, Sj = B_to_C[Bs[i]], B_to_C[Bs[j]]
            inter = len(Si & Sj)
            union = len(Si | Sj)
            jac = inter / union if union else 0.0
            overlaps.append((Bs[i], Bs[j], jac, inter))
    overlaps.sort(key=lambda x: -x[2])

    # fraction of Cs that are shared (fanin >= 2)
    nC = len(C_to_B)
    nC_shared = sum(1 for C, Bs in C_to_B.items() if len(Bs) >= 2)
    frac_C_shared = (nC_shared / nC) if nC else 0.0

    return {
        "dbg_topB_fanout": topB,
        "dbg_topC_fanin": topC,
        "dbg_topB_max_jaccard": overlaps[0] if overlaps else None,
        "dbg_frac_C_shared": float(frac_C_shared),
    }

def summarize_3hop_paths_for_debug(G, node_paths):
    """
    node_paths: list of [A,B,C,D] paths (simple, exact length 3)
    Computes overlap/coupling features that correlate with BDD blow-ups.
    """
    B_to_C = defaultdict(set)
    C_to_B = defaultdict(set)

    used_arcs = set()
    var_occ = defaultdict(int)          # support var id -> count of arc-clauses it appears in
    step_vars = [set(), set(), set()]   # vars used at step 0/1/2 (A->B, B->C, C->D)

    for path in node_paths:
        if len(path) != 4:
            continue
        A, B, C, D = path
        B_to_C[B].add(C)
        C_to_B[C].add(B)

        for step in range(3):
            arc = (path[step], path[step + 1])
            used_arcs.add(arc)
            vids = G.supports.get(arc, ())
            for vid in vids:
                var_occ[vid] += 1
                step_vars[step].add(vid)

    n_paths = len(node_paths)
    nB = len(B_to_C)
    nC = len(C_to_B)
    density = (n_paths / (nB * nC)) if (nB and nC) else 0.0

    diamond_score = sum(_choose2(len(bs)) for bs in C_to_B.values())
    max_fanin_C = max((len(bs) for bs in C_to_B.values()), default=0)
    max_fanout_B = max((len(cs) for cs in B_to_C.values()), default=0)

    occ_vals = list(var_occ.values())
    var_occ_max = max(occ_vals) if occ_vals else 0
    var_occ_p95 = int(sorted(occ_vals)[int(0.95 * (len(occ_vals) - 1))]) if len(occ_vals) >= 2 else var_occ_max

    # how much the *same* support variable appears in multiple steps (often happens with consider_undirected=True)
    step01 = len(step_vars[0] & step_vars[1])
    step12 = len(step_vars[1] & step_vars[2])
    step02 = len(step_vars[0] & step_vars[2])

    supports_per_arc = [len(G.supports.get(a, ())) for a in used_arcs]
    sup_mean = (sum(supports_per_arc) / len(supports_per_arc)) if supports_per_arc else 0.0
    sup_max = max(supports_per_arc) if supports_per_arc else 0

    return {
        "dbg_nB": int(nB),
        "dbg_nC": int(nC),
        "dbg_n_paths": int(n_paths),
        "dbg_density": float(density),
        "dbg_diamond_score": int(diamond_score),
        "dbg_max_fanin_C": int(max_fanin_C),
        "dbg_max_fanout_B": int(max_fanout_B),
        "dbg_used_arcs": int(len(used_arcs)),
        "dbg_unique_vars": int(len(var_occ)),
        "dbg_var_occ_max": int(var_occ_max),
        "dbg_var_occ_p95": int(var_occ_p95),
        "dbg_step_var_overlap_01": int(step01),
        "dbg_step_var_overlap_12": int(step12),
        "dbg_step_var_overlap_02": int(step02),
        "dbg_supports_per_arc_mean": float(sup_mean),
        "dbg_supports_per_arc_max": int(sup_max),
    }

def run_bdd_exact_with_progress(
    G, queries, k, max_paths_per_query, min_path_probability,
    diagnostics_jsonl: str | None = None,
    slow_seconds: float = 10.0,
    big_bdd_nodes: int = 200_000,
    size_check_every: int = 0,  # keep 0 to avoid overhead; you can add growth tracking later if needed
):
    """
    Runs BDD exact per query and logs diagnostics for slow / large-BDD cases to JSONL.

    diagnostics_jsonl: path to write JSONL records (one per flagged pair)
    slow_seconds: log if a single pair takes >= this many seconds
    big_bdd_nodes: log if final BDD DAG size >= this threshold
    """
    out = {}
    f = open(diagnostics_jsonl, "a", encoding="utf-8") if diagnostics_jsonl else None

    try:
        for q in tqdm(queries, desc=f"BDD exact ({k}-hop)", unit="pair"):
            t0 = time.perf_counter()
            res1 = exact_k_hop_probability_bdd(
                G, [q], k=k,
                max_paths_per_query=max_paths_per_query,
                min_path_probability=min_path_probability,
            )[q]
            dt = time.perf_counter() - t0

            # attach timing (so you can also store it in parquet)
            res1 = dict(res1)
            res1["time_s"] = float(dt)

            out[q] = res1

            # decide whether to log extra debug info
            bdd_nodes = res1.get("bdd_nodes", 0) or 0
            if (dt >= slow_seconds) or (bdd_nodes is not None and int(bdd_nodes) >= big_bdd_nodes):
                dbg = {"source": str(q[0]), "target": str(q[1]), "k": int(k)}
                dbg.update({f"bdd_{k}": v for k, v in res1.items()})  # avoid collisions

                # For 3-hop, compute overlap stats from (truncated) node paths.
                if k == 3:
                    node_paths = enumerate_simple_node_paths(
                        G, q[0], q[1], k,
                        max_paths=max_paths_per_query,
                        min_path_probability=min_path_probability,
                    )
                    dbg.update(summarize_3hop_paths_for_debug(G, node_paths))

                    # NEW: heavy hitters + overlap
                    dbg.update(explain_core_heavy_hitters(node_paths, topk=5))

                if f:
                    f.write(json.dumps(dbg, ensure_ascii=False) + "\n")
                    f.flush()

                # also print a one-line hint to console
                print(
                    f"\n[BDD DIAG] {q} time={dt:.2f}s bdd_nodes={bdd_nodes} "
                    f"num_paths={res1.get('num_paths')} num_vars={res1.get('num_vars')}"
                )

    finally:
        if f:
            f.close()

    return out


def run_path_noisy_or_with_progress(G, queries, k, max_paths_per_query, min_path_probability):
    out = {}
    for (A, Z) in tqdm(queries, desc=f"Path noisy-OR ({k}-hop)", unit="pair"):
        # enumerate paths once for this query (uses p_arc pruning)
        node_paths = enumerate_simple_node_paths(
            G, A, Z, k,
            max_paths=max_paths_per_query,
            min_path_probability=min_path_probability,
        )
        if not node_paths:
            out[(A, Z)] = {"probability": 0.0, "num_paths": 0, "truncated": bool(max_paths_per_query)}
            continue

        # compute 1 - prod(1 - prod(p_arc))
        # (reuse library function if you prefer; doing inline keeps per-query progress)
        sum_log = 0.0
        for path in node_paths:
            path_prob = 1.0
            for i in range(len(path) - 1):
                path_prob *= float(G.p_arc[(path[i], path[i + 1])])
            # stable log(1 - x)
            x = min(max(path_prob, 0.0), 1.0 - 1e-15)
            sum_log += float(np.log1p(-x))
        prob = float(-np.expm1(sum_log))

        out[(A, Z)] = {
            "probability": prob,
            "num_paths": int(len(node_paths)),
            "truncated": bool(max_paths_per_query),
        }
    return out


def run_hierarchical_with_progress(G, queries, k, min_path_probability):
    out = {}
    if k == 2:
        # compute per query with tqdm
        for q in tqdm(queries, desc="Hierarchical PSR (2-hop)", unit="pair"):
            res = two_hop_probability_hierarchical(G, [q], min_path_probability=min_path_probability)
            out[q] = res[q]
    elif k == 3:
        for q in tqdm(queries, desc="Hierarchical PSR (3-hop)", unit="pair"):
            res = three_hop_probability_hierarchical(G, [q], min_path_probability=min_path_probability)
            out[q] = res[q]
    else:
        raise ValueError("Hierarchical runner only implemented for k=2 or k=3")
    return out


def run_monte_carlo_with_progress(G, queries, k, n_samples, seed, alpha):
    """
    Calls the library MC estimator; it has internal loops but no tqdm.
    For a draft, we expose progress by iterating samples here would require
    duplicating the estimator. Instead, we show a single tqdm for call-level.
    """
    # If you want sample-level tqdm, I can provide a fully inlined MC loop too.
    with tqdm(total=1, desc=f"Monte Carlo ({k}-hop), n={n_samples}", unit="run") as pbar:
        res = estimate_k_hop_probability_monte_carlo(
            G, queries, k=k,
            n_samples=n_samples, seed=seed, alpha=alpha
        )
        pbar.update(1)
    return res


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-graph", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)

    ap.add_argument("--hops", type=int, default=None, choices=[2, 3], help="Run only one hop length. Default: both.")
    ap.add_argument("--consider-undirected", type=str, default="false", choices=["false", "true", "both"])

    ap.add_argument("--methods", type=str, default="bdd_exact,monte_carlo,hierarchical,path_noisy_or",
                    help="Comma-separated list from: bdd_exact, monte_carlo, hierarchical, path_noisy_or")

    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--max-sources", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--min-edge-probability", type=float, default=0.0)
    ap.add_argument("--min-path-probability", type=float, default=0.0)
    ap.add_argument("--max-paths-per-query", type=int, default=None)

    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)

    ap.add_argument("--write-metadata-json", action="store_true",
                    help="Also write a metadata JSON sidecar per output parquet.")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    allowed = {"bdd_exact", "monte_carlo", "hierarchical", "path_noisy_or"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    hops_list = [args.hops] if args.hops else [2, 3]
    undirected_list = (
        [False, True] if args.consider_undirected == "both"
        else [args.consider_undirected == "true"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading KG: {args.input_graph}")
    kg = KnowledgeGraph.import_graph(str(args.input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    # Try-import dd if bdd_exact requested
    if "bdd_exact" in methods:
        try:
            import dd  # noqa: F401
        except Exception:
            print("WARNING: 'dd' not installed. Skipping bdd_exact. Install with: pip install dd")
            methods = [m for m in methods if m != "bdd_exact"]

    for consider_undirected in undirected_list:
        # Build support graph once per undirected setting
        print(f"\nBuilding SupportGraph (consider_undirected={consider_undirected}) ...")
        G = build_support_graph(
            kg,
            consider_undirected=consider_undirected,
            base_edges_only=True,
            min_edge_probability=args.min_edge_probability,
        )
        print(f"  Arcs: {len(G.supports):,}, Vars: {len(G.p_var):,}")

        for hops in hops_list:
            print(f"\n=== Running comparisons: hops={hops}, undirected={consider_undirected} ===")

            # 1) Discover query pairs (same queries for all methods in this condition)
            queries = discover_query_pairs_exact_k(
                G,
                k=hops,
                max_pairs=args.max_pairs,
                max_sources=args.max_sources,
                seed=args.seed + (10 if consider_undirected else 0) + hops,
            )
            print(f"Discovered {len(queries):,} query pairs")

            # 2) Run each method
            dfs = []

            if "bdd_exact" in methods:
                res_bdd = run_bdd_exact_with_progress(
                    G, queries, hops,
                    max_paths_per_query=args.max_paths_per_query,
                    min_path_probability=args.min_path_probability,
                    diagnostics_jsonl=str(args.output_dir / f"bdd_diag_{hops}hop_undirected{int(consider_undirected)}.jsonl"),
                    slow_seconds=10.0,
                    big_bdd_nodes=200_000,
                )
                dfs.append(results_dict_to_df(res_bdd, "bdd_exact", hops, consider_undirected, kg))

            if "monte_carlo" in methods:
                res_mc = run_monte_carlo_with_progress(
                    G, queries, hops,
                    n_samples=args.n_samples,
                    seed=args.seed,
                    alpha=args.alpha,
                )
                dfs.append(results_dict_to_df(res_mc, "monte_carlo", hops, consider_undirected, kg))

            if "hierarchical" in methods:
                res_h = run_hierarchical_with_progress(
                    G, queries, hops,
                    min_path_probability=args.min_path_probability,
                )
                dfs.append(results_dict_to_df(res_h, "hierarchical", hops, consider_undirected, kg))

            if "path_noisy_or" in methods:
                res_por = run_path_noisy_or_with_progress(
                    G, queries, hops,
                    max_paths_per_query=args.max_paths_per_query,
                    min_path_probability=args.min_path_probability,
                )
                dfs.append(results_dict_to_df(res_por, "path_noisy_or", hops, consider_undirected, kg))

            df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            if df_all.empty:
                print("No results produced; skipping write.")
                continue

            # 3) Write parquet
            out_path = args.output_dir / f"results_{hops}hop_undirected{int(consider_undirected)}.parquet"
            df_all.to_parquet(out_path, index=False)
            print(f"Wrote: {out_path} ({len(df_all):,} rows)")

            if args.write_metadata_json:
                meta = {
                    "input_graph": str(args.input_graph),
                    "hops": int(hops),
                    "consider_undirected": bool(consider_undirected),
                    "methods": methods,
                    "max_pairs": int(args.max_pairs),
                    "max_sources": args.max_sources,
                    "seed": int(args.seed),
                    "min_edge_probability": float(args.min_edge_probability),
                    "min_path_probability": float(args.min_path_probability),
                    "max_paths_per_query": args.max_paths_per_query,
                    "n_samples": int(args.n_samples),
                    "alpha": float(args.alpha),
                }
                meta_path = out_path.with_suffix(".metadata.json")
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                print(f"Wrote: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
