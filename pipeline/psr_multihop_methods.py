#!/usr/bin/env python3
"""
Multi-hop PSR probability methods with shared-variable awareness.

Provides four methods for computing k-hop reachability probabilities:
  - bdd_exact:        Exact probability via Binary Decision Diagrams
  - monte_carlo:      Unbiased MC estimate (handles shared variables correctly)
  - hierarchical:     Fast PSR-style approximation (2-hop or 3-hop)
  - path_noisy_or:    Noisy-OR over full paths (original PSR — may overcount)

Core data structure:
  SupportGraph — directed adjacency where each arc (u,v) is backed by one
  or more independent Bernoulli "support variables".  Undirected edges map
  to a single shared variable supporting both arc orientations, which is
  the key reason the naive path_noisy_or inflates probabilities.

Originally developed for RQ1 method comparisons.  Integrated into the RQ2
tissue-aware pipeline to replace inflated 3-hop probabilities.

Note:
  BDD requires `pip install dd`.  If not installed, bdd_exact will raise
  a clear error at call time; all other methods remain available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
from statistics import NormalDist
import numpy as np


# ---------------------------------------------------------------------------
# Lazy BDD import — module loads fine even without `dd`
# ---------------------------------------------------------------------------
_BDD_AVAILABLE = False
_BDD_BACKEND = None
BDD = None

try:
    from dd.cudd import BDD as _BDD_cudd
    BDD = _BDD_cudd
    _BDD_BACKEND = "cudd"
    _BDD_AVAILABLE = True
except Exception:
    try:
        from dd.autoref import BDD as _BDD_autoref
        BDD = _BDD_autoref
        _BDD_BACKEND = "autoref"
        _BDD_AVAILABLE = True
    except Exception:
        pass

if _BDD_AVAILABLE:
    print(f"[psr_multihop_methods] BDD backend: {_BDD_BACKEND}")
else:
    print("[psr_multihop_methods] BDD not available (install with: pip install dd)")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ordered_pair(u, v):
    try:
        return (u, v) if u <= v else (v, u)
    except TypeError:
        su, sv = str(u), str(v)
        return (u, v) if su <= sv else (v, u)


def default_is_base_edge(data: Dict[str, Any]) -> bool:
    """Heuristic to exclude inferred edges."""
    if data.get("inferred", False):
        return False
    etype = str(data.get("type", ""))
    kind = str(data.get("kind", ""))
    if etype.startswith("inferred") or kind.startswith("inferred"):
        return False
    pl = data.get("path_length", None)
    if pl is not None:
        try:
            if int(pl) > 1:
                return False
        except Exception:
            pass
    return True


def _log1m(x: float, eps: float = 1e-15) -> float:
    x = float(np.clip(x, 0.0, 1.0 - eps))
    return float(np.log1p(-x))


def _noisy_or_prob(ps: Iterable[float]) -> float:
    s = 0.0
    for p in ps:
        s += _log1m(float(p))
    return float(-np.expm1(s))


def wilson_ci(hits: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score interval without scipy."""
    if n <= 0:
        return (0.0, 0.0)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    phat = hits / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * np.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Graph representation with "support variables"
# ---------------------------------------------------------------------------
@dataclass
class SupportGraph:
    # adjacency: u -> list of (v, supports_tuple, p_arc)
    out: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]]
    inn: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]]

    # supports for a directed arc (u,v): tuple of variable IDs that can
    # open that arc
    supports: Dict[Tuple[Any, Any], Tuple[int, ...]]

    # probabilities for underlying independent variables
    p_var: np.ndarray

    # arc probability after OR over its supports (used by approximate methods)
    p_arc: Dict[Tuple[Any, Any], float]


def build_support_graph(
    kg,
    consider_undirected: bool = False,
    base_edges_only: bool = True,
    is_base_edge: Callable[[Dict[str, Any]], bool] = default_is_base_edge,
    min_edge_probability: float = 0.0,
) -> SupportGraph:
    """
    Build a directed adjacency where each arc (u,v) is supported by 1+
    shared Bernoulli variables.

    Directed edge (direction != '0') => variable key ('dir', u, v)
    Undirected edge (direction == '0') => variable key ('undir', a, b)
      with a,b = ordered_pair(u,v).
      If consider_undirected=True, it supports BOTH arcs (a,b) and (b,a)
      using the SAME variable id.

    Parallel edges within the same key are collapsed by noisy-OR into one
    variable probability.
    """
    # 1) collapse raw edges into variable-groups
    log_comp_sum: Dict[Tuple[Any, ...], float] = defaultdict(float)

    for u, v, data in kg.edges(data=True):
        if base_edges_only and not is_base_edge(data):
            continue
        p = float(data.get("probability", 0.0))
        if p < min_edge_probability:
            continue
        direction = str(data.get("direction", "1"))

        if direction == "0":
            if not consider_undirected:
                continue
            a, b = ordered_pair(u, v)
            if a == b:
                continue
            key = ("undir", a, b)
        else:
            if u == v:
                continue
            key = ("dir", u, v)

        log_comp_sum[key] += _log1m(p)

    # 2) assign variable ids
    var_keys = list(log_comp_sum.keys())
    var_id_of = {k: i for i, k in enumerate(var_keys)}
    p_var = np.empty(len(var_keys), dtype=np.float64)
    for k, i in var_id_of.items():
        p_var[i] = float(-np.expm1(log_comp_sum[k]))

    # 3) arc supports: (u,v) -> [var_ids]
    supports_list: Dict[Tuple[Any, Any], List[int]] = defaultdict(list)

    for k, vid in var_id_of.items():
        if k[0] == "dir":
            _, u, v = k
            supports_list[(u, v)].append(vid)
        else:
            # undirected supports both orientations (shared variable)
            _, a, b = k
            supports_list[(a, b)].append(vid)
            supports_list[(b, a)].append(vid)

    supports: Dict[Tuple[Any, Any], Tuple[int, ...]] = {
        arc: tuple(sorted(set(vs))) for arc, vs in supports_list.items()
    }

    # 4) precompute arc probabilities (OR over supports)
    p_arc: Dict[Tuple[Any, Any], float] = {}
    for arc, vids in supports.items():
        p_arc[arc] = _noisy_or_prob(p_var[vid] for vid in vids)

    # 5) adjacency lists with supports + p_arc
    out: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]] = defaultdict(list)
    inn: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]] = defaultdict(list)
    for (u, v), vids in supports.items():
        pav = p_arc[(u, v)]
        if pav <= 0.0:
            continue
        out[u].append((v, vids, pav))
        inn[v].append((u, vids, pav))

    return SupportGraph(
        out=dict(out), inn=dict(inn),
        supports=supports, p_var=p_var, p_arc=p_arc,
    )


# ---------------------------------------------------------------------------
# Simple-path enumeration (exact length k)
# ---------------------------------------------------------------------------
def enumerate_simple_node_paths(
    G: SupportGraph,
    source: Any,
    target: Any,
    k: int,
    max_paths: Optional[int] = None,
    min_path_probability: float = 0.0,
) -> List[List[Any]]:
    """
    Enumerate SIMPLE directed node-paths of EXACT length k from source to
    target.  Returns list of node sequences of length (k+1).
    Path pruning uses product of p_arc along the path.
    """
    if k <= 0:
        return []
    paths: List[List[Any]] = []
    visited = {source}

    def dfs(u: Any, depth: int, acc_prob: float, acc_nodes: List[Any]) -> None:
        if max_paths is not None and len(paths) >= max_paths:
            return
        if depth == k:
            if u == target:
                paths.append(list(acc_nodes))
            return

        for v, _vids, pav in G.out.get(u, []):
            if v in visited:
                continue
            new_prob = acc_prob * float(pav)
            if new_prob < min_path_probability:
                continue
            visited.add(v)
            acc_nodes.append(v)
            dfs(v, depth + 1, new_prob, acc_nodes)
            acc_nodes.pop()
            visited.remove(v)

    dfs(source, 0, 1.0, [source])
    return paths


def find_query_pairs_with_k_hop_structure(
    G: SupportGraph,
    k: int,
    max_pairs: Optional[int] = None,
) -> List[Tuple[Any, Any]]:
    """Structural discovery of (A,Z) pairs with >=1 simple path of exact
    length k (no probabilities)."""
    pairs = []
    nodes = list(set(list(G.out.keys()) + list(G.inn.keys())))
    for A in nodes:
        visited = {A}

        def dfs(u: Any, depth: int):
            nonlocal pairs
            if max_pairs is not None and len(pairs) >= max_pairs:
                return
            if depth == k:
                pairs.append((A, u))
                return
            for v, _vids, _pav in G.out.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                dfs(v, depth + 1)
                visited.remove(v)

        dfs(A, 0)

    uniq = list(dict.fromkeys(pairs))
    if max_pairs is not None:
        uniq = uniq[:max_pairs]
    return uniq


# ---------------------------------------------------------------------------
# BDD exact probability (internal helpers)
# ---------------------------------------------------------------------------
def _bdd_weighted_probability(bdd, root, var_probs: Dict[str, float]) -> float:
    """
    Compute weighted probability of a BDD under independent variable
    probabilities.
    """
    memo: Dict[int, float] = {}

    level_prob: Dict[int, float] = {}
    if hasattr(bdd, "level_of_var"):
        for name, p in var_probs.items():
            try:
                level_prob[int(bdd.level_of_var(name))] = float(p)
            except Exception:
                pass

    def node_id(f) -> int:
        return int(getattr(f, "node", f))

    def level_to_name(level: int) -> str:
        if hasattr(bdd, "var_at_level"):
            return bdd.var_at_level(int(level))
        if hasattr(bdd, "_bdd") and hasattr(bdd._bdd, "var_at_level"):
            return bdd._bdd.var_at_level(int(level))
        return str(level)

    def rec(f) -> float:
        if f == bdd.true:
            return 1.0
        if f == bdd.false:
            return 0.0

        nid = node_id(f)
        if nid in memo:
            return memo[nid]

        var, low, high = bdd.succ(f)

        if isinstance(var, (int, np.integer)):
            lvl = int(var)
            if lvl in level_prob:
                p = level_prob[lvl]
            else:
                name = level_to_name(lvl)
                p = float(var_probs[name])
        else:
            p = float(var_probs[var])

        val = (1.0 - p) * rec(low) + p * rec(high)
        memo[nid] = val
        return val

    return float(rec(root))


# ---------------------------------------------------------------------------
# Option 1: Exact probability via BDD
# ---------------------------------------------------------------------------
def exact_k_hop_probability_bdd(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    k: int,
    max_paths_per_query: Optional[int] = None,
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    Exact (given the enumerated path set): builds formula
      OR_paths  AND_steps ( OR_{var in supports(step)} X_var )
    and computes exact probability under independent variables.
    """
    if not _BDD_AVAILABLE:
        raise RuntimeError(
            "BDD not available. Install with: pip install dd\n"
            "Or use --method monte_carlo or --method path_noisy_or"
        )

    results: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

    for (A, Z) in queries:
        node_paths = enumerate_simple_node_paths(
            G, A, Z, k,
            max_paths=max_paths_per_query,
            min_path_probability=min_path_probability,
        )
        prob = compute_bdd_probability_for_paths(G, node_paths)
        results[(A, Z)] = {
            "probability": prob,
            "num_paths": len(node_paths),
            "truncated": bool(max_paths_per_query),
        }

    return results


def compute_bdd_probability_for_paths(
    G: SupportGraph,
    node_paths: List[List[Any]],
) -> float:
    """
    Compute exact probability for a given set of node-paths using BDD.

    This is the core function used by the RQ2 pipeline for per-metapath-group
    probability computation.  It builds a Boolean formula over the support
    variables referenced by the paths and evaluates it exactly.

    Args:
        G: SupportGraph with support variable mappings.
        node_paths: List of node sequences (each of length k+1).

    Returns:
        Exact probability that at least one path is fully realised.
    """
    if not _BDD_AVAILABLE:
        raise RuntimeError(
            "BDD not available. Install with: pip install dd\n"
            "Or use --method monte_carlo or --method path_noisy_or"
        )

    if not node_paths:
        return 0.0

    # Collect all variables and arcs used
    used_vars = set()
    used_arcs = set()
    for path in node_paths:
        for i in range(len(path) - 1):
            arc = (path[i], path[i + 1])
            vids = G.supports.get(arc, ())
            if not vids:
                continue
            used_arcs.add(arc)
            used_vars.update(vids)

    if not used_vars:
        return 0.0

    used_vars = sorted(used_vars)
    var_name = {vid: f"v{vid}" for vid in used_vars}
    var_probs = {var_name[vid]: float(G.p_var[vid]) for vid in used_vars}

    bdd = BDD()
    bdd.declare(*[var_name[vid] for vid in used_vars])

    # Cache each arc clause as a BDD: clause(arc) = OR over support vars
    clause_cache = {}
    for arc in used_arcs:
        vids = G.supports[arc]
        clause = bdd.false
        for vid in vids:
            if vid in var_name:
                clause = bdd.apply("or", clause, bdd.var(var_name[vid]))
        clause_cache[arc] = clause

    # Build formula: OR over paths, each path = AND of arc clauses
    root = bdd.false
    for path in node_paths:
        arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        # Skip paths with missing arcs (defensive)
        if not all(a in clause_cache for a in arcs):
            continue
        conj = bdd.true
        for a in arcs:
            conj = bdd.apply("and", conj, clause_cache[a])
        root = bdd.apply("or", root, conj)

    return _bdd_weighted_probability(bdd, root, var_probs)


# ---------------------------------------------------------------------------
# Option 2: Monte Carlo reachability
# ---------------------------------------------------------------------------
def _arc_open(present: np.ndarray, vids: Tuple[int, ...]) -> bool:
    for vid in vids:
        if present[vid]:
            return True
    return False


def _reachable_targets_exact_k_one_sample(
    G: SupportGraph, present: np.ndarray, source: Any, k: int,
) -> set:
    targets = set()
    visited = {source}

    def dfs(u: Any, depth: int):
        if depth == k:
            targets.add(u)
            return
        for v, vids, _pav in G.out.get(u, []):
            if v in visited:
                continue
            if not _arc_open(present, vids):
                continue
            visited.add(v)
            dfs(v, depth + 1)
            visited.remove(v)

    dfs(source, 0)
    return targets


def estimate_k_hop_probability_monte_carlo(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    k: int,
    n_samples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    Monte Carlo estimate of k-hop reachability per (source, target) pair.

    Samples all variables jointly, then checks reachability via DFS.
    Handles shared undirected variables correctly because each variable
    is sampled once and used consistently across all arcs it supports.
    """
    rng = np.random.default_rng(seed)
    m = len(G.p_var)

    queries_by_source: Dict[Any, List[Any]] = defaultdict(list)
    for A, Z in queries:
        queries_by_source[A].append(Z)

    hits = {q: 0 for q in queries}

    for _ in range(n_samples):
        present = rng.random(m) < G.p_var
        for A, Zs in queries_by_source.items():
            reachable = _reachable_targets_exact_k_one_sample(G, present, A, k)
            for Z in Zs:
                if Z in reachable:
                    hits[(A, Z)] += 1

    out = {}
    for q in queries:
        h = hits[q]
        lo, hi = wilson_ci(h, n_samples, alpha=alpha)
        out[q] = {
            "probability": float(h / n_samples),
            "n_samples": int(n_samples),
            "hits": int(h),
            "ci_wilson": (lo, hi),
        }
    return out


def estimate_probability_mc_for_path_groups(
    G: SupportGraph,
    groups_node_paths: Dict[Any, List[List[Any]]],
    n_samples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[Any, Dict[str, Any]]:
    """
    Monte Carlo probability estimate per metapath group.

    For each MC sample, draws all support variables, then checks whether
    any node-path in each group has all its arcs open.

    Args:
        G: SupportGraph with support variable mappings.
        groups_node_paths: dict mapping group_key -> list of node-paths.
        n_samples: number of MC samples.
        seed: RNG seed.
        alpha: confidence level for Wilson CI.

    Returns:
        dict mapping group_key -> {probability, ci_low, ci_high, ...}
    """
    rng = np.random.default_rng(seed)
    m = len(G.p_var)

    # Pre-compute support variable tuples per arc per path per group
    # to avoid dict lookups in the hot loop
    groups_precomputed: Dict[Any, List[List[Tuple[int, ...]]]] = {}
    for gk, paths in groups_node_paths.items():
        path_vids = []
        for path in paths:
            arc_vids = []
            valid = True
            for i in range(len(path) - 1):
                arc = (path[i], path[i + 1])
                vids = G.supports.get(arc, ())
                if not vids:
                    valid = False
                    break
                arc_vids.append(vids)
            if valid:
                path_vids.append(arc_vids)
        groups_precomputed[gk] = path_vids

    hits = {gk: 0 for gk in groups_node_paths}

    for _ in range(n_samples):
        present = rng.random(m) < G.p_var

        for gk, path_vids_list in groups_precomputed.items():
            for arc_vids_list in path_vids_list:
                path_open = True
                for vids in arc_vids_list:
                    if not any(present[vid] for vid in vids):
                        path_open = False
                        break
                if path_open:
                    hits[gk] += 1
                    break  # at least one path open -> hit

    results = {}
    for gk in groups_node_paths:
        h = hits[gk]
        lo, hi = wilson_ci(h, n_samples, alpha)
        results[gk] = {
            "probability": float(h / n_samples),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "n_samples": int(n_samples),
            "hits": int(h),
        }
    return results


# ---------------------------------------------------------------------------
# Option 3: Hierarchical PSR approximation (fast; uses p_arc)
# ---------------------------------------------------------------------------
def two_hop_probability_hierarchical(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    P(A->C) = 1 - prod_B (1 - p(A,B)*p(B,C))
    """
    results = {}
    for (A, C) in queries:
        sum_log = 0.0
        used_B = 0
        for B, _vids_ab, p_ab in G.out.get(A, []):
            if B == A or B == C:
                continue
            p_bc = G.p_arc.get((B, C), 0.0)
            if p_bc <= 0.0:
                continue
            x = float(p_ab) * float(p_bc)
            if x < min_path_probability:
                continue
            sum_log += _log1m(x)
            used_B += 1
        prob = float(-np.expm1(sum_log)) if used_B > 0 else 0.0
        results[(A, C)] = {"probability": prob, "num_B": int(used_B)}
    return results


def three_hop_probability_hierarchical(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    P(A->D) = 1 - prod_C (1 - P(A->C)*p(C,D))
    with P(A->C) computed as 2-hop hierarchical over B.
    Enforces simple-path distinctness A,B,C,D.
    """
    results = {}
    for (A, D) in queries:
        sum_log_final = 0.0
        used_C = 0

        for C, _vids_cd, p_cd in G.inn.get(D, []):
            if C == A or C == D:
                continue

            sum_log_ac = 0.0
            used_B = 0
            for B, _vids_ab, p_ab in G.out.get(A, []):
                if B == A or B == C or B == D:
                    continue
                p_bc = G.p_arc.get((B, C), 0.0)
                if p_bc <= 0.0:
                    continue
                if float(p_ab) * float(p_bc) * float(p_cd) < min_path_probability:
                    continue
                x = float(p_ab) * float(p_bc)
                sum_log_ac += _log1m(x)
                used_B += 1

            if used_B == 0:
                continue

            p_ac = float(-np.expm1(sum_log_ac))
            q = p_ac * float(p_cd)
            if q <= 0.0:
                continue

            sum_log_final += _log1m(q)
            used_C += 1

        prob = float(-np.expm1(sum_log_final)) if used_C > 0 else 0.0
        results[(A, D)] = {"probability": prob, "num_C": int(used_C)}
    return results


# ---------------------------------------------------------------------------
# Baseline: path-level noisy-OR (original PSR style)
# ---------------------------------------------------------------------------
def k_hop_probability_path_noisy_or(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    k: int,
    max_paths_per_query: Optional[int] = None,
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    P = 1 - prod_paths (1 - prod_steps p_arc(step))
    """
    results = {}
    for (A, Z) in queries:
        node_paths = enumerate_simple_node_paths(
            G, A, Z, k,
            max_paths=max_paths_per_query,
            min_path_probability=min_path_probability,
        )
        if not node_paths:
            results[(A, Z)] = {"probability": 0.0, "num_paths": 0}
            continue

        sum_log = 0.0
        for path in node_paths:
            path_prob = 1.0
            for i in range(len(path) - 1):
                path_prob *= float(G.p_arc[(path[i], path[i + 1])])
            sum_log += _log1m(path_prob)

        results[(A, Z)] = {
            "probability": float(-np.expm1(sum_log)),
            "num_paths": int(len(node_paths)),
            "truncated": bool(max_paths_per_query),
        }
    return results
