from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from collections import defaultdict
from statistics import NormalDist
import numpy as np


try:
    from dd.cudd import BDD  # much faster
    _BDD_BACKEND = "cudd"
except Exception:
    _BDD_BACKEND = "autoref"
    try:
        from dd.autoref import BDD
    except Exception as e:
        raise RuntimeError("Option 1 requires `pip install dd`") from e

print(f"Using BDD backend {_BDD_BACKEND}.")

# -----------------------------
# Utilities
# -----------------------------
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

# -----------------------------
# Graph representation with "support variables"
# -----------------------------
@dataclass
class SupportGraph:
    # adjacency: u -> list of (v, supports_tuple, p_arc)
    out: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]]
    inn: Dict[Any, List[Tuple[Any, Tuple[int, ...], float]]]

    # supports for a directed arc (u,v): tuple of variable IDs that can open that arc
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
    Build a directed adjacency where each arc (u,v) is supported by 1+ *shared* Bernoulli variables.

    Directed edge (direction != '0') => variable key ('dir', u, v)
    Undirected edge (direction == '0') => variable key ('undir', a, b) with a,b=ordered_pair(u,v)
      If consider_undirected=True, it supports BOTH arcs (a,b) and (b,a) using the SAME variable id.

    Parallel edges within the same key are collapsed by noisy-OR into one variable probability.
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

    return SupportGraph(out=dict(out), inn=dict(inn), supports=supports, p_var=p_var, p_arc=p_arc)


# -----------------------------
# Simple-path enumeration (exact length k)
# -----------------------------
def enumerate_simple_node_paths(
    G: SupportGraph,
    source: Any,
    target: Any,
    k: int,
    max_paths: Optional[int] = None,
    min_path_probability: float = 0.0,
) -> List[List[Any]]:
    """
    Enumerate SIMPLE directed node-paths of EXACT length k from source to target.
    Returns list of node sequences length (k+1).
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
    """Structural discovery of (A,Z) pairs with ≥1 simple path of exact length k (no probabilities)."""
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


# -----------------------------
# Option 1: Exact probability via BDD (handles shared undirected vars correctly)
# -----------------------------
def _bdd_weighted_probability(bdd, root, var_probs: Dict[str, float]) -> float:
    """
    Compute weighted probability of a BDD under independent variable probabilities.

    dd.bdd / dd.autoref return var as an *integer level* in succ().
    We therefore precompute a mapping level -> probability for speed/robustness.
    """
    memo: Dict[int, float] = {}

    # Precompute level -> p(var) if possible
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
        # dd.autoref.BDD exposes var_at_level; fall back if needed
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

        # dd commonly returns var as an integer level
        if isinstance(var, (int, np.integer)):
            lvl = int(var)
            if lvl in level_prob:
                p = level_prob[lvl]
            else:
                name = level_to_name(lvl)
                p = float(var_probs[name])
        else:
            # some backends may return the variable name directly
            p = float(var_probs[var])

        val = (1.0 - p) * rec(low) + p * rec(high)
        memo[nid] = val
        return val

    return float(rec(root))


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
   
    results: Dict[Tuple[Any, Any], Dict[str, Any]] = {}

    for (A, Z) in queries:
        node_paths = enumerate_simple_node_paths(
            G, A, Z, k,
            max_paths=max_paths_per_query,
            min_path_probability=min_path_probability,
        )
        if not node_paths:
            results[(A, Z)] = {
                "probability": 0.0,
                "num_paths": 0,
                "num_vars": 0,
                "bdd_nodes": 0,
                "truncated": bool(max_paths_per_query),
            }
            continue

        # collect vars used by all clauses on all paths
        used_vars = set()
        used_arcs = set()
        for path in node_paths:
            for i in range(len(path) - 1):
                arc = (path[i], path[i + 1])
                used_arcs.add(arc)
                used_vars.update(G.supports[arc])

        used_vars = sorted(used_vars)
        var_name = {vid: f"v{vid}" for vid in used_vars}
        var_probs = {var_name[vid]: float(G.p_var[vid]) for vid in used_vars}

        bdd = BDD()
        bdd.declare(*[var_name[vid] for vid in used_vars])

        # cache each arc clause as a BDD: clause(arc) = OR vars
        clause_cache = {}
        for arc in used_arcs:
            vids = G.supports[arc]
            clause = bdd.false
            for vid in vids:
                # vids not in used_vars shouldn't happen, but guard anyway
                if vid in var_name:
                    clause = bdd.apply("or", clause, bdd.var(var_name[vid]))
            clause_cache[arc] = clause

        root = bdd.false
        for path in node_paths:
            conj = bdd.true
            for i in range(len(path) - 1):
                conj = bdd.apply("and", conj, clause_cache[(path[i], path[i + 1])])
            root = bdd.apply("or", root, conj)

        prob = _bdd_weighted_probability(bdd, root, var_probs)
        try:
            root_id = getattr(root, "node", root)  # dd.cudd.Function has .node
            bdd_nodes = int(len(root_id)) # dag_size
        except Exception:
            try:
                bdd_nodes = int(bdd._bdd.dag_size(root_id))  # fallback
            except Exception:
                bdd_nodes = None

        results[(A, Z)] = {
            "probability": float(prob),
            "num_paths": int(len(node_paths)),
            "num_vars": int(len(used_vars)),
            "bdd_nodes": bdd_nodes,
            "truncated": bool(max_paths_per_query),
        }

    return results


# -----------------------------
# Option 2: Monte Carlo reachability (unbiased; handles shared undirected vars correctly)
# -----------------------------
def _arc_open(present: np.ndarray, vids: Tuple[int, ...]) -> bool:
    # vids is typically tiny (1–2); linear scan is fine
    for vid in vids:
        if present[vid]:
            return True
    return False


def _reachable_targets_exact_k_one_sample(G: SupportGraph, present: np.ndarray, source: Any, k: int) -> set:
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


# -----------------------------
# Option 3: Hierarchical PSR approximation (fast; uses p_arc)
# -----------------------------
def two_hop_probability_hierarchical(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    P(A->C) = 1 - ∏_B (1 - p(A,B)*p(B,C))
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
    P(A->D) = 1 - ∏_C (1 - P(A->C)*p(C,D))
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

            # compute P(A->C) from B (2-hop)
            sum_log_ac = 0.0
            used_B = 0
            for B, _vids_ab, p_ab in G.out.get(A, []):
                if B == A or B == C or B == D:
                    continue
                p_bc = G.p_arc.get((B, C), 0.0)
                if p_bc <= 0.0:
                    continue
                # optional 3-hop path thresholding
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


# -----------------------------
# Baseline: path-level noisy-OR (your current 3-hop style)
# -----------------------------
def k_hop_probability_path_noisy_or(
    G: SupportGraph,
    queries: Sequence[Tuple[Any, Any]],
    k: int,
    max_paths_per_query: Optional[int] = None,
    min_path_probability: float = 0.0,
) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """
    P = 1 - ∏_paths (1 - ∏_steps p_arc(step))
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
