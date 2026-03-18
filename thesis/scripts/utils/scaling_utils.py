"""
Utilities for edge-type scaling.
Provides:
- global frequency distributions
- global weighting (self-information style)
- context distributions
- context KL ratio weighting
"""

import collections
import math


# -------------------------------------------------------------------------
# GLOBAL DISTRIBUTIONS
# -------------------------------------------------------------------------

def global_type_distribution(kg, type_attr="type", smooth=1.0):
    """
    Compute the global distribution of relation types.
    
    Returns:
        P_bg (dict): type -> P_bg(type)
        counts (dict): type -> raw counts
    """
    type_counts = collections.Counter()
    total_edges = 0

    for _, _, _, data in kg.edges(keys=True, data=True):
        t = data.get(type_attr)
        if t is None:
            continue
        type_counts[t] += 1
        total_edges += 1

    if total_edges == 0:
        raise ValueError("No edges found with attribute %r" % type_attr)

    types = list(type_counts.keys())
    num_types = len(types)

    # Laplace smoothing
    denom = total_edges + smooth * num_types

    P_bg = {t: (type_counts[t] + smooth) / denom for t in types}

    return P_bg, dict(type_counts)


# -------------------------------------------------------------------------
# GLOBAL WEIGHTING (BEFORE AGGREGATION)
# -------------------------------------------------------------------------

def global_weighting(
    kg,
    alpha=0.3,
    type_attr="type",
    prob_attr="probability",
    out_attr="probability_scaled",
    smooth=1.0,
):
    """
    Apply global frequency/self-information weighting BEFORE PSR aggregation.

    w(t) = P_bg(t)^(-alpha)
    """
    P_bg, counts = global_type_distribution(kg, type_attr=type_attr, smooth=smooth)

    # compute weights
    w_type = {t: P_bg[t] ** (-alpha) for t in P_bg}

    # apply to edges
    for u, v, k, data in kg.edges(keys=True, data=True):
        p = float(data.get(prob_attr, 0.0))
        t = data.get(type_attr)
        w = w_type.get(t, 1.0)
        p_new = max(0.0, min(1.0, p * w))
        data[out_attr] = p_new

    return {
        "P_bg": P_bg,
        "type_counts": counts,
        "weights": w_type,
        "prob_attr_out": out_attr,
    }


# -------------------------------------------------------------------------
# CONTEXT DISTRIBUTIONS (FOR SUBSETTING SCRIPTS)
# -------------------------------------------------------------------------

def context_type_distribution(sub_edges, type_attr="type", smooth=1.0, types=None):
    """
    Compute P_ctx(t) for a list of edges (from a subgraph or subset).

    sub_edges: list of edge attribute dicts
    """
    ctx_counts = collections.Counter()
    N_ctx = 0

    for e in sub_edges:
        t = e.get(type_attr)
        if t is None:
            continue
        ctx_counts[t] += 1
        N_ctx += 1

    if types is None:
        types = list(ctx_counts.keys())

    num_types = len(types)
    denom = N_ctx + smooth * num_types

    P_ctx = {t: (ctx_counts.get(t, 0) + smooth) / denom for t in types}
    return P_ctx, dict(ctx_counts)


# -------------------------------------------------------------------------
# CONTEXT KL-RATIO WEIGHTING (INSIDE SUBSETTING/PSR PIPELINES)
# -------------------------------------------------------------------------

def context_kl_ratio_weighting(
    sub_edges,
    P_bg,
    alpha=0.2,
    type_attr="type",
    prob_attr="probability",
    out_attr="probability_scaled_ctx",
    smooth=1.0,
):
    """
    Apply context-specific enrichment ratio (KL component) weighting.

    w(t,c) = (P_ctx(t) / P_bg(t))^alpha
    """
    types = list(P_bg.keys())

    # Compute context distribution
    P_ctx, counts_ctx = context_type_distribution(
        sub_edges, type_attr=type_attr, smooth=smooth, types=types
    )

    # Compute weights
    w_tc = {}
    for t in types:
        ratio = P_ctx[t] / P_bg[t] if P_bg[t] > 0 else 1.0
        w_tc[t] = ratio ** alpha

    # Apply to subset edges
    for e in sub_edges:
        p = float(e.get(prob_attr, 0.0))
        t = e.get(type_attr)
        w = w_tc.get(t, 1.0)
        p_new = max(0.0, min(1.0, p * w))
        e[out_attr] = p_new

    return {
        "P_ctx": P_ctx,
        "context_counts": counts_ctx,
        "weights": w_tc,
        "prob_attr_out": out_attr,
        "sub_edges": sub_edges,
    }