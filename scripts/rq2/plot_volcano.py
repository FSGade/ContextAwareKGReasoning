#!/usr/bin/env python3
"""
RQ2 Volcano Plots — Simplified

Produces two volcano-style plots per tissue comparison:

  1. Effect-size vs Evidence:
        x = log₂(coverage_A / coverage_B)
        y = log₁₀(evidence_A + evidence_B)

  2. Traditional Volcano (effect-size vs significance):
        x = log₂(coverage_A / coverage_B)
        y = −log₁₀(p-value)
     with a horizontal line at the BH-adjusted p-value threshold
     corresponding to FDR q < 0.05.

     Using p-values on the y-axis (rather than q-values) gives better
     visual separation. The BH threshold line shows where significance
     is achieved — everything above the line has q < 0.05.

Usage:
    python plot_volcano.py \
        --comparison subcut_vs_visceral \
        --hops 2 \
        --config config.yaml \
        [--top-n-labels 15] \
        [--bias-threshold 0.3] \
        [--target-filter inflammation]
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import load_config

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pretty_tissue(name: str) -> str:
    return name.replace("_", " ").title()


PALETTE = {
    "tissue_A": "#E64B35",   # warm red
    "tissue_B": "#4DBBD5",   # cool teal
    "neutral":  "#B0B0B0",   # grey
    "sig":      "#7B2D8E",   # purple — significant after FDR
    "corner":   "#FF8C00",   # orange — sig AND beyond effect-size cutoff
}

# ---------------------------------------------------------------------------
# Thesis-quality defaults
# ---------------------------------------------------------------------------
MIN_FONTSIZE = 12
MIN_DPI = 600


def compute_bh_cutoff(pvalues: np.ndarray, alpha: float = 0.05):
    """
    Find the Benjamini-Hochberg p-value cutoff for a given FDR level.

    This is the largest p(i) where p(i) <= (i/m) * alpha.
    Everything with p <= this cutoff has q < alpha.

    Returns:
        cutoff (float or None): The p-value cutoff, or None if nothing passes.
        n_significant (int): Number of tests passing the threshold.
    """
    valid = pvalues[~np.isnan(pvalues)]
    if len(valid) == 0:
        return None, 0

    sorted_p = np.sort(valid)
    m = len(sorted_p)
    thresholds = np.arange(1, m + 1) / m * alpha

    passing = sorted_p <= thresholds
    if passing.any():
        last_passing_idx = np.where(passing)[0][-1]
        return float(sorted_p[last_passing_idx]), int(last_passing_idx + 1)
    else:
        return None, 0


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(df: pd.DataFrame, epsilon: float = 0.01,
                 bias_threshold: float = 0.3,
                 fdr_threshold: float = 0.05,
                 log2_effect_cutoff: float = None) -> pd.DataFrame:
    df = df.copy()

    # x-axis: log2 coverage ratio
    cov_A = df["coverage_A"].fillna(0.0) + epsilon
    cov_B = df["coverage_B"].fillna(0.0) + epsilon
    df["log2_ratio"] = np.log2(cov_A / cov_B)

    # Evidence (sum)
    ev_A = df["evidence_A"].fillna(0.0)
    ev_B = df["evidence_B"].fillna(0.0)
    df["sum_evidence"] = ev_A + ev_B
    df["log10_evidence"] = np.log10(df["sum_evidence"].clip(lower=1e-6))

    # -log10(p-value) for traditional volcano (better spread than q-values)
    if "perm_pvalue" in df.columns:
        pv = df["perm_pvalue"].clip(lower=1e-300)
        df["neg_log10_p"] = -np.log10(pv)
    else:
        df["neg_log10_p"] = np.nan

    # Keep q-values for colouring significance
    if "perm_qvalue" in df.columns:
        qv = df["perm_qvalue"].clip(lower=1e-300)
        df["neg_log10_q"] = -np.log10(qv)
    else:
        df["neg_log10_q"] = np.nan

    # Colour logic:
    #   1. Significant (q < fdr_threshold) → "sig"
    #   2. Everything else → "neutral"
    has_q = "perm_qvalue" in df.columns and df["perm_qvalue"].notna().any()

    if has_q:
        sig = df["perm_qvalue"].fillna(1.0) < fdr_threshold
        df["colour"] = np.where(sig, "sig", "neutral")
    else:
        df["colour"] = "neutral"

    # Promote significant points beyond effect-size cutoff to "corner"
    if log2_effect_cutoff is not None and log2_effect_cutoff > 0:
        is_sig = df["colour"] == "sig"
        beyond_cutoff = df["log2_ratio"].abs() > log2_effect_cutoff
        df.loc[is_sig & beyond_cutoff, "colour"] = "corner"

    return df


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_scatter(ax, x, y, colours, top_n_labels, df,
                  label_col="source_gene"):
    """Draw scatter with optional extreme-point labels."""
    # Drop rows where y is NaN
    valid = y.notna()
    if not valid.any():
        return

    x_v = x[valid]
    y_v = y[valid]
    colours_v = colours[valid]
    df_v = df.loc[valid]

    # Draw neutral first
    is_neutral = (colours_v == "neutral").values
    order = np.concatenate([np.where(is_neutral)[0],
                            np.where(~is_neutral)[0]])

    ax.scatter(
        x_v.values[order], y_v.values[order],
        c=[PALETTE.get(c, PALETTE["neutral"]) for c in colours_v.values[order]],
        s=8, alpha=0.45, linewidths=0, rasterized=True,
    )
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)

    if top_n_labels > 0:
        x_rank = x_v.abs().rank(pct=True)
        y_rank = y_v.rank(pct=True)
        dist = x_rank * y_rank
        dist[colours_v == "neutral"] = -np.inf
        top_idx = dist.nlargest(top_n_labels).index

        if HAS_ADJUST_TEXT:
            texts = []
            for idx in top_idx:
                label = str(df_v.loc[idx, label_col])
                if len(label) > 22:
                    label = label[:20] + "…"
                t = ax.text(
                    x_v.loc[idx], y_v.loc[idx], label,
                    fontsize=MIN_FONTSIZE, alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="0.7",
                              alpha=0.75, linewidth=0.4),
                )
                texts.append(t)
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="0.4", linewidth=0.4),
                expand=(1.4, 1.6),
                force_text=(0.8, 1.0),
                force_points=(0.3, 0.4),
            )
        else:
            for idx in top_idx:
                label = str(df_v.loc[idx, label_col])
                if len(label) > 22:
                    label = label[:20] + "…"
                ax.annotate(
                    label, (x_v.loc[idx], y_v.loc[idx]),
                    fontsize=MIN_FONTSIZE, alpha=0.85,
                    ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3),
                )


def _save_figure(fig, path, dpi):
    dpi = max(dpi, MIN_DPI)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _symmetric_xlim(ax, x_data):
    """Force x-axis to be symmetric around 0, showing full range both sides."""
    x_valid = x_data.dropna()
    if len(x_valid) == 0:
        return
    abs_max = max(abs(x_valid.min()), abs(x_valid.max())) * 1.05
    ax.set_xlim(-abs_max, abs_max)


def _draw_vertical_cutoffs(ax, log2_effect_cutoff):
    """Draw vertical dashed lines at ±log2_effect_cutoff."""
    if log2_effect_cutoff is None or log2_effect_cutoff <= 0:
        return
    for sign in [-1, 1]:
        ax.axvline(sign * log2_effect_cutoff, color="#666",
                   linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(log2_effect_cutoff, ax.get_ylim()[1] * 0.98,
            f"  |log₂| = {log2_effect_cutoff}",
            fontsize=MIN_FONTSIZE, color="#666", va="top", ha="left",
            alpha=0.6)


def _draw_extra_fdr_lines(ax, df, fdr_alpha):
    """
    For stricter FDR thresholds beyond the primary one, draw additional
    horizontal lines if any points pass them.
    """
    if "perm_pvalue" not in df.columns:
        return
    pvalues = df["perm_pvalue"].values

    stricter = [t for t in [0.01, 0.001] if t < fdr_alpha or t != fdr_alpha]
    # Only draw thresholds stricter than the primary
    stricter = [t for t in stricter if t < fdr_alpha]

    for alpha_s in stricter:
        cutoff_s, n_sig_s = compute_bh_cutoff(pvalues, alpha=alpha_s)
        if cutoff_s is not None and n_sig_s > 0:
            y_s = -np.log10(cutoff_s)
            ax.axhline(y_s, color="#d32f2f", linewidth=0.7,
                       linestyle=":", alpha=0.5)
            ax.text(ax.get_xlim()[1], y_s,
                    f" q < {alpha_s} (n={n_sig_s:,})",
                    fontsize=MIN_FONTSIZE, color="#d32f2f",
                    va="bottom", ha="right", alpha=0.7)


def _draw_extra_q_lines(ax, df, fdr_alpha):
    """
    For the q-value volcano: draw horizontal lines at stricter thresholds
    directly on the −log₁₀(q) scale.
    """
    if "perm_qvalue" not in df.columns:
        return
    q = df["perm_qvalue"].fillna(1.0)

    stricter = [t for t in [0.01, 0.001] if t < fdr_alpha]
    for alpha_s in stricter:
        n_pass = (q < alpha_s).sum()
        if n_pass > 0:
            y_s = -np.log10(alpha_s)
            ax.axhline(y_s, color="#d32f2f", linewidth=0.7,
                       linestyle=":", alpha=0.5)
            ax.text(ax.get_xlim()[1], y_s,
                    f" q < {alpha_s} (n={n_pass:,})",
                    fontsize=MIN_FONTSIZE, color="#d32f2f",
                    va="bottom", ha="right", alpha=0.7)


def _counts(df):
    c = df["colour"]
    n_sig = (c == "sig").sum()
    n_corner = (c == "corner").sum()
    n_A = (c == "tissue_A").sum()
    n_B = (c == "tissue_B").sum()
    n_N = (c == "neutral").sum()
    return n_A, n_B, n_N, n_sig, n_corner


def _add_legend(ax, handles):
    """Place legend outside the plot area so it never overlaps data."""
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.15),
        ncol=2,
        fontsize=MIN_FONTSIZE,
        framealpha=0.9,
        borderaxespad=0.0,
        columnspacing=1.0,
        handletextpad=0.5,
    )


# ---------------------------------------------------------------------------
# Plot 1: Evidence volcano
# ---------------------------------------------------------------------------

def plot_evidence_volcano(df, tissue_A, tissue_B, hops, top_n_labels,
                          output_path, bias_threshold,
                          log2_effect_cutoff=None,
                          subtitle="", dpi=600):
    """log₂(coverage ratio)  vs  log₁₀(Σ evidence)."""
    dpi = max(dpi, MIN_DPI)
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    _draw_scatter(ax, df["log2_ratio"], df["log10_evidence"],
                  df["colour"], top_n_labels, df)

    _symmetric_xlim(ax, df["log2_ratio"])
    _draw_vertical_cutoffs(ax, log2_effect_cutoff)

    n_A, n_B, n_N, n_sig, n_corner = _counts(df)
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["neutral"], markersize=7,
               label=f"Not significant (n={n_N + n_A + n_B:,})"),
    ]
    if n_sig:
        handles.insert(0, Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=PALETTE["sig"], markersize=7,
            label=f"FDR significant (n={n_sig:,})"))
    if n_corner:
        handles.insert(0, Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=PALETTE["corner"], markersize=7,
            label=f"Sig + |log₂|>{log2_effect_cutoff} (n={n_corner:,})"))

    _add_legend(ax, handles)
    ax.set_xlabel(f"← {tB}      log₂(coverage ratio)      {tA} →",
                  fontsize=MIN_FONTSIZE)
    ax.set_ylabel("log₁₀(Σ evidence score)", fontsize=MIN_FONTSIZE)
    ax.tick_params(labelsize=MIN_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = f"{tA} vs {tB} — {hops}-hop  (evidence)"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=MIN_FONTSIZE + 2, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Plot 2: Traditional volcano (p-value with BH threshold)
# ---------------------------------------------------------------------------

def plot_significance_volcano(df, tissue_A, tissue_B, hops, top_n_labels,
                              output_path, bias_threshold, n_perms,
                              fdr_alpha=0.05, log2_effect_cutoff=None,
                              subtitle="", dpi=600):
    """
    log₂(coverage ratio)  vs  −log₁₀(p-value).

    Uses raw p-values on the y-axis for better visual spread, with the
    Benjamini-Hochberg adjusted threshold drawn as a horizontal line
    to indicate where FDR significance (q < alpha) is achieved.

    This is standard practice (e.g. EnhancedVolcano in R, DESeq2 tutorials).
    """
    dpi = max(dpi, MIN_DPI)
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    if df["neg_log10_p"].isna().all():
        print("  WARNING: No p-values available — skipping significance volcano")
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))

    _draw_scatter(ax, df["log2_ratio"], df["neg_log10_p"],
                  df["colour"], top_n_labels, df)

    _symmetric_xlim(ax, df["log2_ratio"])
    _draw_vertical_cutoffs(ax, log2_effect_cutoff)

    # ---- BH-adjusted threshold line ----
    # This is the p-value cutoff where q < alpha: everything above passes FDR
    if "perm_pvalue" in df.columns:
        pvalues = df["perm_pvalue"].values
        bh_cutoff, n_sig_bh = compute_bh_cutoff(pvalues, alpha=fdr_alpha)

        if bh_cutoff is not None:
            y_bh = -np.log10(bh_cutoff)
            ax.axhline(y_bh, color="#d32f2f", linewidth=1.0,
                       linestyle="--", alpha=0.7)
            ax.text(ax.get_xlim()[1], y_bh,
                    f" FDR q < {fdr_alpha} (n={n_sig_bh:,})",
                    fontsize=MIN_FONTSIZE, color="#d32f2f",
                    va="bottom", ha="right", alpha=0.8)
        else:
            # Nothing passes — show where the line would need to be
            # BH threshold for rank 1 = alpha / m
            m = (~np.isnan(pvalues)).sum()
            if m > 0:
                needed_p = fdr_alpha / m
                y_needed = -np.log10(needed_p)
                # Only draw if it's within a reasonable range of the data
                y_max = df["neg_log10_p"].max()
                if np.isfinite(y_needed) and y_needed < y_max * 3:
                    ax.axhline(y_needed, color="#d32f2f", linewidth=0.8,
                               linestyle="--", alpha=0.4)
                    ax.text(ax.get_xlim()[1], y_needed,
                            f" FDR q < {fdr_alpha} threshold (none pass)",
                            fontsize=MIN_FONTSIZE, color="#d32f2f",
                            va="bottom", ha="right", alpha=0.5)

    # ---- Stricter FDR lines if points pass them ----
    _draw_extra_fdr_lines(ax, df, fdr_alpha)

    # ---- Nominal p = 0.05 line ----
    y_nominal = -np.log10(0.05)
    ax.axhline(y_nominal, color="#888", linewidth=0.6,
               linestyle=":", alpha=0.5)
    ax.text(ax.get_xlim()[0], y_nominal,
            f"  p = 0.05 (nominal)  ",
            fontsize=MIN_FONTSIZE, color="#888", va="bottom",
            ha="left", alpha=0.6)

    # ---- Permutation floor ----
    if n_perms and n_perms > 0:
        min_p = 1.0 / (n_perms + 1)
        max_y = -np.log10(min_p)
        ax.text(ax.get_xlim()[0], max_y * 0.97,
                f"  min p = {min_p:.4f} ({n_perms} perms)",
                fontsize=MIN_FONTSIZE, color="grey", va="top",
                ha="left", alpha=0.5)

    # ---- Legend ----
    n_A, n_B, n_N, n_sig, n_corner = _counts(df)
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["sig"], markersize=7,
               label=f"FDR q<{fdr_alpha} (n={n_sig:,})"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["neutral"], markersize=7,
               label=f"Not significant (n={n_N + n_A + n_B:,})"),
    ]
    if n_corner:
        handles.insert(0, Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=PALETTE["corner"], markersize=7,
            label=f"Sig + |log₂|>{log2_effect_cutoff} (n={n_corner:,})"))

    _add_legend(ax, handles)
    ax.set_xlabel(f"← {tB}      log₂(coverage ratio)      {tA} →",
                  fontsize=MIN_FONTSIZE)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=MIN_FONTSIZE)
    ax.tick_params(labelsize=MIN_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = f"{tA} vs {tB} — {hops}-hop  (significance)"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=MIN_FONTSIZE + 2, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Plot 3: Q-value volcano (effect-size vs −log₁₀ q-value)
# ---------------------------------------------------------------------------

def plot_qvalue_volcano(df, tissue_A, tissue_B, hops, top_n_labels,
                        output_path, bias_threshold, fdr_alpha=0.05,
                        log2_effect_cutoff=None,
                        subtitle="", dpi=600):
    """
    log₂(coverage ratio)  vs  −log₁₀(q-value).

    Complements the p-value volcano with the corrected significance on
    the y-axis directly, making the FDR thresholds easier to read.
    """
    dpi = max(dpi, MIN_DPI)
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    if "neg_log10_q" not in df.columns or df["neg_log10_q"].isna().all():
        print("  WARNING: No q-values available — skipping q-value volcano")
        return

    fig, ax = plt.subplots(figsize=(8, 6.5))

    _draw_scatter(ax, df["log2_ratio"], df["neg_log10_q"],
                  df["colour"], top_n_labels, df)

    _symmetric_xlim(ax, df["log2_ratio"])
    _draw_vertical_cutoffs(ax, log2_effect_cutoff)

    # ---- Primary FDR line ----
    y_fdr = -np.log10(fdr_alpha)
    q = df["perm_qvalue"].fillna(1.0) if "perm_qvalue" in df.columns else pd.Series(1.0, index=df.index)
    n_pass = (q < fdr_alpha).sum()
    ax.axhline(y_fdr, color="#d32f2f", linewidth=1.0,
               linestyle="--", alpha=0.7)
    ax.text(ax.get_xlim()[1], y_fdr,
            f" q < {fdr_alpha} (n={n_pass:,})",
            fontsize=MIN_FONTSIZE, color="#d32f2f",
            va="bottom", ha="right", alpha=0.8)

    # ---- Stricter thresholds ----
    _draw_extra_q_lines(ax, df, fdr_alpha)

    # ---- Legend ----
    n_A, n_B, n_N, n_sig, n_corner = _counts(df)
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["sig"], markersize=7,
               label=f"FDR q<{fdr_alpha} (n={n_sig:,})"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PALETTE["neutral"], markersize=7,
               label=f"Not significant (n={n_N + n_A + n_B:,})"),
    ]
    if n_corner:
        handles.insert(0, Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=PALETTE["corner"], markersize=7,
            label=f"Sig + |log₂|>{log2_effect_cutoff} (n={n_corner:,})"))

    _add_legend(ax, handles)
    ax.set_xlabel(f"← {tB}      log₂(coverage ratio)      {tA} →",
                  fontsize=MIN_FONTSIZE)
    ax.set_ylabel("−log₁₀(q-value)", fontsize=MIN_FONTSIZE)
    ax.tick_params(labelsize=MIN_FONTSIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    title = f"{tA} vs {tB} — {hops}-hop  (q-value)"
    if subtitle:
        title += f"\n{subtitle}"
    ax.set_title(title, fontsize=MIN_FONTSIZE + 2, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# LaTeX table generation — Top N significant tissue-biased inferences
# ---------------------------------------------------------------------------

TISSUE_ABBREVS = {
    "subcutaneous": "SC", "visceral": "VS", "white": "W", "brown": "B",
}


def _tissue_abbrev(name: str) -> str:
    return TISSUE_ABBREVS.get(name.lower(), name[:2].upper())


def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    for char, repl in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                       ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(char, repl)
    return s


def _fmt_q(q) -> str:
    if q is None or pd.isna(q):
        return "---"
    if q < 0.001:
        return r"$<$0.001"
    return f"{q:.3f}"


def _fmt_cov(c) -> str:
    if c is None or pd.isna(c):
        return "0.000"
    return f"{c:.3f}"


def _fmt_diff(d) -> str:
    if d is None or pd.isna(d):
        return "---"
    sign = "$+$" if d > 0 else "$-$" if d < 0 else ""
    return f"{sign}{abs(d):.3f}"


def _fmt_corr(c) -> str:
    if c is None or pd.isna(c):
        return "?"
    c = int(c)
    if c == 1:
        return "$+$"
    if c == -1:
        return "$-$"
    return "0"


def _fmt_prob(p) -> str:
    """Format probability for LaTeX tables."""
    if p is None or pd.isna(p):
        return "---"
    return f"{p:.3f}"


def _fmt_ev(e) -> str:
    """Format evidence score for LaTeX tables."""
    if e is None or pd.isna(e):
        return "---"
    if e >= 100:
        return f"{e:,.0f}"
    return f"{e:.1f}"


def _shorten_rel(rel: str) -> str:
    if not isinstance(rel, str):
        return "?"
    mapping = {
        "positive_correlation": r"Pos\_Corr",
        "negative_correlation": r"Neg\_Corr",
        "association": "Assoc",
        "target_disease": r"Targ\_Dis",
        "marker_mechanism": r"Mark\_Mech",
        "therapeutic": "Therap",
    }
    rel_lower = rel.lower().strip()
    for key, val in mapping.items():
        if key in rel_lower:
            return val
    short = rel.replace("_", " ").title()
    if len(short) > 12:
        short = short[:10] + "."
    return _latex_escape(short)


def _safe_list(val):
    """Safely convert a value to a plain Python list.

    Handles numpy arrays, pandas Series, plain lists, None, and scalars.
    Avoids the 'truth value of an array is ambiguous' error that occurs
    when using ``val or []`` on numpy arrays.
    """
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, pd.Series):
        return val.tolist()
    # scalar or string — wrap in list
    if isinstance(val, str):
        return [val]
    return [val]


def _compact_metapath(metapath: str, n_int_b=None, n_int_c=None,
                      num_intermediates=None) -> str:
    """Compact LaTeX metapath with intermediate counts in parentheses."""
    if not isinstance(metapath, str):
        return r"\texttt{?}"

    parts = re.split(r'(-\[.*?\]-)', metapath)
    parts = [p for p in parts if p]

    node_abbrevs = {
        "gene": "Gene", "chemical": "Chem", "disease": "Dis",
        "compound": "Cmpd", "protein": "Prot", "pathway": "Path",
        "anatomy": "Anat", "biologicalprocess": "BP",
        "molecularfunction": "MF", "cellularcomponent": "CC",
    }

    def shorten_node(s):
        key = s.lower().strip()
        for k, v in node_abbrevs.items():
            if k in key:
                return v
        return s[:5] + "." if len(s) > 6 else s

    def shorten_edge(s):
        inner = s.strip("-[]")
        mapping = {
            "positivecorr": "PosCorr", "negativecorr": "NegCorr",
            "association": "Assoc", "targetdiseas": "TargDis",
            "markermechan": "MarkMech", "therapeutic": "Therap",
        }
        for k, v in mapping.items():
            if k in inner.lower():
                return f"-[{v}]-"
        short = inner[:8] if len(inner) > 8 else inner
        return f"-[{short}]-"

    result_parts = []
    for p in parts:
        if p.startswith("-["):
            result_parts.append(shorten_edge(p))
        else:
            result_parts.append(shorten_node(p))

    if n_int_b is not None and n_int_c is not None and len(result_parts) >= 7:
        result_parts[2] += f"({int(n_int_b)})"
        result_parts[4] += f"({int(n_int_c)})"
    elif num_intermediates is not None and len(result_parts) >= 3:
        result_parts[2] += f"({int(num_intermediates)})"

    return r"\texttt{" + "".join(result_parts) + "}"


def _generate_2hop_table(df, tissue_A, tissue_B, direction, top_n):
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    abA, abB = _tissue_abbrev(tissue_A), _tissue_abbrev(tissue_B)
    biased = tA if direction == "A" else tB
    label_t = tissue_A if direction == "A" else tissue_B
    shown = min(len(df), top_n)

    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        rf"\caption[Top {biased}-biased inferences (2-hop)]"
        rf"{{\textbf{{Top {shown} {biased}-biased inferred relations (2-hop).}} "
        rf"Ranked by $|\Delta\text{{Cov}}|$; only $q < 0.05$.}}",
        rf"\label{{tab:top{shown}_{label_t}_2hop}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{rlllclcrrrcrr}",
        r"\toprule",
        rf"\textbf{{\#}} & \textbf{{Source}} & \textbf{{Rel$_1$}} & "
        rf"\textbf{{\# Int.}} & \textbf{{Rel$_2$}} & \textbf{{Target}} & "
        rf"\textbf{{Corr}} & \textbf{{Cov$_{{\text{{{abA}}}}}$}} & "
        rf"\textbf{{Cov$_{{\text{{{abB}}}}}$}} & \textbf{{$\Delta$Cov}} & "
        rf"\textbf{{Prob}} & \textbf{{Ev}} & "
        rf"\textbf{{$q$}} \\",
        r"\midrule",
    ]

    # Resolve probability/evidence column names:
    # comparison parquets use prob_A/prob_B, evidence_A/evidence_B
    # per-tissue parquets use probability, evidence_score
    prob_col = f"prob_{direction}" if f"prob_{direction}" in df.columns else "probability"
    ev_col = f"evidence_{direction}" if f"evidence_{direction}" in df.columns else "evidence_score"

    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        rels = _safe_list(row.get("relationship_types"))
        rel1 = _shorten_rel(rels[0] if len(rels) > 0 else "?")
        rel2 = _shorten_rel(rels[1] if len(rels) > 1 else "?")
        n_int = int(row.get("num_intermediates", 0) or 0)
        corr = _fmt_corr(row.get("correlation_A") or row.get("correlation_B"))
        lines.append(
            f"{i:>2} & {_latex_escape(str(row.get('source_gene', '?')))} & "
            f"{rel1} & {n_int} & {rel2} & "
            f"{_latex_escape(str(row.get('target_phenotype', '?')))} & "
            f"{corr} & {_fmt_cov(row.get('coverage_A'))} & "
            f"{_fmt_cov(row.get('coverage_B'))} & "
            f"{_fmt_diff(row.get('diff_coverage'))} & "
            f"{_fmt_prob(row.get(prob_col))} & "
            f"{_fmt_ev(row.get(ev_col))} & "
            f"{_fmt_q(row.get('perm_qvalue'))} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def _generate_3hop_table(df, tissue_A, tissue_B, direction, top_n):
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    abA, abB = _tissue_abbrev(tissue_A), _tissue_abbrev(tissue_B)
    biased = tA if direction == "A" else tB
    label_t = tissue_A if direction == "A" else tissue_B
    shown = min(len(df), top_n)

    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        rf"\caption[Top {biased}-biased inferences (3-hop)]"
        rf"{{\textbf{{Top {shown} {biased}-biased inferred relations (3-hop).}} "
        rf"Compact metapath with intermediate counts. "
        rf"Ranked by $|\Delta\text{{Cov}}|$; only $q < 0.05$.}}",
        rf"\label{{tab:top{shown}_{label_t}_3hop}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\small",
        r"\begin{tabular}{rlp{6cm}lrrrcrr}",
        r"\toprule",
        rf"\textbf{{\#}} & \textbf{{Source}} & \textbf{{Metapath}} & "
        rf"\textbf{{Target}} & \textbf{{Cov$_{{\text{{{abA}}}}}$}} & "
        rf"\textbf{{Cov$_{{\text{{{abB}}}}}$}} & \textbf{{$\Delta$Cov}} & "
        rf"\textbf{{Prob}} & \textbf{{Ev}} & "
        rf"\textbf{{$q$}} \\",
        r"\midrule",
    ]

    # Resolve probability/evidence column names (same logic as 2-hop)
    prob_col = f"prob_{direction}" if f"prob_{direction}" in df.columns else "probability"
    ev_col = f"evidence_{direction}" if f"evidence_{direction}" in df.columns else "evidence_score"

    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        mp = _compact_metapath(
            str(row.get("metapath", "?")),
            n_int_b=row.get("n_intermediates_B"),
            n_int_c=row.get("n_intermediates_C"),
            num_intermediates=row.get("num_intermediates"),
        )
        lines.append(
            f"{i:>2} & {_latex_escape(str(row.get('source_gene', '?')))} & "
            f"{mp} & "
            f"{_latex_escape(str(row.get('target_phenotype', '?')))} & "
            f"{_fmt_cov(row.get('coverage_A'))} & "
            f"{_fmt_cov(row.get('coverage_B'))} & "
            f"{_fmt_diff(row.get('diff_coverage'))} & "
            f"{_fmt_prob(row.get(prob_col))} & "
            f"{_fmt_ev(row.get(ev_col))} & "
            f"{_fmt_q(row.get('perm_qvalue'))} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def _generate_intermediates_table(df, tissue_A, tissue_B, direction, top_n, hops):
    """
    Companion table listing intermediate nodes for each row in the main table.

    Reads from whichever intermediate column(s) exist in the parquet:
      2-hop: 'intermediate_genes' (list of B-node names)
      3-hop: 'intermediates_B' + 'intermediates_C', or fallback to 'intermediate_genes'
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    biased = tA if direction == "A" else tB
    label_t = tissue_A if direction == "A" else tissue_B
    shown = min(len(df), top_n)

    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        rf"\caption[Intermediates for top {biased}-biased ({hops}-hop)]"
        rf"{{\textbf{{Intermediate nodes for top {shown} {biased}-biased "
        rf"inferred relations ({hops}-hop).}} "
        rf"Row numbers match the corresponding results table.}}",
        rf"\label{{tab:intermediates_{label_t}_{hops}hop}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\small",
    ]

    # Check for 3-hop split intermediate columns (try both naming conventions)
    has_3hop_split = False
    col_b, col_c = None, None
    if hops == 3:
        for cb, cc in [("intermediate_genes_B", "intermediate_genes_C"),
                        ("intermediates_B", "intermediates_C")]:
            if cb in df.columns and cc in df.columns:
                has_3hop_split = True
                col_b, col_c = cb, cc
                break

    if has_3hop_split:
        # 3-hop with separate B and C intermediate columns
        lines += [
            r"\begin{tabular}{rllp{5cm}p{5cm}}",
            r"\toprule",
            r"\textbf{\#} & \textbf{Source} & \textbf{Target} & "
            r"\textbf{B-intermediates} & \textbf{C-intermediates} \\",
            r"\midrule",
        ]
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            src = _latex_escape(str(row.get("source_gene", "?")))
            tgt = _latex_escape(str(row.get("target_phenotype", "?")))
            ints_b = _safe_list(row.get(col_b))
            ints_c = _safe_list(row.get(col_c))
            b_str = ", ".join(_latex_escape(str(x)) for x in ints_b) if ints_b else "---"
            c_str = ", ".join(_latex_escape(str(x)) for x in ints_c) if ints_c else "---"
            lines.append(
                f"{i:>2} & {src} & {tgt} & {b_str} & {c_str} \\\\"
            )
    else:
        # 2-hop or 3-hop with single intermediates column
        lines += [
            r"\begin{tabular}{rllp{9cm}}",
            r"\toprule",
            r"\textbf{\#} & \textbf{Source} & \textbf{Target} & "
            r"\textbf{Intermediates} \\",
            r"\midrule",
        ]
        for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
            src = _latex_escape(str(row.get("source_gene", "?")))
            tgt = _latex_escape(str(row.get("target_phenotype", "?")))
            # Try multiple possible column names; comparison parquets use
            # tissue-suffixed names (intermediate_genes_A / _B)
            ints = None
            for col in ["intermediate_genes",
                         "intermediate_genes_A", "intermediate_genes_B",
                         "intermediates", "intermediate_nodes"]:
                if col in df.columns:
                    ints = _safe_list(row.get(col))
                    if ints:
                        break
            if not ints:
                ints_str = "---"
            else:
                ints_str = ", ".join(_latex_escape(str(x)) for x in ints)
            lines.append(
                f"{i:>2} & {src} & {tgt} & {ints_str} \\\\"
            )

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_tables(df, tissue_A, tissue_B, hops, top_n,
                          fdr_threshold, rank_by, output_dir):
    """
    Generate and save LaTeX top-N tables for both tissue directions.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    has_q = "perm_qvalue" in df.columns and df["perm_qvalue"].notna().any()
    if has_q:
        sig = df[df["perm_qvalue"].fillna(1.0) < fdr_threshold].copy()
    else:
        print("  WARNING: No q-values — using all rows for tables")
        sig = df.copy()

    print(f"\n  LaTeX tables: {len(sig):,} significant of {len(df):,} total")

    sig_A = sig[sig["diff_coverage"].fillna(0) > 0].copy()
    sig_B = sig[sig["diff_coverage"].fillna(0) < 0].copy()
    print(f"    {tA}-biased: {len(sig_A):,}")
    print(f"    {tB}-biased: {len(sig_B):,}")

    # Rank
    rank_map = {
        "coverage":    ("_abs_diff", True),
        "probability": ("_prob_max", True),
        "evidence":    ("_ev_max", True),
        "qvalue":      ("perm_qvalue", False),
    }
    if rank_by not in rank_map:
        rank_by = "coverage"

    for subset in [sig_A, sig_B]:
        subset["_abs_diff"] = subset["diff_coverage"].abs()
        pa = subset.get("prob_A", pd.Series(0, index=subset.index)).fillna(0)
        pb = subset.get("prob_B", pd.Series(0, index=subset.index)).fillna(0)
        subset["_prob_max"] = pd.concat([pa, pb], axis=1).max(axis=1)
        ea = subset.get("evidence_A", pd.Series(0, index=subset.index)).fillna(0)
        eb = subset.get("evidence_B", pd.Series(0, index=subset.index)).fillna(0)
        subset["_ev_max"] = pd.concat([ea, eb], axis=1).max(axis=1)

    sort_col, sort_desc = rank_map[rank_by]
    sig_A = sig_A.sort_values(sort_col, ascending=not sort_desc)
    sig_B = sig_B.sort_values(sort_col, ascending=not sort_desc)

    gen = _generate_2hop_table if hops == 2 else _generate_3hop_table

    tables_dir = output_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    PREAMBLE = (
        "% Auto-generated by rq2_plot_volcano.py\n"
        "% Requires: booktabs, adjustbox\n"
    )

    prefix = f"top{top_n}_{hops}hop"

    for direction, label, subset in [("A", tissue_A, sig_A),
                                      ("B", tissue_B, sig_B)]:
        if len(subset) > 0:
            tex = gen(subset, tissue_A, tissue_B, direction, top_n)
            tex_int = _generate_intermediates_table(
                subset, tissue_A, tissue_B, direction, top_n, hops)
        else:
            tex = f"% No significant {pretty_tissue(label)}-biased results at q < {fdr_threshold}"
            tex_int = tex

        out_path = tables_dir / f"{prefix}_{label}.tex"
        with open(out_path, "w") as f:
            f.write(PREAMBLE + "\n" + tex + "\n")
        print(f"    Saved: {out_path}")

        int_path = tables_dir / f"{prefix}_{label}_intermediates.tex"
        with open(int_path, "w") as f:
            f.write(PREAMBLE + "\n" + tex_int + "\n")
        print(f"    Saved: {int_path}")

    # Combined
    combined_path = tables_dir / f"{prefix}_combined.tex"
    with open(combined_path, "w") as f:
        f.write(PREAMBLE + "\n")
        for direction, subset in [("A", sig_A), ("B", sig_B)]:
            if len(subset) > 0:
                f.write(gen(subset, tissue_A, tissue_B, direction, top_n))
                f.write("\n\n")
                f.write(_generate_intermediates_table(
                    subset, tissue_A, tissue_B, direction, top_n, hops))
            else:
                lab = tissue_A if direction == "A" else tissue_B
                f.write(f"% No significant {pretty_tissue(lab)}-biased results")
            f.write("\n\n\n")
    print(f"    Saved: {combined_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df, tissue_A, tissue_B, bias_threshold, target_filter=None):
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    n_A, n_B, n_N, n_sig, n_corner = _counts(df)
    print(f"\n{'='*60}")
    print("Volcano plot data summary")
    print(f"{'='*60}")
    if target_filter:
        print(f"Target filter:       '{target_filter}'")
    print(f"Total associations:  {len(df):,}")
    print(f"Bias threshold:      |diff_coverage| > {bias_threshold}")
    print(f"  {tA}-biased:  {n_A:,}")
    print(f"  {tB}-biased:  {n_B:,}")
    print(f"  Neutral:            {n_N:,}")
    print(f"  FDR significant:    {n_sig:,}")
    if n_corner:
        print(f"  Sig + beyond cutoff: {n_corner:,}")

    has_p = df["perm_pvalue"].notna().sum() if "perm_pvalue" in df.columns else 0
    has_q = df["perm_qvalue"].notna().sum() if "perm_qvalue" in df.columns else 0
    print(f"  Triples with p-value: {has_p:,}")
    print(f"  Triples with q-value: {has_q:,}")
    if has_q:
        print(f"  q < 0.05: {(df['perm_qvalue'] < 0.05).sum():,}")
        print(f"  q < 0.01: {(df['perm_qvalue'] < 0.01).sum():,}")
    if has_p:
        print(f"  p < 0.05: {(df['perm_pvalue'] < 0.05).sum():,}")
        print(f"  p < 0.01: {(df['perm_pvalue'] < 0.01).sum():,}")

    print(f"\nlog₂ ratio range:  [{df['log2_ratio'].min():.2f}, "
          f"{df['log2_ratio'].max():.2f}]")
    print(f"log₁₀(evidence):   [{df['log10_evidence'].min():.2f}, "
          f"{df['log10_evidence'].max():.2f}]")
    if not HAS_ADJUST_TEXT:
        print("\nWARNING: adjustText not installed — labels may overlap.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 volcano plots (evidence + significance)")
    parser.add_argument("--comparison", required=True,
                        choices=["subcut_vs_visceral", "white_vs_brown"])
    parser.add_argument("--hops", type=int, required=True, choices=[2, 3])
    parser.add_argument("--config", required=True)
    parser.add_argument("--top-n-labels", type=int, default=0)
    parser.add_argument("--bias-threshold", type=float, default=0.3)
    parser.add_argument("--target-filter", type=str, default=None)
    parser.add_argument("--fdr-alpha", type=float, default=0.05,
                        help="FDR threshold for BH significance line")
    parser.add_argument("--log2-effect-cutoff", type=float, default=None,
                        help="Vertical line at ±this log₂ ratio "
                             "(e.g. 5 for subcut_vs_visceral)")
    parser.add_argument("--top-n-table", type=int, default=20,
                        help="Number of rows per LaTeX table (default: 20)")
    parser.add_argument("--rank-by", type=str, default="coverage",
                        choices=["coverage", "probability", "evidence", "qvalue"],
                        help="How to rank rows in LaTeX tables (default: coverage)")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])

    comp_cfg = next(
        (c for c in config["comparisons"] if c["name"] == args.comparison),
        None)
    if comp_cfg is None:
        sys.exit(f"ERROR: comparison '{args.comparison}' not found in config")

    tissue_A = comp_cfg["tissue_A"]
    tissue_B = comp_cfg["tissue_B"]
    epsilon = config["coverage"].get("epsilon", 0.01)
    bt = args.bias_threshold
    l2c = args.log2_effect_cutoff

    # Load data for volcano plots (uses the --hops argument)
    comp_dir = output_dir / "comparisons" / args.comparison
    path = comp_dir / f"comparison_{args.hops}hop.parquet"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — run compare.py first.")
    df = pd.read_parquet(path)
    print(f"Loaded: {path}  ({len(df):,} rows)")

    # Optional target filter
    if args.target_filter:
        if "target_phenotype" not in df.columns:
            sys.exit("ERROR: no 'target_phenotype' column for filtering")
        mask = df["target_phenotype"].str.contains(
            args.target_filter, case=False, na=False)
        df = df[mask].reset_index(drop=True)
        print(f"  Filtered to '{args.target_filter}': {len(df):,} rows")
        if len(df) == 0:
            sys.exit("ERROR: no rows remain after filter")

    n_perms = config["permutation"].get("n_permutations", 1000)
    df = prepare_data(df, epsilon=epsilon, bias_threshold=bt,
                      fdr_threshold=args.fdr_alpha,
                      log2_effect_cutoff=l2c)
    print_summary(df, tissue_A, tissue_B, bt, args.target_filter)

    # Output
    fig_dir = output_dir / "figures" / "volcano"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # LaTeX dir
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tag = ""
    if args.target_filter:
        tag = "_" + args.target_filter.lower().replace(" ", "_")
    prefix = f"volcano_{args.comparison}_{args.hops}hop{tag}"

    subtitle = ""
    if args.target_filter:
        subtitle = f"[target '{args.target_filter}', n={len(df):,}]"

    common = dict(tissue_A=tissue_A, tissue_B=tissue_B, hops=args.hops,
                  top_n_labels=args.top_n_labels, bias_threshold=bt,
                  log2_effect_cutoff=l2c,
                  subtitle=subtitle, dpi=args.dpi)

    print("Generating evidence volcano...")
    plot_evidence_volcano(
        df, output_path=str(fig_dir / f"{prefix}_evidence.png"), **common)

    print("Generating significance volcano (p-value)...")
    plot_significance_volcano(
        df, output_path=str(fig_dir / f"{prefix}_significance.png"),
        n_perms=n_perms, fdr_alpha=args.fdr_alpha, **common)

    print("Generating q-value volcano...")
    plot_qvalue_volcano(
        df, output_path=str(fig_dir / f"{prefix}_qvalue.png"),
        fdr_alpha=args.fdr_alpha, **common)

    # LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(
        df, tissue_A, tissue_B, args.hops,
        top_n=args.top_n_table,
        fdr_threshold=args.fdr_alpha,
        rank_by=args.rank_by,
        output_dir=tables_dir,
    )

    print("Done")


if __name__ == "__main__":
    main()