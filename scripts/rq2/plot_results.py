#!/usr/bin/env python3
"""
RQ2 Result Visualizations — Multi-Panel Figures

Produces a set of publication-ready figures that characterize the tissue-
differential results:

  1. P-value distribution (histogram + π₀ estimate)
  2. Coverage decay: 2-hop vs 3-hop scatter
  3. Concordance scatter: diff_coverage 2-hop vs 3-hop
  4. Bar chart: tissue-specific counts by metapath
  5. MA-style plot: mean coverage vs diff_coverage
  6. Stacked significance summary across comparisons / hops

Usage:
    python plot_results.py \
        --comparison subcut_vs_visceral \
        --config config.yaml \
        [--dpi 600]
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

from utils import load_config


# ---------------------------------------------------------------------------
# Global font / style settings — thesis-ready sizes
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size":          13,
    "axes.titlesize":     15,
    "axes.labelsize":     13,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "figure.titlesize":   16,
    "font.family":        "serif",
    "mathtext.fontset":   "dejavuserif",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pretty_tissue(name: str) -> str:
    return name.replace("_", " ").title()


def load_comparison(comp_dir: Path, hops: int) -> pd.DataFrame:
    path = comp_dir / f"comparison_{hops}hop.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


COLORS = {
    "tissue_A": "#E64B35",
    "tissue_B": "#4DBBD5",
    "sig":      "#7B2D8E",
    "neutral":  "#B0B0B0",
    "2hop":     "#2166AC",
    "3hop":     "#B2182B",
}


# ---------------------------------------------------------------------------
# 1. P-value distribution
# ---------------------------------------------------------------------------

def plot_pvalue_distribution(df2, df3, tissue_A, tissue_B,
                              output_path, dpi=600):
    """Histogram of permutation p-values for 2-hop and 3-hop."""
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    panels = []
    labels = []
    if df2 is not None and "perm_pvalue" in df2.columns:
        panels.append(df2["perm_pvalue"].dropna())
        labels.append("2-hop")
    if df3 is not None and "perm_pvalue" in df3.columns:
        panels.append(df3["perm_pvalue"].dropna())
        labels.append("3-hop")

    if not panels:
        print("  No p-value data available — skipping distribution plot")
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5),
                              squeeze=False)

    for i, (pv, label) in enumerate(zip(panels, labels)):
        ax = axes[0, i]
        ax.hist(pv, bins=50, color=COLORS.get(label.replace("-", ""),
                                                "#666"),
                alpha=0.7, edgecolor="white", linewidth=0.3)
        ax.axhline(len(pv) / 50, color="red", linestyle="--", linewidth=0.8,
                    alpha=0.6, label="Uniform expectation")

        # Fraction significant
        n_sig_05 = (pv < 0.05).sum()
        n_sig_01 = (pv < 0.01).sum()
        ax.text(0.95, 0.95,
                f"n = {len(pv):,}\np<0.05: {n_sig_05:,} ({100*n_sig_05/len(pv):.1f}%)"
                f"\np<0.01: {n_sig_01:,} ({100*n_sig_01/len(pv):.1f}%)",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.8))

        ax.set_xlabel("Permutation p-value")
        ax.set_ylabel("Count")
        ax.set_title(f"{tA} vs {tB} — {label}", fontweight="bold")
        ax.legend(framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 2. Coverage decay: 2-hop vs 3-hop
# ---------------------------------------------------------------------------

def plot_coverage_decay(df2, df3, tissue_A, tissue_B,
                         output_path, dpi=600):
    """Scatter of mean coverage per source gene: 2-hop vs 3-hop."""
    if df2 is None or df3 is None:
        print("  Need both 2-hop and 3-hop for coverage decay plot — skipping")
        return

    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for ax, cov_col, label in zip(axes, ["coverage_A", "coverage_B"],
                                    [tA, tB]):
        # Mean coverage per source gene
        mean2 = df2.groupby("source_gene")[cov_col].mean()
        mean3 = df3.groupby("source_gene")[cov_col].mean()
        common = mean2.index.intersection(mean3.index)

        if len(common) == 0:
            ax.text(0.5, 0.5, "No overlapping genes", transform=ax.transAxes,
                    ha="center")
            continue

        x = mean2.loc[common]
        y = mean3.loc[common]

        ax.scatter(x, y, s=5, alpha=0.3, color=COLORS["2hop"], rasterized=True)
        lims = [0, max(x.max(), y.max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.4)

        ax.set_xlabel("Mean coverage (2-hop)")
        ax.set_ylabel("Mean coverage (3-hop)")
        ax.set_title(f"{label} coverage: 2-hop vs 3-hop")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Median ratio
        ratios = y / x.clip(lower=1e-6)
        median_ratio = ratios.median()
        ax.text(0.05, 0.92,
                f"n = {len(common):,}\nmedian decay = {median_ratio:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="0.7", alpha=0.8))

    fig.suptitle(f"{tA} vs {tB} — Coverage Decay", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 3. Concordance: diff_coverage 2-hop vs 3-hop
# ---------------------------------------------------------------------------

def plot_diff_concordance(df2, df3, tissue_A, tissue_B,
                          output_path, dpi=600):
    """Scatter of diff_coverage for shared triples between 2-hop and 3-hop."""
    if df2 is None or df3 is None:
        print("  Need both 2-hop and 3-hop for concordance plot — skipping")
        return

    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    # Key columns
    key_cols = ["source_gene", "target_phenotype"]
    df2_k = df2[key_cols + ["diff_coverage"]].rename(
        columns={"diff_coverage": "diff_2hop"})
    df3_k = df3[key_cols + ["diff_coverage"]].rename(
        columns={"diff_coverage": "diff_3hop"})

    # Aggregate by (source_gene, target_phenotype) — take mean across metapaths
    df2_g = df2_k.groupby(key_cols, as_index=False)["diff_2hop"].mean()
    df3_g = df3_k.groupby(key_cols, as_index=False)["diff_3hop"].mean()

    merged = df2_g.merge(df3_g, on=key_cols)

    if len(merged) == 0:
        print("  No shared gene-phenotype pairs for concordance — skipping")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(merged["diff_2hop"], merged["diff_3hop"],
               s=5, alpha=0.3, color="#555", rasterized=True)

    lims = [min(merged["diff_2hop"].min(), merged["diff_3hop"].min()) - 0.05,
            max(merged["diff_2hop"].max(), merged["diff_3hop"].max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.5, alpha=0.4)
    ax.axhline(0, color="grey", linewidth=0.3, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.3, alpha=0.5)

    # Correlation
    corr = merged["diff_2hop"].corr(merged["diff_3hop"])
    ax.text(0.05, 0.92,
            f"n = {len(merged):,}\nPearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.8))

    ax.set_xlabel("diff_coverage (2-hop)")
    ax.set_ylabel("diff_coverage (3-hop)")
    ax.set_title(f"{tA} vs {tB} — Tissue Bias Concordance (2 vs 3 hop)",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 4. Metapath bar chart
# ---------------------------------------------------------------------------

def plot_metapath_bars(df2, df3, tissue_A, tissue_B,
                       output_path, diff_threshold=0.3, dpi=600):
    """
    Stacked bar chart: count of tissue-specific triples per metapath,
    comparing 2-hop and 3-hop side by side.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    panels = []
    labels = []
    if df2 is not None:
        panels.append(df2)
        labels.append("2-hop")
    if df3 is not None:
        panels.append(df3)
        labels.append("3-hop")

    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.5),
                              squeeze=False)

    for ax, df, label in zip(axes[0], panels, labels):
        diff = df["diff_coverage"].fillna(0)
        df_plot = df.copy()
        df_plot["bias"] = np.where(
            diff > diff_threshold, f"{tA}",
            np.where(diff < -diff_threshold, f"{tB}", "Neutral"))

        # Count per metapath
        counts = df_plot.groupby(["metapath", "bias"]).size().unstack(fill_value=0)
        # Keep top 15 metapaths by total tissue-specific
        cols_to_sum = [c for c in counts.columns if c != "Neutral"]
        if cols_to_sum:
            counts["_total_specific"] = counts[cols_to_sum].sum(axis=1)
        else:
            counts["_total_specific"] = 0
        counts = counts.nlargest(15, "_total_specific").drop(columns=["_total_specific"])

        if counts.empty:
            ax.text(0.5, 0.5, "No tissue-specific results", transform=ax.transAxes,
                    ha="center")
            continue

        colors_map = {tA: COLORS["tissue_A"], tB: COLORS["tissue_B"],
                      "Neutral": COLORS["neutral"]}
        bar_colors = [colors_map.get(c, "#999") for c in counts.columns]

        counts.plot.barh(stacked=True, ax=ax, color=bar_colors, edgecolor="white",
                         linewidth=0.3)
        ax.set_xlabel("Number of triples")
        ax.set_ylabel("")
        ax.set_title(f"{label} — Tissue Specificity by Metapath")
        ax.legend(framealpha=0.8, loc="lower right")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"{tA} vs {tB}", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 5. MA-style plot: mean coverage vs diff_coverage
# ---------------------------------------------------------------------------

def plot_ma(df, tissue_A, tissue_B, hops, output_path,
            fdr_threshold=0.05, dpi=600):
    """
    MA plot: x = (coverage_A + coverage_B) / 2, y = diff_coverage.
    Points coloured by FDR significance.
    """
    if df is None:
        return

    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    cov_A = df["coverage_A"].fillna(0)
    cov_B = df["coverage_B"].fillna(0)
    mean_cov = (cov_A + cov_B) / 2
    diff_cov = df["diff_coverage"].fillna(0)

    has_q = "perm_qvalue" in df.columns and df["perm_qvalue"].notna().any()
    if has_q:
        sig = df["perm_qvalue"].fillna(1.0) < fdr_threshold
    else:
        sig = pd.Series(False, index=df.index)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Non-significant first
    ns = ~sig
    ax.scatter(mean_cov[ns], diff_cov[ns], s=5, alpha=0.3,
               color=COLORS["neutral"], rasterized=True, label="Not significant")
    if sig.any():
        ax.scatter(mean_cov[sig], diff_cov[sig], s=7, alpha=0.6,
                   color=COLORS["sig"], rasterized=True,
                   label=f"FDR q < {fdr_threshold}")

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)

    ax.set_xlabel("Mean coverage  (A + B) / 2")
    ax.set_ylabel("diff_coverage  (A − B)")
    ax.set_title(f"{tA} vs {tB} — {hops}-hop MA Plot", fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 6. Significance summary panel
# ---------------------------------------------------------------------------

def plot_significance_summary(df2, df3, tissue_A, tissue_B,
                               output_path, dpi=600):
    """
    Bar chart summarizing significant triples at various thresholds
    for 2-hop vs 3-hop.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    thresholds = [
        ("|Δcov| > 0.3", lambda d: d["diff_coverage"].abs() > 0.3),
        ("p < 0.05", lambda d: d.get("perm_pvalue", pd.Series(1.0, index=d.index)) < 0.05),
        ("q < 0.05", lambda d: d.get("perm_qvalue", pd.Series(1.0, index=d.index)) < 0.05),
        ("q < 0.01", lambda d: d.get("perm_qvalue", pd.Series(1.0, index=d.index)) < 0.01),
        ("|Δcov|>0.3 & q<0.05", lambda d: (d["diff_coverage"].abs() > 0.3) &
         (d.get("perm_qvalue", pd.Series(1.0, index=d.index)) < 0.05)),
    ]

    counts_2hop = []
    counts_3hop = []
    labels = [t[0] for t in thresholds]

    for _, fn in thresholds:
        if df2 is not None:
            try:
                counts_2hop.append(fn(df2).sum())
            except Exception:
                counts_2hop.append(0)
        else:
            counts_2hop.append(0)
        if df3 is not None:
            try:
                counts_3hop.append(fn(df3).sum())
            except Exception:
                counts_3hop.append(0)
        else:
            counts_3hop.append(0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, counts_2hop, width, label="2-hop",
                    color=COLORS["2hop"], alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, counts_3hop, width, label="3-hop",
                    color=COLORS["3hop"], alpha=0.8, edgecolor="white")

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f"{int(h):,}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of triples")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(framealpha=0.8)
    ax.set_title(f"{tA} vs {tB} — Significance Summary", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 7. Q-value vs diff_coverage heatmap-density (2D histogram)
# ---------------------------------------------------------------------------

def plot_qvalue_vs_diff_density(df, tissue_A, tissue_B, hops,
                                 output_path, dpi=600):
    """2D hexbin: diff_coverage vs -log10(q-value)."""
    if df is None or "perm_qvalue" not in df.columns:
        return

    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    valid = df["perm_qvalue"].notna() & df["diff_coverage"].notna()
    if valid.sum() < 50:
        return

    x = df.loc[valid, "diff_coverage"]
    y = -np.log10(df.loc[valid, "perm_qvalue"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    hb = ax.hexbin(x, y, gridsize=40, cmap="YlOrRd", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")

    ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axhline(-np.log10(0.05), color="red", linewidth=0.8,
               linestyle="--", alpha=0.5)

    ax.set_xlabel("diff_coverage  (A − B)")
    ax.set_ylabel("−log₁₀(q-value)")
    ax.set_title(f"{tA} vs {tB} — {hops}-hop Density", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 8. Enrichment: dot plot of significant topics per tissue
# ---------------------------------------------------------------------------

def _load_enrichment_csvs(output_dir: Path, comparison_name: str,
                           hops: int) -> Dict[str, pd.DataFrame]:
    """Load enrichment CSVs for all tissues and fields."""
    enrichment_dir = output_dir / 'enrichment'
    dfs = {}
    for path in sorted(enrichment_dir.glob(
            f"{comparison_name}_{hops}hop_*_enrichment.csv")):
        # Parse tissue and field from filename
        parts = path.stem.replace(f"{comparison_name}_{hops}hop_", "").replace(
            "_enrichment", "").rsplit("_", 1)
        if len(parts) == 2:
            tissue, field = parts
            key = f"{tissue}_{field}"
            try:
                dfs[key] = pd.read_csv(path)
            except Exception as e:
                print(f"  WARNING: failed to load {path}: {e}")
    return dfs


def plot_enrichment_dotplot(output_dir: Path, comparison_name: str,
                             tissue_A: str, tissue_B: str,
                             hops: int, output_path: str, dpi=600):
    """
    Dot plot of significant enrichments per tissue.
    Each dot = one significant topic-triple test.
    x = log₂ fold change, y = topic label, size = total count, colour = tissue.
    Horizontal guide lines connect dots for the same topic across tissues.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    dfs = _load_enrichment_csvs(output_dir, comparison_name, hops)
    if not dfs:
        print(f"  No enrichment CSVs found for {comparison_name} {hops}-hop — skipping dotplot")
        return

    # Combine all significant results
    sig_rows = []
    for key, df in dfs.items():
        sig = df[df['significant'] == True].copy()
        if sig.empty:
            continue
        tissue = key.rsplit("_", 1)[0]
        sig['tissue_label'] = pretty_tissue(tissue)
        sig['field_label'] = key.rsplit("_", 1)[1].title()
        sig_rows.append(sig)

    if not sig_rows:
        print(f"  No significant enrichments for {hops}-hop — skipping dotplot")
        return

    combined = pd.concat(sig_rows, ignore_index=True)

    # Top N topics by frequency of significance
    topic_counts = combined['topic_label'].value_counts().head(25)
    top_topics = topic_counts.index.tolist()
    plot_df = combined[combined['topic_label'].isin(top_topics)].copy()

    if plot_df.empty:
        return

    ## Tissue labels (no vertical offset — colour distinguishes tissues)
    tissue_labels = sorted(plot_df['tissue_label'].unique())

    fig, ax = plt.subplots(figsize=(9, max(5, len(top_topics) * 0.45)))

    # ---- Horizontal guide lines for each topic row ----
    for y_idx in range(len(top_topics)):
        ax.axhline(y_idx, color="#E0E0E0", linewidth=0.6, zorder=0)

    # ---- Plot dots per tissue ----
    for tissue_label in tissue_labels:
        subset = plot_df[plot_df['tissue_label'] == tissue_label]

        # Aggregate per topic: mean log2FC, best q, sum count
        agg = subset.groupby('topic_label').agg(
            log2fc=('log2_fold_change', 'mean'),
            best_q=('fdr_q', 'min'),
            total_count=('total_count', 'sum'),
            n_sig=('significant', 'sum'),
        ).reindex(top_topics).dropna(subset=['log2fc'])

        if agg.empty:
            continue

        y_positions = [top_topics.index(t) for t in agg.index]

        color = COLORS["tissue_A"] if tissue_label == tA else COLORS["tissue_B"]
        sizes = agg['total_count'].clip(lower=5, upper=200) * 0.6

        ax.scatter(agg['log2fc'], y_positions,
                   s=sizes, c=color, alpha=0.75, edgecolors="white",
                   linewidths=0.4, label=tissue_label, zorder=2)

    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_yticks(range(len(top_topics)))
    ax.set_yticklabels(top_topics, fontsize=10)
    ax.set_xlabel("Mean log₂ fold change")
    ax.set_title(f"Enriched Topics — {tA} vs {tB} ({hops}-hop)",
                 fontweight="bold")

    # De-duplicate legend entries
    handles, labs = ax.get_legend_handles_labels()
    by_label = dict(zip(labs, handles))
    ax.legend(by_label.values(), by_label.keys(), framealpha=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 9. Enrichment: counts bar chart (mechanisms vs pathways, 2-hop vs 3-hop)
# ---------------------------------------------------------------------------

def plot_enrichment_counts(output_dir: Path, comparison_name: str,
                            tissue_A: str, tissue_B: str,
                            output_path: str, dpi=600):
    """
    Grouped bar chart: number of significant enrichments per tissue × field,
    comparing 2-hop vs 3-hop.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    categories = []
    counts_2hop = []
    counts_3hop = []

    for tissue in [tissue_A, tissue_B]:
        t_label = pretty_tissue(tissue)
        for field in ['mechanisms', 'pathways']:
            categories.append(f"{t_label}\n{field.title()}")

            for hops, count_list in [(2, counts_2hop), (3, counts_3hop)]:
                dfs = _load_enrichment_csvs(output_dir, comparison_name, hops)
                key = f"{tissue}_{field}"
                if key in dfs:
                    count_list.append(int(dfs[key]['significant'].sum()))
                else:
                    count_list.append(0)

    if not any(counts_2hop) and not any(counts_3hop):
        print("  No enrichment data for counts chart — skipping")
        return

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width/2, counts_2hop, width, label="2-hop",
           color=COLORS["2hop"], alpha=0.8, edgecolor="white")
    ax.bar(x + width/2, counts_3hop, width, label="3-hop",
           color=COLORS["3hop"], alpha=0.8, edgecolor="white")

    for bars_data, offset in [(counts_2hop, -width/2), (counts_3hop, width/2)]:
        for i, v in enumerate(bars_data):
            if v > 0:
                ax.text(i + offset, v + 0.3, str(v), ha="center",
                        va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Significant enrichments (FDR < 0.05)")
    ax.set_title(f"Topic Enrichment Counts — {tA} vs {tB}", fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 10. Enrichment: tissue comparison heatmap
# ---------------------------------------------------------------------------

def plot_enrichment_tissue_comparison(output_dir: Path, comparison_name: str,
                                       tissue_A: str, tissue_B: str,
                                       hops: int, output_path: str, dpi=600):
    """
    For each topic that is significant in at least one tissue, show a side-by-side
    comparison of log₂FC values as a diverging heatmap-style plot.
    """
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)

    dfs = _load_enrichment_csvs(output_dir, comparison_name, hops)

    for field in ['mechanisms', 'pathways']:
        key_A = f"{tissue_A}_{field}"
        key_B = f"{tissue_B}_{field}"
        df_A = dfs.get(key_A)
        df_B = dfs.get(key_B)

        if df_A is None and df_B is None:
            continue

        # Get significant topics from either
        sig_topics = set()
        for df in [df_A, df_B]:
            if df is not None:
                sig_topics.update(
                    df[df['significant'] == True]['topic_label'].unique())

        if not sig_topics or len(sig_topics) > 40:
            all_sig = []
            for df in [df_A, df_B]:
                if df is not None:
                    all_sig.append(df[df['significant'] == True])
            if all_sig:
                combined_sig = pd.concat(all_sig)
                sig_topics = set(
                    combined_sig.nsmallest(30, 'fdr_q')['topic_label'].unique())

        if not sig_topics:
            continue

        # Compute mean log2FC per topic per tissue
        data_rows = []
        for topic in sorted(sig_topics):
            row = {'topic': topic}
            for label, df in [(tA, df_A), (tB, df_B)]:
                if df is not None:
                    t_df = df[df['topic_label'] == topic]
                    if not t_df.empty:
                        row[label] = t_df['log2_fold_change'].mean()
                        row[f"{label}_sig"] = t_df['significant'].any()
                    else:
                        row[label] = 0
                        row[f"{label}_sig"] = False
                else:
                    row[label] = 0
                    row[f"{label}_sig"] = False
            data_rows.append(row)

        if not data_rows:
            continue

        plot_df = pd.DataFrame(data_rows).set_index('topic')

        fig, axes = plt.subplots(1, 2, figsize=(9, max(3.5, len(plot_df) * 0.35)),
                                  sharey=True)

        for ax, tissue_label, color in zip(axes, [tA, tB],
                                            [COLORS["tissue_A"], COLORS["tissue_B"]]):
            vals = plot_df[tissue_label]
            sig_mask = plot_df[f"{tissue_label}_sig"]

            # Horizontal guide lines
            for y_idx in range(len(vals)):
                ax.axhline(y_idx, color="#E8E8E8", linewidth=0.5, zorder=0)

            bars = ax.barh(range(len(vals)), vals,
                           color=[color if s else "#ddd" for s in sig_mask],
                           alpha=0.8, edgecolor="white", linewidth=0.3,
                           zorder=1)
            ax.axvline(0, color="grey", linewidth=0.5)
            ax.set_xlabel("log₂ fold change")
            ax.set_title(tissue_label, fontweight="bold")
            ax.tick_params(labelsize=9)

        axes[0].set_yticks(range(len(plot_df)))
        axes[0].set_yticklabels(plot_df.index, fontsize=9)
        axes[0].invert_yaxis()

        fig.suptitle(f"{field.title()} Enrichment — {tA} vs {tB} ({hops}-hop)",
                     fontweight="bold", y=1.02)
        fig.tight_layout()

        field_path = output_path.replace(".png", f"_{field}.png")
        _save(fig, field_path, dpi)


# ---------------------------------------------------------------------------
# 11. Intermediate count distribution: 2-hop vs 3-hop
# ---------------------------------------------------------------------------

def _stat_box(ax, lines, x=0.95, y=0.95, ha="right"):
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, fontsize=9,
            va="top", ha=ha, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.85))


def plot_intermediates_dist(df2, df3, tissue_A, tissue_B,
                            output_path, dpi=600):
    """Violin + boxplot of num_intermediates, 2-hop vs 3-hop."""
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    col = "num_intermediates"

    series = {}
    if df2 is not None and col in df2.columns:
        series["2-hop"] = df2[col].dropna()
    if df3 is not None and col in df3.columns:
        series["3-hop"] = df3[col].dropna()

    if not series:
        print("  No num_intermediates data — skipping")
        return

    labels = list(series.keys())
    data = [series[l].values for l in labels]
    colors = [COLORS[l.replace("-", "")] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Violin plot
    parts = ax.violinplot(data, positions=range(len(labels)),
                          showextrema=False, widths=0.7)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.35)
        body.set_edgecolor(colors[i])

    # Boxplot overlay (narrow, no fliers — violin already shows tails)
    bp = ax.boxplot(data, positions=range(len(labels)),
                    widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)
    for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
        box.set_edgecolor("black")
        box.set_linewidth(0.8)
        median.set_color("white")
        median.set_linewidth(1.5)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.8)

    # Log scale to handle the long tail
    upper = max(s.quantile(0.99) for s in series.values())
    ax.set_ylim(-0.5, upper * 1.1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of unique intermediates")
    ax.set_title(f"{tA} vs {tB} — Intermediate Count Distribution",
                 fontweight="bold")

    # Stats box
    lines = []
    for label, s in series.items():
        lines.append(f"{label}:  n={len(s):,}  median={s.median():.0f}  "
                     f"mean={s.mean():.1f}  Q3={s.quantile(0.75):.0f}  "
                     f"max={s.max():.0f}")
    _stat_box(ax, lines, x=0.95, y=0.95)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 12. 3-hop intermediates by position (B vs C)
# ---------------------------------------------------------------------------

def plot_intermediates_bc(df3, tissue_A, tissue_B,
                          output_path, dpi=600):
    """3-hop only: violin + boxplot of intermediates at position B vs C."""
    if df3 is None:
        print("  No 3-hop data — skipping B/C plot")
        return
    for col in ["n_intermediates_B", "n_intermediates_C"]:
        if col not in df3.columns:
            print(f"  Missing {col} — skipping B/C plot")
            return

    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    b_vals = df3["n_intermediates_B"].dropna()
    c_vals = df3["n_intermediates_C"].dropna()

    labels = ["Position B\n(1st intermediate)", "Position C\n(2nd intermediate)"]
    data = [b_vals.values, c_vals.values]
    colors = ["#E78AC3", "#66C2A5"]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Violin plot
    parts = ax.violinplot(data, positions=range(len(labels)),
                          showextrema=False, widths=0.7)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_alpha(0.35)
        body.set_edgecolor(colors[i])

    # Boxplot overlay
    bp = ax.boxplot(data, positions=range(len(labels)),
                    widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)
    for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
        box.set_facecolor(colors[i])
        box.set_alpha(0.7)
        box.set_edgecolor("black")
        box.set_linewidth(0.8)
        median.set_color("white")
        median.set_linewidth(1.5)
    for element in ['whiskers', 'caps']:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(0.8)

    upper = max(b_vals.quantile(0.99), c_vals.quantile(0.99))
    ax.set_ylim(-0.5, upper * 1.1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of unique intermediates at position")
    ax.set_title(f"{tA} vs {tB} — 3-hop Intermediates by Position (B vs C)",
                 fontweight="bold")

    lines = [
        f"Pos B: n={len(b_vals):,}  median={b_vals.median():.0f}  "
        f"mean={b_vals.mean():.1f}  Q3={b_vals.quantile(0.75):.0f}  "
        f"max={b_vals.max():.0f}",
        f"Pos C: n={len(c_vals):,}  median={c_vals.median():.0f}  "
        f"mean={c_vals.mean():.1f}  Q3={c_vals.quantile(0.75):.0f}  "
        f"max={c_vals.max():.0f}",
    ]
    _stat_box(ax, lines)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)

# ---------------------------------------------------------------------------
# 13. Coverage distribution per tissue (single tissue, 2-hop vs 3-hop)
# ---------------------------------------------------------------------------

def plot_coverage_dist_tissue(df2, df3, col, tissue_label,
                              tissue_A, tissue_B,
                              output_path, dpi=600):
    """Single-tissue coverage histogram with 2-hop vs 3-hop overlay."""
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    dfs = {"2-hop": df2, "3-hop": df3}

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 61)

    for label, df in dfs.items():
        if df is None or col not in df.columns:
            continue
        s = df[col].dropna()
        c = COLORS[label.replace("-", "")]
        ax.hist(s, bins=bins, color=c, alpha=0.5, edgecolor="white",
                linewidth=0.3, label=label, density=True)

    lines = []
    for label, df in dfs.items():
        if df is None or col not in df.columns:
            continue
        s = df[col].dropna()
        pct_zero = 100 * (s == 0).mean()
        lines.append(f"{label}: n={len(s):,}  median={s.median():.4f}  "
                     f"mean={s.mean():.4f}  zero={pct_zero:.1f}%")
    _stat_box(ax, lines)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Density")
    ax.set_title(f"{tA} vs {tB} — {tissue_label} Coverage ({col})",
                 fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 14. Differential coverage distribution (2-hop vs 3-hop)
# ---------------------------------------------------------------------------

def plot_diff_coverage_dist(df2, df3, tissue_A, tissue_B,
                            output_path, dpi=600):
    """diff_coverage histogram, 2-hop vs 3-hop."""
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    dfs = {"2-hop": df2, "3-hop": df3}

    fig, ax = plt.subplots(figsize=(7, 5))

    for label, df in dfs.items():
        if df is None or "diff_coverage" not in df.columns:
            continue
        s = df["diff_coverage"].dropna()
        c = COLORS[label.replace("-", "")]
        lo, hi = s.quantile(0.001), s.quantile(0.999)
        bins = np.linspace(lo, hi, 61)
        ax.hist(s, bins=bins, color=c, alpha=0.5, edgecolor="white",
                linewidth=0.3, label=label, density=True)

    ax.axvline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    lines = []
    for label, df in dfs.items():
        if df is None or "diff_coverage" not in df.columns:
            continue
        s = df["diff_coverage"].dropna()
        pct_pos = 100 * (s > 0).mean()
        lines.append(f"{label}: n={len(s):,}  median={s.median():.4f}  "
                     f">{tA}: {pct_pos:.1f}%")
    _stat_box(ax, lines)

    ax.set_xlabel(f"diff_coverage  (← {tB}  |  {tA} →)")
    ax.set_ylabel("Density")
    ax.set_title(f"{tA} vs {tB} — Differential Coverage Distribution",
                 fontweight="bold")
    ax.legend(framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# 15. Coverage ECDF (4 curves: 2 hops × 2 tissues)
# ---------------------------------------------------------------------------

def plot_coverage_ecdf(df2, df3, tissue_A, tissue_B,
                       output_path, dpi=600):
    """ECDF of coverage — 4 curves: 2 hops × 2 tissues."""
    tA, tB = pretty_tissue(tissue_A), pretty_tissue(tissue_B)
    dfs = {"2-hop": df2, "3-hop": df3}

    fig, ax = plt.subplots(figsize=(7, 5))

    styles = {
        ("2-hop", "coverage_A"): (COLORS["2hop"], "-",  f"2-hop {tA}"),
        ("2-hop", "coverage_B"): (COLORS["2hop"], "--", f"2-hop {tB}"),
        ("3-hop", "coverage_A"): (COLORS["3hop"], "-",  f"3-hop {tA}"),
        ("3-hop", "coverage_B"): (COLORS["3hop"], "--", f"3-hop {tB}"),
    }

    for (hop_label, col), (color, ls, legend) in styles.items():
        if hop_label not in dfs or dfs[hop_label] is None:
            continue
        if col not in dfs[hop_label].columns:
            continue
        s = dfs[hop_label][col].dropna().sort_values()
        if len(s) == 0:
            continue
        y = np.arange(1, len(s) + 1) / len(s)
        ax.step(s, y, color=color, linestyle=ls, linewidth=1.4,
                label=legend, alpha=0.8)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title(f"{tA} vs {tB} — Coverage ECDF", fontweight="bold")
    ax.legend(framealpha=0.8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path, dpi)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save(fig, path, dpi):
    """Save figure as PNG only."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 result visualizations (2-hop vs 3-hop comparison)")
    parser.add_argument("--comparison", required=True,
                        choices=["subcut_vs_visceral", "white_vs_brown"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])
    comp_dir = output_dir / "comparisons" / args.comparison

    comp_cfg = next(
        (c for c in config["comparisons"] if c["name"] == args.comparison), None)
    if comp_cfg is None:
        sys.exit(f"ERROR: comparison '{args.comparison}' not found")

    tissue_A = comp_cfg["tissue_A"]
    tissue_B = comp_cfg["tissue_B"]

    fig_dir = output_dir / "figures" / "res"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"results_{args.comparison}"

    # Load both hops
    df2 = load_comparison(comp_dir, 2)
    df3 = load_comparison(comp_dir, 3)
    if df2 is not None:
        print(f"Loaded 2-hop: {len(df2):,} rows")
    if df3 is not None:
        print(f"Loaded 3-hop: {len(df3):,} rows")

    if df2 is None and df3 is None:
        sys.exit("ERROR: no comparison data found")

    d = args.dpi
    ta, tb = tissue_A, tissue_B

    # 1. P-value distribution
    print("\n[1/16] P-value distribution...")
    plot_pvalue_distribution(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_pvalue_dist.png"), d)

    # 2. Coverage decay
    print("[2/16] Coverage decay...")
    plot_coverage_decay(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_coverage_decay.png"), d)

    # 3. Diff concordance 2 vs 3 hop
    print("[3/16] Diff concordance...")
    plot_diff_concordance(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_diff_concordance.png"), d)

    # 4. Metapath bars
    print("[4/16] Metapath bars...")
    plot_metapath_bars(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_metapath_bars.png"), dpi=d)

    # 5. MA plots (one per hop)
    print("[5/16] MA plots...")
    for hops, df in [(2, df2), (3, df3)]:
        if df is not None:
            plot_ma(df, ta, tb, hops,
                    str(fig_dir / f"{prefix}_ma_{hops}hop.png"), dpi=d)

    # 6. Significance summary
    print("[6/16] Significance summary...")
    plot_significance_summary(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_sig_summary.png"), d)

    # 7. Density plots
    print("[7/16] Q-value density...")
    for hops, df in [(2, df2), (3, df3)]:
        if df is not None and "perm_qvalue" in df.columns:
            plot_qvalue_vs_diff_density(
                df, ta, tb, hops,
                str(fig_dir / f"{prefix}_density_{hops}hop.png"), d)

    # 8. Enrichment dot plot (per hop)
    print("[8/16] Enrichment dot plots...")
    for hops in [2, 3]:
        plot_enrichment_dotplot(
            output_dir, args.comparison, ta, tb, hops,
            str(fig_dir / f"{prefix}_enrich_dotplot_{hops}hop.png"), d)

    # 9. Enrichment counts (2-hop vs 3-hop)
    print("[9/16] Enrichment counts...")
    plot_enrichment_counts(
        output_dir, args.comparison, ta, tb,
        str(fig_dir / f"{prefix}_enrich_counts.png"), d)

    # 10. Enrichment tissue comparison (per hop)
    print("[10/16] Enrichment tissue comparison...")
    for hops in [2, 3]:
        plot_enrichment_tissue_comparison(
            output_dir, args.comparison, ta, tb, hops,
            str(fig_dir / f"{prefix}_enrich_tissue_{hops}hop.png"), d)

    # 11. Intermediate count distribution
    print("[11/16] Intermediates distribution...")
    plot_intermediates_dist(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_intermediates_dist.png"), d)

    # 12. 3-hop intermediates by position (B vs C)
    print("[12/16] Intermediates B vs C...")
    plot_intermediates_bc(
        df3, ta, tb,
        str(fig_dir / f"{prefix}_intermediates_bc.png"), d)

    # 13. Coverage distribution per tissue
    print("[13/16] Coverage distribution (tissue A)...")
    plot_coverage_dist_tissue(
        df2, df3, "coverage_A", pretty_tissue(ta), ta, tb,
        str(fig_dir / f"{prefix}_coverage_dist_A.png"), d)
    print("[14/16] Coverage distribution (tissue B)...")
    plot_coverage_dist_tissue(
        df2, df3, "coverage_B", pretty_tissue(tb), ta, tb,
        str(fig_dir / f"{prefix}_coverage_dist_B.png"), d)

    # 15. Differential coverage distribution
    print("[15/16] Diff coverage distribution...")
    plot_diff_coverage_dist(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_diff_coverage_dist.png"), d)

    # 16. Coverage ECDF
    print("[16/16] Coverage ECDF...")
    plot_coverage_ecdf(
        df2, df3, ta, tb,
        str(fig_dir / f"{prefix}_coverage_ecdf.png"), d)

    print("\n✓ All figures generated")


if __name__ == "__main__":
    main()