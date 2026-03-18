#!/usr/bin/env python3
"""
LDA Topic Visualizations

Generates publication-quality plots from LDA model artifacts.
Each topic gets its own individual plot file.

Plots generated:
  1. Word cloud — one PNG per topic
  2. Top terms bar chart — one PNG per topic
  3. Topic size distribution — single bar chart
  4. Topic similarity heatmap — Jensen-Shannon divergence matrix
  5. Coherence & perplexity vs k — from explore results (if available)
  6. Vocabulary frequency distribution — top terms by doc frequency

Usage:
    python plot_lda.py --lda-dir lda_output/ --fields mechanisms pathways
    python plot_lda.py --lda-dir lda_output/ --fields mechanisms --output-dir plots/
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# COLORBLIND-FRIENDLY PALETTE
# =============================================================================
# Combines Wong (2011) Nature Methods palette with Tol's qualitative scheme.
# Tested against deuteranopia, protanopia, and tritanopia.

CB_PALETTE = [
    '#0077BB',  # blue
    '#EE7733',  # orange
    '#009988',  # teal
    '#CC3311',  # red
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    '#BBBBBB',  # grey
    '#332288',  # indigo
    '#DDCC77',  # sand/yellow
    '#AA3377',  # wine
    '#44BB99',  # mint
    '#882255',  # dark magenta
    '#999933',  # olive
    '#661100',  # dark red
    '#6699CC',  # steel blue
    '#117733',  # green
]

# Sequential colormap for heatmaps (colorblind-safe: white → cyan → blue → indigo)
CB_SEQ_CMAP = LinearSegmentedColormap.from_list(
    'cb_seq', ['#FFFFFF', '#33BBEE', '#0077BB', '#332288'])


def get_color(i):
    return CB_PALETTE[i % len(CB_PALETTE)]


def setup_style():
    """Clean scientific figure style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 15,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': 200,
    })


# =============================================================================
# LOADING
# =============================================================================

def load_model_artifacts(field_dir: Path) -> dict:
    """Load all LDA artifacts for a field."""
    artifacts = {}

    topics_path = field_dir / 'topics.json'
    if topics_path.exists():
        with open(topics_path) as f:
            artifacts['topics_meta'] = json.load(f)

    model_path = field_dir / 'lda_model.pkl'
    if model_path.exists():
        with open(model_path, 'rb') as f:
            artifacts['model_data'] = pickle.load(f)

    vocab_path = field_dir / 'vocabulary.json'
    if vocab_path.exists():
        with open(vocab_path) as f:
            artifacts['vocabulary'] = json.load(f)

    explore_path = field_dir / 'explore_summary.json'
    if explore_path.exists():
        with open(explore_path) as f:
            artifacts['explore'] = json.load(f)

    return artifacts


# =============================================================================
# PLOT 1: INDIVIDUAL WORD CLOUDS
# =============================================================================

def plot_wordclouds(topics_meta: dict, field: str, output_dir: Path):
    """One word cloud PNG per topic."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print(f"  [!] wordcloud not installed (pip install wordcloud). Skipping.")
        return

    topics = topics_meta['topics']

    for topic in topics:
        tid = topic['topic_id']
        color = get_color(tid)

        # Build frequency dict
        freq = {}
        for t in topic['top_terms']:
            term = t['term']
            if len(term) > 45:
                term = term[:42] + '...'
            freq[term] = t['weight']

        # Monochrome color function derived from the topic color
        r0, g0, b0 = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        def make_color_func(r0, g0, b0):
            def color_func(*args, **kwargs):
                f = np.random.uniform(0.55, 1.0)
                return (int(r0 * f), int(g0 * f), int(b0 * f))
            return color_func

        wc = WordCloud(
            width=800, height=500,
            background_color='white',
            color_func=make_color_func(r0, g0, b0),
            max_words=15,
            prefer_horizontal=0.8,
            min_font_size=10,
            max_font_size=90,
            relative_scaling=0.5,
        )
        wc.generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.imshow(wc, interpolation='bilinear')
        label = topic['label'][:60]
        n_edges = topic.get('n_dominant_edges', '?')
        ax.set_title(f'Topic {tid}: {label}\n({n_edges:,} edges)',
                     fontsize=13, fontweight='bold', pad=12)
        ax.axis('off')

        plt.tight_layout()
        path = output_dir / f'wordcloud_{field}_topic{tid}.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(topics)} word clouds → wordcloud_{field}_topic*.png")


# =============================================================================
# PLOT 2: INDIVIDUAL TOP TERMS BAR CHARTS
# =============================================================================

def plot_top_terms(topics_meta: dict, field: str, output_dir: Path,
                   top_n: int = 12):
    """One horizontal bar chart per topic showing term weights."""
    topics = topics_meta['topics']

    for topic in topics:
        tid = topic['topic_id']
        color = get_color(tid)
        terms = topic['top_terms'][:top_n]

        names = [t['term'][:50] for t in reversed(terms)]
        weights = [t['weight'] for t in reversed(terms)]

        fig, ax = plt.subplots(figsize=(8, max(3.5, len(terms) * 0.4)))

        bars = ax.barh(range(len(names)), weights, color=color,
                       edgecolor='white', linewidth=0.4, height=0.7, alpha=0.85)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Weight (proportion of topic)', fontsize=10)

        label = topic['label'][:55]
        n_edges = topic.get('n_dominant_edges', '?')
        ax.set_title(f'Topic {tid}: {label}  ({n_edges:,} edges)',
                     fontsize=12, fontweight='bold')

        # Weight labels on bars
        for bar, w in zip(bars, weights):
            ax.text(bar.get_width() + max(weights) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{w:.3f}', va='center', fontsize=8, color='#444')

        ax.set_xlim(0, max(weights) * 1.18)

        plt.tight_layout()
        path = output_dir / f'terms_{field}_topic{tid}.png'
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(topics)} term charts → terms_{field}_topic*.png")


# =============================================================================
# PLOT 3: TOPIC SIZE DISTRIBUTION
# =============================================================================

def plot_topic_sizes(topics_meta: dict, field: str, output_dir: Path):
    """Bar chart of edge count per topic."""
    topics = topics_meta['topics']

    tids = [t['topic_id'] for t in topics]
    sizes = [t.get('n_dominant_edges', 0) for t in topics]
    labels = [f"T{t['topic_id']}: {t['label'][:30]}" for t in topics]
    colors = [get_color(i) for i in range(len(topics))]

    # Sort descending
    order = np.argsort(sizes)[::-1]
    tids = [tids[i] for i in order]
    sizes = [sizes[i] for i in order]
    labels = [labels[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(topics) * 0.5)))

    bars = ax.barh(range(len(topics)), sizes, color=colors,
                   edgecolor='white', linewidth=0.5, height=0.7)
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Edges (dominant topic)', fontsize=11)
    ax.set_title(f'{field.capitalize()} — Topic Size Distribution',
                 fontsize=13, fontweight='bold')

    for bar, s in zip(bars, sizes):
        ax.text(bar.get_width() + max(sizes) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{s:,}', va='center', fontsize=8, color='#444')

    ax.set_xlim(0, max(sizes) * 1.12)

    total = sum(sizes)
    ax.text(0.98, 0.02, f'Total: {total:,} edges',
            transform=ax.transAxes, ha='right', fontsize=9, color='#666')

    plt.tight_layout()
    path = output_dir / f'topic_sizes_{field}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 4: TOPIC SIMILARITY HEATMAP
# =============================================================================

def plot_topic_similarity(model_data: dict, topics_meta: dict,
                          field: str, output_dir: Path):
    """Jensen-Shannon divergence heatmap between topic word distributions."""
    from scipy.spatial.distance import jensenshannon

    lda = model_data['model']
    topic_word = lda.components_
    topic_dists = topic_word / topic_word.sum(axis=1, keepdims=True)

    n_topics = topic_dists.shape[0]
    js_matrix = np.zeros((n_topics, n_topics))
    for i in range(n_topics):
        for j in range(n_topics):
            js_matrix[i, j] = jensenshannon(topic_dists[i], topic_dists[j])

    sim_matrix = 1 - js_matrix

    topics = topics_meta['topics']
    labels = [f"T{t['topic_id']}: {t['label'][:22]}" for t in topics]

    fig, ax = plt.subplots(figsize=(max(7, n_topics * 0.75),
                                    max(6, n_topics * 0.65)))

    im = ax.imshow(sim_matrix, cmap=CB_SEQ_CMAP, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(n_topics))
    ax.set_yticks(range(n_topics))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(n_topics):
        for j in range(n_topics):
            val = sim_matrix[i, j]
            color = 'white' if val > 0.55 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label='Similarity (1 − JS divergence)', shrink=0.8)
    ax.set_title(f'{field.capitalize()} — Topic Similarity',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = output_dir / f'topic_similarity_{field}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 5: COHERENCE & PERPLEXITY VS K
# =============================================================================

def plot_explore_metrics(explore: dict, field: str, output_dir: Path):
    """Separate coherence and perplexity plots from explore sweep."""
    results = explore.get('results', [])
    if not results:
        return

    ks = [r['k'] for r in results]
    coherences = [r['coherence_umass'] for r in results]
    perplexities = [r['perplexity'] for r in results]

    # --- Coherence ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, coherences, 'o-', color=CB_PALETTE[0], linewidth=2,
            markersize=9, markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor=CB_PALETTE[0])
    best_idx = coherences.index(max(coherences))
    ax.plot(ks[best_idx], coherences[best_idx], 's', color=CB_PALETTE[3],
            markersize=13, zorder=5, label=f'Best: k={ks[best_idx]}')
    ax.fill_between(ks, [min(coherences)] * len(ks), coherences,
                    alpha=0.07, color=CB_PALETTE[0])
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('UMass Coherence (higher = better)')
    ax.set_title(f'{field.capitalize()} — Coherence vs k', fontweight='bold')
    ax.legend(frameon=False, fontsize=10)
    ax.set_xticks(ks)
    plt.tight_layout()
    path = output_dir / f'coherence_vs_k_{field}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Perplexity ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, perplexities, 'o-', color=CB_PALETTE[1], linewidth=2,
            markersize=9, markerfacecolor='white', markeredgewidth=2.5,
            markeredgecolor=CB_PALETTE[1])
    ax.fill_between(ks, perplexities, [max(perplexities)] * len(ks),
                    alpha=0.07, color=CB_PALETTE[1])
    ax.set_xlabel('Number of Topics (k)')
    ax.set_ylabel('Perplexity (lower = better)')
    ax.set_title(f'{field.capitalize()} — Perplexity vs k', fontweight='bold')
    ax.set_xticks(ks)
    plt.tight_layout()
    path = output_dir / f'perplexity_vs_k_{field}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# PLOT 6: VOCABULARY FREQUENCY DISTRIBUTION
# =============================================================================

def plot_vocabulary_dist(vocabulary: list, field: str, output_dir: Path,
                         top_n: int = 40):
    """Top terms by document frequency."""
    if not vocabulary:
        return

    terms = vocabulary[:top_n]
    names = [t['term'][:50] for t in reversed(terms)]
    freqs = [t['doc_freq'] for t in reversed(terms)]

    median_f = np.median(freqs)
    colors = [CB_PALETTE[0] if f > median_f else CB_PALETTE[4] for f in freqs]

    fig, ax = plt.subplots(figsize=(9, max(5, len(terms) * 0.32)))
    ax.barh(range(len(names)), freqs, color=colors,
            edgecolor='white', linewidth=0.3, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Document Frequency (# edges)', fontsize=11)
    ax.set_title(f'{field.capitalize()} — Top {top_n} Terms by Frequency',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = output_dir / f'vocabulary_dist_{field}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate LDA topic visualizations (individual plots)')
    parser.add_argument('--lda-dir', type=str, required=True,
                        help='LDA output directory (contains mechanisms/ '
                             'and/or pathways/ subdirs)')
    parser.add_argument('--fields', type=str, nargs='+',
                        default=['mechanisms', 'pathways'],
                        help='Which fields to plot')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: <lda-dir>/plots)')
    args = parser.parse_args()

    lda_dir = Path(args.lda_dir)
    output_dir = Path(args.output_dir) if args.output_dir else lda_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"LDA Topic Visualization")
    print(f"  LDA dir: {lda_dir}")
    print(f"  Output:  {output_dir}")

    for field in args.fields:
        field_dir = lda_dir / field
        if not field_dir.exists():
            print(f"\n  WARNING: {field_dir} not found, skipping {field}")
            continue

        print(f"\n{'='*60}")
        print(f"  {field.upper()}")
        print(f"{'='*60}")

        artifacts = load_model_artifacts(field_dir)

        if not artifacts.get('topics_meta'):
            print(f"  No topics.json found in {field_dir}")
            continue

        topics_meta = artifacts['topics_meta']
        model_data = artifacts.get('model_data')
        vocabulary = artifacts.get('vocabulary')
        explore = artifacts.get('explore')

        n_topics = len(topics_meta['topics'])
        print(f"  Topics: {n_topics}")
        print(f"  Model loaded: {'yes' if model_data else 'no'}")
        print(f"  Explore data: {'yes' if explore else 'no'}")

        # Individual word clouds (1 per topic)
        plot_wordclouds(topics_meta, field, output_dir)

        # Individual term bar charts (1 per topic)
        plot_top_terms(topics_meta, field, output_dir)

        # Single overview charts
        plot_topic_sizes(topics_meta, field, output_dir)

        if model_data:
            plot_topic_similarity(model_data, topics_meta, field, output_dir)

        if explore:
            plot_explore_metrics(explore, field, output_dir)

        if vocabulary:
            plot_vocabulary_dist(vocabulary, field, output_dir)

    print(f"\n✓ All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()