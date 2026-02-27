#!/usr/bin/env python3
"""
Comprehensive EDA and Characterization of Augmented Knowledge Graph

Analyzes a knowledge graph with LLM-extracted context annotations.
Focus areas: general statistics, evidence quality, context field coverage,
adipose tissue biology, and inflammation.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm
from contextlib import redirect_stdout
import csv
import json
import re
from datetime import datetime

# Add project to path for knowledge_graph module
sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DEFAULT_INPUT_GRAPH = "/home/projects2/ContextAwareKGReasoning/data/graphs/augmented_graph_normalized.pkl"
DEFAULT_OUTPUT_DIR = "/home/projects2/ContextAwareKGReasoning/results/normalized_kg_eda"

# Plot settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

FIGURE_SIZE = (8, 6)
FIGURE_SIZE_WIDE = (10, 6)
FIGURE_SIZE_TALL = (8, 10)

# Okabe-Ito colorblind-friendly palette
COLORS = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#000000',  # Black
    '#999999',  # Gray
]

# Context fields to analyze
CONTEXT_FIELDS = [
    'Tissue', 'Detailed_Tissue', 'Organ', 'System',
    'Organism', 'Detailed_Organism',
    'Cell_Type', 'Detailed_Cell_Type', 'Cell_Line',
    'Target_Disease_Role', 'Mechanisms', 'Gene_Regulation',
    'Pathways', 'Study_Type', 'Model_System', 'Assays_Techniques',
    'Statistical_Methodology', 'Physical_Phenotype', 'Molecular_Phenotype',
    'Disease_Progression_Stage', 'Temporal_Response', 'Population_Demographics'
]

# Inflammation-related keywords
INFLAMMATION_KEYWORDS = [
    'nf-κb', 'nf-kb', 'nfkb', 'inflamm', 'cytokine', 'tnf', 'il-1', 'il-6',
    'il1', 'il6', 'macrophage', 'm1', 'm2', 'polarization', 'tlr', 'nlrp3',
    'interleukin', 'chemokine', 'adipokine', 'ccl', 'cxcl'
]

TOP_N = 20
TOP_N_BY_FIELD = {
    'Tissue': 30,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_graph(path: str):
    """Load knowledge graph from pickle file using KnowledgeGraph.import_graph()."""
    print(f"Loading graph from: {path}")
    kg = KnowledgeGraph.import_graph(path)
    print(f"  Type: {type(kg).__name__}")
    print(f"  Nodes: {kg.number_of_nodes():,}")
    print(f"  Edges: {kg.number_of_edges():,}")
    return kg


def get_node_name(node) -> str:
    """Extract node name from node object."""
    if hasattr(node, 'name'):
        return node.name
    elif isinstance(node, tuple) and len(node) >= 1:
        return str(node[0])
    return str(node)


def get_node_type(node) -> str:
    """Extract node type from node object."""
    if hasattr(node, 'type'):
        return node.type
    elif isinstance(node, tuple) and len(node) >= 2:
        return str(node[1])
    return 'Unknown'


def is_specified(value: Any) -> bool:
    """Check if a context field value is meaningfully specified."""
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    if isinstance(value, str):
        val_lower = value.lower().strip()
        return val_lower not in ['', 'not specified', 'not specified.', 'na', 'n/a', 'none', 'nan', '[]', '{}', '()']
    return True


def contains_any(text: str, keywords: List[str]) -> bool:
    """Check if text contains any of the keywords (case-insensitive)."""
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def get_context_value(data: dict, field: str) -> Optional[str]:
    """Safely get a context field value."""
    ctx = data.get('context', {})
    if ctx is None:
        return None
    return ctx.get(field)


def save_plot(fig, path: Path, name: str):
    """Save figure and close."""
    filepath = path / f"{name}.png"
    fig.savefig(filepath)
    plt.close(fig)
    print(f"  Saved: {filepath.name}")


def explode_items_from_series(s: pd.Series, delimiters: Optional[List[str]] = None) -> pd.Series:
    """
    Explode a Series of possibly-multi-valued strings into per-item values.

    - `delimiters` is a list of delimiter strings (e.g. [';'] or [',']).
    - If None, defaults to [';'].
    - Filters out non-specified values using `is_specified()`.
    Returns a 1D Series of cleaned items (one item per row).
    """
    if delimiters is None:
        delimiters = [';']

    if s is None or s.empty:
        return pd.Series(dtype=str)

    s = s[s.apply(is_specified)]
    if s.empty:
        return pd.Series(dtype=str)

    # Build a regex pattern that splits on any of the provided delimiters
    # Escape delimiters for regex
    esc = [re.escape(d) for d in delimiters]
    pattern = '|'.join(esc)

    exploded = (
        s.astype(str)
         .str.split(pattern, expand=False)
         .explode()
         .astype(str)
         .str.strip()
    )

    exploded = exploded[exploded.apply(is_specified)]
    exploded = exploded.str.replace(r"\s+", " ", regex=True)
    return exploded


def plot_top_counts(counts: pd.Series, output_dir: Path, filename: str, title: str, top_n: int = TOP_N):
    """Plot a horizontal bar chart of top counts and save it."""
    if counts is None or counts.empty:
        return

    top_counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, len(top_counts) * 0.35)))
    bars = ax.barh(range(len(top_counts)), top_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(top_counts))])

    ax.set_yticks(range(len(top_counts)))
    labels = [(str(v)[:60] + '...') if len(str(v)) > 60 else str(v) for v in top_counts.index[::-1]]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count')
    ax.set_title(title)

    annotate_barh(ax, bars, top_counts.values)

    plt.tight_layout()
    save_plot(fig, output_dir, filename)

def _srgb_to_linear(u: float) -> float:
    # WCAG conversion
    return u / 12.92 if u <= 0.04045 else ((u + 0.055) / 1.055) ** 2.4


def _relative_luminance(rgb) -> float:
    r, g, b = rgb
    r_lin, g_lin, b_lin = map(_srgb_to_linear, (r, g, b))
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def pick_text_color_for_bg(bg_color, light="#FFFFFF", dark="#000000", threshold=0.5) -> str:
    """
    Return black/white text for best readability on a given background color.

    bg_color can be:
      - hex string like "#RRGGBB"
      - any matplotlib color spec
      - an (r,g,b) or (r,g,b,a) tuple in 0..1
    threshold: larger -> more likely to choose dark text.
    """
    import matplotlib.colors as mcolors

    rgba = mcolors.to_rgba(bg_color)
    r, g, b, a = rgba

    # If the patch is transparent, assume it's on a white background.
    # Effective color = alpha*color + (1-alpha)*white
    r = a * r + (1 - a) * 1.0
    g = a * g + (1 - a) * 1.0
    b = a * b + (1 - a) * 1.0

    L = _relative_luminance((r, g, b))
    return dark if L > threshold else light


def annotate_barh(ax, bars, values, pad_frac=0.01, inside_frac=0.9, fmt=None, fontsize=12):
    """
    Annotate horizontal bars robustly.
    - Expands xlim to make room
    - Puts label inside the bar if it would run off the edge
    - Auto-picks black/white text for readability based on bar color
    """
    if values is None or len(values) == 0:
        return

    vmax = float(np.max(values))
    if vmax <= 0:
        return

    if fmt is None:
        fmt = lambda w: f"{int(w):,}"

    cur_xlim = ax.get_xlim()
    ax.set_xlim(0, max(cur_xlim[1], vmax * 1.15))

    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2

        # Choose text color based on the bar fill color
        txt_color = pick_text_color_for_bg(bar.get_facecolor())

        if w >= inside_frac * vmax:
            ax.text(
                w - vmax * pad_frac, y, fmt(w),
                va="center", ha="right", fontsize=fontsize, clip_on=False,
                color=txt_color
            )
        else:
            ax.text(
                w + vmax * pad_frac, y, fmt(w),
                va="center", ha="left", fontsize=fontsize, clip_on=False,
                color="#000000"  # outside labels: assume white background
            )

# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_edge_dataframe(kg) -> pd.DataFrame:
    """Extract all edge data into a DataFrame with flattened context."""
    print("Extracting edge data to DataFrame...")
    
    rows = []
    for u, v, key, data in tqdm(kg.edges(keys=True, data=True), 
                                 desc="Processing edges", 
                                 total=kg.number_of_edges()):
        row = {
            'source_node': get_node_name(u),
            'target_node': get_node_name(v),
            'source_type': get_node_type(u),
            'target_type': get_node_type(v),
            'edge_key': key,
        }
        
        # Add standard edge attributes
        for attr in ['type', 'source', 'correlation_type', 'direction', 'score',
                     'probability', 'novelty', 'document_id', 'species_id', 
                     'species', 'journal_score', 'year', 'source_subset']:
            row[attr] = data.get(attr)
        
        # Add context fields with prefix (normalize lists/dicts to strings)
        def _normalize_ctx_value(val):
            """Normalize context values for storage:
            - Return None for empty containers
            - Join lists/tuples/sets of primitives into a '; '-separated string
            - Fall back to str() for complex objects
            """
            if val is None:
                return None
            # Lists
            if isinstance(val, list):
                if len(val) == 0:
                    return None
                try:
                    return '; '.join(map(str, val))
                except Exception:
                    return str(val)
            # Tuples
            if isinstance(val, tuple):
                if len(val) == 0:
                    return None
                try:
                    return '; '.join(map(str, val))
                except Exception:
                    return str(val)
            # Sets
            if isinstance(val, set):
                if len(val) == 0:
                    return None
                try:
                    return '; '.join(map(str, sorted(val)))
                except Exception:
                    return str(val)
            # Dicts
            if isinstance(val, dict):
                if len(val) == 0:
                    return None
                try:
                    return json.dumps(val, sort_keys=True, ensure_ascii=False)
                except Exception:
                    return str(val)
            # Ensure primitive values are returned as strings for downstream
            # string-based operations (contains, lower, etc.).
            try:
                return str(val)
            except Exception:
                return val

        ctx = data.get('context', {}) or {}
        for field in CONTEXT_FIELDS:
            row[f'ctx_{field}'] = _normalize_ctx_value(ctx.get(field))
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"  DataFrame shape: {df.shape}")
    return df


def extract_node_dataframe(kg) -> pd.DataFrame:
    """Extract node data into a DataFrame."""
    print("Extracting node data to DataFrame...")
    
    rows = []
    for node in tqdm(kg.nodes(), desc="Processing nodes", total=kg.number_of_nodes()):
        node_data = kg.nodes[node]
        row = {
            'name': get_node_name(node),
            'type': get_node_type(node),
            'degree': kg.degree(node),
        }
        # Try to get in/out degree if available
        try:
            row['in_degree'] = kg.in_degree(node)
            row['out_degree'] = kg.out_degree(node)
        except (AttributeError, TypeError):
            row['in_degree'] = None
            row['out_degree'] = None
            
        # Add any other node attributes from kg.nodes[node]
        for k, v in node_data.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"  DataFrame shape: {df.shape}")
    return df


# =============================================================================
# PART 1: BASIC GRAPH STATISTICS
# =============================================================================

def analyze_basic_stats(kg, node_df: pd.DataFrame, edge_df: pd.DataFrame, 
                        output_dir: Path, report_lines: List[str]):
    """Compute and plot basic graph statistics."""
    
    print("\n" + "="*70)
    print("PART 1: BASIC GRAPH STATISTICS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 1: BASIC GRAPH STATISTICS")
    report_lines.append("="*70 + "\n")
    
    # Basic counts
    n_nodes = kg.number_of_nodes()
    n_edges = kg.number_of_edges()
    
    # Density: handle directed vs undirected correctly (undirected uses 2E / N(N-1))
    if n_nodes > 1:
        try:
            is_directed = kg.is_directed()
        except AttributeError:
            is_directed = isinstance(kg, nx.DiGraph) or isinstance(kg, nx.MultiDiGraph)

        if is_directed:
            density = n_edges / (n_nodes * (n_nodes - 1))
        else:
            density = 2 * n_edges / (n_nodes * (n_nodes - 1))
    else:
        density = 0
    
    # Check graph properties safely
    # is_directed already computed above for density, but ensure variable exists
    try:
        is_directed
    except NameError:
        try:
            is_directed = kg.is_directed()
        except AttributeError:
            is_directed = isinstance(kg, nx.DiGraph) or isinstance(kg, nx.MultiDiGraph)
    
    try:
        is_multigraph = kg.is_multigraph()
    except AttributeError:
        is_multigraph = isinstance(kg, nx.MultiGraph) or isinstance(kg, nx.MultiDiGraph)
    
    # Connected components (treat as undirected for this)
    try:
        undirected = kg.to_undirected() if is_directed else kg
    except AttributeError:
        undirected = nx.Graph(kg)  # Convert to simple undirected graph
    
    try:
        components = list(nx.connected_components(undirected))
        n_components = len(components)
        largest_cc_size = max(len(c) for c in components) if components else 0
    except Exception as e:
        print(f"  Warning: Could not compute connected components: {e}")
        n_components = -1
        largest_cc_size = -1
    
    # Degree stats
    degrees = [d for n, d in kg.degree()]
    degree_stats = {
        'mean': np.mean(degrees),
        'median': np.median(degrees),
        'std': np.std(degrees),
        'min': np.min(degrees),
        'max': np.max(degrees),
    }
    
    # Report
    report_lines.append(f"Total nodes: {n_nodes:,}")
    report_lines.append(f"Total edges: {n_edges:,}")
    report_lines.append(f"Graph density: {density:.6f}")
    report_lines.append(f"Is directed: {is_directed}")
    report_lines.append(f"Is multigraph: {is_multigraph}")
    report_lines.append(f"Connected components: {n_components:,}")
    if largest_cc_size >= 0:
        report_lines.append(f"Largest component size: {largest_cc_size:,} ({100*largest_cc_size/n_nodes:.1f}%)")
    report_lines.append(f"\nDegree statistics:")
    report_lines.append(f"  Mean: {degree_stats['mean']:.2f}")
    report_lines.append(f"  Median: {degree_stats['median']:.1f}")
    report_lines.append(f"  Std: {degree_stats['std']:.2f}")
    report_lines.append(f"  Min: {degree_stats['min']}")
    report_lines.append(f"  Max: {degree_stats['max']}")
    
    print(f"  Nodes: {n_nodes:,}, Edges: {n_edges:,}")
    print(f"  Components: {n_components:,}, Largest: {largest_cc_size:,}")
    
    # --- Plot: Degree Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(degrees, bins=50, color=COLORS[0], edgecolor='white', alpha=0.8)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Degree Distribution')
    ax1.axvline(degree_stats['mean'], color=COLORS[4], linestyle='--', 
                linewidth=2, label=f"Mean: {degree_stats['mean']:.1f}")
    ax1.axvline(degree_stats['median'], color=COLORS[5], linestyle=':', 
                linewidth=2, label=f"Median: {degree_stats['median']:.1f}")
    ax1.legend()
    
    # Log-log plot
    ax2 = axes[1]
    degree_counts = Counter(degrees)
    x = np.array(sorted(degree_counts.keys()))
    y = np.array([degree_counts[k] for k in x])

    # Filter out zero degrees before log-scaling (isolated nodes)
    mask = x > 0
    if mask.sum() > 0:
        ax2.scatter(x[mask], y[mask], color=COLORS[1], alpha=0.7, s=30)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
    else:
        ax2.scatter(x, y, color=COLORS[1], alpha=0.7, s=30)

    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Degree Distribution (Log-Log)')
    
    plt.tight_layout()
    save_plot(fig, output_dir, 'degree_distribution')
    
    # --- Plot: Node Type Distribution ---
    node_type_counts = node_df['type'].value_counts()
    report_lines.append(f"\nNode type distribution:")
    for ntype, count in node_type_counts.items():
        report_lines.append(f"  {ntype}: {count:,} ({100*count/n_nodes:.1f}%)")
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    bars = ax.barh(node_type_counts.index[::-1], node_type_counts.values[::-1], 
                   color=[COLORS[i % len(COLORS)] for i in range(len(node_type_counts))])
    ax.set_xlabel('Count')
    ax.set_title('Node Type Distribution')
    # Add count labels
    annotate_barh(ax, bars, node_type_counts.values)
    plt.tight_layout()
    save_plot(fig, output_dir, 'node_type_distribution')
    
    # --- Plot: Edge Type Distribution ---
    edge_type_counts = edge_df['type'].value_counts()
    report_lines.append(f"\nEdge type distribution:")
    for etype, count in edge_type_counts.items():
        report_lines.append(f"  {etype}: {count:,} ({100*count/n_edges:.1f}%)")

    # --- Direction Distribution ---
    if 'direction' in edge_df.columns:
        dir_counts = edge_df['direction'].fillna('0').astype(str).value_counts()
        n_directed = dir_counts.get('1', 0)
        n_undirected = dir_counts.get('0', 0)
        
        report_lines.append(f"\nEdge Direction Distribution:")
        report_lines.append(f"  Directed (direction=1): {n_directed:,} ({100*n_directed/n_edges:.1f}%)")
        report_lines.append(f"  Undirected (direction=0): {n_undirected:,} ({100*n_undirected/n_edges:.1f}%)")
        
        # Cross-tab with edge type
        if 'type' in edge_df.columns:
            dir_by_type = edge_df.groupby(['type', 'direction']).size().unstack(fill_value=0)
            dir_by_type.to_csv(output_dir / 'direction_by_edge_type.csv')
    
    # Show top 20 if many types
    plot_counts = edge_type_counts.head(TOP_N)
    
    fig, ax = plt.subplots(figsize=(8, max(6, len(plot_counts)*0.35)))
    bars = ax.barh(plot_counts.index[::-1], plot_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(plot_counts))])
    ax.set_xlabel('Count')
    ax.set_title('Edge Type Distribution')
    annotate_barh(ax, bars, plot_counts.values)
    plt.tight_layout()
    save_plot(fig, output_dir, 'edge_type_distribution')
    
    # --- Entity type pairs ---
    pair_counts = edge_df.groupby(['source_type', 'target_type']).size().reset_index(name='count')
    pair_counts = pair_counts.sort_values('count', ascending=False)
    
    report_lines.append(f"\nEntity type pairs (top 20):")
    for _, row in pair_counts.head(20).iterrows():
        report_lines.append(f"  {row['source_type']} -> {row['target_type']}: {row['count']:,}")
    
    return degree_stats


# =============================================================================
# PART 2: EVIDENCE QUALITY
# =============================================================================

def analyze_evidence_quality(edge_df: pd.DataFrame, output_dir: Path, 
                             report_lines: List[str]):
    """Analyze evidence quality metrics."""
    
    print("\n" + "="*70)
    print("PART 2: EVIDENCE QUALITY")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 2: EVIDENCE QUALITY")
    report_lines.append("="*70 + "\n")
    
    n_edges = len(edge_df)
    
    # --- Journal Score Distribution ---
    if 'journal_score' in edge_df.columns:
        js_counts = edge_df['journal_score'].value_counts().sort_index()
        
        report_lines.append("Journal Score Distribution:")
        for score, count in js_counts.items():
            report_lines.append(f"  Score {score}: {count:,} ({100*count/n_edges:.1f}%)")
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        x_labels = [str(s) for s in js_counts.index]
        bars = ax.bar(range(len(js_counts)), js_counts.values, 
                      color=[COLORS[i % len(COLORS)] for i in range(len(js_counts))])
        ax.set_xticks(range(len(js_counts)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Journal Score')
        ax.set_ylabel('Edge Count')
        ax.set_title('Journal Score Distribution')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=13)
        plt.tight_layout()
        save_plot(fig, output_dir, 'journal_score_distribution')
    
    # --- Publication Year Distribution ---
    if 'year' in edge_df.columns:
        years = pd.to_numeric(edge_df['year'], errors='coerce').dropna()
        current_year = datetime.now().year
        years = years[(years >= 1980) & (years <= current_year)]
        
        if len(years) > 0:
            year_counts = years.value_counts().sort_index()
            
            report_lines.append(f"\nPublication Year Distribution:")
            report_lines.append(f"  Range: {int(years.min())} - {int(years.max())}")
            report_lines.append(f"  Median: {int(years.median())}")
            report_lines.append(f"  Edges with valid year: {len(years):,} ({100*len(years)/n_edges:.1f}%)")
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE_WIDE)
            ax.bar(year_counts.index, year_counts.values, color=COLORS[1], alpha=0.8)
            ax.set_xlabel('Publication Year')
            ax.set_ylabel('Edge Count')
            ax.set_title('Publication Year Distribution')
            
            # Add trend line
            z = np.polyfit(year_counts.index, year_counts.values, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(year_counts.index.min(), year_counts.index.max(), 100)
            ax.plot(x_smooth, p(x_smooth), color=COLORS[5], linestyle='--', 
                    linewidth=2, label='Trend')
            ax.legend()
            plt.tight_layout()
            save_plot(fig, output_dir, 'publication_year_distribution')
    
    # --- Novelty Distribution ---
    if 'novelty' in edge_df.columns:
        novelty_counts = edge_df['novelty'].value_counts()
        
        report_lines.append(f"\nNovelty Distribution:")
        for val, count in novelty_counts.items():
            report_lines.append(f"  {val}: {count:,} ({100*count/n_edges:.1f}%)")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = [str(v) for v in novelty_counts.index]
        bars = ax.bar(labels, novelty_counts.values, 
                      color=[COLORS[0] if v else COLORS[1] for v in novelty_counts.index])
        ax.set_xlabel('Novelty')
        ax.set_ylabel('Edge Count')
        ax.set_title('Novelty Distribution')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=13)
        plt.tight_layout()
        save_plot(fig, output_dir, 'novelty_distribution')
    
    # --- Probability Distribution ---
    if 'probability' in edge_df.columns:
        probs = edge_df['probability'].dropna()
        
        if len(probs) > 0:
            report_lines.append(f"\nProbability Distribution:")
            report_lines.append(f"  Mean: {probs.mean():.3f}")
            report_lines.append(f"  Median: {probs.median():.3f}")
            report_lines.append(f"  Std: {probs.std():.3f}")
            report_lines.append(f"  Range: [{probs.min():.3f}, {probs.max():.3f}]")
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            ax.hist(probs, bins=30, color=COLORS[2], edgecolor='white', alpha=0.8)
            ax.set_xlabel('Probability')
            ax.set_ylabel('Edge Count')
            ax.set_title('Probability Distribution')
            ax.axvline(probs.mean(), color=COLORS[4], linestyle='--', 
                       linewidth=2, label=f"Mean: {probs.mean():.3f}")
            ax.legend()
            plt.tight_layout()
            save_plot(fig, output_dir, 'probability_distribution')
    
    # --- Score Distribution ---
    if 'score' in edge_df.columns:
        scores = edge_df['score'].dropna()
        
        if len(scores) > 0:
            report_lines.append(f"\nScore Distribution:")
            report_lines.append(f"  Mean: {scores.mean():.1f}")
            report_lines.append(f"  Median: {scores.median():.1f}")
            report_lines.append(f"  Std: {scores.std():.1f}")
            report_lines.append(f"  Range: [{scores.min():.1f}, {scores.max():.1f}]")
            
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            # Use log scale if scores vary widely
            if scores.max() / (scores.min() + 1) > 100:
                ax.hist(scores, bins=50, color=COLORS[3], edgecolor='white', alpha=0.8)
                ax.set_yscale('log')
            else:
                ax.hist(scores, bins=30, color=COLORS[3], edgecolor='white', alpha=0.8)
            ax.set_xlabel('Score')
            ax.set_ylabel('Edge Count')
            ax.set_title('Score Distribution')
            plt.tight_layout()
            save_plot(fig, output_dir, 'score_distribution')

def investigate_year_spike(edge_df, output_dir: Path, report_lines: list[str], year: int = None, top_k: int = 20):
    years = pd.to_numeric(edge_df["year"], errors="coerce").dropna().astype(int)
    tmp = edge_df.loc[years.index].copy()
    tmp["year_int"] = years.values

    year_counts = tmp["year_int"].value_counts()
    if year is None:
        year = int(year_counts.idxmax())

    sub = tmp[tmp["year_int"] == year].copy()
    n = len(sub)
    report_lines.append(f"\nYear spike investigation: {year} has {n:,} edges.")

    if "document_id" in sub.columns:
        pmid_counts = sub["document_id"].fillna("NA").value_counts()
        top = pmid_counts.head(top_k)

        # how concentrated is it?
        share_top1 = (top.iloc[0] / n) if len(top) else 0
        share_top10 = (top.head(10).sum() / n) if len(top) else 0
        report_lines.append(f"  Unique document_id values: {pmid_counts.size:,}")
        report_lines.append(f"  Top PMID share: {share_top1*100:.1f}%")
        report_lines.append(f"  Top 10 PMIDs share: {share_top10*100:.1f}%")

        out = output_dir / f"year_{year}_top_pmids.csv"
        top.rename_axis("document_id").reset_index(name="edge_count").to_csv(out, index=False)
        report_lines.append(f"  Saved: {out.name}")

    # also check if it’s dominated by one source/subset/type
    for col in ["source_subset", "source", "type", "correlation_type"]:
        if col in sub.columns:
            vc = sub[col].fillna("NA").value_counts().head(10)
            report_lines.append(f"\n  Top {col} in {year}:")
            for k, v in vc.items():
                report_lines.append(f"    {k}: {v:,}")

# =============================================================================
# PART 3: CONTEXT FIELD OVERVIEW
# =============================================================================

def analyze_context_coverage(edge_df: pd.DataFrame, output_dir: Path,
                             report_lines: List[str]) -> pd.DataFrame:
    """Analyze coverage of all context fields."""
    
    print("\n" + "="*70)
    print("PART 3: CONTEXT FIELD COVERAGE")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 3: CONTEXT FIELD COVERAGE")
    report_lines.append("="*70 + "\n")
    
    n_edges = len(edge_df)
    coverage_data = []
    
    def _is_unspecified_string(x):
        if not isinstance(x, str):
            return False
        v = x.lower().strip()
        return v in ['', 'not specified', 'not specified.', 'na', 'n/a', 'none', '[]', '{}', '()']

    for field in CONTEXT_FIELDS:
        col = f'ctx_{field}'
        if col not in edge_df.columns:
            continue
        
        values = edge_df[col]
        
        # Count specified / unspecified / null
        n_specified = int(values.apply(is_specified).sum())
        n_unspecified = int(values.apply(_is_unspecified_string).sum())
        n_null = int(values.isna().sum())
        n_unique = int(values[values.apply(is_specified)].nunique())

        coverage_data.append({
            'field': field,
            'n_specified': n_specified,
            'pct_specified': 100 * n_specified / n_edges,
            'n_unspecified': n_unspecified,
            'pct_unspecified': 100 * n_unspecified / n_edges,
            'n_null': n_null,
            'pct_null': 100 * n_null / n_edges,
            'n_unique': n_unique,
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df = coverage_df.sort_values('pct_specified', ascending=False)
    
    # Report
    report_lines.append("Context Field Coverage Summary:")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Field':<30} {'Specified':>10} {'Unspec.':>9} {'Null':>10} {'Unique':>8}")
    report_lines.append("-" * 70)

    for _, row in coverage_df.iterrows():
        report_lines.append(
            f"{row['field']:<30} {row['pct_specified']:>9.1f}% {row['pct_unspecified']:>9.1f}% "
            f"{row['pct_null']:>9.1f}% {row['n_unique']:>8}"
        )
    
    # Save CSV
    coverage_df.to_csv(output_dir / 'context_field_coverage.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"  Saved: context_field_coverage.csv")
    
    # Plot coverage overview
    fig, ax = plt.subplots(figsize=(10, max(6, len(coverage_df)*0.35)))
    
    y_pos = range(len(coverage_df))
    bars = ax.barh(y_pos, coverage_df['pct_specified'].values,
                   color=[COLORS[i % len(COLORS)] for i in range(len(coverage_df))])
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coverage_df['field'].values)
    ax.set_xlabel('Coverage (%)')
    ax.set_title('Context Field Coverage (% edges with specified value)')
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    
    # Add percentage labels
    annotate_barh(ax, bars, coverage_df['pct_specified'].values, fmt=lambda w: f"{w:.1f}%")
    
    plt.tight_layout()
    save_plot(fig, output_dir, 'context_coverage_overview')
    
    return coverage_df


# =============================================================================
# PART 4: INDIVIDUAL CONTEXT FIELD DISTRIBUTIONS
# =============================================================================

def analyze_context_distributions(edge_df: pd.DataFrame, coverage_df: pd.DataFrame,
                                  output_dir: Path, report_lines: List[str]):
    """Analyze distribution of values for each context field."""
    
    print("\n" + "="*70)
    print("PART 4: CONTEXT FIELD DISTRIBUTIONS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 4: CONTEXT FIELD DISTRIBUTIONS")
    report_lines.append("="*70 + "\n")
    
    # Only plot fields with >10% coverage
    fields_to_plot = coverage_df[coverage_df['pct_specified'] >= 10]['field'].tolist()
    
    for field in fields_to_plot:
        col = f'ctx_{field}'
        if col not in edge_df.columns:
            continue
        
        # Get values, filter out not specified
        values = edge_df[col][edge_df[col].apply(is_specified)]
        if values.empty:
            continue

        # ---- SPECIAL CASE: Tissue gets two outputs (Top 20 and Top 30) ----
        if field == "Tissue":
            for top_n in (20, 30):
                value_counts = values.value_counts().head(top_n)
                if value_counts.empty:
                    continue

                report_lines.append(f"\n{field} - Top {len(value_counts)} values (Top {top_n}):")
                for val, count in value_counts.items():
                    report_lines.append(f"  {val}: {count:,}")

                fig, ax = plt.subplots(figsize=(8, max(5, len(value_counts)*0.35)))
                bars = ax.barh(
                    range(len(value_counts)),
                    value_counts.values[::-1],
                    color=[COLORS[i % len(COLORS)] for i in range(len(value_counts))]
                )

                ax.set_yticks(range(len(value_counts)))
                labels = [str(v)[:50] + '...' if len(str(v)) > 50 else str(v)
                        for v in value_counts.index[::-1]]
                ax.set_yticklabels(labels)
                ax.set_xlabel('Count')
                ax.set_title(f'{field} Distribution (Top {top_n})')

                annotate_barh(ax, bars, value_counts.values)
                plt.tight_layout()
                save_plot(fig, output_dir, f'context_{field}_distribution_top{top_n}')

            continue   

        # ---- GENERAL CASE ----
        top_n = TOP_N_BY_FIELD.get(field, TOP_N_DEFAULT)
        value_counts = values.value_counts().head(top_n)
        
        if value_counts.empty:
            continue
        
        report_lines.append(f"\n{field} - Top {min(top_n, len(value_counts))} values:")
        for val, count in value_counts.items():
            report_lines.append(f"  {val}: {count:,}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, max(5, len(value_counts)*0.35)))
        
        bars = ax.barh(range(len(value_counts)), value_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(value_counts))])
        
        ax.set_yticks(range(len(value_counts)))
        # Truncate long labels
        labels = [str(v)[:50] + '...' if len(str(v)) > 50 else str(v) 
                  for v in value_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title(f'{field} Distribution (Top {top_n})')
        
        annotate_barh(ax, bars, value_counts.values)
        
        plt.tight_layout()
        save_plot(fig, output_dir, f'context_{field}_distribution')


def analyze_multivalued_fields(
    edge_df: pd.DataFrame,
    output_dir: Path,
    report_lines: List[str],
    fields: Optional[List[str]] = None,
):
    """Explode and count per-item frequencies for multi-valued context fields."""
    if fields is None:
        fields = ['Pathways', 'Mechanisms', 'Assays_Techniques', 'Gene_Regulation']

    report_lines.append("\nMulti-valued field item counts (per-item, exploded):")

    for field in fields:
        col = f'ctx_{field}'
        if col not in edge_df.columns:
            continue

        s = edge_df[col]
        if s is None or s.empty:
            continue

        # Choose delimiters heuristically per field
        if field in ('Assays_Techniques',):
            # Prefer comma+space to avoid splitting tokens that use commas within phrases
            delimiters = [', ']
        else:
            delimiters = [';']

        exploded = explode_items_from_series(s, delimiters=delimiters)
        if exploded.empty:
            continue

        counts = exploded.value_counts()

        report_lines.append(f"\n{field} - Top {min(TOP_N, len(counts))} items:")
        for val, count in counts.head(TOP_N).items():
            short_val = val[:120] + '...' if len(val) > 120 else val
            report_lines.append(f"  {short_val}: {count:,}")

        # Save CSV
        out_csv = output_dir / f"{field.lower()}_item_counts.csv"
        counts_df = counts.rename_axis(field).reset_index(name="count")
        counts_df.to_csv(out_csv, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        print(f"  Saved: {out_csv.name}")

        # Plot
        top_counts = counts.head(TOP_N)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top_counts)*0.35)))
        bars = ax.barh(range(len(top_counts)), top_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(top_counts))])
        ax.set_yticks(range(len(top_counts)))
        labels = [str(v)[:60] + '...' if len(str(v)) > 60 else str(v) for v in top_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title(f'{field} - Item Counts (Top {len(top_counts)})')
        annotate_barh(ax, bars, top_counts.values)
        plt.tight_layout()
        save_plot(fig, output_dir, f'{field.lower()}_item_counts')



# =============================================================================
# PART 5: CROSS-FIELD CONSISTENCY
# =============================================================================

def analyze_cross_field_relationships(edge_df: pd.DataFrame, output_dir: Path,
                                      report_lines: List[str]):
    """Analyze relationships between related context fields."""
    
    print("\n" + "="*70)
    print("PART 5: CROSS-FIELD RELATIONSHIPS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 5: CROSS-FIELD RELATIONSHIPS")
    report_lines.append("="*70 + "\n")
    
    # --- Tissue vs Organ ---
    report_lines.append("Tissue vs Organ Consistency:")
    report_lines.append("-" * 50)
    
    tissue_col = 'ctx_Tissue'
    organ_col = 'ctx_Organ'
    
    if tissue_col in edge_df.columns and organ_col in edge_df.columns:
        # Get unique Tissue values
        tissue_values = edge_df[tissue_col][edge_df[tissue_col].apply(is_specified)].unique()
        
        cross_data = []
        for tissue in tissue_values[:20]:  # Top 20 tissues
            mask = edge_df[tissue_col] == tissue
            organs = edge_df.loc[mask, organ_col].value_counts().head(5)
            
            if len(organs) > 0:
                report_lines.append(f"\n  Tissue: {tissue}")
                for organ, count in organs.items():
                    report_lines.append(f"    -> Organ: {organ} ({count:,})")
                    cross_data.append({'Tissue': tissue, 'Organ': organ, 'count': count})
        
        if cross_data:
            cross_df = pd.DataFrame(cross_data)
            cross_df.to_csv(output_dir / 'cross_field_tissue_organ.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: cross_field_tissue_organ.csv")
    
    # --- Tissue vs Detailed_Tissue ---
    report_lines.append("\n\nTissue vs Detailed_Tissue Breakdown:")
    report_lines.append("-" * 50)
    
    detailed_col = 'ctx_Detailed_Tissue'
    
    if tissue_col in edge_df.columns and detailed_col in edge_df.columns:
        # Focus on top tissues
        top_tissues = edge_df[tissue_col][edge_df[tissue_col].apply(is_specified)].value_counts().head(10).index
        
        cross_data = []
        for tissue in top_tissues:
            mask = edge_df[tissue_col] == tissue
            detailed = edge_df.loc[mask, detailed_col]
            detailed = detailed[detailed.apply(is_specified)]
            detailed_counts = detailed.value_counts().head(10)
            
            if len(detailed_counts) > 0:
                report_lines.append(f"\n  Tissue: {tissue}")
                for det, count in detailed_counts.items():
                    report_lines.append(f"    -> {det}: {count:,}")
                    cross_data.append({'Tissue': tissue, 'Detailed_Tissue': det, 'count': count})
        
        if cross_data:
            cross_df = pd.DataFrame(cross_data)
            cross_df.to_csv(output_dir / 'cross_field_tissue_detailed.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: cross_field_tissue_detailed.csv")
    
    # --- Cell_Type vs Detailed_Cell_Type ---
    report_lines.append("\n\nCell_Type vs Detailed_Cell_Type Breakdown:")
    report_lines.append("-" * 50)
    
    cell_col = 'ctx_Cell_Type'
    detailed_cell_col = 'ctx_Detailed_Cell_Type'
    
    if cell_col in edge_df.columns and detailed_cell_col in edge_df.columns:
        top_cells = edge_df[cell_col][edge_df[cell_col].apply(is_specified)].value_counts().head(10).index
        
        cross_data = []
        for cell in top_cells:
            mask = edge_df[cell_col] == cell
            detailed = edge_df.loc[mask, detailed_cell_col]
            detailed = detailed[detailed.apply(is_specified)]
            detailed_counts = detailed.value_counts().head(10)
            
            if len(detailed_counts) > 0:
                report_lines.append(f"\n  Cell_Type: {cell}")
                for det, count in detailed_counts.items():
                    report_lines.append(f"    -> {det}: {count:,}")
                    cross_data.append({'Cell_Type': cell, 'Detailed_Cell_Type': det, 'count': count})
        
        if cross_data:
            cross_df = pd.DataFrame(cross_data)
            cross_df.to_csv(output_dir / 'cross_field_celltype_detailed.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: cross_field_celltype_detailed.csv")
    
    # --- Organism vs Detailed_Organism ---
    report_lines.append("\n\nOrganism vs Detailed_Organism Breakdown:")
    report_lines.append("-" * 50)
    
    org_col = 'ctx_Organism'
    detailed_org_col = 'ctx_Detailed_Organism'
    
    if org_col in edge_df.columns and detailed_org_col in edge_df.columns:
        top_orgs = edge_df[org_col][edge_df[org_col].apply(is_specified)].value_counts().head(10).index
        
        cross_data = []
        for org in top_orgs:
            mask = edge_df[org_col] == org
            detailed = edge_df.loc[mask, detailed_org_col]
            detailed = detailed[detailed.apply(is_specified)]
            detailed_counts = detailed.value_counts().head(10)
            
            if len(detailed_counts) > 0:
                report_lines.append(f"\n  Organism: {org}")
                for det, count in detailed_counts.items():
                    report_lines.append(f"    -> {det}: {count:,}")
                    cross_data.append({'Organism': org, 'Detailed_Organism': det, 'count': count})
        
        if cross_data:
            cross_df = pd.DataFrame(cross_data)
            cross_df.to_csv(output_dir / 'cross_field_organism_detailed.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: cross_field_organism_detailed.csv")


def analyze_species_context_mismatch(edge_df: pd.DataFrame, output_dir: Path, report_lines: List[str]):
    """
    Flag edges where 'species'/'species_id' disagree with context 'Organism'.
    Uses a simple heuristic mapping (rat/mouse/human) to detect mismatches.
    Saves a focused CSV of mismatches when found.
    """
    if 'species' not in edge_df.columns or 'ctx_Organism' not in edge_df.columns:
        return

    def norm(s):
        return str(s).lower().strip() if pd.notna(s) else ""

    species = edge_df['species'].apply(norm)
    org = edge_df['ctx_Organism'].apply(norm)

    def coarse_label(x):
        if 'rat' in x or 'rattus norvegicus' in x:
            return 'rat'
        if 'mouse' in x or 'mice' in x or 'mus musculus' in x:
            return 'mouse'
        if 'human' in x or 'homo sapiens' in x:
            return 'human'
        if x == '' or 'not specified' in x:
            return 'unspecified'
        return 'other'

    sp_lab = species.apply(coarse_label)
    org_lab = org.apply(coarse_label)

    mask = (sp_lab != 'unspecified') & (org_lab != 'unspecified') & (sp_lab != org_lab)
    mismatch = edge_df[mask].copy()

    report_lines.append("\nSpecies vs Context Organism mismatches:")
    report_lines.append(f"  Mismatched edges: {len(mismatch):,} / {len(edge_df):,} ({100*len(mismatch)/len(edge_df):.2f}%)")

    if len(mismatch) > 0:
        # save a focused table
        cols = ['document_id', 'species_id', 'species', 'ctx_Organism', 'ctx_Detailed_Organism', 'ctx_Tissue', 'ctx_Detailed_Tissue']
        cols = [c for c in cols if c in mismatch.columns]
        mismatch[cols].to_csv(output_dir / "species_context_mismatches.csv", index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        report_lines.append("  Saved: species_context_mismatches.csv")


# =============================================================================
# PART 6: ADIPOSE-FOCUSED ANALYSIS
# =============================================================================

def analyze_adipose(edge_df: pd.DataFrame, kg, output_dir: Path,
                    report_lines: List[str]) -> pd.DataFrame:
    """Deep dive into adipose tissue-related edges."""
    
    print("\n" + "="*70)
    print("PART 6: ADIPOSE TISSUE ANALYSIS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 6: ADIPOSE TISSUE ANALYSIS")
    report_lines.append("="*70 + "\n")
    
    # Vectorized adipose filter (faster than row-wise apply)
    adipose_mask = (
        edge_df.get('ctx_Tissue', pd.Series(dtype=str)).fillna('').str.contains('adipos', case=False, na=False) |
        edge_df.get('ctx_Organ', pd.Series(dtype=str)).fillna('').str.contains('adipos', case=False, na=False) |
        edge_df.get('ctx_Detailed_Tissue', pd.Series(dtype=str)).fillna('').str.contains('adipos', case=False, na=False)
    )
    adipose_df = edge_df[adipose_mask].copy()
    n_adipose = len(adipose_df)
    n_total = len(edge_df)
    
    report_lines.append(f"Adipose-related edges: {n_adipose:,} / {n_total:,} ({100*n_adipose/n_total:.1f}%)")
    
    # Filter to directed edges for causal analysis
    if 'direction' in adipose_df.columns:
        adipose_directed = adipose_df[adipose_df['direction'].astype(str) == '1'].copy()
        n_adipose_directed = len(adipose_directed)
        report_lines.append(f"  Directed adipose edges: {n_adipose_directed:,} / {n_adipose:,} ({100*n_adipose_directed/n_adipose:.1f}%)")
        
        if n_adipose_directed == 0:
            report_lines.append("  WARNING: No directed adipose edges found. Using all adipose edges.")
        else:
            adipose_df = adipose_directed
            n_adipose = n_adipose_directed
    else:
        report_lines.append("  WARNING: No 'direction' column found. Using all adipose edges.")

    if n_adipose == 0:
        report_lines.append("No adipose-related edges found.")
        print("  No adipose-related edges found.")
        return pd.DataFrame()
    
    print(f"  Found {n_adipose:,} adipose-related edges")
    
    # --- Detailed Tissue breakdown ---
    report_lines.append("\nDetailed_Tissue within adipose edges:")
    detailed_tissue = adipose_df['ctx_Detailed_Tissue'][adipose_df['ctx_Detailed_Tissue'].apply(is_specified)]
    # dt_counts = detailed_tissue.value_counts().head(TOP_N_DEFAULT)
    # for val, count in dt_counts.items():
    #     report_lines.append(f"  {val}: {count:,}")

    dt_counts = detailed_tissue.value_counts()  # ALL values
    
    # Report ALL to text
    for val, count in dt_counts.items():
        report_lines.append(f"  {val}: {count:,}")

    # Save ALL to CSV
    dt_counts.to_csv(output_dir / 'adipose_detailed_tissue_full.csv', header=['count'])
    report_lines.append(f"  (Full list: {len(dt_counts)} unique values saved to adipose_detailed_tissue_full.csv)")

    dt_plot = dt_counts.head(TOP_N_DEFAULT)
    
    if len(dt_plot) > 0:
        fig, ax = plt.subplots(figsize=(8, max(5, len(dt_plot)*0.35)))
        bars = ax.barh(range(len(dt_plot)), dt_plot.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(dt_plot))])
        ax.set_yticks(range(len(dt_plot)))
        labels = [str(v)[:45] + '...' if len(str(v)) > 45 else str(v) for v in dt_plot.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title('Adipose Edges: Detailed Tissue Types')
        annotate_barh(ax, bars, dt_plot.values)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_detailed_tissue')
    
    # --- Cell Types in adipose ---
    report_lines.append("\nCell_Type within adipose edges:")
    cell_types = adipose_df['ctx_Cell_Type'][adipose_df['ctx_Cell_Type'].apply(is_specified)]
    ct_counts = cell_types.value_counts().head(TOP_N_DEFAULT)
    for val, count in ct_counts.items():
        report_lines.append(f"  {val}: {count:,}")
    
    if len(ct_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, max(5, len(ct_counts)*0.35)))
        bars = ax.barh(range(len(ct_counts)), ct_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(ct_counts))])
        ax.set_yticks(range(len(ct_counts)))
        labels = [str(v)[:45] + '...' if len(str(v)) > 45 else str(v) for v in ct_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title('Adipose Edges: Cell Types')
        annotate_barh(ax, bars, ct_counts.values)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_cell_types')
    
    # --- Mechanisms in adipose (EXPLODED) ---
    report_lines.append("\nMechanisms within adipose edges (exploded per-item):")
    mech_items = explode_items_from_series(adipose_df['ctx_Mechanisms'])
    mech_counts = mech_items.value_counts()

    for val, count in mech_counts.head(TOP_N_DEFAULT).items():
        short_val = val[:120] + '...' if len(val) > 120 else val
        report_lines.append(f"  {short_val}: {count:,}")

    if not mech_counts.empty:
        # Save exploded CSV
        mech_counts.rename_axis('Mechanism').reset_index(name='count').to_csv(
            output_dir / 'adipose_mechanisms_exploded_counts.csv',
            index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        print(f"  Saved: adipose_mechanisms_exploded_counts.csv")

        plot_top_counts(
            mech_counts,
            output_dir,
            filename='adipose_mechanisms_exploded',
            title='Adipose Edges: Mechanisms (Exploded per-item)'
        )

    # --- Pathways in adipose (EXPLODED) ---
    report_lines.append("\nPathways within adipose edges (exploded per-item):")
    path_items = explode_items_from_series(adipose_df['ctx_Pathways'])
    pw_counts = path_items.value_counts()

    for val, count in pw_counts.head(TOP_N_DEFAULT).items():
        short_val = val[:120] + '...' if len(val) > 120 else val
        report_lines.append(f"  {short_val}: {count:,}")

    if not pw_counts.empty:
        pw_counts.rename_axis('Pathway').reset_index(name='count').to_csv(
            output_dir / 'adipose_pathways_exploded_counts.csv',
            index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        print(f"  Saved: adipose_pathways_exploded_counts.csv")

        plot_top_counts(
            pw_counts,
            output_dir,
            filename='adipose_pathways_exploded',
            title='Adipose Edges: Pathways (Exploded per-item)'
        )
    
    # --- Model Systems in adipose ---
    report_lines.append("\nModel_System within adipose edges:")
    models = adipose_df['ctx_Model_System'][adipose_df['ctx_Model_System'].apply(is_specified)]
    model_counts = models.value_counts().head(TOP_N_DEFAULT)
    for val, count in model_counts.items():
        report_lines.append(f"  {val}: {count:,}")
    
    if len(model_counts) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(model_counts)*0.4)))
        bars = ax.barh(range(len(model_counts)), model_counts.values[::-1],
                   color=[COLORS[i % len(COLORS)] for i in range(len(model_counts))])
        ax.set_yticks(range(len(model_counts)))
        labels = [str(v)[:55] + '...' if len(str(v)) > 55 else str(v) for v in model_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title('Adipose Edges: Model Systems')
        annotate_barh(ax, bars, model_counts.values)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_model_systems')
    
    # --- Organisms in adipose ---
    report_lines.append("\nOrganism within adipose edges:")
    organisms = adipose_df['ctx_Organism'][adipose_df['ctx_Organism'].apply(is_specified)]
    org_counts = organisms.value_counts().head(TOP_N_DEFAULT)
    for val, count in org_counts.items():
        report_lines.append(f"  {val}: {count:,}")

    
    if len(org_counts) > 0:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        bars = ax.bar(range(len(org_counts)), org_counts.values,
                      color=[COLORS[i % len(COLORS)] for i in range(len(org_counts))])
        ax.set_xticks(range(len(org_counts)))
        ax.set_xticklabels(org_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Adipose Edges: Organisms')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=13)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_organisms')
    
    # --- Top genes/entities in adipose subgraph ---
    report_lines.append("\nTop entities (by degree) in adipose edges:")
    
    # Count node occurrences
    node_counts = Counter()
    for _, row in adipose_df.iterrows():
        node_counts[row['source_node']] += 1
        node_counts[row['target_node']] += 1
    
    top_nodes = node_counts.most_common(TOP_N_DEFAULT)
    for node, count in top_nodes:
        report_lines.append(f"  {node}: {count:,}")
    
    if len(top_nodes) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_nodes)*0.35)))
        nodes = [n for n, c in top_nodes]
        counts = [c for n, c in top_nodes]
        bars = ax.barh(range(len(nodes)), counts[::-1],
                       color=[COLORS[i % len(COLORS)] for i in range(len(nodes))])
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes[::-1])
        ax.set_xlabel('Edge Count')
        ax.set_title('Adipose Edges: Top Entities by Degree')
        annotate_barh(ax, bars, counts)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_top_entities')
    
    # Save adipose edge data
    adipose_df.to_csv(output_dir / 'adipose_edges.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"  Saved: adipose_edges.csv")
    
    return adipose_df


# =============================================================================
# PART 7: INFLAMMATION-FOCUSED ANALYSIS
# =============================================================================

def analyze_inflammation(edge_df: pd.DataFrame, adipose_df: pd.DataFrame,
                         output_dir: Path, report_lines: List[str]) -> pd.DataFrame:
    """Analyze inflammation-related edges."""
    
    print("\n" + "="*70)
    print("PART 7: INFLAMMATION ANALYSIS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 7: INFLAMMATION ANALYSIS")
    report_lines.append("="*70 + "\n")
    
    # Vectorized inflammation filter using keyword pattern (faster than row-wise apply)
    # For short tokens (e.g., m1, m2, tnf, il1, il6, ccl, cxcl, tlr, nfkb) use
    # word boundaries to avoid massive overmatching inside other tokens.
    short_tokens = {"m1", "m2", "tnf", "il1", "il6", "tlr", "nfkb"}
    parts = []
    for k in INFLAMMATION_KEYWORDS:
        kl = k.lower()
        if kl in short_tokens:
            parts.append(rf"\b{re.escape(kl)}\b")
        elif kl in {"ccl", "cxcl"}:
            # skip raw token; will add digit-aware chemokine patterns below
            continue
        else:
            parts.append(re.escape(kl))
    # Add chemokine digit-aware patterns (e.g., CCL2, CXCL8)
    chemokine_patterns = [r"\bccl-?\d+\b", r"\bcxcl-?\d+\b"]
    parts.extend(chemokine_patterns)
    kw_pattern = "|".join(parts)
    inflam_mask = (
        edge_df.get('ctx_Mechanisms', pd.Series(dtype=str)).fillna('').str.lower().str.contains(kw_pattern, na=False, regex=True) |
        edge_df.get('ctx_Target_Disease_Role', pd.Series(dtype=str)).fillna('').str.lower().str.contains(kw_pattern, na=False, regex=True) |
        edge_df.get('ctx_Pathways', pd.Series(dtype=str)).fillna('').str.lower().str.contains(kw_pattern, na=False, regex=True) |
        edge_df.get('ctx_Molecular_Phenotype', pd.Series(dtype=str)).fillna('').str.lower().str.contains(kw_pattern, na=False, regex=True)
    )
    inflam_df = edge_df[inflam_mask].copy()
    n_inflam = len(inflam_df)
    n_total = len(edge_df)
    
    report_lines.append(f"Inflammation-related edges: {n_inflam:,} / {n_total:,} ({100*n_inflam/n_total:.1f}%)")

    # Filter to directed edges for causal analysis
    if 'direction' in inflam_df.columns:
        inflam_directed = inflam_df[inflam_df['direction'].astype(str) == '1'].copy()
        n_inflam_directed = len(inflam_directed)
        report_lines.append(f"  Directed inflammation edges: {n_inflam_directed:,} / {n_inflam:,} ({100*n_inflam_directed/n_inflam:.1f}%)")
        
        if n_inflam_directed == 0:
            report_lines.append("  WARNING: No directed inflammation edges found. Using all inflammation edges.")
        else:
            inflam_df = inflam_directed
            n_inflam = n_inflam_directed
    else:
        report_lines.append("  WARNING: No 'direction' column found. Using all inflammation edges.")

    report_lines.append(f"\nInflammation keywords used: {', '.join(INFLAMMATION_KEYWORDS[:10])}...")
    
    if n_inflam == 0:
        report_lines.append("No inflammation-related edges found.")
        print("  No inflammation-related edges found.")
        return pd.DataFrame()
    
    print(f"  Found {n_inflam:,} inflammation-related edges")
    
    # --- Overlap with adipose ---
    if len(adipose_df) > 0:
        adipose_indices = set(adipose_df.index)
        inflam_indices = set(inflam_df.index)
        overlap = adipose_indices & inflam_indices
        n_overlap = len(overlap)
        
        report_lines.append(f"\nOverlap with adipose edges: {n_overlap:,}")
        report_lines.append(f"  % of adipose edges that are inflammation-related: {100*n_overlap/len(adipose_df):.1f}%")
        report_lines.append(f"  % of inflammation edges that are adipose-related: {100*n_overlap/n_inflam:.1f}%")
        
        # Venn-style bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ['Adipose only', 'Both', 'Inflammation only']
        values = [
            len(adipose_indices - inflam_indices),
            n_overlap,
            len(inflam_indices - adipose_indices)
        ]
        bars = ax.bar(categories, values, color=[COLORS[0], COLORS[2], COLORS[1]])
        ax.set_ylabel('Edge Count')
        ax.set_title('Adipose vs Inflammation Edge Overlap')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=13)
        plt.tight_layout()
        save_plot(fig, output_dir, 'adipose_inflammation_overlap')
    
    # --- Pathways in inflammation (EXPLODED) ---
    report_lines.append("\nPathways within inflammation edges (exploded per-item):")
    path_items = explode_items_from_series(inflam_df['ctx_Pathways'])
    pw_counts = path_items.value_counts()

    for val, count in pw_counts.head(TOP_N_DEFAULT).items():
        short_val = val[:120] + '...' if len(val) > 120 else val
        report_lines.append(f"  {short_val}: {count:,}")

    if not pw_counts.empty:
        pw_counts.rename_axis('Pathway').reset_index(name='count').to_csv(
            output_dir / 'inflammation_pathways_exploded_counts.csv',
            index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        print(f"  Saved: inflammation_pathways_exploded_counts.csv")

        plot_top_counts(
            pw_counts,
            output_dir,
            filename='inflammation_pathways_exploded',
            title='Inflammation Edges: Pathways (Exploded per-item)'
        )
    
    # --- Cell Types in inflammation ---
    report_lines.append("\nCell_Type within inflammation edges:")
    cell_types = inflam_df['ctx_Cell_Type'][inflam_df['ctx_Cell_Type'].apply(is_specified)]
    ct_counts = cell_types.value_counts().head(TOP_N_DEFAULT)
    for val, count in ct_counts.items():
        report_lines.append(f"  {val}: {count:,}")
    
    if len(ct_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, max(5, len(ct_counts)*0.35)))
        bars = ax.barh(range(len(ct_counts)), ct_counts.values[::-1],
                       color=[COLORS[i % len(COLORS)] for i in range(len(ct_counts))])
        ax.set_yticks(range(len(ct_counts)))
        labels = [str(v)[:45] + '...' if len(str(v)) > 45 else str(v) for v in ct_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title('Inflammation Edges: Cell Types')
        annotate_barh(ax, bars, ct_counts.values)
        plt.tight_layout()
        save_plot(fig, output_dir, 'inflammation_cell_types')
    
    # --- Mechanisms in inflammation (EXPLODED) ---
    report_lines.append("\nMechanisms within inflammation edges (exploded per-item):")
    mech_items = explode_items_from_series(inflam_df['ctx_Mechanisms'])
    mech_counts = mech_items.value_counts()

    for val, count in mech_counts.head(TOP_N_DEFAULT).items():
        short_val = val[:120] + '...' if len(val) > 120 else val
        report_lines.append(f"  {short_val}: {count:,}")

    if not mech_counts.empty:
        mech_counts.rename_axis('Mechanism').reset_index(name='count').to_csv(
            output_dir / 'inflammation_mechanisms_exploded_counts.csv',
            index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        print(f"  Saved: inflammation_mechanisms_exploded_counts.csv")

        plot_top_counts(
            mech_counts,
            output_dir,
            filename='inflammation_mechanisms_exploded',
            title='Inflammation Edges: Mechanisms (Exploded per-item)'
        )
    
    # --- Tissues in inflammation ---
    report_lines.append("\nTissue within inflammation edges:")
    tissues = inflam_df['ctx_Tissue'][inflam_df['ctx_Tissue'].apply(is_specified)]
    tissue_counts = tissues.value_counts().head(TOP_N_BY_FIELD.get('Tissue', TOP_N_DEFAULT))
    for val, count in tissue_counts.items():
        report_lines.append(f"  {val}: {count:,}")
    
    if len(tissue_counts) > 0:
        fig, ax = plt.subplots(figsize=(8, max(5, len(tissue_counts)*0.35)))
        bars = ax.barh(range(len(tissue_counts)), tissue_counts.values[::-1],
                       color=[COLORS[i % len(COLORS)] for i in range(len(tissue_counts))])
        ax.set_yticks(range(len(tissue_counts)))
        labels = [str(v)[:45] + '...' if len(str(v)) > 45 else str(v) for v in tissue_counts.index[::-1]]
        ax.set_yticklabels(labels)
        ax.set_xlabel('Count')
        ax.set_title('Inflammation Edges: Tissues')
        annotate_barh(ax, bars, tissue_counts.values)
        plt.tight_layout()
        save_plot(fig, output_dir, 'inflammation_tissues')
    
    # Save inflammation edge data
    inflam_df.to_csv(output_dir / 'inflammation_edges.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"  Saved: inflammation_edges.csv")
    
    return inflam_df


# =============================================================================
# PART 8: GAP/COVERAGE ANALYSIS
# =============================================================================

def analyze_coverage_gaps(edge_df: pd.DataFrame, output_dir: Path,
                          report_lines: List[str]):
    """Analyze coverage gaps across field combinations."""
    
    print("\n" + "="*70)
    print("PART 8: COVERAGE GAP ANALYSIS")
    print("="*70)
    
    report_lines.append("\n" + "="*70)
    report_lines.append("PART 8: COVERAGE GAP ANALYSIS")
    report_lines.append("="*70 + "\n")
    
    # --- Organism × Tissue matrix ---
    report_lines.append("Organism × Tissue Coverage Matrix:")
    report_lines.append("-" * 50)
    
    org_col = 'ctx_Organism'
    tissue_col = 'ctx_Tissue'
    
    if org_col in edge_df.columns and tissue_col in edge_df.columns:
        # Filter to specified values
        df_filtered = edge_df[
            edge_df[org_col].apply(is_specified) & 
            edge_df[tissue_col].apply(is_specified)
        ]
        
        if len(df_filtered) > 0:
            cross_tab = pd.crosstab(df_filtered[org_col], df_filtered[tissue_col])
            
            # Keep top N for each dimension
            top_orgs = df_filtered[org_col].value_counts().head(10).index
            top_tissues = df_filtered[tissue_col].value_counts().head(15).index
            
            cross_tab_filtered = cross_tab.loc[
                cross_tab.index.isin(top_orgs),
                cross_tab.columns.isin(top_tissues)
            ]
            
            # Save full crosstab
            cross_tab.to_csv(output_dir / 'coverage_organism_tissue.csv', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: coverage_organism_tissue.csv")
            
            # Plot heatmap
            if len(cross_tab_filtered) > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(cross_tab_filtered, annot=True, fmt='d', cmap='YlOrRd',
                            ax=ax, cbar_kws={'label': 'Edge Count'})
                ax.set_xlabel('Tissue')
                ax.set_ylabel('Organism')
                ax.set_title('Organism × Tissue Coverage')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                save_plot(fig, output_dir, 'coverage_organism_tissue_heatmap')
            
            # Report top combinations
            report_lines.append("\nTop Organism-Tissue combinations:")
            combo_counts = df_filtered.groupby([org_col, tissue_col]).size().sort_values(ascending=False)
            for (org, tissue), count in combo_counts.head(20).items():
                report_lines.append(f"  {org} × {tissue}: {count:,}")
    
    # --- Model System × Tissue matrix ---
    report_lines.append("\n\nModel_System × Tissue Coverage Matrix:")
    report_lines.append("-" * 50)
    
    model_col = 'ctx_Model_System'
    
    if model_col in edge_df.columns and tissue_col in edge_df.columns:
        df_filtered = edge_df[
            edge_df[model_col].apply(is_specified) & 
            edge_df[tissue_col].apply(is_specified)
        ]
        
        if len(df_filtered) > 0:
            cross_tab = pd.crosstab(df_filtered[model_col], df_filtered[tissue_col])
            cross_tab.to_csv(output_dir / 'coverage_model_tissue.csv', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: coverage_model_tissue.csv")
            
            # Report top combinations
            report_lines.append("\nTop Model_System-Tissue combinations:")
            combo_counts = df_filtered.groupby([model_col, tissue_col]).size().sort_values(ascending=False)
            for (model, tissue), count in combo_counts.head(20).items():
                short_model = model[:40] + '...' if len(str(model)) > 40 else model
                report_lines.append(f"  {short_model} × {tissue}: {count:,}")
    
    # --- Cell Type × Tissue matrix ---
    report_lines.append("\n\nCell_Type × Tissue Coverage Matrix:")
    report_lines.append("-" * 50)
    
    cell_col = 'ctx_Cell_Type'
    
    if cell_col in edge_df.columns and tissue_col in edge_df.columns:
        df_filtered = edge_df[
            edge_df[cell_col].apply(is_specified) & 
            edge_df[tissue_col].apply(is_specified)
        ]
        
        if len(df_filtered) > 0:
            cross_tab = pd.crosstab(df_filtered[cell_col], df_filtered[tissue_col])
            
            # Keep top N for each dimension
            top_cells = df_filtered[cell_col].value_counts().head(15).index
            top_tissues = df_filtered[tissue_col].value_counts().head(15).index
            
            cross_tab_filtered = cross_tab.loc[
                cross_tab.index.isin(top_cells),
                cross_tab.columns.isin(top_tissues)
            ]
            
            cross_tab.to_csv(output_dir / 'coverage_celltype_tissue.csv', encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
            print(f"  Saved: coverage_celltype_tissue.csv")
            
            # Plot heatmap
            if len(cross_tab_filtered) > 0:
                fig, ax = plt.subplots(figsize=(14, 10))
                sns.heatmap(cross_tab_filtered, annot=True, fmt='d', cmap='YlOrRd',
                            ax=ax, cbar_kws={'label': 'Edge Count'})
                ax.set_xlabel('Tissue')
                ax.set_ylabel('Cell Type')
                ax.set_title('Cell Type × Tissue Coverage')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
                save_plot(fig, output_dir, 'coverage_celltype_tissue_heatmap')
            
            # Report top combinations
            report_lines.append("\nTop Cell_Type-Tissue combinations:")
            combo_counts = df_filtered.groupby([cell_col, tissue_col]).size().sort_values(ascending=False)
            for (cell, tissue), count in combo_counts.head(20).items():
                report_lines.append(f"  {cell} × {tissue}: {count:,}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_analysis(input_path: str = DEFAULT_INPUT_GRAPH, 
                 output_dir: str = DEFAULT_OUTPUT_DIR):
    """Run the complete analysis pipeline."""
    
    print("\n" + "="*70)
    print("AUGMENTED KNOWLEDGE GRAPH EDA")
    print("="*70)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize report
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("AUGMENTED KNOWLEDGE GRAPH - EXPLORATORY DATA ANALYSIS")
    report_lines.append("="*70)
    report_lines.append(f"\nInput graph: {input_path}")
    report_lines.append(f"Output directory: {output_dir}")
    
    # Load graph
    kg = load_graph(input_path)
    
    # Extract DataFrames
    node_df = extract_node_dataframe(kg)
    edge_df = extract_edge_dataframe(kg)
    
    # Quick QA: species vs context organism mismatches
    analyze_species_context_mismatch(edge_df, output_path, report_lines)

    # Save DataFrames
    node_df.to_csv(output_path / 'node_data.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    edge_df.to_csv(output_path / 'edge_data.csv', index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    print(f"  Saved: node_data.csv, edge_data.csv")
    
    # Run analyses
    analyze_basic_stats(kg, node_df, edge_df, output_path, report_lines)
    analyze_evidence_quality(edge_df, output_path, report_lines)
    investigate_year_spike(edge_df, output_path, report_lines)
    coverage_df = analyze_context_coverage(edge_df, output_path, report_lines)
    analyze_context_distributions(edge_df, coverage_df, output_path, report_lines)
    analyze_multivalued_fields(edge_df, output_path, report_lines)
    analyze_cross_field_relationships(edge_df, output_path, report_lines)
    adipose_df = analyze_adipose(edge_df, kg, output_path, report_lines)
    inflam_df = analyze_inflammation(edge_df, adipose_df, output_path, report_lines)
    analyze_coverage_gaps(edge_df, output_path, report_lines)
    
    # Save report
    report_text = '\n'.join(report_lines)
    report_path = output_path / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  Saved: analysis_report.txt")

    # Save stats and schema
    print(f"\nSaving graph schema to {output_path / 'schema.txt'}...")
    try:
        with open(output_path / "schema.txt", "w") as f:
            if hasattr(kg, 'schema'):
                f.write(str(kg.schema))
            else:
                f.write("Schema not available for this graph type.\n")
                f.write(f"Graph type: {type(kg).__name__}\n")
                f.write(f"Nodes: {kg.number_of_nodes()}\n")
                f.write(f"Edges: {kg.number_of_edges()}\n")
        print("Done!")
    except Exception as e:
        print(f"Warning: Could not save schema: {e}")
    
    print(f"\nSaving graph stats to {output_path / 'stats.txt'}...")
    try:
        with open(output_path / "stats.txt", "w") as f:
            with redirect_stdout(f):
                print_kg_stats(kg)
        print("Done!")
    except Exception as e:
        print(f"Warning: Could not save stats: {e}")

    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {output_path}")
    
    return kg, node_df, edge_df


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive EDA of LLM-annotated Knowledge Graph'
    )
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT_GRAPH,
        help=f'Path to input graph pickle file (default: {DEFAULT_INPUT_GRAPH})'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    run_analysis(args.input, args.output)


if __name__ == "__main__":
    main()