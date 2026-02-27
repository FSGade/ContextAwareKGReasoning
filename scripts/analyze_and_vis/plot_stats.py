#!/usr/bin/env python3
"""
Plot barplots of node and edge type distributions from subsetting experiment stats.
Creates one plot per configuration.
"""

import re
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


class KGStatsParser:
    """Parse knowledge graph statistics from text files."""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.data = self._parse_file()
    
    def _parse_file(self) -> Dict:
        """Parse the statistics file and extract node/edge distributions."""
        with open(self.filepath, 'r') as f:
            content = f.read()
        
        # Extract node type distribution
        node_section = re.search(
            r'Node Type Distribution:(.*?)(?=Edge Type Distribution:|$)', 
            content, 
            re.DOTALL
        )
        nodes = {}
        if node_section:
            for line in node_section.group(1).strip().split('\n'):
                match = re.search(r'(\w+(?:\s+\w+)*?):\s*(\d+)\s+nodes', line.strip())
                if match:
                    nodes[match.group(1)] = int(match.group(2))
        
        # Extract edge type distribution
        edge_section = re.search(
            r'Edge Type Distribution:(.*?)$', 
            content, 
            re.DOTALL
        )
        edges = {}
        if edge_section:
            for line in edge_section.group(1).strip().split('\n'):
                match = re.search(r'(\w+(?:_\w+)*?):\s*(\d+)\s+edges', line.strip())
                if match:
                    edges[match.group(1)] = int(match.group(2))
        
        return {'nodes': nodes, 'edges': edges}
    
    def get_nodes(self) -> Dict[str, int]:
        """Get node type distribution."""
        return self.data['nodes']
    
    def get_edges(self) -> Dict[str, int]:
        """Get edge type distribution."""
        return self.data['edges']


def parse_filename(filename: str) -> Dict[str, str]:
    """Extract metadata from filename."""
    # Remove _stats.txt suffix
    name = filename.replace('_stats.txt', '')
    
    # Parse components
    parts = name.split('_')
    
    # Strategy is first part (baseline, obesity_filter, multi_disease)
    if parts[0] == 'multi':
        strategy = f"{parts[0]}_{parts[1]}"
        rest = parts[2:]
    else:
        strategy = parts[0]
        rest = parts[1:]
    
    # Article type is next (no_reviews, all_articles, reviews_only)
    if rest[0] == 'no':
        article_type = f"{rest[0]}_{rest[1]}"
        mode = rest[2]
    elif rest[0] == 'reviews':
        article_type = f"{rest[0]}_{rest[1]}"
        mode = rest[2]
    else:
        article_type = f"{rest[0]}_{rest[1]}"
        mode = rest[2]
    
    return {
        'strategy': strategy,
        'article_type': article_type,
        'mode': mode,
        'full_name': name
    }


def create_node_barplot(nodes: Dict[str, int], config_name: str, output_file: Path):
    """Create a barplot for node type distribution."""
    
    if not nodes:
        print(f"    Warning: No node data for {config_name}")
        return
    
    # Sort by count descending
    sorted_nodes = sorted(nodes.items(), key=lambda x: x[1], reverse=True)
    node_types = [nt for nt, _ in sorted_nodes]
    counts = [c for _, c in sorted_nodes]
    
    # Color palette (colorblind-friendly Okabe-Ito)
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7']
    bar_colors = [colors[i % len(colors)] for i in range(len(node_types))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(node_types))
    bars = ax.bar(x, counts, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Node Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Node Type Distribution\n{config_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(node_types, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_edge_barplot(edges: Dict[str, int], config_name: str, output_file: Path):
    """Create a barplot for edge type distribution."""
    
    if not edges:
        print(f"    Warning: No edge data for {config_name}")
        return
    
    # Sort by count descending
    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
    edge_types = [et.replace('_', ' ') for et, _ in sorted_edges]
    counts = [c for _, c in sorted_edges]
    
    # Extended color palette (colorblind-friendly)
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', 
              '#D55E00', '#CC79A7', '#999999', '#88CCEE', '#DDCC77']
    bar_colors = [colors[i % len(colors)] for i in range(len(edge_types))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(edge_types))
    bars = ax.bar(x, counts, color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Edge Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Edge Type Distribution\n{config_name.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(edge_types, rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def process_graph(stats_dir: Path, output_dir: Path, graph_name: str):
    """Process all stats files for a graph and create individual plots."""
    
    stats_files = sorted(stats_dir.glob('*_stats.txt'))
    
    if not stats_files:
        print(f"  No stats files found in {stats_dir}")
        return
    
    print(f"  Found {len(stats_files)} configurations")
    
    for stats_file in stats_files:
        parser = KGStatsParser(stats_file)
        metadata = parse_filename(stats_file.name)
        
        config_name = metadata['full_name']
        
        # Create node plot
        node_output = output_dir / f"{config_name}_nodes.png"
        create_node_barplot(parser.get_nodes(), config_name, node_output)
        
        # Create edge plot
        edge_output = output_dir / f"{config_name}_edges.png"
        create_edge_barplot(parser.get_edges(), config_name, edge_output)
        
        print(f"    ✓ {config_name}")


def main():
    """Main function to process all graph types."""
    
    base_results = Path("/home/projects2/ContextAwareKGReasoning/results/search_subset_2")
    
    graph_configs = {
        'ikraph_pubmed_human': base_results / 'ikraph_pubmed_human' / 'stats',
        'ikraph_pubmed': base_results / 'ikraph_pubmed' / 'stats',
        'ikraph_full': base_results / 'ikraph_full' / 'stats',
    }
    
    output_base = base_results / 'plots_final'
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SUBSETTING EXPERIMENT VISUALIZATION")
    print("="*80)
    
    for graph_name, stats_dir in graph_configs.items():
        if not stats_dir.exists():
            print(f"\n❌ Skipping {graph_name}: directory not found at {stats_dir}")
            continue
        
        print(f"\n📊 Processing {graph_name}...")
        
        output_dir = output_base / graph_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        process_graph(stats_dir, output_dir, graph_name)
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {output_base}/")
    
    # Print summary
    print("\nOutput structure:")
    for graph_name in graph_configs.keys():
        graph_output = output_base / graph_name
        if graph_output.exists():
            node_plots = list(graph_output.glob('*_nodes.png'))
            edge_plots = list(graph_output.glob('*_edges.png'))
            print(f"  {graph_name}/")
            print(f"    - {len(node_plots)} node distribution plots")
            print(f"    - {len(edge_plots)} edge distribution plots")


if __name__ == "__main__":
    main()