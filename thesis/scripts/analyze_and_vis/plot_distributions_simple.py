"""
Simple script to plot probability distributions for all graphs.
Shows how probabilities change through PSR and inference steps.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


def plot_graph_distributions(graphs, output_dir):
    """Plot probability distributions for all graphs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, path in graphs.items():
        if not path.exists():
            print(f"Skipping {name} - file not found")
            continue
            
        print(f"\nProcessing: {name}")
        
        # Load graph
        kg = KnowledgeGraph.import_graph(str(path))
        print(f"  Nodes: {kg.number_of_nodes():,}, Edges: {kg.number_of_edges():,}")
        
        # Extract probabilities
        probs = []
        for u, v, data in tqdm(kg.edges(data=True), desc="  Extracting probs", total=kg.number_of_edges()):
            probs.append(data.get('probability', 0.0))
        
        probs = np.array(probs)
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Histogram
        axes[0, 0].hist(probs, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.median(probs), color='red', linestyle='--', label=f'Median: {np.median(probs):.3f}')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # CDF
        sorted_probs = np.sort(probs)
        cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        axes[0, 1].plot(sorted_probs, cdf, linewidth=2)
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution')
        axes[0, 1].grid(alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot([probs], labels=[name], vert=True)
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].grid(alpha=0.3)
        
        # Statistics text
        axes[1, 1].axis('off')
        stats_text = f"""
        Statistics for {name}
        
        N edges:  {len(probs):,}
        Mean:     {np.mean(probs):.4f}
        Median:   {np.median(probs):.4f}
        Std Dev:  {np.std(probs):.4f}
        Min:      {np.min(probs):.4f}
        Max:      {np.max(probs):.4f}
        Q1:       {np.percentile(probs, 25):.4f}
        Q3:       {np.percentile(probs, 75):.4f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.suptitle(f'{name}\nProbability Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        safe_name = name.replace(' ', '_').replace('/', '_')
        plt.savefig(output_dir / f'{safe_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir / f'{safe_name}.png'}")
        plt.close()


def main():
    base_path = Path("/home/projects2/ContextAwareKGReasoning")
    
    # Define graphs to analyze
    graphs = {
        'prototype_original': base_path / "data/graphs/prototypes/prototype_8seeds_12nodes.pkl",
        'prototype_aggregated': base_path / "data/graphs/subsets/prototype_8_12_aggregated.pkl",
        'prototype_inferred': base_path / "data/graphs/subsets/inferred/prototype_8_12_aggregated_with_inferred.pkl",
        'prototype_3hop_inferred': base_path / "data/graphs/subsets/inferred/prototype_8_12_aggregated_three_hop_with_inferred.pkl",
    }
    
    output_dir = base_path / "results/probability_distributions_prot"
    
    print("="*80)
    print("PROBABILITY DISTRIBUTION PLOTS")
    print("="*80)
    
    plot_graph_distributions(graphs, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
