"""
Analyze database source composition of knowledge graph edges.
Creates bar-of-pie charts showing all database sources.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


def categorize_edge_type(edge_type):
    """Categorize edge type into entity categories."""
    edge_type_lower = edge_type.lower()
    
    if 'anatomy' in edge_type_lower or 'protein_present' in edge_type_lower or 'protein_absent' in edge_type_lower:
        return 'Anatomy'
    elif 'bioprocess' in edge_type_lower:
        return 'Biological Process'
    elif 'pathway' in edge_type_lower:
        return 'Pathway'
    elif 'cellcomp' in edge_type_lower:
        return 'Cellular Component'
    elif 'molfunc' in edge_type_lower:
        return 'Molecular Function'
    else:
        return 'Other'


def extract_source_data(kg):
    """Extract source information from all edges."""
    print("Extracting source data from edges...")
    
    all_sources = []
    sources_by_category = defaultdict(list)
    
    for u, v, key, data in tqdm(kg.edges(keys=True, data=True), 
                                 desc="Processing edges", 
                                 total=kg.number_of_edges()):
        
        source = data.get('source', 'Unknown')
        edge_type = data.get('type', 'unknown')
        
        # Handle multiple sources (comma-separated)
        if isinstance(source, str) and ',' in source:
            sources_list = [s.strip() for s in source.split(',')]
        else:
            sources_list = [source]
        
        # Add to overall collection
        all_sources.extend(sources_list)
        
        # Categorize and add to category-specific collection
        category = categorize_edge_type(edge_type)
        sources_by_category[category].extend(sources_list)
    
    return all_sources, sources_by_category


def create_bar_of_pie(source_counts, title, output_path, top_n=5):
    """Create a bar-of-pie chart with connection lines showing top N sources in pie, rest in stacked bar."""
    
    if not source_counts:
        print(f"  No data for {title}")
        return
    
    # Get all sources sorted by count
    all_sources = source_counts.most_common()
    total = sum(source_counts.values())
    
    if len(all_sources) <= top_n:
        top_n = len(all_sources)
    
    # Split into top N and others
    top_sources = all_sources[:top_n]
    other_sources = all_sources[top_n:]
    
    # Okabe-Ito color palette (colorblind-friendly)
    okabe_ito = [
        '#E69F00',  # Orange
        '#56B4E9',  # Sky Blue
        '#009E73',  # Bluish Green
        '#F0E442',  # Yellow
        '#0072B2',  # Blue
        '#D55E00',  # Vermillion
        '#CC79A7',  # Reddish Purple
        '#000000',  # Black
        '#999999',  # Gray (for "Others")
    ]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0)
    
    # Prepare data for pie chart
    pie_sizes = []
    pie_labels = []
    pie_colors = []
    explode = []
    
    for i, (source, count) in enumerate(top_sources):
        percentage = 100 * count / total
        pie_labels.append(f"{source}\n({percentage:.1f}%)")
        pie_sizes.append(count)
        pie_colors.append(okabe_ito[i % len(okabe_ito)])
        explode.append(0)
    
    # Add "Others" slice if there are remaining sources
    if other_sources:
        other_total = sum(count for source, count in other_sources)
        other_pct = 100 * other_total / total
        pie_labels.append(f"Others ({len(other_sources)})\n({other_pct:.1f}%)")
        pie_sizes.append(other_total)
        pie_colors.append(okabe_ito[top_n % len(okabe_ito)])
        explode.append(0.1)  # Explode the "Others" slice
    
    # Calculate start angle to position "Others" slice on the right side
    if other_sources:
        # Calculate percentages
        pct_before_others = sum(pie_sizes[:-1]) / sum(pie_sizes)
        pct_others_half = pie_sizes[-1] / (2 * sum(pie_sizes))
        # Rotate so "Others" is centered at 0 degrees (pointing right)
        angle = -(pct_before_others + pct_others_half) * 360
    else:
        angle = 90
    
    # Create pie chart
    wedges, texts, autotexts = ax1.pie(
        pie_sizes,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=angle,
        colors=pie_colors,
        explode=explode,
        pctdistance=0.85,
        textprops={'fontsize': 12}
    )
    
    for text in texts:
        text.set_fontsize(11)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax1.set_title(f"{title}\n(Top {top_n} Sources)", fontsize=16, fontweight='bold', pad=20)
    
    # Create stacked bar chart for "Others" breakdown
    if other_sources:
        bar_ratios = [count / other_total for source, count in other_sources]
        bar_labels = [source for source, count in other_sources]
        
        bottom = 1
        width = 0.2
        
        # Stack bars from top to bottom
        # Use colors starting from where pie chart left off
        for j, (height, label, (source, count)) in enumerate(reversed(list(zip(bar_ratios, bar_labels, other_sources)))):
            bottom -= height
            # Continue color palette from where pie left off
            color_idx = (top_n + len(other_sources) - j - 1) % len(okabe_ito)
            bc = ax2.bar(0, height, width, bottom=bottom, 
                        color=okabe_ito[color_idx], 
                        label=label,
                        alpha=0.7,
                        edgecolor='white',
                        linewidth=1)
            
            # Add percentage label
            percentage = 100 * count / total
            ax2.bar_label(bc, labels=[f"{percentage:.2f}%"], label_type='center', fontsize=11, fontweight='bold')
        
        ax2.set_title(f'"Others" Breakdown\n({len(other_sources)} databases)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        ax2.axis('off')
        ax2.set_xlim(-2.5 * width, 2.5 * width)
        
        # Draw connection lines between pie slice and bar
        if other_sources:
            others_wedge = wedges[-1]  # The "Others" wedge
            theta1, theta2 = others_wedge.theta1, others_wedge.theta2
            center, r = others_wedge.center, others_wedge.r
            bar_height = sum(bar_ratios)
            
            # Draw top connecting line
            x = r * np.cos(np.pi / 180 * theta2) + center[0]
            y = r * np.sin(np.pi / 180 * theta2) + center[1]
            con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                                xyB=(x, y), coordsB=ax1.transData)
            con.set_color([0, 0, 0])
            con.set_linewidth(2)
            ax2.add_artist(con)
            
            # Draw bottom connecting line
            x = r * np.cos(np.pi / 180 * theta1) + center[0]
            y = r * np.sin(np.pi / 180 * theta1) + center[1]
            con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                                xyB=(x, y), coordsB=ax1.transData)
            con.set_color([0, 0, 0])
            con.set_linewidth(2)
            ax2.add_artist(con)
    else:
        ax2.text(0.5, 0.5, 'All sources shown in pie chart',
                ha='center', va='center', fontsize=14)
        ax2.axis('off')
    
    # Overall title
    fig.suptitle(f"Database Source Composition", fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_standalone_pie(source_counts, title, output_path, top_n=5):
    """Create a standalone pie chart for category-specific analysis."""
    
    if not source_counts:
        print(f"  No data for {title}")
        return
    
    # Get all sources sorted by count
    all_sources = source_counts.most_common()
    total = sum(source_counts.values())
    
    if len(all_sources) <= top_n:
        top_n = len(all_sources)
    
    # Split into top N and others
    top_sources = all_sources[:top_n]
    other_sources = all_sources[top_n:]
    
    # Okabe-Ito color palette
    okabe_ito = [
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
    
    # Prepare data for pie chart
    pie_sizes = []
    pie_labels = []
    pie_colors = []
    
    for i, (source, count) in enumerate(top_sources):
        percentage = 100 * count / total
        pie_labels.append(f"{source}\n({percentage:.1f}%)")
        pie_sizes.append(count)
        pie_colors.append(okabe_ito[i % len(okabe_ito)])
    
    # Add "Others" if present
    if other_sources:
        other_total = sum(count for source, count in other_sources)
        other_pct = 100 * other_total / total
        pie_labels.append(f"Others ({len(other_sources)})\n({other_pct:.1f}%)")
        pie_sizes.append(other_total)
        pie_colors.append(okabe_ito[top_n % len(okabe_ito)])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        pie_sizes,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        pctdistance=0.85,
        textprops={'fontsize': 12}
    )
    
    for text in texts:
        text.set_fontsize(11)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title(f"{title}\nDatabase Source Distribution", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_summary_report(all_sources, sources_by_category, output_path):
    """Create a text summary report."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DATABASE SOURCE COMPOSITION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        total_edges = len(all_sources)
        unique_sources = len(set(all_sources))
        source_counts = Counter(all_sources)
        
        f.write(f"OVERALL STATISTICS\n")
        f.write(f"-" * 80 + "\n")
        f.write(f"Total edges: {total_edges:,}\n")
        f.write(f"Unique sources: {unique_sources}\n\n")
        
        f.write(f"ALL SOURCES (Overall):\n")
        for i, (source, count) in enumerate(source_counts.most_common(), 1):
            percentage = 100 * count / total_edges
            f.write(f"{i:2d}. {source:30s} {count:>10,} ({percentage:>5.2f}%)\n")
        
        # Category-specific statistics
        f.write(f"\n\n" + "="*80 + "\n")
        f.write("SOURCES BY CATEGORY\n")
        f.write("="*80 + "\n\n")
        
        for category in sorted(sources_by_category.keys()):
            sources_list = sources_by_category[category]
            if not sources_list:
                continue
            
            cat_total = len(sources_list)
            cat_counts = Counter(sources_list)
            
            f.write(f"\n{category.upper()}\n")
            f.write(f"-" * 80 + "\n")
            f.write(f"Total edges: {cat_total:,}\n")
            f.write(f"Unique sources: {len(cat_counts)}\n\n")
            
            f.write(f"All sources:\n")
            for i, (source, count) in enumerate(cat_counts.most_common(), 1):
                percentage = 100 * count / cat_total
                f.write(f"{i:2d}. {source:30s} {count:>10,} ({percentage:>5.2f}%)\n")
    
    print(f"  Saved: {output_path}")


def main():
    # Configuration
    base_path = Path("/home/projects2/ContextAwareKGReasoning/data")
    input_graph = base_path / "graphs/ikraph.pkl"
    output_dir = base_path / "results/eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    print("="*80)
    print("KNOWLEDGE GRAPH SOURCE COMPOSITION ANALYSIS")
    print("="*80)
    print(f"\nLoading graph: {input_graph}")
    kg = KnowledgeGraph.import_graph(str(input_graph))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges\n")
    
    # Extract source data
    all_sources, sources_by_category = extract_source_data(kg)
    
    if not all_sources:
        print("No source information found in edges!")
        return
    
    print(f"\nTotal edge-source pairs: {len(all_sources):,}")
    print(f"Unique sources: {len(set(all_sources))}")
    print(f"Categories found: {', '.join(sorted(sources_by_category.keys()))}\n")
    
    # Create overall bar-of-pie chart
    print("Creating visualizations...")
    overall_counts = Counter(all_sources)
    create_bar_of_pie(
        overall_counts,
        "All Edges",
        output_dir / "overall_source_distribution.png",
        top_n=3
    )
    
    # Create category-specific standalone pie charts
    categories_of_interest = ['Anatomy', 'Pathway', 'Biological Process', 
                              'Cellular Component', 'Molecular Function']
    
    for category in categories_of_interest:
        if category in sources_by_category and sources_by_category[category]:
            cat_counts = Counter(sources_by_category[category])
            safe_name = category.lower().replace(' ', '_')
            create_standalone_pie(
                cat_counts,
                f"{category} Edges",
                output_dir / f"{safe_name}_source_distribution.png",
                top_n=5
            )
    
    # Create summary report
    print("\nCreating summary report...")
    create_summary_report(
        all_sources, 
        sources_by_category, 
        output_dir / "source_composition_report.txt"
    )
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - overall_source_distribution.png (bar-of-pie)")
    for category in categories_of_interest:
        if category in sources_by_category and sources_by_category[category]:
            safe_name = category.lower().replace(' ', '_')
            print(f"  - {safe_name}_source_distribution.png (standalone pie)")
    print("  - source_composition_report.txt")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)