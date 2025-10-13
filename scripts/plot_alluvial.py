import plotly.graph_objects as go
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd


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


class AlluvialDiagramGenerator:
    """Generate alluvial diagrams for knowledge graph composition changes."""
    
    def __init__(self, input_path: Path, output_path: Path, 
                 file_patterns: List[str], subset_labels: List[str]):
        """
        Initialize with path-based configuration.
        
        Args:
            input_path: Path to directory containing stats files
            output_path: Path to directory for saving output
            file_patterns: List of filename patterns to match
            subset_labels: List of labels for each subset
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load files
        subset_files = [self.input_path / pattern for pattern in file_patterns]
        self.parsers = [KGStatsParser(f) for f in subset_files]
        self.labels = subset_labels
    
    def _get_colorblind_palette(self, n: int) -> List[str]:
        """
        Generate colorblind-friendly palette based on Okabe-Ito colors.
        Extended with additional distinct colors for larger n.
        """
        # Okabe-Ito colorblind-friendly palette
        base_colors = [
            (230, 159, 0),    # Orange
            (86, 180, 233),   # Sky Blue
            (0, 158, 115),    # Bluish Green
            (240, 228, 66),   # Yellow
            (0, 114, 178),    # Blue
            (213, 94, 0),     # Vermillion
            (204, 121, 167),  # Reddish Purple
            (102, 102, 102),  # Gray
        ]
        
        # Additional complementary colors for larger sets
        extended_colors = [
            (180, 119, 31),   # Brown
            (14, 209, 69),    # Green
            (255, 133, 27),   # Bright Orange
            (128, 0, 128),    # Purple
            (0, 191, 196),    # Cyan
            (255, 62, 165),   # Pink
            (122, 73, 46),    # Dark Brown
            (70, 130, 180),   # Steel Blue
            (154, 205, 50),   # Yellow Green
            (220, 20, 60),    # Crimson
            (64, 224, 208),   # Turquoise
            (184, 134, 11),   # Dark Goldenrod
        ]
        
        all_colors = base_colors + extended_colors
        
        # If we need more colors, interpolate
        if n > len(all_colors):
            import colorsys
            colors = []
            for i in range(n):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
                colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
            all_colors = colors
        
        return [f'rgb({r}, {g}, {b})' for r, g, b in all_colors[:n]]
    
    def _lighten_color(self, color_str: str, factor: float = 0.3) -> str:
        """
        Lighten an RGB color string.
        
        Args:
            color_str: RGB color string like 'rgb(230, 159, 0)'
            factor: How much to lighten (0 = original, 1 = white)
        """
        # Parse RGB values
        rgb = re.findall(r'\d+', color_str)
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        
        # Lighten by moving towards white
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        
        return f'rgba({r}, {g}, {b}, 0.4)'
    
    def _prepare_sankey_data(self, data_type: str = 'nodes', 
                        top_n: Optional[int] = None) -> Tuple:
        """
        Prepare data for Sankey diagram with hidden sink node for proper sizing.
        
        Args:
            data_type: 'nodes' or 'edges'
            top_n: Show only top N categories (None for all)
        """
        # Collect all data
        all_data = []
        for parser in self.parsers:
            if data_type == 'nodes':
                all_data.append(parser.get_nodes())
            else:
                all_data.append(parser.get_edges())
        
        # Get all unique categories
        all_categories = set()
        for data in all_data:
            all_categories.update(data.keys())
        
        # Filter to top N if specified
        if top_n:
            category_totals = {cat: sum(data.get(cat, 0) for data in all_data) 
                            for cat in all_categories}
            top_categories = sorted(category_totals.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_n]
            all_categories = {cat for cat, _ in top_categories}
            
            for i, data in enumerate(all_data):
                other_count = sum(count for cat, count in data.items() 
                                if cat not in all_categories)
                if other_count > 0:
                    all_data[i]['Other'] = other_count
            all_categories.add('Other')
        
        # Sort categories by total count (descending)
        category_totals = {cat: sum(data.get(cat, 0) for data in all_data) 
                        for cat in all_categories}
        all_categories = sorted(all_categories, key=lambda x: category_totals[x], reverse=True)
        
        # Create node labels
        node_labels = []
        node_map = {}
        node_idx = 0
        
        # Add actual subset nodes
        for i, label in enumerate(self.labels):
            for category in all_categories:
                node_labels.append(f"{category}")
                node_map[(i, category)] = node_idx
                node_idx += 1
        
        # Add hidden sink node
        sink_idx = node_idx
        node_labels.append("")
        
        # Create links and colors
        sources = []
        targets = []
        values = []
        link_colors = []
        
        colors = self._get_colorblind_palette(len(all_categories))
        category_colors = {cat: colors[i] for i, cat in enumerate(all_categories)}
        
        # Create flows between subsets and handle shrinkage/growth
        for i in range(len(self.parsers) - 1):
            for category in all_categories:
                source_count = all_data[i].get(category, 0)
                target_count = all_data[i + 1].get(category, 0)
                
                # Visible flow to next subset
                if target_count > 0:
                    sources.append(node_map[(i, category)])
                    targets.append(node_map[(i + 1, category)])
                    values.append(target_count)
                    link_colors.append(self._lighten_color(category_colors[category]))
                
                # Invisible flow to sink for items that disappear
                if source_count > target_count:
                    sources.append(node_map[(i, category)])
                    targets.append(sink_idx)
                    values.append(source_count - target_count)
                    link_colors.append('rgba(0, 0, 0, 0)')
        
        # DO NOT send anything from the last column to the sink!
        # The last column should just end without outflows
        
        # Assign colors to nodes
        node_colors = []
        for i, label in enumerate(self.labels):
            for category in all_categories:
                node_colors.append(category_colors[category])
        node_colors.append('rgba(0, 0, 0, 0)')  # Sink invisible
        
        # Create positions
        num_categories = len(all_categories)
        y_positions = []
        x_positions = []
        
        if num_categories == 1:
            for i in range(len(self.labels)):
                x_pos = i / (len(self.labels) - 1) if len(self.labels) > 1 else 0.5
                y_positions.append(0.5)
                x_positions.append(x_pos)
        else:
            for i in range(len(self.labels)):
                x_pos = i / (len(self.labels) - 1) if len(self.labels) > 1 else 0.5
                for j in range(num_categories):
                    y_val = 0.1 + (j * 0.8 / (num_categories - 1))
                    y_positions.append(y_val)
                    x_positions.append(x_pos)
        
        # Sink position (way off-screen in both dimensions so it doesn't affect layout)
        y_positions.append(10)  # Far below the visible area
        x_positions.append(10)  # Far to the right of the visible area
        
        return node_labels, sources, targets, values, link_colors, node_colors, y_positions, x_positions
        
    def create_node_diagram(self, top_n: Optional[int] = None, 
                       title: str = "Node Type Composition Changes",
                       filename: str = "node_composition_alluvial.html") -> go.Figure:
        """Create alluvial diagram for node composition."""
        node_labels, sources, targets, values, link_colors, node_colors, y_positions, x_positions = \
            self._prepare_sankey_data('nodes', top_n)
        
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=10,
                thickness=20,
                line=dict(color="white", width=0),  # No border at all
                label=node_labels,
                color=node_colors,
                x=x_positions,
                y=y_positions
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            font=dict(size=12),
            height=800,
            margin=dict(l=100, r=100, t=120, b=30)
        )
        
        # Add subset labels
        for i, label in enumerate(self.labels):
            x_pos = i/(len(self.labels)-1) if len(self.labels) > 1 else 0.5
            fig.add_annotation(
                x=x_pos,
                y=1.12,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=14)
            )
        
        # Save to output path
        output_file = self.output_path / filename
        fig.write_html(str(output_file))
        print(f"Node diagram saved to: {output_file}")
        
        return fig

    def create_edge_diagram(self, top_n: Optional[int] = None, 
                        title: str = "Edge Type Composition Changes",
                        filename: str = "edge_composition_alluvial.html") -> go.Figure:
        """Create alluvial diagram for edge composition."""
        node_labels, sources, targets, values, link_colors, node_colors, y_positions, x_positions = \
            self._prepare_sankey_data('edges', top_n)
        
        fig = go.Figure(data=[go.Sankey(
            arrangement='snap',
            node=dict(
                pad=10,
                thickness=20,
                line=dict(color="white", width=0),  # No border at all
                label=node_labels,
                color=node_colors,
                x=x_positions,
                y=y_positions
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        )])
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=18)),
            font=dict(size=12),
            height=1000,
            margin=dict(l=100, r=100, t=120, b=20)
        )
        
        # Add subset labels
        for i, label in enumerate(self.labels):
            x_pos = i/(len(self.labels)-1) if len(self.labels) > 1 else 0.5
            fig.add_annotation(
                x=x_pos,
                y=1.12,
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=14)
            )
        
        # Save to output path
        output_file = self.output_path / filename
        fig.write_html(str(output_file))
        print(f"Edge diagram saved to: {output_file}")
        
        return fig


# Example usage
if __name__ == "__main__":
    # Setup paths
    base_path = Path("/home/projects2/ContextAwareKGReasoning")
    input_path = base_path / "data/graphs/info"
    output_path = base_path / "results/eda/alluvial_aggr_5"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define your files and labels
    file_patterns = [
        'ikraph_stats.txt',
        'ikraph_aggregated_stats.txt',
        'pubmed_human_aggregated_stats.txt'
    ]
    
    subset_labels = ['Full iKGraph', 'Aggregated iKGraph', 'Pubmed Human Aggregated']
    
    # Create generator
    generator = AlluvialDiagramGenerator(
        input_path=input_path,
        output_path=output_path,
        file_patterns=file_patterns,
        subset_labels=subset_labels
    )
    
    print("Generating diagrams with all categories...")
    
    # Generate diagrams with ALL categories (no limit)
    node_fig = generator.create_node_diagram(
        top_n=None,
        title="iKGraph Node Type Evolution",
        filename="node_composition_aggr_full.html"
    )
 
    edge_fig = generator.create_edge_diagram(
        top_n=None,
        title="iKGraph Edge Type Evolution",
        filename="edge_composition_aggr_full.html"
    )

    print("Generating diagrams with top 20 categories...")
    
    # Plot with top 20 as well
    node_fig_20 = generator.create_node_diagram(
        top_n=20,
        title="iKGraph Node Type Evolution (Top 20)",
        filename="node_composition_aggr_top20.html"
    )

    edge_fig_20 = generator.create_edge_diagram(
        top_n=20,
        title="iKGraph Edge Type Evolution (Top 20)",
        filename="edge_composition_aggr_top20.html"
    )
    
    print(f"\nAll diagrams generated successfully!")
    print(f"Output saved to: {output_path}")