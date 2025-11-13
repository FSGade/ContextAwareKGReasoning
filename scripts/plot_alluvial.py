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
    
    # def _prepare_sankey_data(self, data_type: str = 'nodes', 
    #                     top_n: Optional[int] = None) -> Tuple:
    #     """
    #     Prepare data for Sankey diagram with hidden sink node for proper sizing.
        
    #     Args:
    #         data_type: 'nodes' or 'edges'
    #         top_n: Show only top N categories (None for all)
    #     """
    #     # Collect all data
    #     all_data = []
    #     for parser in self.parsers:
    #         if data_type == 'nodes':
    #             all_data.append(parser.get_nodes())
    #         else:
    #             all_data.append(parser.get_edges())
        
    #     # Get all unique categories
    #     all_categories = set()
    #     for data in all_data:
    #         all_categories.update(data.keys())
        
    #     # Filter to top N if specified
    #     if top_n:
    #         category_totals = {cat: sum(data.get(cat, 0) for data in all_data) 
    #                         for cat in all_categories}
    #         top_categories = sorted(category_totals.items(), 
    #                             key=lambda x: x[1], 
    #                             reverse=True)[:top_n]
    #         all_categories = {cat for cat, _ in top_categories}
            
    #         for i, data in enumerate(all_data):
    #             other_count = sum(count for cat, count in data.items() 
    #                             if cat not in all_categories)
    #             if other_count > 0:
    #                 all_data[i]['Other'] = other_count
    #         all_categories.add('Other')
        
    #     # Sort categories by total count (descending)
    #     category_totals = {cat: sum(data.get(cat, 0) for data in all_data) 
    #                     for cat in all_categories}
    #     all_categories = sorted(all_categories, key=lambda x: category_totals[x], reverse=True)
        
    #     # Create node labels
    #     node_labels = []
    #     node_map = {}
    #     node_idx = 0
        
    #     # Add actual subset nodes
    #     for i, label in enumerate(self.labels):
    #         for category in all_categories:
    #             node_labels.append(f"{category}")
    #             node_map[(i, category)] = node_idx
    #             node_idx += 1
        
    #     # Add hidden sink node
    #     sink_idx = node_idx
    #     node_labels.append("")
        
    #     # Create links and colors
    #     sources = []
    #     targets = []
    #     values = []
    #     link_colors = []
        
    #     colors = self._get_colorblind_palette(len(all_categories))
    #     category_colors = {cat: colors[i] for i, cat in enumerate(all_categories)}
        
    #     # Create flows between subsets and handle shrinkage/growth
    #     for i in range(len(self.parsers) - 1):
    #         for category in all_categories:
    #             source_count = all_data[i].get(category, 0)
    #             target_count = all_data[i + 1].get(category, 0)
                
    #             # Visible flow to next subset
    #             if target_count > 0:
    #                 sources.append(node_map[(i, category)])
    #                 targets.append(node_map[(i + 1, category)])
    #                 values.append(target_count)
    #                 link_colors.append(self._lighten_color(category_colors[category]))
                
    #             # Invisible flow to sink for items that disappear
    #             if source_count > target_count:
    #                 sources.append(node_map[(i, category)])
    #                 targets.append(sink_idx)
    #                 values.append(source_count - target_count)
    #                 link_colors.append('rgba(0, 0, 0, 0)')
        
    #     # DO NOT send anything from the last column to the sink!
    #     # The last column should just end without outflows
        
    #     # Assign colors to nodes
    #     node_colors = []
    #     for i, label in enumerate(self.labels):
    #         for category in all_categories:
    #             node_colors.append(category_colors[category])
    #     node_colors.append('rgba(0, 0, 0, 0)')  # Sink invisible
        
    #     # Create positions
    #     num_categories = len(all_categories)
    #     y_positions = []
    #     x_positions = []
        
    #     if num_categories == 1:
    #         for i in range(len(self.labels)):
    #             x_pos = i / (len(self.labels) - 1) if len(self.labels) > 1 else 0.5
    #             y_positions.append(0.5)
    #             x_positions.append(x_pos)
    #     else:
    #         for i in range(len(self.labels)):
    #             x_pos = i / (len(self.labels) - 1) if len(self.labels) > 1 else 0.5
    #             for j in range(num_categories):
    #                 y_val = 0.1 + (j * 0.8 / (num_categories - 1))
    #                 y_positions.append(y_val)
    #                 x_positions.append(x_pos)
        
    #     # Sink position (way off-screen in both dimensions so it doesn't affect layout)
    #     y_positions.append(10)  # Far below the visible area
    #     x_positions.append(10)  # Far to the right of the visible area
        
    #     return node_labels, sources, targets, values, link_colors, node_colors, y_positions, x_positions
    def _prepare_sankey_data(self, data_type: str = 'nodes', 
                         top_n: Optional[int] = None) -> Tuple:
        """
        Prepare data for Sankey with conservation using a single global hidden source/sink:
        - Visible carry-over = min(source, target)
        - Shrink (source > target) -> one global hidden sink (far right in snap)
        - Growth (target > source) <- one global hidden source (far left in snap)
        We deliberately do NOT return x/y positions and will use arrangement='snap'.
        """
        # 1) Collect parsed distributions
        all_data = []
        for parser in self.parsers:
            all_data.append(parser.get_nodes() if data_type == 'nodes' else parser.get_edges())

        # 2) Category union
        all_categories = set()
        for d in all_data:
            all_categories.update(d.keys())

        # 3) Optional Top-N (aggregate remainder to "Other")
        if top_n:
            totals = {c: sum(d.get(c, 0) for d in all_data) for c in all_categories}
            keep = {c for c, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:top_n]}
            new_all = []
            for d in all_data:
                nd = {k: v for k, v in d.items() if k in keep}
                other = sum(v for k, v in d.items() if k not in keep)
                if other > 0:
                    nd['Other'] = nd.get('Other', 0) + other
                new_all.append(nd)
            all_data = new_all
            all_categories = set(keep)
            if any('Other' in d for d in all_data):
                all_categories.add('Other')

        # 4) Stable category order by total
        totals = {c: sum(d.get(c, 0) for d in all_data) for c in all_categories}
        cats = sorted(all_categories, key=lambda c: totals[c], reverse=True)

        # 5) Visible nodes (grid: columns × categories)
        node_labels, node_map = [], {}
        node_idx = 0
        n_cols = len(self.labels)
        for col in range(n_cols):
            for c in cats:
                node_labels.append(c)
                node_map[(col, c)] = node_idx
                node_idx += 1

        # 6) One global hidden sink + one global hidden source
        sink_idx = node_idx
        node_labels.append("")   # hidden sink (far right in snap)
        node_idx += 1

        source_idx = node_idx
        node_labels.append("")   # hidden source (far left in snap)
        node_idx += 1

        # 7) Links + colors
        sources, targets, values, link_colors = [], [], [], []
        colors = self._get_colorblind_palette(len(cats))
        cat_color = {c: colors[i] for i, c in enumerate(cats)}

        for i in range(n_cols - 1):
            left, right = all_data[i], all_data[i + 1]
            for c in cats:
                s = left.get(c, 0)
                t = right.get(c, 0)
                overlap = min(s, t)

                # Visible carry-over
                if overlap > 0:
                    sources.append(node_map[(i, c)])
                    targets.append(node_map[(i + 1, c)])
                    values.append(overlap)
                    link_colors.append(self._lighten_color(cat_color[c]))

                # Shrink -> global sink
                if s > t:
                    sources.append(node_map[(i, c)])
                    targets.append(sink_idx)
                    values.append(s - t)
                    link_colors.append('rgba(0,0,0,0)')

                # Growth <- global source
                if t > s:
                    sources.append(source_idx)
                    targets.append(node_map[(i + 1, c)])
                    values.append(t - s)
                    link_colors.append('rgba(0,0,0,0)')

        # 8) Node colors: visible nodes colored; hidden nodes transparent
        node_colors = []
        for _ in range(n_cols):
            for c in cats:
                node_colors.append(cat_color[c])
        node_colors.append('rgba(0,0,0,0)')  # sink
        node_colors.append('rgba(0,0,0,0)')  # source

        # We return empty positions (we won't pass them to Plotly in snap mode)
        y_positions, x_positions = [], []

        return (
            node_labels, 
            sources, 
            targets, 
            values, 
            link_colors, 
            node_colors, 
            y_positions, 
            x_positions
        )



        
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
    output_prot = base_path / "results/eda/snap/alluvial_prot_2"
    output_sub = base_path / "results/eda/snap/alluvial_sub_2"
    output_aggr = base_path / "results/eda/snap/alluvial_aggr_2"

    # Create output directories if they don't exist
    # output_prot.mkdir(parents=True, exist_ok=True)
    # output_sub.mkdir(parents=True, exist_ok=True)
    output_aggr.mkdir(parents=True, exist_ok=True)

    # Define your files and labels
    file_prot = [
        'prototype_stats.txt',
        'prototype_8_12_aggregated_schema.txt',
        'prototype_8_12_aggregated_with_inferred_stats.txt'
    ]

    file_sub = ["ikraph_stats.txt",
                "ikraph_pubmed_stats.txt",
                "ikraph_pubmed_human_stats.txt"]

    file_aggr = ["ikraph_stats.txt",
                "ikraph_aggregated_stats.txt",
                "pubmed_human_aggregated_stats.txt",
                'prototype_8_12_aggregated_schema.txt']
    
                
    
    labels_prot = ['Prototype', 'Aggregated Prototype', 'Prototype + Inferred']
    labels_sub = ['iKraph', 'Pubmed Subset', 'Pubmed Human Subset']
    labels_aggr = ['iKraph', 'iKraph Aggregated', 'Pubmed Human Aggregated', 'Aggregated Prototype']
    
    # Create configurations for the three sets we want to plot
    sets = [
        # {
        #     'name': 'prototype',
        #     'file_patterns': file_prot,
        #     'labels': labels_prot,
        #     'output_dir': output_prot,
        # },
        # {
        #     'name': 'subset',
        #     'file_patterns': file_sub,
        #     'labels': labels_sub,
        #     'output_dir': output_sub,
        # },
        {
            'name': 'aggregated',
            'file_patterns': file_aggr,
            'labels': labels_aggr,
            'output_dir': output_aggr,
        },
    ]

    for s in sets:
        print(f"\nGenerating diagrams for set: {s['name']}")
        s['output_dir'].mkdir(parents=True, exist_ok=True)

        generator = AlluvialDiagramGenerator(
            input_path=input_path,
            output_path=s['output_dir'],
            file_patterns=s['file_patterns'],
            subset_labels=s['labels']
        )

        # Full (all categories)
        generator.create_node_diagram(
            top_n=None,
            title=f"{s['name'].title()} Node Type Evolution",
            filename=f"{s['name']}_node_composition_full.html",
        )
        generator.create_edge_diagram(
            top_n=None,
            title=f"{s['name'].title()} Edge Type Evolution",
            filename=f"{s['name']}_edge_composition_full.html",
        )

        # Top 20
        generator.create_node_diagram(
            top_n=20,
            title=f"{s['name'].title()} Node Type Evolution (Top 20)",
            filename=f"{s['name']}_node_composition_top20.html",
        )
        generator.create_edge_diagram(
            top_n=20,
            title=f"{s['name'].title()} Edge Type Evolution (Top 20)",
            filename=f"{s['name']}_edge_composition_top20.html",
        )

    print(f"\nAll diagrams generated successfully!")
    print(f"Outputs saved to: {output_prot}, {output_sub}, {output_aggr}")