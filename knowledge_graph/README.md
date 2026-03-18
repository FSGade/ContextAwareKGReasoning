# Knowledge Graph Package

A comprehensive Python package for creating, manipulating, and analyzing schema-validated knowledge graphs. Built on NetworkX's MultiDiGraph, this package provides robust features for knowledge graph operations, including schema validation, visualization, format conversion, and integration with popular graph learning frameworks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
  - [Basic Operations](#basic-operations)
  - [Schema Management](#schema-management)
  - [Graph Manipulation](#graph-manipulation)
  - [Visualization](#visualization)
  - [Import/Export](#importexport)
  - [Framework Integration](#framework-integration)
  - [Utilities](#utilities)
- [License](#license)

## Features

### Core Features

- **Schema Validation**: Enforce data consistency with type validation for nodes and edges
- **Flexible Entity Management**: Unique node identification through name-type pairs
- **Dynamic Schema Updates**: Automatic schema updates for new types (when not frozen)
- **Multi-edge Support**: Handle multiple edges between the same nodes

### Visualization

- Static visualization using matplotlib
- Interactive visualization using pyvis
- Customizable node colors, sizes, and labels
- Support for various output formats (PNG, PDF, SVG, HTML)

### Data Integration

- Multiple format support (JSON, CSV, GEXF, GML, GraphML, Pickle)
- Integration with graph learning frameworks:
  - PyTorch Geometric (PyG)
  - ULTRA
  - StATIK

### Utilities

- Comprehensive graph statistics
- Advanced filtering capabilities
- Graph sampling and component analysis
- Degree distribution analysis

## Installation

### Development Installation

```bash
cd knowledge_graph
pip install -e ".[dev]"
```

### Dependencies

- **Required**:

  - Python >= 3.7
  - networkx
  - numpy
  - pandas

- **Optional**:

  - pytorch >= 1.8.0 (for deep learning integrations)
  - torch-geometric (for PyG integration)
  - matplotlib (for static visualization)
  - pyvis (for interactive visualization)
  - colorsys (for visualization color management)

## Architecture

### Core Components

#### KnowledgeGraph Class

The main class extending NetworkX's MultiDiGraph:

```python
from knowledge_graph import KnowledgeGraph

# Create a new knowledge graph
kg = KnowledgeGraph()

# Add nodes and edges with types
kg.add_typed_node("John", "Person")
kg.add_typed_edge(
    head_name="John",
    head_type="Person",
    tail_name="Acme",
    tail_type="Company",
    edge_type="WORKS_FOR",
)
```

#### KnowledgeGraphSchema Class

Manages and enforces the graph's structure:

```python
from knowledge_graph import KnowledgeGraphSchema

# Create a schema with predefined types
schema = KnowledgeGraphSchema(
    node_types={"Person", "Organization", "Project"},
    edge_types={"WORKS_FOR", "MANAGES", "CONTRIBUTES_TO"},
    frozen=True,  # Prevent schema modifications
)

# Create a graph with the schema
kg = KnowledgeGraph(schema=schema)
```

## Usage Guide

### Basic Operations

#### Creating Nodes and Edges

```python
# Add single node
entity = kg.add_node(("John", "Person"))

# Add multiple nodes
kg.add_nodes_from([("Alice", "Person"), ("Bob", "Person"), ("Acme", "Company")])

# Add edge with attributes
kg.add_edge(
    ("John", "Person"), ("Acme", "Company"), type="WORKS_FOR", start_date="2023-01-01"
)
```

#### Querying the Graph

```python
# Get nodes by type
persons = kg.get_nodes_by_type("Person")

# Get nodes by name
johns = kg.get_nodes_by_name("John")

# Get edges by type
work_relations = kg.get_edges_by_type("WORKS_FOR")
```

### Schema Management

#### Creating and Modifying Schema

```python
# Create flexible schema
schema = KnowledgeGraphSchema(frozen=False)

# Add new types dynamically
schema.add_node_type("Customer")
schema.add_edge_type("PURCHASES")

# Freeze schema to prevent modifications
schema.frozen = True
```

#### Validation

```python
# Will raise ValueError if type is invalid
try:
    kg.add_node(("John", "InvalidType"))
except ValueError as e:
    print(f"Validation error: {e}")
```

### Graph Manipulation

#### Filtering

```python
from knowledge_graph.utils.filtering import filter_graph

# Filter edges based on attributes
filtered_kg = filter_graph(
    kg,
    filter_criterion=lambda d: (
        d.get("weight", 0) > 1.0 and d.get("type") == "WORKS_FOR"
    ),
)

# Remove rare relations
from knowledge_graph.utils.filtering import remove_rare_relations

cleaned_kg = remove_rare_relations(kg, n=5)
```

#### Sampling

```python
from knowledge_graph.utils.filtering import sample_and_get_largest_component

# Get largest component with 1000 edges
sampled_kg = sample_and_get_largest_component(kg, n_edges=1000, seed=42)
```

### Visualization

#### Static Visualization

```python
# Basic visualization
kg.visualize("graph.png", figsize=(12, 8), node_size=2000, font_size=10)

# Customized visualization
kg.visualize(
    "graph.pdf",
    title="Company Structure",
    node_size=3000,
    font_size=12,
    edge_font_size=8,
)
```

#### Interactive Visualization

```python
# Create interactive HTML visualization
kg.visualize("graph.html", title="Interactive Knowledge Graph")
```

### Import/Export

#### Exporting Graphs

```python
# Export to various formats
kg.export_graph("graph.json")
kg.export_graph("graph.csv")
kg.export_graph("graph.gexf")
kg.export_graph("graph.pkl", file_format="pickle")
```

#### Importing Graphs

```python
# Import from file
kg = KnowledgeGraph.import_graph("graph.json", file_format="json")
```

### Framework Integration

#### PyTorch Geometric Integration

```python
from knowledge_graph.io.adapters.ultra import ULTRAAdapter

# Convert to PyG dataset
dataset = ULTRAAdapter.to_dataset(kg, split_ratios=(0.8, 0.1, 0.1), seed=42)

# Access splits
train_data = dataset[0]
valid_data = dataset[1]
test_data = dataset[2]
```

#### StATIK Integration

```python
from knowledge_graph.io.adapters.statik import StATIKAdapter

# Convert to StATIK format
processed_dataset = StATIKAdapter.create_processed_dataset(
    kg, save_dir="processed_data", embedding_model_name="bert-base-cased"
)
```

### Utilities

#### Graph Statistics

```python
from knowledge_graph.utils.stats import print_kg_stats

# Print comprehensive statistics
print_kg_stats(kg)
```

## License

MIT License

## Citation

No citation for this package yet exists. Please contact fzsg@novonordisk.com or fstga@dtu.dk
