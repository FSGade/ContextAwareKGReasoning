#!/usr/bin/env python3
"""
Shared utilities for the RQ2 pipeline.

Centralizes functions that were previously duplicated across multiple scripts:
config loading, graph helpers, and numerical constants.
"""

from pathlib import Path
from typing import Any, Tuple

import yaml
import numpy as np

# Numerical stability constant for log computations
EPS = 0.01


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ordered_pair(u: Any, v: Any) -> Tuple[Any, Any]:
    """Return lexicographically ordered pair for undirected edges."""
    return (u, v) if str(u) <= str(v) else (v, u)


def get_node_name(kg, node_id: Any) -> str:
    """Get human-readable name for a node."""
    attrs = kg.nodes.get(node_id, {})
    return attrs.get('name', attrs.get('label', str(node_id)))


def get_node_type(kg, node_id: Any) -> str:
    """Get the type attribute of a node."""
    attrs = kg.nodes.get(node_id, {})
    return attrs.get('type', attrs.get('kind', 'Unknown'))


def format_edge_type(etype: str) -> str:
    """Shorten edge type for metapath strings."""
    return etype.replace('_', '')[:12]


def extract_metapath(edge_types: list, node_types: list) -> str:
    """Build a metapath string like Gene-[assoc]-Protein-[regulates]-Disease."""
    if not node_types or not edge_types:
        return "Unknown"
    parts = [node_types[0]]
    for i, etype in enumerate(edge_types):
        parts.append(f"-[{format_edge_type(etype)}]-")
        if i + 1 < len(node_types):
            parts.append(node_types[i + 1])
    return ''.join(parts)