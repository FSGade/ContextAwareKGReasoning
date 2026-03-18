#!/usr/bin/env python3
"""
RQ2 Tissue Mapping — Canonical tissue group definitions and coverage utilities.

Defines:
- Tissue group mappings for Subcutaneous/Visceral and White/Brown comparisons
- Functions to match detailed_tissue values against groups
- Coverage propagation methods (geometric mean, min, product)

Matching rules use STRICT labels only — see RQ2_TISSUE_GROUPS for details.
"""

from typing import Any, Dict, List
from collections import Counter

import numpy as np


# ============================================================================
# Tissue Group Definitions
# ============================================================================
#
# Based on normalized detailed_tissue values from the graph.
# Counts from exploration (approximate):
#     4,260  White Adipose Tissue
#     3,473  Visceral Adipose Tissue
#     2,093  Subcutaneous Adipose Tissue
#     1,782  ['Subcutaneous Adipose Tissue', 'Visceral Adipose Tissue']
#       658  ['Brown Adipose Tissue', 'White Adipose Tissue']
#       648  Subcutaneous Abdominal Adipose Tissue
#       504  Brown Adipose Tissue
#       311  Visceral White Adipose Tissue
#       226  Subcutaneous White Adipose Tissue

RQ2_TISSUE_GROUPS = {
    # Comparison 1: Subcutaneous vs Visceral (depot location)
    'subcutaneous': {
        'exact_matches': [
            'Subcutaneous Adipose Tissue',
            'Subcutaneous Abdominal Adipose Tissue',
            'Subcutaneous White Adipose Tissue',
        ],
        'list_component_matches': ['Subcutaneous Adipose Tissue'],
        'description': 'Subcutaneous adipose depot (strict)',
    },
    'visceral': {
        'exact_matches': [
            'Visceral Adipose Tissue',
            'Visceral White Adipose Tissue',
        ],
        'list_component_matches': ['Visceral Adipose Tissue'],
        'description': 'Visceral adipose depot (strict — only explicit Visceral label)',
    },

    # Comparison 2: White vs Brown (adipose cell type)
    'white': {
        'exact_matches': ['White Adipose Tissue'],
        'list_component_matches': ['White Adipose Tissue'],
        'description': 'White adipose tissue (strict — only "White Adipose Tissue")',
    },
    'brown': {
        'exact_matches': ['Brown Adipose Tissue'],
        'list_component_matches': ['Brown Adipose Tissue'],
        'description': 'Brown adipose tissue (strict)',
    },
}

RQ2_COMPARISONS = [
    {'name': 'subcut_vs_visceral', 'tissue_A': 'subcutaneous', 'tissue_B': 'visceral',
     'description': 'Subcutaneous vs Visceral adipose depot location'},
    {'name': 'white_vs_brown', 'tissue_A': 'white', 'tissue_B': 'brown',
     'description': 'White vs Brown adipose type (strict matching)'},
]

# Tissues that don't map to any group (for documentation)
AMBIGUOUS_TISSUES = {
    'Adipocytes': 'Ambiguous — could be white or brown adipocytes',
    'Obese Adipose Tissue': 'Ambiguous — could be any depot',
    'Adipose Tissue': 'Generic — no depot or type specified',
    'Epididymal Adipose Tissue': 'Mouse visceral depot — excluded from strict matching',
    'Epididymal White Adipose Tissue': 'Mouse WAT — excluded from strict white matching',
    'Omental Adipose Tissue': 'Human visceral depot — excluded from strict matching',
}


# ============================================================================
# Tissue Matching
# ============================================================================

def matches_tissue_group(detailed_tissue: Any, group_name: str) -> bool:
    """
    Check if a detailed_tissue value matches a tissue group.

    Args:
        detailed_tissue: A string, a list of strings, None, or 'Not specified'.
        group_name: One of 'subcutaneous', 'visceral', 'white', 'brown'.

    Returns:
        True if the value matches the group.
    """
    if group_name not in RQ2_TISSUE_GROUPS:
        raise ValueError(f"Unknown tissue group: {group_name}")

    group = RQ2_TISSUE_GROUPS[group_name]
    exact = set(group.get('exact_matches', []))
    list_components = set(group.get('list_component_matches', []))

    if detailed_tissue is None:
        return False
    if isinstance(detailed_tissue, str):
        if detailed_tissue.lower() in ('not specified', ''):
            return False
        return detailed_tissue in exact
    if isinstance(detailed_tissue, list):
        return any(t in list_components for t in detailed_tissue)

    return False


def compute_tissue_coverage(edges_data: List[dict], group_name: str) -> float:
    """Fraction of edges with evidence from the given tissue group."""
    if not edges_data:
        return 0.0
    count = sum(1 for e in edges_data
                if matches_tissue_group(e.get('detailed_tissue'), group_name))
    return count / len(edges_data)


def compute_all_tissue_coverages(edges_data: List[dict]) -> Dict[str, float]:
    """Compute coverage for every RQ2 tissue group."""
    return {g: compute_tissue_coverage(edges_data, g) for g in RQ2_TISSUE_GROUPS}


# ============================================================================
# Coverage Propagation Along Paths
# ============================================================================

def propagate_coverage_geometric_mean(coverages: List[float]) -> float:
    """Geometric mean of edge coverages. Returns 0 if any edge has zero coverage."""
    if not coverages:
        return 0.0
    arr = np.array(coverages)
    if np.any(arr == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(arr))))


def propagate_coverage_min(coverages: List[float]) -> float:
    """Bottleneck: path coverage = weakest edge."""
    return min(coverages) if coverages else 0.0


def propagate_coverage_product(coverages: List[float]) -> float:
    """Joint probability — conservative for long paths."""
    return float(np.prod(coverages)) if coverages else 0.0


def propagate_coverage(coverages: List[float],
                       method: str = 'geometric_mean') -> float:
    """Propagate coverage along a path using the specified method."""
    dispatch = {
        'geometric_mean': propagate_coverage_geometric_mean,
        'min': propagate_coverage_min,
        'product': propagate_coverage_product,
    }
    if method not in dispatch:
        raise ValueError(f"Unknown propagation method: {method}")
    return dispatch[method](coverages)


# ============================================================================
# Analysis Utilities
# ============================================================================

def analyze_detailed_tissue_distribution(edges_data: List[dict]) -> Dict:
    """
    Summarise the detailed_tissue distribution across edges.

    Returns dict with 'counts', 'group_counts', 'unmapped', 'total_edges'.
    """
    tissue_counts: Counter = Counter()
    group_counts = {g: 0 for g in RQ2_TISSUE_GROUPS}
    unmapped: Counter = Counter()

    for edge in edges_data:
        detailed = edge.get('detailed_tissue')
        detailed_str = str(detailed) if isinstance(detailed, list) else (str(detailed) if detailed else 'Not specified')
        tissue_counts[detailed_str] += 1

        matched_any = False
        for group_name in RQ2_TISSUE_GROUPS:
            if matches_tissue_group(detailed, group_name):
                group_counts[group_name] += 1
                matched_any = True

        if not matched_any and detailed_str not in ('Not specified', 'None', ''):
            unmapped[detailed_str] += 1

    return {
        'counts': tissue_counts,
        'group_counts': group_counts,
        'unmapped': unmapped,
        'total_edges': len(edges_data),
    }


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == '__main__':
    test_cases = [
        ('Subcutaneous Adipose Tissue', ['subcutaneous']),
        ('Visceral Adipose Tissue', ['visceral']),
        ('White Adipose Tissue', ['white']),
        ('Brown Adipose Tissue', ['brown']),
        ('Subcutaneous Abdominal Adipose Tissue', ['subcutaneous']),
        ('Visceral White Adipose Tissue', ['visceral']),
        (['Subcutaneous Adipose Tissue', 'Visceral Adipose Tissue'],
         ['subcutaneous', 'visceral']),
        (['Brown Adipose Tissue', 'White Adipose Tissue'], ['white', 'brown']),
        ('Epididymal Adipose Tissue', []),
        ('Not specified', []),
        (None, []),
    ]

    all_passed = True
    for value, expected in test_cases:
        actual = [g for g in RQ2_TISSUE_GROUPS if matches_tissue_group(value, g)]
        passed = set(actual) == set(expected)
        if not passed:
            all_passed = False
        status = "✓" if passed else "✗"
        print(f"  {status} {str(value)[:55]:55s} -> {actual}")

    print(f"\nAll tests passed: {all_passed}")