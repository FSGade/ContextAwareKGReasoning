"""
Metapath grouping utilities for PSR inference.
Groups paths by metapath signature before probability aggregation.

Note: This module handles grouping and naming logic only.
Actual probability/evidence aggregation should be done by the calling script
since different aggregation strategies (PSR, noisy-OR, etc.) may be used.
"""

from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional


def create_metapath_signature(
    path_info: Dict[str, Any], 
    strategy: str = 'mechanistic'
) -> Tuple:
    """
    Create metapath signature based on strategy.
    
    Args:
        path_info: Dictionary with path information including:
            - node_types: Tuple of node types
            - relations: Tuple of relation types
            - correlation: Integer correlation (-1, 0, 1)
        strategy: 'mechanistic' or 'semantic'
    
    Returns:
        Tuple that can be used as dict key
        - mechanistic: (node_types, relations)
        - semantic: (node_types, correlation)
    """
    if strategy == 'mechanistic':
        return (path_info['node_types'], path_info['relations'])
    elif strategy == 'semantic':
        return (path_info['node_types'], path_info['correlation'])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def group_paths_by_metapath(
    paths: List[Dict[str, Any]], 
    strategy: str = 'mechanistic',
    split_inconsistent_correlations: bool = True
) -> Dict[Tuple, Dict[str, Any]]:
    """
    Group paths by metapath signature.
    
    Args:
        paths: List of path dictionaries with keys:
            - node_types: Tuple of node types
            - relations: Tuple of relation types
            - correlation: Integer (-1, 0, 1)
            - probability: Float
            - Other metadata
        strategy: 'mechanistic' or 'semantic'
        split_inconsistent_correlations: For mechanistic, split if correlations disagree
    
    Returns:
        Dict mapping metapath_signature -> {
            'paths': list of paths,
            'was_split': bool,
            'correlation': int
        }
    """
    if not paths:
        return {}
    
    groups = defaultdict(list)
    
    # Initial grouping by strategy
    for path_info in paths:
        sig = create_metapath_signature(path_info, strategy)
        groups[sig].append(path_info)
    
    # For mechanistic: validate correlation consistency
    if strategy == 'mechanistic' and split_inconsistent_correlations:
        final_groups = {}
        
        for sig, path_list in groups.items():
            correlations = set(p['correlation'] for p in path_list)
            
            if len(correlations) > 1:
                # Split by correlation - this shouldn't happen but handle it
                print(f"WARNING: Inconsistent correlations {correlations} for metapath")
                print(f"  Node types: {sig[0]}")
                print(f"  Relations: {sig[1]}")
                print(f"  Splitting into {len(correlations)} sub-groups by correlation")
                
                for corr in sorted(correlations):
                    corr_paths = [p for p in path_list if p['correlation'] == corr]
                    # Add correlation to signature to distinguish split groups
                    new_sig = (sig[0], sig[1], corr)
                    final_groups[new_sig] = {
                        'paths': corr_paths,
                        'was_split': True,
                        'correlation': corr
                    }
            else:
                # No split needed
                final_groups[sig] = {
                    'paths': path_list,
                    'was_split': False,
                    'correlation': list(correlations)[0]
                }
        
        return final_groups
    
    # For semantic or no splitting
    result = {}
    for sig, path_list in groups.items():
        correlation = path_list[0]['correlation']
        result[sig] = {
            'paths': path_list,
            'was_split': False,
            'correlation': correlation
        }
    
    return result


def count_unique_intermediates(paths: List[Dict], path_length: int) -> Dict[str, Any]:
    """
    Count unique intermediate nodes.
    
    For 2-hop: All intermediates in a metapath group are same type
    For 3-hop: Position B has one type, position C has one type (can differ)
    
    Args:
        paths: List of path dictionaries
        path_length: 2 or 3
    
    Returns:
        Dict with counts and details about intermediates
    """
    if path_length == 2:
        # One intermediate position - all same type within metapath group
        unique_intermediates = set()
        
        for p in paths:
            intermediate_node = p['intermediate']  # The actual node tuple
            unique_intermediates.add(intermediate_node)
        
        # Get the type (same for all in this metapath group)
        intermediate_type = paths[0]['node_types'][1]
        
        return {
            'unique_intermediates': list(unique_intermediates),
            'n_unique_intermediates': len(unique_intermediates),
            'intermediate_type': intermediate_type,
            'intermediate_position': 'single'
        }
    
    elif path_length == 3:
        # Two intermediate positions: B and C
        unique_B_intermediates = set()
        unique_C_intermediates = set()
        
        for p in paths:
            B_node = p['path'][1]
            C_node = p['path'][2]
            
            unique_B_intermediates.add(B_node)
            unique_C_intermediates.add(C_node)
        
        # Get types (same for all in this metapath group at each position)
        B_type = paths[0]['node_types'][1]
        C_type = paths[0]['node_types'][2]
        
        return {
            'unique_intermediates_B': list(unique_B_intermediates),
            'unique_intermediates_C': list(unique_C_intermediates),
            'n_unique_intermediates_B': len(unique_B_intermediates),
            'n_unique_intermediates_C': len(unique_C_intermediates),
            'intermediate_B_type': B_type,
            'intermediate_C_type': C_type,
            'intermediate_position': 'B_and_C'
        }
    
    else:
        raise ValueError(f"Unsupported path_length: {path_length}")


def create_metapath_edge_name(
    metapath_sig: Tuple,
    paths: List[Dict],
    path_length: int,
    was_split: bool = False
) -> str:
    """
    Create compact, descriptive edge name from metapath.
    
    Format 2-hop: {rel1}_{rel2}_{n_unique}_{type}[s][_split]
    Format 3-hop: {rel1}_{rel2}_{rel3}_{n_B}_{type_B}[s]_{n_C}_{type_C}[s][_split]
    
    Examples:
        2-hop: "suppression_upregulation_3_Genes"
        3-hop: "suppression_exhibition_association_4_Genes_2_CellLines"
        split: "downregulation_positive_correlation_2_Genes_split"
    
    Args:
        metapath_sig: Metapath signature tuple
        paths: List of paths in this metapath group
        path_length: 2 or 3
        was_split: Whether this group resulted from correlation splitting
    
    Returns:
        String edge name
    """
    # Extract components from signature
    if len(metapath_sig) == 2:
        node_types, relation_types = metapath_sig
    elif len(metapath_sig) == 3:
        # Was split - has correlation in signature
        node_types, relation_types, correlation = metapath_sig
    else:
        raise ValueError(f"Invalid metapath signature: {metapath_sig}")
    
    # Build relation chain (compact, no linker words)
    relation_chain = "_".join(rel.lower() for rel in relation_types)
    
    # Count unique intermediates
    intermediate_info = count_unique_intermediates(paths, path_length)
    
    if path_length == 2:
        # Format: {rel1}_{rel2}_{count}_{type}[s]
        count = intermediate_info['n_unique_intermediates']
        itype = intermediate_info['intermediate_type']
        
        # Add plural 's' if count > 1
        plural = 's' if count > 1 else ''
        
        # Combine
        name = f"{relation_chain}_{count}_{itype}{plural}"
    
    elif path_length == 3:
        # Format: {rel1}_{rel2}_{rel3}_{n_B}_{type_B}[s]_{n_C}_{type_C}[s]
        count_B = intermediate_info['n_unique_intermediates_B']
        type_B = intermediate_info['intermediate_B_type']
        count_C = intermediate_info['n_unique_intermediates_C']
        type_C = intermediate_info['intermediate_C_type']
        
        # Add plural 's' if count > 1
        plural_B = 's' if count_B > 1 else ''
        plural_C = 's' if count_C > 1 else ''
        
        # Combine
        name = f"{relation_chain}_{count_B}_{type_B}{plural_B}_{count_C}_{type_C}{plural_C}"
    
    # Add split suffix if needed
    if was_split:
        name = f"{name}_split"
    
    return name


def create_metapath_edge_attrs(
    source,
    target,
    kg,
    metapath_sig: Tuple,
    paths: List[Dict],
    combined_prob: float,
    combined_evidence: float,
    correlation: int,
    path_length: int,
    was_split: bool = False,
    grouping_strategy: str = 'mechanistic'
) -> Dict[str, Any]:
    """
    Create complete edge attributes for metapath-specific inferred edge.
    
    Note: combined_prob and combined_evidence should be pre-computed by caller
    using their chosen aggregation method (e.g., PSR formula).
    
    Args:
        source: Source node
        target: Target node
        kg: Knowledge graph (for getting node names)
        metapath_sig: Metapath signature tuple
        paths: List of paths in this metapath group
        combined_prob: Pre-aggregated probability (e.g., using PSR formula)
        combined_evidence: Pre-aggregated evidence score
        correlation: Final correlation type
        path_length: 2 or 3
        was_split: Whether this group resulted from correlation splitting
        grouping_strategy: 'mechanistic' or 'semantic'
    
    Returns:
        Dict with all edge attributes to add to graph
    """
    # Create edge name
    edge_name = create_metapath_edge_name(metapath_sig, paths, path_length, was_split)
    
    # Extract intermediate info
    intermediate_info = count_unique_intermediates(paths, path_length)
    
    # Get node types and relations from signature
    if len(metapath_sig) == 2:
        node_types, relation_types = metapath_sig
    else:  # length 3, was split
        node_types, relation_types, _ = metapath_sig
    
    # Base attributes
    attrs = {
        'type': edge_name,
        'kind': f'inferred_{path_length}_hop',
        'metapath_signature': str(metapath_sig),
        'node_type_sequence': list(node_types),
        'relation_sequence': list(relation_types),
        'probability': round(float(combined_prob), 6),
        'evidence_score': round(float(combined_evidence), 4),
        'correlation_type': int(correlation),
        'direction': '1',
        'num_paths': len(paths),
        'was_correlation_split': was_split,
        'aggregated': False,
        'inferred': True,
        'source': f'PSR_{path_length}_hop_metapath',
        'grouping_strategy': grouping_strategy,
    }
    
    # Add intermediate information
    if path_length == 2:
        attrs.update({
            'n_unique_intermediates': intermediate_info['n_unique_intermediates'],
            'intermediate_type': intermediate_info['intermediate_type'],
            # Store first few example intermediate names for reference
            'example_intermediates': [
                kg.nodes[p['intermediate']].get('name', str(p['intermediate']))
                for p in paths[:10]
            ]
        })
    
    elif path_length == 3:
        attrs.update({
            'n_unique_intermediates_B': intermediate_info['n_unique_intermediates_B'],
            'n_unique_intermediates_C': intermediate_info['n_unique_intermediates_C'],
            'intermediate_B_type': intermediate_info['intermediate_B_type'],
            'intermediate_C_type': intermediate_info['intermediate_C_type'],
            # Store first few example pairs for reference
            'example_intermediate_pairs': [
                (
                    kg.nodes[p['path'][1]].get('name', str(p['path'][1])),
                    kg.nodes[p['path'][2]].get('name', str(p['path'][2]))
                )
                for p in paths[:10]
            ]
        })
    
    return attrs