#!/usr/bin/env python3
"""
RQ2 PSR Inference — Run inference with tissue coverage propagation.

Features:
- No type constraints on intermediates (discovers all metapaths)
- Target must be a Disease node
- Coverage propagated via geometric mean along paths (sequential edges),
  arithmetic mean across paths (independent alternatives)
- Collects LDA topic IDs from edges for downstream enrichment
- Saves inference subgraph for visualization

Usage:
    python run_psr.py --tissue subcutaneous --hops 2 --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph

from utils import load_config, get_node_name, get_node_type, extract_metapath, EPS
from tissue_mapping import propagate_coverage


def _collect_topic_ids_split(edge_data: dict) -> Dict[str, Dict[str, List[int]]]:
    """
    Extract topic ID lists from an edge, split into tissue-matching (foreground)
    and non-matching (background) sets.

    Uses the _list_tissue / _list_other attributes set by aggregate.py.
    Falls back to the flat _list attribute if split is absent.
    Skips topic_id = -1 (unclassified).
    """
    result = {}
    for field in ['mechanisms', 'pathways']:
        all_key = f'{field}_topic_id_list'
        tissue_key = f'{field}_topic_id_list_tissue'
        other_key = f'{field}_topic_id_list_other'
        scalar_key = f'{field}_topic_id'

        def _parse_list(val):
            if val and isinstance(val, list):
                return [int(v) for v in val if v is not None and int(v) != -1]
            return []

        all_ids = _parse_list(edge_data.get(all_key))
        if not all_ids:
            scalar = edge_data.get(scalar_key)
            if scalar is not None and int(scalar) != -1:
                all_ids = [int(scalar)]

        result[field] = {
            'tissue': _parse_list(edge_data.get(tissue_key)),
            'other': _parse_list(edge_data.get(other_key)),
            'all': all_ids,
        }
    return result


def _build_adjacency(kg: KnowledgeGraph) -> Dict:
    """Build outgoing adjacency list, expanding undirected edges both ways."""
    outgoing = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        outgoing[u].append((v, data))
        if data.get('direction', '0') == '0':
            outgoing[v].append((u, data))
    return outgoing


def _accumulate_topics(mp_entry: dict, edge_topics_list: list, edges: list):
    """Accumulate topic counts and edge tallies into a metapath_topics entry."""
    for field in ['mechanisms', 'pathways']:
        for et in edge_topics_list:
            mp_entry['foreground'][field].update(et[field]['tissue'])
            mp_entry['background'][field].update(et[field]['other'])
    for e in edges:
        if e.get('coverage', 0.0) > 0:
            mp_entry['n_fg_edges'] += 1
        else:
            mp_entry['n_bg_edges'] += 1


def find_two_hop_paths(kg: KnowledgeGraph, config: dict) -> Tuple[List[dict], Dict]:
    """Find all 2-hop paths: Source → Intermediate → Target(Disease)."""
    target_types = set(config['psr_params'].get('target_types', ['Disease']))
    min_prob = config['psr_params'].get('min_path_probability', 0.001)
    prop_method = config['coverage'].get('propagation_method', 'geometric_mean')

    print(f"Finding 2-hop paths to {target_types}...")
    outgoing = _build_adjacency(kg)
    target_nodes = {n for n in kg.nodes() if get_node_type(kg, n) in target_types}
    print(f"  Target nodes: {len(target_nodes):,}")

    paths = []
    metapath_topics = defaultdict(lambda: {
        'foreground': {'mechanisms': Counter(), 'pathways': Counter()},
        'background': {'mechanisms': Counter(), 'pathways': Counter()},
        'n_fg_edges': 0, 'n_bg_edges': 0,
    })
    nodes_checked = 0

    for source in kg.nodes():
        if get_node_type(kg, source) in target_types:
            continue
        nodes_checked += 1
        if nodes_checked % 5000 == 0:
            print(f"  Checked {nodes_checked:,} nodes, found {len(paths):,} paths...")

        for intermediate, e1 in outgoing[source]:
            if intermediate == source or get_node_type(kg, intermediate) in target_types:
                continue
            e1_prob = e1.get('probability', 0.5)
            e1_cov = e1.get('coverage', 0.0)

            for target, e2 in outgoing[intermediate]:
                if target not in target_nodes or target in (source, intermediate):
                    continue
                e2_prob = e2.get('probability', 0.5)
                path_prob = e1_prob * e2_prob
                if path_prob < min_prob:
                    continue

                e2_cov = e2.get('coverage', 0.0)
                path_cov = propagate_coverage([e1_cov, e2_cov], prop_method)

                e1_ev = e1.get('evidence_score', -np.log(1 - e1_prob + EPS))
                e2_ev = e2.get('evidence_score', -np.log(1 - e2_prob + EPS))
                c1 = e1.get('correlation_type', 0)
                c2 = e2.get('correlation_type', 0)

                node_types = [get_node_type(kg, n) for n in [source, intermediate, target]]
                edge_types = [e1.get('type', 'Unknown'), e2.get('type', 'Unknown')]
                metapath = extract_metapath(edge_types, node_types)

                # Accumulate topic counts
                e1_topics = _collect_topic_ids_split(e1)
                e2_topics = _collect_topic_ids_split(e2)
                _accumulate_topics(
                    metapath_topics[(source, target, metapath)],
                    [e1_topics, e2_topics], [e1, e2])

                paths.append({
                    'source': source, 'target': target,
                    'path_probability': path_prob,
                    'path_evidence': e1_ev * e2_ev,
                    'path_coverage': path_cov,
                    'path_correlation': c1 * c2 if c1 and c2 else 0,
                    'edge_types': edge_types, 'node_types': node_types,
                    'metapath': metapath, 'intermediates': [intermediate],
                })

    print(f"  Found {len(paths):,} 2-hop paths")
    return paths, dict(metapath_topics)


def find_three_hop_paths(kg: KnowledgeGraph, config: dict) -> Tuple[List[dict], Dict]:
    """Find all 3-hop paths: Source → Int1 → Int2 → Target(Disease)."""
    target_types = set(config['psr_params'].get('target_types', ['Disease']))
    min_prob = config['psr_params'].get('min_path_probability', 0.001)
    prop_method = config['coverage'].get('propagation_method', 'geometric_mean')

    print(f"Finding 3-hop paths to {target_types}...")
    outgoing = _build_adjacency(kg)
    target_nodes = {n for n in kg.nodes() if get_node_type(kg, n) in target_types}
    print(f"  Target nodes: {len(target_nodes):,}")

    paths = []
    metapath_topics = defaultdict(lambda: {
        'foreground': {'mechanisms': Counter(), 'pathways': Counter()},
        'background': {'mechanisms': Counter(), 'pathways': Counter()},
        'n_fg_edges': 0, 'n_bg_edges': 0,
    })
    nodes_checked = 0

    for source in kg.nodes():
        if get_node_type(kg, source) in target_types:
            continue
        nodes_checked += 1
        if nodes_checked % 2000 == 0:
            print(f"  Checked {nodes_checked:,} nodes, found {len(paths):,} paths...")

        for int1, e1 in outgoing[source]:
            if int1 == source or get_node_type(kg, int1) in target_types:
                continue
            e1_prob = e1.get('probability', 0.5)

            for int2, e2 in outgoing[int1]:
                if int2 in (source, int1) or get_node_type(kg, int2) in target_types:
                    continue
                e2_prob = e2.get('probability', 0.5)
                if e1_prob * e2_prob < min_prob:
                    continue

                for target, e3 in outgoing[int2]:
                    if target not in target_nodes or target in (source, int1, int2):
                        continue
                    e3_prob = e3.get('probability', 0.5)
                    path_prob = e1_prob * e2_prob * e3_prob
                    if path_prob < min_prob:
                        continue

                    edge_list = [e1, e2, e3]
                    covs = [e.get('coverage', 0.0) for e in edge_list]
                    path_cov = propagate_coverage(covs, prop_method)
                    evs = [e.get('evidence_score', -np.log(1 - e.get('probability', 0.5) + EPS))
                           for e in edge_list]
                    corrs = [e.get('correlation_type', 0) for e in edge_list]

                    node_types = [get_node_type(kg, n)
                                  for n in [source, int1, int2, target]]
                    edge_types = [e.get('type', 'Unknown') for e in edge_list]
                    metapath = extract_metapath(edge_types, node_types)

                    e_topics = [_collect_topic_ids_split(e) for e in edge_list]
                    _accumulate_topics(
                        metapath_topics[(source, target, metapath)],
                        e_topics, edge_list)

                    paths.append({
                        'source': source, 'target': target,
                        'path_probability': path_prob,
                        'path_evidence': np.prod(evs),
                        'path_coverage': path_cov,
                        'path_correlation': int(np.prod(corrs)) if all(corrs) else 0,
                        'edge_types': edge_types, 'node_types': node_types,
                        'metapath': metapath,
                        'intermediates': [int1, int2],
                        'intermediate_B': int1, 'intermediate_C': int2,
                    })

    print(f"  Found {len(paths):,} 3-hop paths")
    return paths, dict(metapath_topics)


def aggregate_paths(paths: List[dict], kg: KnowledgeGraph) -> pd.DataFrame:
    """
    Aggregate paths into results grouped by (source, target, metapath).

    Aggregation rules:
    - Probability: PSR formula across paths
    - Evidence: sum across paths
    - Coverage: arithmetic mean across paths (independent alternatives)
    - Correlation: evidence-weighted majority vote
    """
    if not paths:
        return pd.DataFrame()

    print(f"\nAggregating {len(paths):,} paths...")
    is_three_hop = 'intermediate_B' in paths[0]

    grouped = defaultdict(list)
    for p in paths:
        grouped[(p['source'], p['target'], p['metapath'])].append(p)
    print(f"  Groups: {len(grouped):,}")

    results = []
    for (source, target, metapath), plist in grouped.items():
        probs = [p['path_probability'] for p in plist]
        agg_prob = 1 - np.prod([1 - p for p in probs])
        agg_ev = sum(p['path_evidence'] for p in plist)
        agg_cov = float(np.mean([p['path_coverage'] for p in plist]))

        # Correlation: evidence-weighted majority vote
        corr_weights = defaultdict(float)
        for p in plist:
            corr_weights[p['path_correlation']] += p['path_evidence']
        agg_corr = max(corr_weights, key=corr_weights.get) if corr_weights else 0

        result = {
            'source_gene': get_node_name(kg, source),
            'source_gene_id': str(source),
            'target_phenotype': get_node_name(kg, target),
            'target_id': str(target),
            'metapath': metapath,
            'probability': agg_prob,
            'evidence_score': agg_ev,
            'coverage': agg_cov,
            'correlation_type': int(agg_corr),
            'num_paths': len(plist),
            'relationship_types': plist[0]['edge_types'],
        }

        if is_three_hop:
            ints_B = list({p['intermediate_B'] for p in plist})
            ints_C = list({p['intermediate_C'] for p in plist})
            result['n_intermediates_B'] = len(ints_B)
            result['n_intermediates_C'] = len(ints_C)
            result['intermediate_genes_B'] = [get_node_name(kg, i) for i in ints_B[:50]]
            result['intermediate_genes_C'] = [get_node_name(kg, i) for i in ints_C[:50]]
            result['num_intermediates'] = len(ints_B) + len(ints_C)
            result['intermediate_genes'] = result['intermediate_genes_B'] + result['intermediate_genes_C']
        else:
            intermediates = list({i for p in plist for i in p['intermediates']})
            result['num_intermediates'] = len(intermediates)
            result['intermediate_genes'] = [get_node_name(kg, i) for i in intermediates[:50]]

        results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values(['probability', 'coverage', 'evidence_score'],
                        ascending=[False, False, False])
    df['rank'] = range(1, len(df) + 1)
    print(f"  Results: {len(df):,} rows")
    return df


def save_inference_subgraph(paths: List[dict], kg: KnowledgeGraph,
                            output_path: Path):
    """Save the subgraph of nodes/edges used in valid inference paths."""
    used_nodes = set()
    for p in paths:
        used_nodes.update([p['source'], p['target']] + p['intermediates'])

    sub = KnowledgeGraph()
    for node in used_nodes:
        sub.add_node(node, **kg.nodes[node])
    for u, v, data in kg.edges(data=True):
        if u in used_nodes and v in used_nodes:
            sub.add_edge(u, v, **data)

    sub.export_graph(str(output_path))
    print(f"  Saved inference subgraph: {len(sub.nodes()):,} nodes, "
          f"{len(sub.edges()):,} edges -> {output_path.name}")


def save_metapath_topics(metapath_topics: Dict, output_path: Path,
                         kg: KnowledgeGraph = None):
    """Save per-(source, target, metapath) topic counts for enrichment."""
    SEP = '|||'
    serializable = {}
    for key, data in metapath_topics.items():
        if isinstance(key, tuple) and len(key) == 3:
            src_id, tgt_id, mp = key
            src = get_node_name(kg, src_id) if kg else str(src_id)
            tgt = get_node_name(kg, tgt_id) if kg else str(tgt_id)
            str_key = SEP.join([src, tgt, mp])
        else:
            str_key = str(key)

        serializable[str_key] = {
            'foreground': {
                f: {str(k): v for k, v in data['foreground'][f].items()}
                for f in ['mechanisms', 'pathways']
            },
            'background': {
                f: {str(k): v for k, v in data['background'][f].items()}
                for f in ['mechanisms', 'pathways']
            },
            'n_fg_edges': data['n_fg_edges'],
            'n_bg_edges': data['n_bg_edges'],
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved topic counts: {output_path} ({len(serializable):,} triples)")


def main():
    parser = argparse.ArgumentParser(description='Run PSR inference')
    parser.add_argument('--tissue', type=str, required=True,
                        choices=['subcutaneous', 'visceral', 'white', 'brown'])
    parser.add_argument('--hops', type=int, required=True, choices=[2, 3])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])

    input_path = output_dir / 'aggregated' / f'aggregated_{args.tissue}.pkl'
    print(f"Loading: {input_path}")
    kg = KnowledgeGraph.import_graph(str(input_path))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    start = datetime.now()
    if args.hops == 2:
        paths, metapath_topics = find_two_hop_paths(kg, config)
    else:
        paths, metapath_topics = find_three_hop_paths(kg, config)

    df = aggregate_paths(paths, kg)
    df['hop_length'] = args.hops

    # Save results
    inference_dir = output_dir / 'inference'
    inference_dir.mkdir(parents=True, exist_ok=True)

    output_path = inference_dir / f'{args.tissue}_{args.hops}hop.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    save_inference_subgraph(paths, kg,
                            inference_dir / f'{args.tissue}_{args.hops}hop_subgraph.pkl')
    save_metapath_topics(metapath_topics,
                         inference_dir / f'{args.tissue}_{args.hops}hop_metapath_topics.json',
                         kg=kg)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"✓ Completed in {elapsed:.1f}s")

    stats = {
        'tissue': args.tissue, 'hops': args.hops,
        'n_paths': len(paths), 'n_results': len(df),
        'n_metapaths_with_topics': len(metapath_topics),
        'elapsed_seconds': elapsed,
    }
    with open(inference_dir / f'{args.tissue}_{args.hops}hop_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()