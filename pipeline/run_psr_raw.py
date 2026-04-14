#!/usr/bin/env python3
"""
RQ2 PSR Inference (Raw) — Full inference from cleaned graph, no pre-aggregation.

Unlike the standard pipeline (aggregate.py -> run_psr.py), this script:
  - Builds a SupportGraph directly from the cleaned (pre-aggregation) graph
  - ALL edge types between a node pair contribute to arc probability
    (no best-type selection that discards evidence)
  - Coverage is computed inline from raw edge tissue labels
  - Edge attributes (topics, evidence, correlation) aggregated on the fly

This produces the same output schema as run_psr.py, so downstream scripts
(compare.py, enrichment.py, run_permutation.py) work unchanged.

Usage:
    python run_psr_raw.py --tissue subcutaneous --hops 2 --method bdd_exact --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph

from utils import load_config, get_node_name, get_node_type, extract_metapath, EPS
from tissue_mapping import (
    matches_tissue_group, propagate_coverage, RQ2_TISSUE_GROUPS,
)
from ContextAwareKGReasoning.rq2.psr_multihop_methods import (
    SupportGraph, build_support_graph,
    compute_bdd_probability_for_paths,
    estimate_probability_mc_for_path_groups,
    _log1m, _BDD_AVAILABLE,
)


# Attributes to collect as topic lists (mirrors aggregate.py)
TOPIC_FIELDS = ['mechanisms_topic_id', 'pathways_topic_id']


# ---------------------------------------------------------------------------
# Arc metadata: per-arc attribute summary from raw edges
# ---------------------------------------------------------------------------
def build_arc_metadata(
    kg: KnowledgeGraph,
    tissue_name: str,
    consider_undirected: bool = True,
) -> Dict[Tuple, dict]:
    """
    For each arc (u,v) implied by the raw graph, aggregate edge attributes
    from all constituent raw edges.

    Unlike aggregate.py, this does NOT select a best type per node pair —
    it summarises attributes across all edges for downstream use (coverage,
    evidence, metapath labelling, topics).

    Undirected edges (direction='0') contribute to both arc (u,v) and (v,u)
    to match the behaviour of _build_adjacency / build_support_graph.

    Returns:
        {(u,v): {coverage, evidence_score, best_type, correlation_type,
                 n_edges, topics_foreground, topics_background}}
    """
    arc_edges: Dict[Tuple, List[dict]] = defaultdict(list)

    for u, v, data in kg.edges(data=True):
        direction = str(data.get('direction', '0'))
        if direction == '0':
            if consider_undirected:
                arc_edges[(u, v)].append(data)
                if u != v:
                    arc_edges[(v, u)].append(data)
        else:
            arc_edges[(u, v)].append(data)

    arc_meta = {}
    for (u, v), edges in arc_edges.items():
        n_total = len(edges)

        # Coverage: fraction of edges matching the tissue group
        n_tissue = sum(
            1 for e in edges
            if matches_tissue_group(e.get('detailed_tissue'), tissue_name)
        )
        coverage = n_tissue / n_total if n_total else 0.0

        # Evidence score: sum of -log(1-p+eps) over all edges
        evidence = sum(
            -np.log(1 - float(e.get('probability', 0.5)) + EPS)
            for e in edges
        )

        # Best edge type for metapath labelling (by max probability, then count)
        type_max_prob = defaultdict(float)
        type_counts = Counter()
        for e in edges:
            etype = e.get('type', 'Unknown')
            type_counts[etype] += 1
            type_max_prob[etype] = max(
                type_max_prob[etype], float(e.get('probability', 0.5)))
        best_type = max(
            type_counts,
            key=lambda t: (type_max_prob[t], type_counts[t]),
        )

        # Correlation: majority vote
        corr_counts = Counter(e.get('correlation_type', 0) for e in edges)
        correlation = corr_counts.most_common(1)[0][0] if corr_counts else 0

        # Topic IDs: split into tissue (foreground) and other (background)
        topics_fg = {'mechanisms': Counter(), 'pathways': Counter()}
        topics_bg = {'mechanisms': Counter(), 'pathways': Counter()}
        for e in edges:
            is_tissue = matches_tissue_group(
                e.get('detailed_tissue'), tissue_name)
            dest = topics_fg if is_tissue else topics_bg
            for field_base in ['mechanisms', 'pathways']:
                # Try list attribute first (from LDA), then scalar
                list_key = f'{field_base}_topic_id'
                val = e.get(list_key)
                if val is not None:
                    vals = val if isinstance(val, list) else [val]
                    for v_ in vals:
                        if v_ is not None and int(v_) != -1:
                            dest[field_base][int(v_)] += 1

        arc_meta[(u, v)] = {
            'coverage': coverage,
            'evidence_score': evidence,
            'best_type': best_type,
            'all_types': list(type_counts.keys()),
            'correlation_type': int(correlation),
            'n_edges': n_total,
            'n_edges_tissue': n_tissue,
            'topics_fg': topics_fg,
            'topics_bg': topics_bg,
        }

    return arc_meta


# ---------------------------------------------------------------------------
# Path enumeration on SupportGraph with attribute lookup
# ---------------------------------------------------------------------------
def find_paths_on_support_graph(
    sg: SupportGraph,
    kg: KnowledgeGraph,
    arc_meta: Dict[Tuple, dict],
    hops: int,
    tissue_name: str,
    config: dict,
) -> Tuple[List[dict], dict]:
    """
    Enumerate k-hop paths on the SupportGraph, collecting edge attributes
    from arc_meta for coverage, evidence, metapath, and topic counts.

    Filters:
      - Target must be a Disease node
      - Simple paths only (no repeated nodes)
      - min_path_probability threshold

    Returns (paths, metapath_topics) in the same format as run_psr.py.
    """
    target_types = set(config['psr_params'].get('target_types', ['Disease']))
    min_prob = config['psr_params'].get('min_path_probability', 0.001)
    prop_method = config['coverage'].get('propagation_method', 'geometric_mean')

    # Identify target nodes
    target_nodes = {
        n for n in kg.nodes() if get_node_type(kg, n) in target_types
    }

    print(f"Finding {hops}-hop paths (raw pipeline) to {target_types}...")
    print(f"  Target nodes: {len(target_nodes):,}")
    print(f"  SupportGraph arcs: {len(sg.supports):,}")

    # All nodes in the SupportGraph
    all_nodes = set(sg.out.keys()) | set(sg.inn.keys())
    source_nodes = [n for n in all_nodes if get_node_type(kg, n) not in target_types]

    paths = []
    metapath_topics = defaultdict(lambda: {
        'foreground': {'mechanisms': Counter(), 'pathways': Counter()},
        'background': {'mechanisms': Counter(), 'pathways': Counter()},
        'n_fg_edges': 0, 'n_bg_edges': 0,
    })

    def _arc_attr(u, v):
        """Look up arc metadata, with fallback defaults."""
        return arc_meta.get((u, v), {
            'coverage': 0.0, 'evidence_score': 0.0,
            'best_type': 'Unknown', 'correlation_type': 0,
            'n_edges': 0, 'n_edges_tissue': 0,
            'topics_fg': {'mechanisms': Counter(), 'pathways': Counter()},
            'topics_bg': {'mechanisms': Counter(), 'pathways': Counter()},
        })

    def _process_path(node_seq: List):
        """Build a path dict from a node sequence, looking up arc metadata."""
        k = len(node_seq) - 1  # number of hops
        arcs = [(node_seq[i], node_seq[i + 1]) for i in range(k)]
        arc_attrs = [_arc_attr(u, v) for u, v in arcs]

        # Path probability from SupportGraph arc probabilities
        path_prob = 1.0
        for u, v in arcs:
            path_prob *= sg.p_arc.get((u, v), 0.0)
        if path_prob < min_prob:
            return None

        # Coverage: geometric mean along path
        coverages = [a['coverage'] for a in arc_attrs]
        path_cov = propagate_coverage(coverages, prop_method)

        # Evidence: product along path (same as run_psr.py)
        evs = [a['evidence_score'] for a in arc_attrs]
        path_evidence = np.prod(evs)

        # Correlation: product of arc correlations
        corrs = [a['correlation_type'] for a in arc_attrs]
        path_corr = int(np.prod(corrs)) if all(corrs) else 0

        # Metapath from node types + best edge types
        node_types = [get_node_type(kg, n) for n in node_seq]
        edge_types = [a['best_type'] for a in arc_attrs]
        metapath = extract_metapath(edge_types, node_types)

        source = node_seq[0]
        target = node_seq[-1]

        # Accumulate topic counts
        mp_key = (source, target, metapath)
        mp_entry = metapath_topics[mp_key]
        for a in arc_attrs:
            for field in ['mechanisms', 'pathways']:
                mp_entry['foreground'][field].update(a['topics_fg'][field])
                mp_entry['background'][field].update(a['topics_bg'][field])
        for a in arc_attrs:
            if a['coverage'] > 0:
                mp_entry['n_fg_edges'] += 1
            else:
                mp_entry['n_bg_edges'] += 1

        path_dict = {
            'source': source,
            'target': target,
            'path_probability': path_prob,
            'path_evidence': path_evidence,
            'path_coverage': path_cov,
            'path_correlation': path_corr,
            'edge_types': edge_types,
            'node_types': node_types,
            'metapath': metapath,
            'node_path': list(node_seq),
        }

        # Intermediate info
        intermediates = list(node_seq[1:-1])
        path_dict['intermediates'] = intermediates
        if k == 3:
            path_dict['intermediate_B'] = intermediates[0]
            path_dict['intermediate_C'] = intermediates[1]

        return path_dict

    # --- DFS path enumeration ---
    nodes_checked = 0
    log_interval = 5000 if hops == 2 else 2000

    for source in source_nodes:
        nodes_checked += 1
        if nodes_checked % log_interval == 0:
            print(f"  Checked {nodes_checked:,} / {len(source_nodes):,} sources, "
                  f"found {len(paths):,} paths...")

        if hops == 2:
            # Source -> B -> Target
            for B, _vids_sb, p_sb in sg.out.get(source, []):
                if B == source or B in target_nodes:
                    continue
                for target, _vids_bt, p_bt in sg.out.get(B, []):
                    if target not in target_nodes or target in (source, B):
                        continue
                    result = _process_path([source, B, target])
                    if result is not None:
                        paths.append(result)

        elif hops == 3:
            # Source -> B -> C -> Target
            for B, _vids_sb, p_sb in sg.out.get(source, []):
                if B == source or B in target_nodes:
                    continue
                for C, _vids_bc, p_bc in sg.out.get(B, []):
                    if C in (source, B) or C in target_nodes:
                        continue
                    # Early pruning
                    if p_sb * p_bc < min_prob:
                        continue
                    for target, _vids_ct, p_ct in sg.out.get(C, []):
                        if target not in target_nodes or target in (source, B, C):
                            continue
                        result = _process_path([source, B, C, target])
                        if result is not None:
                            paths.append(result)

    print(f"  Found {len(paths):,} {hops}-hop paths")
    return paths, dict(metapath_topics)


# ---------------------------------------------------------------------------
# Aggregation — reuses logic from run_psr.py
# ---------------------------------------------------------------------------
def _compute_path_noisy_or(path_probs: List[float]) -> float:
    if not path_probs:
        return 0.0
    sum_log = sum(_log1m(p) for p in path_probs)
    return float(-np.expm1(sum_log))


def compute_group_probabilities(
    grouped: dict, method: str, sg: SupportGraph = None,
    mc_n_samples: int = 5000, mc_seed: int = 42, mc_alpha: float = 0.05,
) -> dict:
    """Compute probability per (source, target, metapath) group."""
    results = {}

    if method == 'path_noisy_or':
        for key, plist in grouped.items():
            probs = [p['path_probability'] for p in plist]
            results[key] = {'probability': _compute_path_noisy_or(probs)}

    elif method == 'bdd_exact':
        if not _BDD_AVAILABLE:
            raise RuntimeError("BDD not available. pip install dd")
        n_groups = len(grouped)
        for idx, (key, plist) in enumerate(grouped.items()):
            if (idx + 1) % 10000 == 0:
                print(f"    BDD progress: {idx + 1:,} / {n_groups:,}")
            node_paths = [p['node_path'] for p in plist]
            prob = compute_bdd_probability_for_paths(sg, node_paths)
            results[key] = {'probability': prob}

    elif method == 'monte_carlo':
        groups_node_paths = {
            key: [p['node_path'] for p in plist]
            for key, plist in grouped.items()
        }
        print(f"    MC: {mc_n_samples:,} samples, {len(groups_node_paths):,} groups...")
        mc_results = estimate_probability_mc_for_path_groups(
            sg, groups_node_paths,
            n_samples=mc_n_samples, seed=mc_seed, alpha=mc_alpha,
        )
        for key in grouped:
            mc_r = mc_results.get(key, {})
            results[key] = {
                'probability': mc_r.get('probability', 0.0),
                'mc_ci_low': mc_r.get('ci_low'),
                'mc_ci_high': mc_r.get('ci_high'),
            }
    else:
        raise ValueError(f"Unknown method: {method}")

    return results


def aggregate_paths(
    paths: List[dict], kg: KnowledgeGraph,
    method: str = 'path_noisy_or', sg: SupportGraph = None,
    mc_n_samples: int = 5000, mc_seed: int = 42, mc_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aggregate paths into (source, target, metapath) groups.

    Same output schema as run_psr.py's aggregate_paths.
    """
    if not paths:
        return pd.DataFrame()

    print(f"\nAggregating {len(paths):,} paths (method={method})...")
    is_three_hop = 'intermediate_B' in paths[0]

    grouped = defaultdict(list)
    for p in paths:
        grouped[(p['source'], p['target'], p['metapath'])].append(p)
    print(f"  Groups: {len(grouped):,}")

    # Compute probabilities
    if method == 'all':
        probs_pno = compute_group_probabilities(grouped, 'path_noisy_or')
        probs_bdd = {}
        if _BDD_AVAILABLE and sg is not None:
            print("  Computing BDD exact probabilities...")
            probs_bdd = compute_group_probabilities(grouped, 'bdd_exact', sg=sg)
        probs_mc = {}
        if sg is not None:
            print("  Computing Monte Carlo probabilities...")
            probs_mc = compute_group_probabilities(
                grouped, 'monte_carlo', sg=sg,
                mc_n_samples=mc_n_samples, mc_seed=mc_seed, mc_alpha=mc_alpha,
            )
    else:
        probs_primary = compute_group_probabilities(
            grouped, method, sg=sg,
            mc_n_samples=mc_n_samples, mc_seed=mc_seed, mc_alpha=mc_alpha,
        )

    results = []
    for (source, target, metapath), plist in grouped.items():
        key = (source, target, metapath)
        agg_ev = sum(p['path_evidence'] for p in plist)
        agg_cov = float(np.mean([p['path_coverage'] for p in plist]))

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
            'evidence_score': agg_ev,
            'coverage': agg_cov,
            'correlation_type': int(agg_corr),
            'num_paths': len(plist),
            'relationship_types': plist[0]['edge_types'],
        }

        if method == 'all':
            if key in probs_bdd:
                result['probability'] = probs_bdd[key]['probability']
            else:
                result['probability'] = probs_pno[key]['probability']
            result['prob_path_noisy_or'] = probs_pno[key]['probability']
            if key in probs_bdd:
                result['prob_bdd'] = probs_bdd[key]['probability']
            if key in probs_mc:
                result['prob_mc'] = probs_mc[key]['probability']
                result['mc_ci_low'] = probs_mc[key].get('mc_ci_low')
                result['mc_ci_high'] = probs_mc[key].get('mc_ci_high')
            result['method'] = 'bdd_exact' if key in probs_bdd else 'path_noisy_or'
        else:
            result['probability'] = probs_primary[key]['probability']
            result['method'] = method
            if method == 'monte_carlo':
                result['mc_ci_low'] = probs_primary[key].get('mc_ci_low')
                result['mc_ci_high'] = probs_primary[key].get('mc_ci_high')

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


# ---------------------------------------------------------------------------
# Saving helpers (same as run_psr.py)
# ---------------------------------------------------------------------------
def save_inference_subgraph(paths, kg, output_path):
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


def save_metapath_topics(metapath_topics, output_path, kg=None):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run PSR inference directly from raw cleaned graph (no pre-aggregation)')
    parser.add_argument('--tissue', type=str, required=True,
                        choices=['subcutaneous', 'visceral', 'white', 'brown'])
    parser.add_argument('--hops', type=int, required=True, choices=[2, 3])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--method', type=str, default='bdd_exact',
                        choices=['path_noisy_or', 'bdd_exact', 'monte_carlo', 'all'])
    parser.add_argument('--mc-n-samples', type=int, default=5000)
    parser.add_argument('--mc-seed', type=int, default=42)
    parser.add_argument('--mc-alpha', type=float, default=0.05)
    parser.add_argument('--output-suffix', type=str, default='_raw',
                        help='Suffix for output filenames (default: _raw)')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])

    if args.method == 'bdd_exact' and not _BDD_AVAILABLE:
        print("ERROR: --method bdd_exact requires 'dd'. pip install dd")
        sys.exit(1)

    # Load cleaned (pre-aggregation) graph
    input_path = output_dir / 'preprocessed' / 'cleaned_graph.pkl'
    if not input_path.exists():
        print(f"ERROR: Cleaned graph not found at {input_path}")
        print("Run preprocess.py first!")
        sys.exit(1)

    print(f"Loading cleaned graph: {input_path}")
    kg = KnowledgeGraph.import_graph(str(input_path))
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    start = datetime.now()

    # Build SupportGraph from raw graph
    print(f"\nBuilding SupportGraph from raw graph...")
    sg = build_support_graph(
        kg,
        consider_undirected=True,
        base_edges_only=False,
        min_edge_probability=0.0,
    )
    print(f"  Arcs: {len(sg.supports):,}, Variables: {len(sg.p_var):,}")

    # Build arc metadata for attribute lookup
    print(f"Building arc metadata for tissue={args.tissue}...")
    arc_meta = build_arc_metadata(kg, args.tissue, consider_undirected=True)
    print(f"  Arc metadata entries: {len(arc_meta):,}")

    # Enumerate paths
    paths, metapath_topics = find_paths_on_support_graph(
        sg, kg, arc_meta, args.hops, args.tissue, config)

    # Aggregate
    df = aggregate_paths(
        paths, kg,
        method=args.method, sg=sg,
        mc_n_samples=args.mc_n_samples, mc_seed=args.mc_seed,
        mc_alpha=args.mc_alpha,
    )
    df['hop_length'] = args.hops
    df['pipeline'] = 'raw'

    # Save
    inference_dir = output_dir / 'inference'
    inference_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    output_path = inference_dir / f'{args.tissue}_{args.hops}hop{suffix}.parquet'
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")

    save_inference_subgraph(
        paths, kg,
        inference_dir / f'{args.tissue}_{args.hops}hop{suffix}_subgraph.pkl')
    save_metapath_topics(
        metapath_topics,
        inference_dir / f'{args.tissue}_{args.hops}hop{suffix}_metapath_topics.json',
        kg=kg)

    elapsed = (datetime.now() - start).total_seconds()
    print(f"✓ Completed in {elapsed:.1f}s")

    stats = {
        'tissue': args.tissue, 'hops': args.hops,
        'method': args.method, 'pipeline': 'raw',
        'n_paths': len(paths), 'n_results': len(df),
        'n_metapaths_with_topics': len(metapath_topics),
        'elapsed_seconds': elapsed,
        'bdd_available': _BDD_AVAILABLE,
        'sg_arcs': len(sg.supports), 'sg_vars': len(sg.p_var),
        'arc_meta_entries': len(arc_meta),
    }
    if args.method in ('monte_carlo', 'all'):
        stats['mc_n_samples'] = args.mc_n_samples
    with open(inference_dir / f'{args.tissue}_{args.hops}hop{suffix}_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == '__main__':
    main()
