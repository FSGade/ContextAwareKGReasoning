#!/usr/bin/env python3
"""
RQ2 Permutation Worker — Per-Triple Null Distribution (Batched)

Runs MULTIPLE permutations per job to amortize graph-loading cost.
Each job:
  1. Stages the graph from NFS to /home/local (once per node, lockfile)
  2. Loads graph from local disk (one load per job)
  3. Loops over N permutations:
       - Skip if output already exists
       - Shuffle tissue labels (in-place, restore after)
       - Aggregate → Infer → Compare → Save result
  4. Each perm result (~100KB) written to NFS immediately after completion

With perms_per_job=100 and 10,000 total permutations → 100 jobs per comparison.
Fits in a single Slurm array (0-99). Graph loaded 100× instead of 10,000×.
Timeouts are safe: completed perms are already saved, resubmit to continue.

Usage:
    python run_permutation.py \
        --perm-start 1 --perm-count 100 \
        --comparison subcut_vs_visceral --config config.yaml
"""

import argparse
import fcntl
import hashlib
import os
import shutil
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple

import yaml
import numpy as np
import pandas as pd

from utils import load_config, ordered_pair, get_node_name, get_node_type, format_edge_type, EPS

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from knowledge_graph import KnowledgeGraph

# Import from other RQ2 modules
from tissue_mapping import (
    matches_tissue_group,
    propagate_coverage,
    RQ2_TISSUE_GROUPS,
)


# Local disk staging directory
LOCAL_WORK_DIR = Path('/home/local') / os.environ.get('USER', 'unknown') / 'perm_work'


# ---------------------------------------------------------------------------
# Local disk staging
# ---------------------------------------------------------------------------

def stage_to_local(src_path: Path, local_dir: Path) -> Path:
    """
    Copy a file from NFS to local disk if not already present.
    Uses a lockfile so only one task per node copies each file.
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    path_hash = hashlib.md5(str(src_path).encode()).hexdigest()[:8]
    local_path = local_dir / f"{path_hash}_{src_path.name}"
    lock_path = local_dir / f".{path_hash}_{src_path.name}.lock"

    if local_path.exists() and local_path.stat().st_size == src_path.stat().st_size:
        print(f"  Local cache hit: {local_path.name}")
        return local_path

    with open(lock_path, 'w') as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock
            if local_path.exists() and local_path.stat().st_size == src_path.stat().st_size:
                print(f"  Local cache hit (after lock): {local_path.name}")
                return local_path

            print(f"  Staging to local disk: {src_path.name} -> {local_path}")
            shutil.copy2(src_path, local_path)
            print(f"  Staged: {local_path.stat().st_size / 1e6:.1f} MB")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)

    return local_path


# ---------------------------------------------------------------------------
# Focused triple set loading
# ---------------------------------------------------------------------------

def load_focused_triple_keys(
    config: dict,
    comparison_name: str,
    local_dir: Path,
) -> Optional[Dict[int, Set[Tuple[str, str, str]]]]:
    """
    Load observed comparison files and return the set of focused triple keys.
    Reads from local disk if staged, otherwise from NFS.
    """
    perm_config = config.get('permutation', {})
    if not perm_config.get('filter_before_save', False):
        return None

    focus_phenotypes = set(perm_config.get('focus_phenotypes', []))
    if not focus_phenotypes:
        print("  WARNING: filter_before_save=true but no focus_phenotypes in config")
        return None

    min_coverage = perm_config.get('min_coverage_both', 0.0)
    hop_lengths = perm_config.get('hops_for_permutation', [2])
    if isinstance(hop_lengths, int):
        hop_lengths = [hop_lengths]

    focus_keys: Dict[int, Set[Tuple[str, str, str]]] = {}

    for hops in hop_lengths:
        nfs_path = (Path(config['paths']['output_dir']) / 'comparisons' /
                    comparison_name / f'comparison_{hops}hop.parquet')

        # Check local first
        path_hash = hashlib.md5(str(nfs_path).encode()).hexdigest()[:8]
        local_path = local_dir / f"{path_hash}_{nfs_path.name}"

        if local_path.exists():
            comp_path = local_path
        elif nfs_path.exists():
            comp_path = nfs_path
        else:
            print(f"  WARNING: {nfs_path} not found -- skipping filter for {hops}-hop")
            continue

        df = pd.read_parquet(comp_path)
        pheno_mask = df['target_phenotype'].isin(focus_phenotypes)
        cov_mask = (
            (df['coverage_A'].fillna(0) > min_coverage) &
            (df['coverage_B'].fillna(0) > min_coverage)
        )
        combined = pheno_mask & cov_mask
        filtered = df[combined]
        keys = set(zip(filtered['source_gene'],
                       filtered['target_phenotype'],
                       filtered['metapath']))
        focus_keys[hops] = keys

        print(f"  Focused filtering ({hops}-hop): "
              f"{len(df):,} total -> {combined.sum():,} focused keys")

    return focus_keys if focus_keys else None


def filter_results_to_focused(
    results: List[Dict],
    focus_keys: Set[Tuple[str, str, str]],
) -> List[Dict]:
    return [
        r for r in results
        if (r['source_gene'], r['target_phenotype'], r['metapath']) in focus_keys
    ]


# ---------------------------------------------------------------------------
# Tissue label shuffle + restore
# ---------------------------------------------------------------------------

def save_original_tissues(kg: KnowledgeGraph) -> List[Tuple]:
    """Save original tissue labels for later restoration.  ~50MB in memory."""
    return [
        (u, v, key, data.get('detailed_tissue'),
         data.get('context', {}).get('Detailed_Tissue') if isinstance(data.get('context'), dict) else None)
        for u, v, key, data in kg.edges(keys=True, data=True)
    ]


def restore_tissue_labels(kg: KnowledgeGraph, originals: List[Tuple]):
    """Restore tissue labels to original values (fast, in-place)."""
    for u, v, key, tissue, context_tissue in originals:
        kg[u][v][key]['detailed_tissue'] = tissue
        if 'context' in kg[u][v][key] and isinstance(kg[u][v][key]['context'], dict):
            kg[u][v][key]['context']['Detailed_Tissue'] = context_tissue


def shuffle_tissue_labels(kg: KnowledgeGraph, seed: int):
    """Shuffle detailed_tissue labels across all edges (in-place)."""
    np.random.seed(seed)

    detailed_tissues = []
    edge_keys = []

    for u, v, key, data in kg.edges(keys=True, data=True):
        detailed_tissues.append(data.get('detailed_tissue'))
        edge_keys.append((u, v, key))

    shuffled_indices = np.random.permutation(len(detailed_tissues))
    shuffled_tissues = [detailed_tissues[i] for i in shuffled_indices]

    for (u, v, key), new_tissue in zip(edge_keys, shuffled_tissues):
        kg[u][v][key]['detailed_tissue'] = new_tissue
        if 'context' in kg[u][v][key] and isinstance(kg[u][v][key]['context'], dict):
            kg[u][v][key]['context']['Detailed_Tissue'] = new_tissue


# ---------------------------------------------------------------------------
# Aggregate per tissue (fast version)
# ---------------------------------------------------------------------------

def aggregate_for_tissue_fast(kg: KnowledgeGraph, tissue_name: str) -> KnowledgeGraph:
    edge_groups = defaultdict(list)

    for u, v, data in kg.edges(data=True):
        edge_type = data.get('type', 'unknown')
        correlation = data.get('correlation_type', 0)
        direction = data.get('direction', '0')

        if direction == '0':
            u, v = ordered_pair(u, v)

        group_key = (u, v, edge_type, correlation, direction)
        edge_groups[group_key].append(data.copy())

    aggregated_by_type = defaultdict(dict)

    for (u, v, edge_type, correlation, direction), edges_data in edge_groups.items():
        probs = [e.get('probability', 0.5) for e in edges_data]
        prob = 1.0 - np.prod([1.0 - p for p in probs])
        evidence = sum(-np.log(1.0 - p + EPS) for p in probs)

        count = sum(1 for e in edges_data
                    if matches_tissue_group(e.get('detailed_tissue'), tissue_name))
        coverage = count / len(edges_data)

        attrs = edges_data[0].copy()
        attrs['probability'] = prob
        attrs['evidence_score'] = evidence
        attrs['coverage'] = coverage

        node_pair = (u, v)
        aggregated_by_type[node_pair][edge_type] = (prob, coverage, evidence, attrs)

    agg_kg = KnowledgeGraph()
    for node in kg.nodes():
        agg_kg.add_node(node, **kg.nodes[node])

    for node_pair, types_dict in aggregated_by_type.items():
        best = max(types_dict.items(), key=lambda x: (x[1][0], x[1][1], x[1][2]))
        u, v = node_pair
        agg_kg.add_edge(u, v, **best[1][3])

    return agg_kg




# ---------------------------------------------------------------------------
# Fast inference (2-hop and 3-hop)
# ---------------------------------------------------------------------------

def run_2hop_inference_fast(kg: KnowledgeGraph, config: dict) -> List[Dict]:
    target_types = set(config['psr_params'].get('target_types', ['Disease']))
    min_prob = config['psr_params'].get('min_path_probability', 0.001)
    prop_method = config['coverage'].get('propagation_method', 'geometric_mean')

    outgoing = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        outgoing[u].append((v, data))
        if data.get('direction', '0') == '0':
            outgoing[v].append((u, data))

    target_nodes = {n for n in kg.nodes() if get_node_type(kg, n) in target_types}
    grouped = defaultdict(list)

    for source in kg.nodes():
        if get_node_type(kg, source) in target_types:
            continue
        src_type = get_node_type(kg, source)

        for intermediate, e1 in outgoing[source]:
            if intermediate == source or get_node_type(kg, intermediate) in target_types:
                continue
            e1_prob = e1.get('probability', 0.5)
            e1_cov = e1.get('coverage', 0.0)
            e1_type = format_edge_type(e1.get('type', e1.get('kind', 'unknown')))
            int_type = get_node_type(kg, intermediate)

            for target, e2 in outgoing[intermediate]:
                if target not in target_nodes or target in (source, intermediate):
                    continue
                e2_prob = e2.get('probability', 0.5)
                path_prob = e1_prob * e2_prob
                if path_prob < min_prob:
                    continue

                e2_cov = e2.get('coverage', 0.0)
                e2_type = format_edge_type(e2.get('type', e2.get('kind', 'unknown')))
                tgt_type = get_node_type(kg, target)

                metapath = f"{src_type}-[{e1_type}]-{int_type}-[{e2_type}]-{tgt_type}"
                path_cov = propagate_coverage([e1_cov, e2_cov], prop_method)
                e1_ev = e1.get('evidence_score', -np.log(1 - e1_prob + EPS))
                e2_ev = e2.get('evidence_score', -np.log(1 - e2_prob + EPS))

                grouped[(source, target, metapath)].append({
                    'prob': path_prob,
                    'evidence': e1_ev * e2_ev,
                    'coverage': path_cov,
                })

    results = []
    for (source, target, metapath), paths in grouped.items():
        probs = [p['prob'] for p in paths]
        agg_prob = 1 - np.prod([1 - p for p in probs])
        agg_ev = sum(p['evidence'] for p in paths)
        path_coverages = [p['coverage'] for p in paths]
        agg_cov = float(np.mean(path_coverages))

        results.append({
            'source': source,
            'source_gene': get_node_name(kg, source),
            'target': target,
            'target_phenotype': get_node_name(kg, target),
            'metapath': metapath,
            'probability': agg_prob,
            'evidence_score': agg_ev,
            'coverage': agg_cov,
        })

    return results


def run_3hop_inference_fast(kg: KnowledgeGraph, config: dict) -> List[Dict]:
    target_types = set(config['psr_params'].get('target_types', ['Disease']))
    min_prob = config['psr_params'].get('min_path_probability', 0.001)
    prop_method = config['coverage'].get('propagation_method', 'geometric_mean')

    outgoing = defaultdict(list)
    for u, v, data in kg.edges(data=True):
        outgoing[u].append((v, data))
        if data.get('direction', '0') == '0':
            outgoing[v].append((u, data))

    target_nodes = {n for n in kg.nodes() if get_node_type(kg, n) in target_types}
    grouped = defaultdict(list)

    for source in kg.nodes():
        if get_node_type(kg, source) in target_types:
            continue
        src_type = get_node_type(kg, source)

        for int1, e1 in outgoing[source]:
            if int1 == source or get_node_type(kg, int1) in target_types:
                continue
            e1_prob = e1.get('probability', 0.5)
            e1_cov = e1.get('coverage', 0.0)
            e1_type = format_edge_type(e1.get('type', e1.get('kind', 'unknown')))
            int1_type = get_node_type(kg, int1)

            for int2, e2 in outgoing[int1]:
                if int2 in (source, int1) or get_node_type(kg, int2) in target_types:
                    continue
                e2_prob = e2.get('probability', 0.5)
                e2_cov = e2.get('coverage', 0.0)
                e2_type = format_edge_type(e2.get('type', e2.get('kind', 'unknown')))
                int2_type = get_node_type(kg, int2)

                for target, e3 in outgoing[int2]:
                    if target not in target_nodes or target in (source, int1, int2):
                        continue
                    e3_prob = e3.get('probability', 0.5)
                    path_prob = e1_prob * e2_prob * e3_prob
                    if path_prob < min_prob:
                        continue

                    e3_cov = e3.get('coverage', 0.0)
                    e3_type = format_edge_type(e3.get('type', e3.get('kind', 'unknown')))
                    tgt_type = get_node_type(kg, target)

                    metapath = (f"{src_type}-[{e1_type}]-{int1_type}"
                                f"-[{e2_type}]-{int2_type}"
                                f"-[{e3_type}]-{tgt_type}")
                    path_cov = propagate_coverage([e1_cov, e2_cov, e3_cov], prop_method)
                    e1_ev = e1.get('evidence_score', -np.log(1 - e1_prob + EPS))
                    e2_ev = e2.get('evidence_score', -np.log(1 - e2_prob + EPS))
                    e3_ev = e3.get('evidence_score', -np.log(1 - e3_prob + EPS))

                    grouped[(source, target, metapath)].append({
                        'prob': path_prob,
                        'evidence': e1_ev * e2_ev * e3_ev,
                        'coverage': path_cov,
                    })

    results = []
    for (source, target, metapath), paths in grouped.items():
        probs = [p['prob'] for p in paths]
        agg_prob = 1 - np.prod([1 - p for p in probs])
        agg_ev = sum(p['evidence'] for p in paths)
        path_coverages = [p['coverage'] for p in paths]
        agg_cov = float(np.mean(path_coverages))

        results.append({
            'source': source,
            'source_gene': get_node_name(kg, source),
            'target': target,
            'target_phenotype': get_node_name(kg, target),
            'metapath': metapath,
            'probability': agg_prob,
            'evidence_score': agg_ev,
            'coverage': agg_cov,
        })

    return results


# ---------------------------------------------------------------------------
# Compare tissues -> per-triple diff_coverage
# ---------------------------------------------------------------------------

def compare_results_to_df(results_A: List[Dict],
                          results_B: List[Dict]) -> pd.DataFrame:
    idx_A = {(r['source_gene'], r['target_phenotype'], r['metapath']): r
             for r in results_A}
    idx_B = {(r['source_gene'], r['target_phenotype'], r['metapath']): r
             for r in results_B}

    all_keys = set(idx_A.keys()) | set(idx_B.keys())

    rows = []
    for src_name, tgt_name, mp in all_keys:
        r_A = idx_A.get((src_name, tgt_name, mp))
        r_B = idx_B.get((src_name, tgt_name, mp))
        cov_A = r_A['coverage'] if r_A else 0.0
        cov_B = r_B['coverage'] if r_B else 0.0
        rows.append({
            'source_gene': src_name,
            'target_phenotype': tgt_name,
            'metapath': mp,
            'diff_coverage': cov_A - cov_B,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single permutation (called in loop)
# ---------------------------------------------------------------------------

def run_one_permutation(perm_id: int, kg: KnowledgeGraph,
                        tissue_A: str, tissue_B: str,
                        config: dict,
                        original_tissues: List[Tuple],
                        focus_keys: Optional[Dict[int, Set]] = None,
                        ) -> Dict[int, pd.DataFrame]:
    """
    Run one permutation: shuffle -> aggregate -> infer -> compare.
    Restores tissue labels afterward so kg is reusable.
    """
    hop_lengths = config['permutation'].get('hops_for_permutation', [2])
    if isinstance(hop_lengths, int):
        hop_lengths = [hop_lengths]

    # Shuffle in-place
    shuffle_tissue_labels(kg, seed=perm_id)

    try:
        # Aggregate for both tissues
        agg_A = aggregate_for_tissue_fast(kg, tissue_A)
        agg_B = aggregate_for_tissue_fast(kg, tissue_B)

        results_by_hop: Dict[int, pd.DataFrame] = {}

        for hops in hop_lengths:
            if hops == 2:
                res_A = run_2hop_inference_fast(agg_A, config)
                res_B = run_2hop_inference_fast(agg_B, config)
            elif hops == 3:
                res_A = run_3hop_inference_fast(agg_A, config)
                res_B = run_3hop_inference_fast(agg_B, config)
            else:
                continue

            if focus_keys and hops in focus_keys:
                fk = focus_keys[hops]
                res_A = filter_results_to_focused(res_A, fk)
                res_B = filter_results_to_focused(res_B, fk)

            comp_df = compare_results_to_df(res_A, res_B)
            results_by_hop[hops] = comp_df

        return results_by_hop

    finally:
        # ALWAYS restore labels, even if something fails
        restore_tissue_labels(kg, original_tissues)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run batched permutations (per-triple null distribution)')
    parser.add_argument('--perm-start', type=int, required=True,
                        help='First permutation ID in this batch')
    parser.add_argument('--perm-count', type=int, required=True,
                        help='Number of permutations to run')
    parser.add_argument('--comparison', type=str, required=True,
                        choices=['subcut_vs_visceral', 'white_vs_brown'])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])
    perm_dir = output_dir / 'permutations' / args.comparison

    hop_lengths = config['permutation'].get('hops_for_permutation', [2])
    if isinstance(hop_lengths, int):
        hop_lengths = [hop_lengths]

    perm_ids = list(range(args.perm_start, args.perm_start + args.perm_count))

    # --- Check how many are already done ---
    todo = []
    for pid in perm_ids:
        expected = [perm_dir / f'perm_{pid:04d}_{h}hop.parquet' for h in hop_lengths]
        if not all(f.exists() for f in expected):
            todo.append(pid)

    if not todo:
        print(f"All {len(perm_ids)} permutations already done, skipping.")
        sys.exit(0)

    print(f"Permutations {args.perm_start}-{args.perm_start + args.perm_count - 1} "
          f"for {args.comparison}: {len(todo)} todo, {len(perm_ids) - len(todo)} already done")

    # --- Resolve comparison ---
    comparison_config = None
    for comp in config['comparisons']:
        if comp['name'] == args.comparison:
            comparison_config = comp
            break
    if comparison_config is None:
        raise ValueError(f"Comparison '{args.comparison}' not found")
    tissue_A = comparison_config['tissue_A']
    tissue_B = comparison_config['tissue_B']

    # --- Stage files to local disk ---
    local_dir = LOCAL_WORK_DIR / args.comparison
    local_dir.mkdir(parents=True, exist_ok=True)

    graph_nfs = output_dir / 'preprocessed' / 'cleaned_graph.pkl'
    graph_local = stage_to_local(graph_nfs, local_dir)

    # Stage comparison parquets for focused filtering
    perm_config = config.get('permutation', {})
    if perm_config.get('filter_before_save', False):
        comp_dir = output_dir / 'comparisons' / args.comparison
        for h in hop_lengths:
            comp_file = comp_dir / f'comparison_{h}hop.parquet'
            if comp_file.exists():
                stage_to_local(comp_file, local_dir)

    # --- Load graph ONCE from local disk ---
    print(f"  Loading graph from local disk: {graph_local}")
    load_start = datetime.now()
    kg = KnowledgeGraph.import_graph(str(graph_local))
    load_elapsed = (datetime.now() - load_start).total_seconds()
    print(f"  Graph loaded in {load_elapsed:.1f}s "
          f"({kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges)")

    # --- Save original tissue labels for restore between perms ---
    original_tissues = save_original_tissues(kg)
    print(f"  Saved {len(original_tissues):,} original tissue labels")

    # --- Load focused triple keys ---
    focus_keys = load_focused_triple_keys(config, args.comparison, local_dir)
    if focus_keys:
        total = sum(len(v) for v in focus_keys.values())
        print(f"  Focused filtering ENABLED: {total:,} total keys")
    else:
        print(f"  Focused filtering DISABLED: saving all triples")

    # --- Run permutations ---
    perm_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    job_start = datetime.now()

    for i, perm_id in enumerate(todo):
        perm_start = datetime.now()

        # Double-check skip (another job might have completed it)
        expected = [perm_dir / f'perm_{perm_id:04d}_{h}hop.parquet' for h in hop_lengths]
        if all(f.exists() for f in expected):
            print(f"  [{i+1}/{len(todo)}] Perm {perm_id}: already exists, skipping")
            completed += 1
            continue

        # Run one permutation
        dfs_by_hop = run_one_permutation(
            perm_id, kg, tissue_A, tissue_B, config,
            original_tissues, focus_keys)

        # Save results immediately (crash-safe)
        for hops, comp_df in dfs_by_hop.items():
            out_path = perm_dir / f'perm_{perm_id:04d}_{hops}hop.parquet'
            comp_df.to_parquet(out_path, index=False)

        perm_elapsed = (datetime.now() - perm_start).total_seconds()
        completed += 1

        # Progress
        total_elapsed = (datetime.now() - job_start).total_seconds()
        avg_per_perm = total_elapsed / completed
        remaining = (len(todo) - completed) * avg_per_perm
        print(f"  [{completed}/{len(todo)}] Perm {perm_id}: "
              f"{perm_elapsed:.0f}s "
              f"(avg {avg_per_perm:.0f}s, ~{remaining/60:.0f}m remaining)")

    total_elapsed = (datetime.now() - job_start).total_seconds()
    print(f"\nCompleted {completed}/{len(todo)} permutations in "
          f"{total_elapsed/60:.1f} min ({total_elapsed/completed:.1f}s avg)")


if __name__ == '__main__':
    main()