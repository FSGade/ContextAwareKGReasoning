#!/usr/bin/env python3
"""
RQ2 Comparison — Compare results between tissue pairs.

Joins results on (source, target, metapath), computes differential metrics
(diff_coverage, log2_ratio), and analyses 2-hop vs 3-hop differences.

Usage:
    python compare.py --comparison subcut_vs_visceral --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from utils import load_config


def compare_tissue_results(df_A: pd.DataFrame, df_B: pd.DataFrame,
                           tissue_A: str, tissue_B: str,
                           epsilon: float = 0.01) -> pd.DataFrame:
    """Join and compare results from two tissue contexts."""
    print(f"\nComparing {tissue_A} vs {tissue_B}...")
    print(f"  Results A ({tissue_A}): {len(df_A):,} rows")
    print(f"  Results B ({tissue_B}): {len(df_B):,} rows")

    def safe_rename(df, rename_map):
        """Only rename columns that exist and won't collide."""
        actual = {}
        existing = set(df.columns)
        for old, new in rename_map.items():
            if old in existing and new not in existing:
                actual[old] = new
                existing.discard(old)
                existing.add(new)
        return df.rename(columns=actual)

    rename_A = {
        'probability': 'prob_A', 'evidence_score': 'evidence_A',
        'coverage': 'coverage_A', 'correlation_type': 'correlation_A',
        'intermediate_genes': 'intermediate_genes_A',
        'num_paths': 'num_paths_A', 'rank': 'rank_A',
    }
    rename_B = {
        'probability': 'prob_B', 'evidence_score': 'evidence_B',
        'coverage': 'coverage_B', 'correlation_type': 'correlation_B',
        'intermediate_genes': 'intermediate_genes_B',
        'num_paths': 'num_paths_B', 'rank': 'rank_B',
    }

    df_A = safe_rename(df_A, rename_A)
    df_B = safe_rename(df_B, rename_B)

    join_cols = ['source_gene', 'source_gene_id', 'target_phenotype',
                 'target_id', 'metapath']

    # Keep only necessary columns
    wanted_A = set(join_cols + [
        'prob_A', 'evidence_A', 'coverage_A', 'correlation_A',
        'intermediate_genes_A', 'num_paths_A', 'rank_A',
        'relationship_types', 'num_intermediates',
    ])
    for extra in ['intermediate_genes_B', 'intermediate_genes_C',
                  'n_intermediates_B', 'n_intermediates_C', 'num_intermediates']:
        if extra in df_A.columns:
            wanted_A.add(extra)

    wanted_B = set(join_cols + [
        'prob_B', 'evidence_B', 'coverage_B', 'correlation_B',
        'intermediate_genes_B', 'num_paths_B', 'rank_B',
    ])

    df_A_sub = df_A[[c for c in df_A.columns if c in wanted_A]]
    df_B_sub = df_B[[c for c in df_B.columns if c in wanted_B]]

    merged = df_A_sub.merge(df_B_sub, on=join_cols, how='outer',
                            suffixes=('', '_dup'))

    # Drop duplicate columns from overlapping non-join columns
    dup_cols = [c for c in merged.columns if c.endswith('_dup')]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)

    print(f"  Merged: {len(merged):,} rows")
    print(f"    In both: {(merged['prob_A'].notna() & merged['prob_B'].notna()).sum():,}")
    print(f"    Only in A: {(merged['prob_A'].notna() & merged['prob_B'].isna()).sum():,}")
    print(f"    Only in B: {(merged['prob_A'].isna() & merged['prob_B'].notna()).sum():,}")

    # Differential metrics
    merged['diff_prob'] = merged['prob_A'].fillna(0) - merged['prob_B'].fillna(0)
    merged['diff_coverage'] = merged['coverage_A'].fillna(0) - merged['coverage_B'].fillna(0)

    cov_A = merged['coverage_A'].fillna(0) + epsilon
    cov_B = merged['coverage_B'].fillna(0) + epsilon
    merged['log2_ratio'] = np.log2(cov_A / cov_B)

    # Flags
    merged['found_in_A'] = merged['prob_A'].notna()
    merged['found_in_B'] = merged['prob_B'].notna()
    merged['tissue_exclusive'] = merged['found_in_A'] != merged['found_in_B']
    merged['dominant_tissue'] = np.where(
        merged['diff_coverage'] > 0, tissue_A,
        np.where(merged['diff_coverage'] < 0, tissue_B, 'neither'))
    merged['tissue_specificity'] = merged['diff_coverage'].abs()

    # Rankings
    merged['rank_by_prob_A'] = merged['prob_A'].rank(ascending=False, method='min', na_option='bottom')
    merged['rank_by_prob_B'] = merged['prob_B'].rank(ascending=False, method='min', na_option='bottom')
    merged['rank_by_diff_cov'] = merged['tissue_specificity'].rank(ascending=False, method='min')

    merged = merged.sort_values('tissue_specificity', ascending=False)
    merged['tissue_A_name'] = tissue_A
    merged['tissue_B_name'] = tissue_B

    if merged.columns.duplicated().any():
        merged = merged.loc[:, ~merged.columns.duplicated()]

    return merged


def analyze_hop_comparison(comparison_2hop: pd.DataFrame,
                           comparison_3hop: pd.DataFrame,
                           k_values: List[int] = [50, 100, 250, 500]) -> Dict:
    """Compare 2-hop and 3-hop results: coverage decay, Jaccard overlap, specificity."""
    print("\n  Analyzing 2-hop vs 3-hop...")

    def sf(val):
        """Safe float conversion."""
        if val is None or pd.isna(val) or (isinstance(val, float) and np.isinf(val)):
            return None
        return float(val)

    results = {}

    # Coverage decay
    cov_2A = comparison_2hop['coverage_A'].dropna().mean()
    cov_2B = comparison_2hop['coverage_B'].dropna().mean()
    cov_3A = comparison_3hop['coverage_A'].dropna().mean()
    cov_3B = comparison_3hop['coverage_B'].dropna().mean()

    results['coverage_decay'] = {
        'mean_coverage_A_2hop': sf(cov_2A),
        'mean_coverage_A_3hop': sf(cov_3A),
        'mean_coverage_B_2hop': sf(cov_2B),
        'mean_coverage_B_3hop': sf(cov_3B),
        'decay_A': sf(cov_3A / cov_2A) if cov_2A and cov_2A > 0 else None,
        'decay_B': sf(cov_3B / cov_2B) if cov_2B and cov_2B > 0 else None,
    }

    # Jaccard overlap of top-K source genes
    results['jaccard_overlap'] = {}
    for k in k_values:
        g2 = set(comparison_2hop.nlargest(k, 'prob_A', keep='first')['source_gene'].dropna())
        g3 = set(comparison_3hop.nlargest(k, 'prob_A', keep='first')['source_gene'].dropna())
        inter = len(g2 & g3)
        union = len(g2 | g3)
        results['jaccard_overlap'][str(k)] = {
            'jaccard': sf(inter / union) if union > 0 else 0,
            'shared': inter, 'union': union,
            'only_2hop': len(g2 - g3), 'only_3hop': len(g3 - g2),
        }

    # Tissue specificity
    results['tissue_specificity'] = {
        'mean_specificity_2hop': sf(comparison_2hop['tissue_specificity'].mean()),
        'mean_specificity_3hop': sf(comparison_3hop['tissue_specificity'].mean()),
        'n_high_diff_2hop': int((comparison_2hop['tissue_specificity'] > 0.3).sum()),
        'n_high_diff_3hop': int((comparison_3hop['tissue_specificity'] > 0.3).sum()),
    }

    results['n_pairs_2hop'] = len(comparison_2hop)
    results['n_pairs_3hop'] = len(comparison_3hop)
    return results


def analyze_metapaths(comparison_df: pd.DataFrame,
                      tissue_A: str, tissue_B: str) -> List[Dict]:
    """Analyse metapath distribution and tissue bias."""
    print("\n  Analyzing metapaths...")
    metapath_stats = []

    for metapath in comparison_df['metapath'].unique():
        sub = comparison_df[comparison_df['metapath'] == metapath]
        stats = {
            'metapath': str(metapath) if metapath else 'Unknown',
            'count': len(sub),
            'mean_prob_A': float(sub['prob_A'].mean()),
            'mean_prob_B': float(sub['prob_B'].mean()),
            'mean_coverage_A': float(sub['coverage_A'].mean()),
            'mean_coverage_B': float(sub['coverage_B'].mean()),
            'mean_diff_coverage': float(sub['diff_coverage'].mean()),
            'mean_abs_diff_coverage': float(sub['diff_coverage'].abs().mean()),
            'pct_A_biased': float((sub['diff_coverage'] > 0.1).mean() * 100),
            'pct_B_biased': float((sub['diff_coverage'] < -0.1).mean() * 100),
            'n_tissue_specific': int((sub['tissue_specificity'] > 0.3).sum()),
        }
        metapath_stats.append(stats)

    metapath_stats.sort(key=lambda x: -x['count'])
    print(f"    Found {len(metapath_stats)} unique metapaths")
    return metapath_stats


def compute_observed_stats(comparison_df: pd.DataFrame,
                           thresholds: Dict = None) -> Dict:
    """Compute summary statistics for the comparison."""
    if thresholds is None:
        thresholds = {'high': 0.3, 'very_high': 0.5}

    diff_cov = comparison_df['diff_coverage'].fillna(0)
    return {
        'n_results': len(comparison_df),
        'n_high_diff': int((diff_cov.abs() > thresholds['high']).sum()),
        'n_very_high_diff': int((diff_cov.abs() > thresholds['very_high']).sum()),
        'mean_abs_diff': float(diff_cov.abs().mean()),
        'median_abs_diff': float(diff_cov.abs().median()),
        'std_diff': float(diff_cov.std()),
        'max_diff': float(diff_cov.max()),
        'min_diff': float(diff_cov.min()),
        'n_A_biased': int((diff_cov > 0.1).sum()),
        'n_B_biased': int((diff_cov < -0.1).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare tissue results')
    parser.add_argument('--comparison', type=str, required=True,
                        choices=['subcut_vs_visceral', 'white_vs_brown'])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])
    inference_dir = output_dir / 'inference'

    comparison_config = next(
        (c for c in config['comparisons'] if c['name'] == args.comparison), None)
    if comparison_config is None:
        print(f"ERROR: Comparison '{args.comparison}' not found in config")
        sys.exit(1)

    tissue_A = comparison_config['tissue_A']
    tissue_B = comparison_config['tissue_B']
    epsilon = config['coverage'].get('epsilon', 0.01)
    k_values = config['analysis'].get('jaccard_k_values', [50, 100, 250, 500])

    print(f"Comparison: {args.comparison} ({tissue_A} vs {tissue_B})")

    comp_dir = output_dir / 'comparisons' / args.comparison
    comp_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for hops in [2, 3]:
        print(f"\n{'=' * 60}\nProcessing {hops}-hop results\n{'=' * 60}")

        path_A = inference_dir / f'{tissue_A}_{hops}hop.parquet'
        path_B = inference_dir / f'{tissue_B}_{hops}hop.parquet'

        if not path_A.exists() or not path_B.exists():
            print(f"  WARNING: Missing results for {hops}-hop")
            continue

        df_A = pd.read_parquet(path_A)
        df_B = pd.read_parquet(path_B)

        comparison = compare_tissue_results(df_A, df_B, tissue_A, tissue_B, epsilon)
        comparison['hop_length'] = hops

        output_path = comp_dir / f'comparison_{hops}hop.parquet'
        comparison.to_parquet(output_path, index=False)
        print(f"\n  Saved: {output_path}")

        # Observed stats and metapath analysis
        thresholds = config['analysis'].get('tissue_specific', {})
        obs_stats = compute_observed_stats(comparison, {
            'high': thresholds.get('diff_coverage_threshold', 0.3),
            'very_high': thresholds.get('high_diff_threshold', 0.5),
        })
        with open(comp_dir / f'observed_stats_{hops}hop.json', 'w') as f:
            json.dump(obs_stats, f, indent=2)

        metapath_stats = analyze_metapaths(comparison, tissue_A, tissue_B)
        with open(comp_dir / f'metapath_analysis_{hops}hop.json', 'w') as f:
            json.dump(metapath_stats, f, indent=2)

        results[f'{hops}hop'] = {
            'comparison_path': str(output_path),
            'n_results': len(comparison),
            'observed_stats': obs_stats,
            'metapath_count': len(metapath_stats),
        }

    # 2-hop vs 3-hop comparison
    if '2hop' in results and '3hop' in results:
        comp_2 = pd.read_parquet(comp_dir / 'comparison_2hop.parquet')
        comp_3 = pd.read_parquet(comp_dir / 'comparison_3hop.parquet')
        hop_comparison = analyze_hop_comparison(comp_2, comp_3, k_values)
        with open(comp_dir / 'hop_comparison.json', 'w') as f:
            json.dump(hop_comparison, f, indent=2)
        results['hop_comparison'] = hop_comparison

    with open(comp_dir / 'comparison_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Comparison complete — output: {comp_dir}")


if __name__ == '__main__':
    main()