#!/usr/bin/env python3
"""
RQ1 Step 3: Aggregate results and compute cross-context comparisons.

This script:
1. Loads PSR results from all contexts
2. If metapath grouping was used, aggregates to (Gene, Disease) level for main metrics
3. Computes Spearman correlations (on common genes)
4. Computes Jaccard overlap at k=50, 100, 250, 500
5. Identifies tissue-exclusive genes
6. Analyzes metapath distributions (if metapath grouping was used)
7. Saves comparison metrics

Usage:
    python compare_contexts.py --config config.yaml
    python compare_contexts.py --input-dir /path/to/psr_results --output-dir /path/to/comparisons
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import yaml
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# Standard sort order used across all ranking operations
RANK_SORT_KEYS = ['path_probability', 'evidence_score', 'num_intermediates', 'source_gene']
RANK_SORT_ASC = [False, False, False, True]


def rank_by_standard_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a results DataFrame by the standard ranking criteria.
    
    Order: path_probability (desc), evidence_score (desc),
           num_intermediates (desc), source_gene (asc).
    """
    return df.sort_values(by=RANK_SORT_KEYS, ascending=RANK_SORT_ASC)


def has_metapath_grouping(df: pd.DataFrame) -> bool:
    """Check if DataFrame has metapath grouping (multiple rows per gene-disease pair)."""
    return 'metapath_name' in df.columns


def aggregate_metapaths_to_gene_disease(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metapath-level results back to (Gene, Disease) level.
    
    For each (Gene, Disease) pair, combines all metapaths using PSR formula:
    - probability: P = 1 - ∏(1 - p_i)
    - evidence: sum of evidence scores
    - correlation: weighted average by evidence
    
    Returns DataFrame with one row per (Gene, Disease) pair.
    """
    if not has_metapath_grouping(df):
        return df  # Already at gene-disease level
    
    # Group by (source_gene, target)
    grouped = df.groupby(['source_gene', 'source_gene_id', 'target', 'target_id', 'source_type', 'target_type'])
    
    aggregated = []
    for (gene, gene_id, disease, disease_id, src_type, tgt_type), group in grouped:
        # PSR aggregation: P = 1 - ∏(1 - p_i)
        probs = group['path_probability'].values
        combined_prob = 1.0 - np.prod(1.0 - probs)
        
        # Sum evidence scores
        combined_evidence = group['evidence_score'].sum()
        
        # Weighted correlation
        if combined_evidence > 0:
            weighted_corr = (group['correlation_type'] * group['evidence_score']).sum() / combined_evidence
            if weighted_corr > 0.5:
                final_corr = 1
            elif weighted_corr < -0.5:
                final_corr = -1
            else:
                final_corr = 0
        else:
            final_corr = 0
        
        # Collect all intermediates and metapaths
        all_intermediates = []
        for ints in group['intermediate_genes']:
            if isinstance(ints, list):
                all_intermediates.extend(ints)
        unique_intermediates = list(set(all_intermediates))
        
        metapath_names = group['metapath_name'].tolist()
        
        aggregated.append({
            'source_gene': gene,
            'source_gene_id': gene_id,
            'source_type': src_type,
            'target': disease,
            'target_id': disease_id,
            'target_type': tgt_type,
            'path_probability': round(float(combined_prob), 6),
            'evidence_score': round(float(combined_evidence), 4),
            'correlation_type': int(final_corr),
            'num_intermediates': len(unique_intermediates),
            'num_metapaths': len(group),
            'metapath_names': metapath_names,
            'intermediate_genes': unique_intermediates[:50],  # Limit for storage
            'hop_length': group['hop_length'].iloc[0],
        })
    
    agg_df = pd.DataFrame(aggregated)
    
    agg_df = rank_by_standard_sort(agg_df)
    agg_df['rank'] = range(1, len(agg_df) + 1)
    
    return agg_df


def load_all_results(input_dir: Path, hops: int, aggregate_metapaths: bool = True) -> dict:
    """
    Load PSR results for all contexts.
    
    Args:
        input_dir: Directory containing results
        hops: Number of hops (2 or 3)
        aggregate_metapaths: If True and data has metapaths, aggregate to (Gene, Disease) level
    
    Returns dict: context_name -> DataFrame
    """
    contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    results = {}
    
    for ctx in contexts:
        parquet_path = input_dir / f"{ctx}_{hops}hop_results.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            
            # Check for metapath grouping
            if has_metapath_grouping(df):
                n_metapath = len(df)
                n_pairs = df.groupby(['source_gene', 'target']).ngroups
                print(f"  {ctx}: {n_metapath:,} metapath results ({n_pairs:,} gene-disease pairs)")
                
                if aggregate_metapaths:
                    df = aggregate_metapaths_to_gene_disease(df)
                    print(f"    -> Aggregated to {len(df):,} gene-disease results")
            else:
                print(f"  {ctx}: {len(df):,} results")
            
            results[ctx] = df
        else:
            print(f"  {ctx}: NOT FOUND ({parquet_path})")
    
    return results


def load_all_results_metapath_level(input_dir: Path, hops: int) -> dict:
    """
    Load PSR results at metapath level (no aggregation).
    
    Returns dict: context_name -> DataFrame (with metapath columns if available)
    """
    return load_all_results(input_dir, hops, aggregate_metapaths=False)


def compute_spearman_for_disease(disease_name: str, results_by_context: dict, 
                                  disease_ids: list = None) -> dict:
    """
    Compute pairwise Spearman correlations for gene rankings for a specific disease.
    Uses only genes present in both contexts being compared.
    Uses consistent multi-key ranking (probability, evidence, num_intermediates, gene name).
    
    Args:
        disease_name: Disease to filter on (case-insensitive partial match)
        results_by_context: Dict[context_name] -> DataFrame
        disease_ids: Optional list of exact disease IDs to match (overrides disease_name)
    
    Returns:
        Dict with correlation matrix and metadata
    """
    contexts = list(results_by_context.keys())
    
    # Filter to disease and build gene rank dicts
    gene_ranks = {}
    for ctx, df in results_by_context.items():
        # Filter to disease (exact ID match if provided, otherwise partial name match)
        if disease_ids:
            disease_df = df[df['target_id'].isin(disease_ids)].copy()
        else:
            disease_df = df[df['target'].str.lower().str.contains(disease_name.lower(), na=False)].copy()
        
        if len(disease_df) == 0:
            gene_ranks[ctx] = {}
            continue
        
        # Sort by: path_probability (desc), evidence_score (desc), num_intermediates (desc), source_gene (asc)
        disease_df = rank_by_standard_sort(disease_df)
        disease_df['disease_rank'] = range(1, len(disease_df) + 1)
        gene_ranks[ctx] = disease_df.set_index('source_gene')['disease_rank'].to_dict()
    
    # Compute pairwise Spearman
    n_ctx = len(contexts)
    corr_matrix = np.zeros((n_ctx, n_ctx))
    pval_matrix = np.zeros((n_ctx, n_ctx))
    n_common_matrix = np.zeros((n_ctx, n_ctx), dtype=int)
    
    for i, c1 in enumerate(contexts):
        for j, c2 in enumerate(contexts):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
                n_common_matrix[i, j] = len(gene_ranks[c1])
                continue
            
            # Find common genes
            common_genes = set(gene_ranks[c1].keys()) & set(gene_ranks[c2].keys())
            n_common_matrix[i, j] = len(common_genes)
            
            if len(common_genes) < 3:
                corr_matrix[i, j] = np.nan
                pval_matrix[i, j] = np.nan
                continue
            
            # Build rank vectors
            ranks1 = [gene_ranks[c1][g] for g in common_genes]
            ranks2 = [gene_ranks[c2][g] for g in common_genes]
            
            rho, pval = spearmanr(ranks1, ranks2)
            corr_matrix[i, j] = rho
            pval_matrix[i, j] = pval
    
    return {
        'disease': disease_name,
        'disease_ids': disease_ids,
        'contexts': contexts,
        'correlation_matrix': corr_matrix.tolist(),
        'pvalue_matrix': pval_matrix.tolist(),
        'n_common_genes': n_common_matrix.tolist(),
        'n_genes_per_context': {ctx: len(gene_ranks[ctx]) for ctx in contexts}
    }


def compute_jaccard_at_k(disease_name: str, results_by_context: dict, k_values: list,
                         disease_ids: list = None) -> dict:
    """
    Compute Jaccard overlap at multiple k values for a specific disease.
    Uses consistent multi-key ranking.
    
    Returns dict with Jaccard matrices for each k.
    """
    contexts = list(results_by_context.keys())
    
    # Get top-k genes per context for this disease
    top_k_genes = {k: {} for k in k_values}
    
    for ctx, df in results_by_context.items():
        # Filter to disease
        if disease_ids:
            disease_df = df[df['target_id'].isin(disease_ids)].copy()
        else:
            disease_df = df[df['target'].str.lower().str.contains(disease_name.lower(), na=False)].copy()
        
        disease_df = rank_by_standard_sort(disease_df)
        
        for k in k_values:
            top_k_genes[k][ctx] = set(disease_df.head(k)['source_gene'])
    
    # Compute Jaccard for each k
    jaccard_results = {}
    
    for k in k_values:
        n_ctx = len(contexts)
        jaccard_matrix = np.zeros((n_ctx, n_ctx))
        
        for i, c1 in enumerate(contexts):
            for j, c2 in enumerate(contexts):
                set1 = top_k_genes[k][c1]
                set2 = top_k_genes[k][c2]
                
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0.0
        
        jaccard_results[k] = {
            'matrix': jaccard_matrix.tolist(),
            'set_sizes': {ctx: len(top_k_genes[k][ctx]) for ctx in contexts}
        }
    
    return {
        'disease': disease_name,
        'disease_ids': disease_ids,
        'contexts': contexts,
        'k_values': k_values,
        'jaccard': jaccard_results
    }


def find_tissue_exclusive_genes(disease_name: str, results_by_context: dict, 
                                 top_k_tissue: int = 100, not_in_top_m_baseline: int = 500,
                                 disease_ids: list = None) -> dict:
    """
    Find genes that are highly ranked in a tissue context but not in baseline.
    Uses consistent multi-key ranking.
    
    Args:
        disease_name: Disease to filter on
        results_by_context: Dict[context_name] -> DataFrame
        top_k_tissue: Gene must be in top-k of tissue context
        not_in_top_m_baseline: Gene must NOT be in top-m of baseline
        disease_ids: Optional list of exact disease IDs
    
    Returns:
        Dict with tissue-exclusive genes per context
    """
    if 'baseline' not in results_by_context:
        return {'error': 'Baseline context not found'}
    
    # Get baseline top-m genes with consistent sorting
    baseline_df = results_by_context['baseline']
    if disease_ids:
        baseline_disease = baseline_df[baseline_df['target_id'].isin(disease_ids)].copy()
    else:
        baseline_disease = baseline_df[baseline_df['target'].str.lower().str.contains(disease_name.lower(), na=False)].copy()
    
    baseline_disease = rank_by_standard_sort(baseline_disease)
    baseline_top_m = set(baseline_disease.head(not_in_top_m_baseline)['source_gene'])
    
    exclusive_genes = {}
    
    for ctx, df in results_by_context.items():
        if ctx == 'baseline':
            continue
        
        # Get top-k genes in this tissue context
        if disease_ids:
            disease_df = df[df['target_id'].isin(disease_ids)].copy()
        else:
            disease_df = df[df['target'].str.lower().str.contains(disease_name.lower(), na=False)].copy()
        
        disease_df = rank_by_standard_sort(disease_df)
        tissue_top_k = disease_df.head(top_k_tissue)
        
        # Find genes NOT in baseline top-m
        exclusive = tissue_top_k[~tissue_top_k['source_gene'].isin(baseline_top_m)]
        
        exclusive_genes[ctx] = {
            'genes': exclusive['source_gene'].tolist(),
            'count': len(exclusive),
            'details': exclusive[['source_gene', 'target', 'path_probability', 'evidence_score', 'num_intermediates']].to_dict(orient='records')
        }
    
    return {
        'disease': disease_name,
        'disease_ids': disease_ids,
        'thresholds': {
            'top_k_tissue': top_k_tissue,
            'not_in_top_m_baseline': not_in_top_m_baseline
        },
        'baseline_top_m_size': len(baseline_top_m),
        'exclusive_genes': exclusive_genes
    }


def compute_rank_shifts(disease_name: str, results_by_context: dict,
                        disease_ids: list = None) -> pd.DataFrame:
    """
    Compute rank shifts between baseline and each tissue context.
    Uses consistent multi-key ranking.
    
    Returns DataFrame with genes and their ranks across contexts.
    """
    if 'baseline' not in results_by_context:
        return pd.DataFrame()
    
    # Get ranks per context with consistent sorting
    all_ranks = {}
    all_probs = {}
    
    for ctx, df in results_by_context.items():
        if disease_ids:
            disease_df = df[df['target_id'].isin(disease_ids)].copy()
        else:
            disease_df = df[df['target'].str.lower().str.contains(disease_name.lower(), na=False)].copy()
        
        disease_df = rank_by_standard_sort(disease_df)
        disease_df['ctx_rank'] = range(1, len(disease_df) + 1)
        
        all_ranks[ctx] = disease_df.set_index('source_gene')['ctx_rank'].to_dict()
        all_probs[ctx] = disease_df.set_index('source_gene')['path_probability'].to_dict()
    
    # Combine into single DataFrame
    all_genes = set()
    for ranks in all_ranks.values():
        all_genes.update(ranks.keys())
    
    rows = []
    for gene in all_genes:
        row = {'gene': gene}
        for ctx in results_by_context.keys():
            row[f'rank_{ctx}'] = all_ranks[ctx].get(gene, np.nan)
            row[f'prob_{ctx}'] = all_probs[ctx].get(gene, np.nan)
        
        # Compute rank shifts (relative to baseline)
        if 'baseline' in all_ranks and gene in all_ranks['baseline']:
            baseline_rank = all_ranks['baseline'][gene]
            for ctx in results_by_context.keys():
                if ctx != 'baseline' and gene in all_ranks[ctx]:
                    row[f'shift_{ctx}'] = baseline_rank - all_ranks[ctx][gene]  # Positive = better in tissue
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by baseline rank if available
    if 'rank_baseline' in df.columns:
        df = df.sort_values('rank_baseline')
    
    return df


def compute_context_diagnostics(results_by_context: dict, hops: int) -> dict:
    """
    Compute diagnostic statistics to assess whether context effects 
    could be driven by graph structure rather than biology.
    
    Returns dict with diagnostic metrics per context.
    """
    diagnostics = {}
    
    for ctx, df in results_by_context.items():
        if len(df) == 0:
            diagnostics[ctx] = {'error': 'No results'}
            continue
        
        ctx_stats = {
            'total_gene_disease_pairs': len(df),
            'unique_genes': df['source_gene'].nunique(),
            'unique_diseases': df['target'].nunique(),
            
            # Path count statistics - important for combinatorics confound
            'path_counts': {
                'min': int(df['num_paths'].min()) if 'num_paths' in df.columns else 0,
                'max': int(df['num_paths'].max()) if 'num_paths' in df.columns else 0,
                'mean': float(df['num_paths'].mean()) if 'num_paths' in df.columns else 0,
                'median': float(df['num_paths'].median()) if 'num_paths' in df.columns else 0,
                'total': int(df['num_paths'].sum()) if 'num_paths' in df.columns else 0,
            },
            
            # Intermediate statistics
            'intermediate_counts': {
                'min': int(df['num_intermediates'].min()),
                'max': int(df['num_intermediates'].max()),
                'mean': float(df['num_intermediates'].mean()),
            },
            
            # Expanded edge usage (sensitivity check)
            'expanded_edge_usage': {},
        }
        
        if 'expanded_edge_fraction' in df.columns:
            ctx_stats['expanded_edge_usage'] = {
                'mean_fraction': float(df['expanded_edge_fraction'].mean()),
                'results_with_any_expanded': int((df['expanded_edge_fraction'] > 0).sum()),
                'results_with_all_expanded': int((df['expanded_edge_fraction'] == 1.0).sum()),
                'pct_with_any_expanded': float((df['expanded_edge_fraction'] > 0).mean() * 100),
            }
        
        diagnostics[ctx] = ctx_stats
    
    return {
        'hops': hops,
        'contexts': diagnostics
    }


def analyze_metapath_distribution(results_by_context: dict, disease_name: str = None,
                                   disease_ids: list = None) -> dict:
    """
    Analyze metapath distribution across contexts.
    
    Returns dict with:
    - Metapath counts per context
    - Tissue-exclusive metapaths (metapaths that appear in tissue but not baseline)
    - Top metapaths per context
    """
    # Check if we have metapath data
    first_df = next(iter(results_by_context.values()))
    if 'metapath_name' not in first_df.columns:
        return {'has_metapaths': False}
    
    metapath_counts = {}
    metapath_evidence = {}
    
    for ctx, df in results_by_context.items():
        # Filter to disease if specified
        if disease_ids:
            df = df[df['target_id'].isin(disease_ids)]
        elif disease_name:
            df = df[df['target'].str.lower().str.contains(disease_name.lower(), na=False)]
        
        # Count metapaths
        counts = df['metapath_name'].value_counts().to_dict()
        metapath_counts[ctx] = counts
        
        # Sum evidence per metapath
        evidence = df.groupby('metapath_name')['evidence_score'].sum().to_dict()
        metapath_evidence[ctx] = evidence
    
    # Find tissue-exclusive metapaths
    baseline_metapaths = set(metapath_counts.get('baseline', {}).keys())
    tissue_exclusive_metapaths = {}
    
    for ctx in ['adipose', 'nonadipose', 'liver']:
        if ctx not in metapath_counts:
            continue
        ctx_metapaths = set(metapath_counts[ctx].keys())
        exclusive = ctx_metapaths - baseline_metapaths
        
        # Get details for exclusive metapaths
        exclusive_details = []
        for mp in exclusive:
            exclusive_details.append({
                'metapath': mp,
                'count': metapath_counts[ctx][mp],
                'evidence': metapath_evidence[ctx].get(mp, 0)
            })
        exclusive_details.sort(key=lambda x: -x['count'])
        
        tissue_exclusive_metapaths[ctx] = {
            'count': len(exclusive),
            'metapaths': exclusive_details
        }
    
    # Top metapaths per context (by count)
    top_metapaths = {}
    for ctx, counts in metapath_counts.items():
        sorted_mps = sorted(counts.items(), key=lambda x: -x[1])[:10]
        top_metapaths[ctx] = [{'metapath': mp, 'count': c} for mp, c in sorted_mps]
    
    return {
        'has_metapaths': True,
        'disease': disease_name,
        'disease_ids': disease_ids,
        'metapath_counts': metapath_counts,
        'tissue_exclusive_metapaths': tissue_exclusive_metapaths,
        'top_metapaths_by_context': top_metapaths,
        'total_unique_metapaths': len(set().union(*[set(c.keys()) for c in metapath_counts.values()]))
    }


def find_metapath_rank_changes(results_by_context: dict, disease_name: str = None,
                                disease_ids: list = None, top_k: int = 100) -> dict:
    """
    Find (Gene, Disease, Metapath) combinations where rank changes significantly
    between baseline and tissue contexts.
    
    Returns dict with notable rank changes.
    """
    # Check if we have metapath data
    first_df = next(iter(results_by_context.values()))
    if 'metapath_name' not in first_df.columns:
        return {'has_metapaths': False}
    
    if 'baseline' not in results_by_context:
        return {'error': 'No baseline context'}
    
    changes = {}
    
    for ctx in ['adipose', 'nonadipose', 'liver']:
        if ctx not in results_by_context:
            continue
        
        baseline_df = results_by_context['baseline']
        tissue_df = results_by_context[ctx]
        
        # Filter to disease
        if disease_ids:
            baseline_df = baseline_df[baseline_df['target_id'].isin(disease_ids)]
            tissue_df = tissue_df[tissue_df['target_id'].isin(disease_ids)]
        elif disease_name:
            baseline_df = baseline_df[baseline_df['target'].str.lower().str.contains(disease_name.lower(), na=False)]
            tissue_df = tissue_df[tissue_df['target'].str.lower().str.contains(disease_name.lower(), na=False)]
        
        # Sort and rank
        baseline_df = rank_by_standard_sort(baseline_df).copy()
        baseline_df['rank'] = range(1, len(baseline_df) + 1)
        
        tissue_df = rank_by_standard_sort(tissue_df).copy()
        tissue_df['rank'] = range(1, len(tissue_df) + 1)
        
        # Create key for matching
        baseline_df['key'] = baseline_df['source_gene'] + '|' + baseline_df['target'] + '|' + baseline_df['metapath_name']
        tissue_df['key'] = tissue_df['source_gene'] + '|' + tissue_df['target'] + '|' + tissue_df['metapath_name']
        
        baseline_ranks = baseline_df.set_index('key')['rank'].to_dict()
        tissue_ranks = tissue_df.set_index('key')['rank'].to_dict()
        
        # Find significant changes (in top-k of tissue but not baseline, or big rank jumps)
        notable_changes = []
        
        for key, tissue_rank in tissue_ranks.items():
            if tissue_rank > top_k:
                continue
            
            baseline_rank = baseline_ranks.get(key)
            
            if baseline_rank is None:
                # New in tissue context
                gene, disease, metapath = key.split('|')
                notable_changes.append({
                    'gene': gene,
                    'disease': disease,
                    'metapath': metapath,
                    'baseline_rank': None,
                    'tissue_rank': tissue_rank,
                    'change': 'new_in_tissue'
                })
            elif tissue_rank < baseline_rank * 0.5:
                # Significant improvement (rank at least halved)
                gene, disease, metapath = key.split('|')
                notable_changes.append({
                    'gene': gene,
                    'disease': disease,
                    'metapath': metapath,
                    'baseline_rank': baseline_rank,
                    'tissue_rank': tissue_rank,
                    'change': f'improved_by_{baseline_rank - tissue_rank}'
                })
        
        changes[ctx] = {
            'count': len(notable_changes),
            'changes': notable_changes[:50]  # Limit to top 50
        }
    
    return {
        'has_metapaths': True,
        'disease': disease_name,
        'top_k': top_k,
        'changes_by_context': changes
    }


def aggregate_results(input_dir: Path, hops: int) -> pd.DataFrame:
    """
    Aggregate all results into a single DataFrame with context column.
    """
    contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    all_dfs = []
    
    for ctx in contexts:
        parquet_path = input_dir / f"{ctx}_{hops}hop_results.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df['tissue_context'] = ctx
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description='Compare PSR results across tissue contexts')
    parser.add_argument('--config', type=Path, help='Path to config.yaml')
    parser.add_argument('--input-dir', type=Path, help='Directory with PSR results')
    parser.add_argument('--output-dir', type=Path, help='Directory for comparison outputs')
    parser.add_argument('--hops', type=int, default=None, help='Analyze specific hop length (2 or 3)')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        input_dir = Path(args.input_dir or config['paths']['output_dir']) / 'psr_results'
        output_dir = Path(args.output_dir or config['paths']['output_dir']) / 'comparisons'
        analysis_config = config.get('analysis', {})
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("Either --config or both --input-dir and --output-dir are required")
        input_dir = args.input_dir
        output_dir = args.output_dir
        analysis_config = {
            'jaccard_k_values': [50, 100, 250, 500],
            'tissue_exclusive': {'top_k_tissue': 100, 'not_in_top_m_baseline': 500},
            'disease_focus': ['inflammation', 'obesity', 'insulin resistance', 'type 2 diabetes']
        }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which hops to analyze
    hops_to_analyze = [args.hops] if args.hops else [2, 3]
    
    print("=" * 80)
    print("RQ1 STEP 3: CROSS-CONTEXT COMPARISON")
    print("=" * 80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    for hops in hops_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING {hops}-HOP RESULTS")
        print(f"{'='*80}")
        
        # Load results (aggregated to gene-disease level for main metrics)
        print("\nLoading results (aggregated to gene-disease level)...")
        results_by_context = load_all_results(input_dir, hops, aggregate_metapaths=True)
        
        if not results_by_context:
            print(f"No results found for {hops}-hop, skipping...")
            continue
        
        # Also load at metapath level for metapath-specific analysis
        print("\nLoading results at metapath level...")
        results_metapath_level = load_all_results_metapath_level(input_dir, hops)
        has_metapaths = has_metapath_grouping(next(iter(results_metapath_level.values())))
        if has_metapaths:
            print("  Metapath grouping detected - will perform metapath-level analysis")
        
        # Aggregate all results (at metapath level for combined file)
        print("\nAggregating results...")
        combined_df = aggregate_results(input_dir, hops)
        combined_path = output_dir / f"combined_{hops}hop.parquet"
        combined_df.to_parquet(combined_path, index=False)
        print(f"Saved combined results: {combined_path}")
        
        # Get diseases to analyze
        diseases = analysis_config.get('disease_focus', ['inflammation'])
        jaccard_k_values = analysis_config.get('jaccard_k_values', [50, 100, 250, 500])
        tissue_exclusive_config = analysis_config.get('tissue_exclusive', {})
        
        all_spearman = []
        all_jaccard = []
        all_exclusive = []
        all_rank_shifts = {}
        all_metapath_analysis = []
        all_metapath_rank_changes = []
        
        for disease in diseases:
            print(f"\n--- Analyzing disease: {disease} ---")
            
            # Spearman correlation (on aggregated gene-disease data)
            print("  Computing Spearman correlations (gene-disease level)...")
            spearman_result = compute_spearman_for_disease(disease, results_by_context)
            all_spearman.append(spearman_result)
            
            # Print correlation matrix
            contexts = spearman_result['contexts']
            print(f"  Correlation matrix (n_common shown):")
            print(f"  {'':12}", end='')
            for ctx in contexts:
                print(f"{ctx:12}", end='')
            print()
            for i, c1 in enumerate(contexts):
                print(f"  {c1:12}", end='')
                for j, c2 in enumerate(contexts):
                    rho = spearman_result['correlation_matrix'][i][j]
                    n = spearman_result['n_common_genes'][i][j]
                    if np.isnan(rho):
                        print(f"{'N/A':12}", end='')
                    else:
                        print(f"{rho:.3f}({n:4d})", end=' ')
                print()
            
            # Jaccard overlap (on aggregated gene-disease data)
            print(f"  Computing Jaccard overlap at k={jaccard_k_values}...")
            jaccard_result = compute_jaccard_at_k(disease, results_by_context, jaccard_k_values)
            all_jaccard.append(jaccard_result)
            
            # Print Jaccard for k=100
            if 100 in jaccard_k_values:
                print(f"  Jaccard@100:")
                jac_matrix = jaccard_result['jaccard'][100]['matrix']
                for i, c1 in enumerate(contexts):
                    print(f"    {c1:12}", end='')
                    for j, c2 in enumerate(contexts):
                        print(f"{jac_matrix[i][j]:.3f} ", end='')
                    print()
            
            # Tissue-exclusive genes (on aggregated data)
            print(f"  Finding tissue-exclusive genes...")
            exclusive_result = find_tissue_exclusive_genes(
                disease, results_by_context,
                top_k_tissue=tissue_exclusive_config.get('top_k_tissue', 100),
                not_in_top_m_baseline=tissue_exclusive_config.get('not_in_top_m_baseline', 500)
            )
            all_exclusive.append(exclusive_result)
            
            if 'exclusive_genes' in exclusive_result:
                for ctx, info in exclusive_result['exclusive_genes'].items():
                    print(f"    {ctx}: {info['count']} exclusive genes")
                    if info['count'] > 0:
                        top_3 = info['genes'][:3]
                        print(f"      Top 3: {', '.join(top_3)}")
            
            # Rank shifts (on aggregated data)
            print(f"  Computing rank shifts...")
            rank_shift_df = compute_rank_shifts(disease, results_by_context)
            all_rank_shifts[disease] = rank_shift_df
            
            # Metapath-specific analysis (if metapaths available)
            if has_metapaths:
                print(f"  Analyzing metapath distribution...")
                metapath_analysis = analyze_metapath_distribution(
                    results_metapath_level, disease_name=disease
                )
                all_metapath_analysis.append(metapath_analysis)
                
                if metapath_analysis.get('has_metapaths'):
                    print(f"    Total unique metapaths: {metapath_analysis['total_unique_metapaths']}")
                    for ctx, info in metapath_analysis.get('tissue_exclusive_metapaths', {}).items():
                        if info['count'] > 0:
                            print(f"    {ctx}: {info['count']} tissue-exclusive metapaths")
                
                print(f"  Finding metapath rank changes...")
                metapath_changes = find_metapath_rank_changes(
                    results_metapath_level, disease_name=disease, top_k=100
                )
                all_metapath_rank_changes.append(metapath_changes)
                
                if metapath_changes.get('has_metapaths'):
                    for ctx, info in metapath_changes.get('changes_by_context', {}).items():
                        print(f"    {ctx}: {info['count']} notable metapath rank changes")
        
        # Save all comparison results
        print("\nSaving comparison results...")
        
        # Spearman
        spearman_path = output_dir / f"spearman_{hops}hop.json"
        with open(spearman_path, 'w') as f:
            json.dump({'hops': hops, 'results': all_spearman}, f, indent=2)
        print(f"  Saved: {spearman_path}")
        
        # Jaccard
        jaccard_path = output_dir / f"jaccard_{hops}hop.json"
        with open(jaccard_path, 'w') as f:
            json.dump({'hops': hops, 'k_values': jaccard_k_values, 'results': all_jaccard}, f, indent=2)
        print(f"  Saved: {jaccard_path}")
        
        # Tissue-exclusive
        exclusive_path = output_dir / f"tissue_exclusive_{hops}hop.json"
        with open(exclusive_path, 'w') as f:
            json.dump({'hops': hops, 'results': all_exclusive}, f, indent=2, default=str)
        print(f"  Saved: {exclusive_path}")
        
        # Rank shifts (one file per disease)
        for disease, df in all_rank_shifts.items():
            disease_slug = disease.lower().replace(' ', '_')
            shift_path = output_dir / f"rank_shifts_{hops}hop_{disease_slug}.parquet"
            df.to_parquet(shift_path, index=False)
        print(f"  Saved rank shift files for {len(all_rank_shifts)} diseases")
        
        # Metapath analysis (if available)
        if has_metapaths and all_metapath_analysis:
            metapath_path = output_dir / f"metapath_analysis_{hops}hop.json"
            with open(metapath_path, 'w') as f:
                json.dump({'hops': hops, 'results': all_metapath_analysis}, f, indent=2, default=str)
            print(f"  Saved: {metapath_path}")
            
            metapath_changes_path = output_dir / f"metapath_rank_changes_{hops}hop.json"
            with open(metapath_changes_path, 'w') as f:
                json.dump({'hops': hops, 'results': all_metapath_rank_changes}, f, indent=2, default=str)
            print(f"  Saved: {metapath_changes_path}")
        
        # Diagnostics - path count analysis to detect combinatorics confounds
        print("\nComputing context diagnostics...")
        diagnostics = compute_context_diagnostics(results_by_context, hops)
        diagnostics_path = output_dir / f"diagnostics_{hops}hop.json"
        with open(diagnostics_path, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"  Saved: {diagnostics_path}")
        
        # Print diagnostic summary
        print("\n  Context diagnostics summary:")
        for ctx, stats in diagnostics['contexts'].items():
            if 'error' in stats:
                continue
            print(f"    {ctx}:")
            print(f"      Gene-Disease pairs: {stats['total_gene_disease_pairs']:,}")
            print(f"      Total paths: {stats['path_counts'].get('total', 'N/A'):,}")
            if stats.get('expanded_edge_usage'):
                pct = stats['expanded_edge_usage'].get('pct_with_any_expanded', 0)
                print(f"      Results using expanded edges: {pct:.1f}%")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()