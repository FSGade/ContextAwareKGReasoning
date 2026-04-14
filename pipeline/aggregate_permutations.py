#!/usr/bin/env python3
"""
RQ2 Aggregate Permutations — Per-Triple P-values with FDR

Enhanced version with:
  - tqdm progress bar
  - Optional focused subset filtering (phenotype + shared coverage)
  - Handles both pre-filtered and unfiltered permutation files
  - Saves focused subset parquet alongside full comparison

Usage:
    # Default: compute p-values for ALL triples in comparison file
    python aggregate_permutations.py \
        --comparison subcut_vs_visceral --config config.yaml

    # Focused: filter to phenotype whitelist + shared coverage first
    python aggregate_permutations.py \
        --comparison subcut_vs_visceral --config config.yaml \
        --phenotype-filter --shared-only

    # Check for missing permutations
    python aggregate_permutations.py \
        --comparison subcut_vs_visceral --config config.yaml --check-missing
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import yaml
import numpy as np
import pandas as pd

from utils import load_config

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False



def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (NaN-safe)."""
    n = len(pvalues)
    qvalues = np.full(n, np.nan)
    valid = ~np.isnan(pvalues)
    m = valid.sum()
    if m == 0:
        return qvalues

    idx = np.where(valid)[0]
    pv = pvalues[idx]

    order = np.argsort(pv)
    sorted_pv = pv[order]

    ranks = np.arange(1, m + 1)
    adjusted = sorted_pv * m / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(m)
    result[order] = adjusted
    qvalues[idx] = result

    return qvalues


def discover_permutation_files(perm_dir: Path, hops: int) -> List[Path]:
    """Find all perm_XXXX_{hops}hop.parquet files."""
    pattern = f"perm_*_{hops}hop.parquet"
    files = sorted(perm_dir.glob(pattern))
    return files


def load_observed_comparison(comp_dir: Path, hops: int) -> pd.DataFrame:
    """Load the observed comparison parquet."""
    path = comp_dir / f'comparison_{hops}hop.parquet'
    if not path.exists():
        raise FileNotFoundError(f"Observed comparison not found: {path}")
    return pd.read_parquet(path)


def apply_phenotype_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter to focused phenotype whitelist."""
    # Use config phenotypes if available, otherwise use hardcoded set
    perm_config = config.get('permutation', {})
    phenotypes = set(perm_config['focus_phenotypes'])

    mask = df['target_phenotype'].isin(phenotypes)
    filtered = df[mask].copy()
    print(f"    Phenotype filter: {len(df):,} -> {len(filtered):,} "
          f"({len(phenotypes)} phenotypes)")
    return filtered


def apply_shared_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Filter to triples with both coverages > threshold."""
    perm_config = config.get('permutation', {})
    min_cov = perm_config.get('min_coverage_both', 0.05)

    if 'coverage_A' not in df.columns or 'coverage_B' not in df.columns:
        print(f"    WARNING: coverage_A/B columns not found — skipping shared filter")
        return df

    mask = (
        (df['coverage_A'].fillna(0) > min_cov) &
        (df['coverage_B'].fillna(0) > min_cov)
    )
    filtered = df[mask].copy()
    print(f"    Shared filter (> {min_cov}): {len(df):,} -> {len(filtered):,}")
    return filtered


def compute_per_triple_pvalues(
    observed_df: pd.DataFrame,
    perm_dir: Path,
    hops: int,
) -> pd.DataFrame:
    """
    Compute per-triple empirical p-values from permutation files.

    Enhanced with tqdm progress bar or frequent print updates.
    Handles both pre-filtered (small) and unfiltered (large) perm files.
    """
    KEY_COLS = ['source_gene', 'target_phenotype', 'metapath']
    SEP = '\x00'

    # Verify columns exist
    for col in KEY_COLS + ['diff_coverage']:
        if col not in observed_df.columns:
            raise KeyError(f"Observed comparison missing column: {col}")

    # Build composite key -> integer index
    print(f"  Building key index from observed data...", flush=True)
    ref = observed_df.reset_index(drop=True)
    n_triples = len(ref)

    obs_keys = (ref['source_gene'].astype(str).values + SEP +
                ref['target_phenotype'].astype(str).values + SEP +
                ref['metapath'].astype(str).values)

    key_to_idx = {}
    for i, k in enumerate(obs_keys):
        key_to_idx[k] = i

    abs_diff_obs = ref['diff_coverage'].abs().values
    print(f"  Index built: {len(key_to_idx):,} unique keys from {n_triples:,} rows", flush=True)

    # Discover permutation files
    perm_files = discover_permutation_files(perm_dir, hops)
    n_perms = len(perm_files)

    if n_perms == 0:
        print(f"  WARNING: No permutation files found for {hops}-hop in {perm_dir}", flush=True)
        result = ref[KEY_COLS].copy()
        result['perm_pvalue'] = np.nan
        result['perm_qvalue'] = np.nan
        result['perm_n_seen'] = 0
        return result

    print(f"  Processing {n_perms} permutation files for {hops}-hop...", flush=True)
    if HAS_TQDM:
        print(f"  Using tqdm progress bar", flush=True)
    else:
        print(f"  Install tqdm for better progress tracking: pip install tqdm", flush=True)
        print(f"  Progress updates every 10 files...", flush=True)

    # Peek at first file to check structure
    peek = pd.read_parquet(perm_files[0])
    print(f"  Perm file columns: {peek.columns.tolist()}", flush=True)
    print(f"  Perm file rows: {len(peek):,} "
          f"({'pre-filtered' if len(peek) < 10000 else 'unfiltered'})", flush=True)
    for col in KEY_COLS + ['diff_coverage']:
        if col not in peek.columns:
            raise KeyError(
                f"Permutation file missing column: {col}. "
                f"Available: {peek.columns.tolist()}. "
                f"Re-run permutations with updated run_permutation.py."
            )

    # Accumulators
    n_extreme = np.zeros(n_triples, dtype=np.int32)
    n_seen = np.zeros(n_triples, dtype=np.int32)

    t0 = time.time()

    # Progress tracking with tqdm or manual updates
    if HAS_TQDM:
        iterator = tqdm(enumerate(perm_files), total=n_perms,
                       desc="  Aggregating", unit="file",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        iterator = enumerate(perm_files)

    for i, pf in iterator:
        # Manual progress update every 10 files if no tqdm
        if not HAS_TQDM and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_perms - i - 1) / rate
            print(f"    [{i + 1:4d}/{n_perms}] {rate:5.1f} files/s | "
                  f"Elapsed: {elapsed/60:5.1f}m | ETA: {eta/60:5.1f}m", flush=True)

        try:
            pdf = pd.read_parquet(pf, columns=KEY_COLS + ['diff_coverage'])
        except Exception as e:
            if not HAS_TQDM:
                print(f"    WARNING: Failed to load {pf}: {e}", flush=True)
            continue

        # Build composite keys
        perm_keys = (pdf['source_gene'].astype(str).values + SEP +
                     pdf['target_phenotype'].astype(str).values + SEP +
                     pdf['metapath'].astype(str).values)

        # Dict lookup
        idx_arr = np.array([key_to_idx.get(k, -1) for k in perm_keys],
                           dtype=np.int32)

        # Filter to matched rows
        matched_mask = idx_arr >= 0
        if not matched_mask.any():
            if i == 0:
                print(f"  ⚠ First file: ZERO matches!", flush=True)
            continue

        matched_idx = idx_arr[matched_mask]
        perm_abs = np.abs(pdf['diff_coverage'].values[matched_mask])
        obs_abs_matched = abs_diff_obs[matched_idx]

        # Track testability
        seen_counts = np.bincount(matched_idx, minlength=n_triples)
        n_seen += (seen_counts > 0).astype(np.int32)

        # Count extreme values
        extreme_mask = perm_abs >= obs_abs_matched
        if extreme_mask.any():
            extreme_idx = matched_idx[extreme_mask]
            n_extreme += np.bincount(extreme_idx,
                                     minlength=n_triples).astype(np.int32)

        # Diagnostic on first file
        if i == 0:
            n_matched = int(matched_mask.sum())
            msg = (f"  First file: {len(pdf):,} rows, {n_matched:,} matched "
                   f"({100*n_matched/max(len(pdf),1):.1f}%)")
            if HAS_TQDM:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    elapsed_total = time.time() - t0
    print(f"\n  Done in {elapsed_total/60:.1f} min ({n_perms/max(elapsed_total,1):.1f} files/s)", flush=True)

    # Compute p-values
    pvalues = np.full(n_triples, np.nan)
    testable = n_seen > 0
    pvalues[testable] = (n_extreme[testable] + 1).astype(np.float64) / (n_perms + 1)

    qvalues = benjamini_hochberg(pvalues)

    # Testability report
    total_testable = testable.sum()
    total_untestable = n_triples - total_testable
    print(f"\n  Testability report:", flush=True)
    print(f"    Total observed triples: {n_triples:,}", flush=True)
    print(f"    Testable (seen >= 1):   {total_testable:,} ({100*total_testable/n_triples:.1f}%)", flush=True)
    print(f"    Untestable (seen = 0):  {total_untestable:,} ({100*total_untestable/n_triples:.1f}%)", flush=True)
    for thresh in [10, 100, 500, 900]:
        count = (n_seen >= thresh).sum()
        print(f"    Seen >= {thresh:4d} perms:     {count:,}", flush=True)

    # Build result
    result = ref[KEY_COLS].copy()
    result['perm_pvalue'] = pvalues
    result['perm_qvalue'] = qvalues
    result['perm_n_seen'] = n_seen
    return result


def merge_pvalues_into_comparison(comp_dir: Path, pvalues_df: pd.DataFrame,
                                   hops: int) -> pd.DataFrame:
    """Merge perm_pvalue and perm_qvalue into comparison parquet."""
    KEY_COLS = ['source_gene', 'target_phenotype', 'metapath']
    comp_path = comp_dir / f'comparison_{hops}hop.parquet'
    comp_df = pd.read_parquet(comp_path)

    # Drop old columns if re-running
    for col in ['perm_pvalue', 'perm_qvalue', 'perm_n_seen']:
        if col in comp_df.columns:
            comp_df = comp_df.drop(columns=[col])

    merged = comp_df.merge(pvalues_df, on=KEY_COLS, how='left')
    merged.to_parquet(comp_path, index=False)

    print(f"  Updated {comp_path}", flush=True)
    print(f"    Triples with p-value: {merged['perm_pvalue'].notna().sum():,} / {len(merged):,}", flush=True)

    return merged


def save_focused_subset(comp_dir: Path, merged_df: pd.DataFrame,
                         hops: int, config: dict) -> Optional[pd.DataFrame]:
    """
    Save a separate parquet with only the focused subset (phenotype + shared).
    This is the subset that was actually tested and has valid p-values.
    Useful for downstream focused analysis and volcano plots.
    """
    perm_config = config.get('permutation', {})
    phenotypes = set(perm_config['focus_phenotypes'])
    min_cov = perm_config.get('min_coverage_both', 0.05)

    mask_pheno = merged_df['target_phenotype'].isin(phenotypes)

    mask_cov = pd.Series(True, index=merged_df.index)
    if 'coverage_A' in merged_df.columns and 'coverage_B' in merged_df.columns:
        mask_cov = (
            (merged_df['coverage_A'].fillna(0) > min_cov) &
            (merged_df['coverage_B'].fillna(0) > min_cov)
        )

    focused = merged_df[mask_pheno & mask_cov].copy()

    if len(focused) == 0:
        print(f"  WARNING: No triples in focused subset for {hops}-hop")
        return None

    out_path = comp_dir / f'comparison_{hops}hop_focused.parquet'
    focused.to_parquet(out_path, index=False)
    print(f"  Saved focused subset: {out_path} ({len(focused):,} triples)")

    # Summary for focused subset
    if 'perm_qvalue' in focused.columns:
        n_sig_005 = (focused['perm_qvalue'].fillna(1) < 0.05).sum()
        n_sig_010 = (focused['perm_qvalue'].fillna(1) < 0.10).sum()
        print(f"    FDR q < 0.05: {n_sig_005:,}")
        print(f"    FDR q < 0.10: {n_sig_010:,}")

    return focused


def compute_summary_stats(merged_df: pd.DataFrame, n_perms: int,
                          config: dict) -> Dict:
    """Compute summary statistics."""
    thresholds = config['analysis'].get('tissue_specific', {})
    high_thresh = thresholds.get('diff_coverage_threshold', 0.3)
    alpha = config['permutation'].get('alpha', 0.05)

    diff_cov = merged_df['diff_coverage'].fillna(0)
    pv = merged_df['perm_pvalue']
    qv = merged_df['perm_qvalue']

    stats = {
        'n_permutations': n_perms,
        'n_triples_total': len(merged_df),
        'n_testable': int(pv.notna().sum()),
        'n_untestable': int(pv.isna().sum()),
        'n_significant_raw_005': int((pv < 0.05).sum()),
        'n_significant_raw_001': int((pv < 0.01).sum()),
        'n_significant_fdr_005': int((qv < 0.05).sum()),
        'n_significant_fdr_010': int((qv < 0.10).sum()),
        'n_significant_fdr_001': int((qv < 0.01).sum()),
        'n_high_diff': int((diff_cov.abs() > high_thresh).sum()),
        'n_high_diff_and_sig': int(
            ((diff_cov.abs() > high_thresh) & (qv < alpha)).sum()),
        'mean_pvalue': float(pv.mean()) if pv.notna().any() else None,
        'median_pvalue': float(pv.median()) if pv.notna().any() else None,
        'min_pvalue': float(pv.min()) if pv.notna().any() else None,
    }

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate permutations into per-triple p-values')
    parser.add_argument('--comparison', type=str, required=True,
                        choices=['subcut_vs_visceral', 'white_vs_brown'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--check-missing', action='store_true',
                        help='Only check for missing permutations')
    parser.add_argument('--phenotype-filter', action='store_true',
                        help='Filter to focused phenotype whitelist before computing p-values')
    parser.add_argument('--shared-only', action='store_true',
                        help='Filter to triples with both coverages > threshold')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])
    perm_dir = output_dir / 'permutations' / args.comparison
    comp_dir = output_dir / 'comparisons' / args.comparison

    n_permutations = config['permutation'].get('n_permutations', 1000)
    hop_lengths = config['permutation'].get('hops_for_permutation', [2])
    if isinstance(hop_lengths, int):
        hop_lengths = [hop_lengths]

    # Check for missing permutations
    if args.check_missing:
        for hops in hop_lengths:
            found = len(discover_permutation_files(perm_dir, hops))
            missing = n_permutations - found
            if missing > 0:
                print(f"  {hops}-hop: {found}/{n_permutations} complete "
                      f"({missing} missing)")
            else:
                print(f"  {hops}-hop: all {n_permutations} complete")
        return

    # Process each hop length
    overall_summary = {}

    for hops in hop_lengths:
        print(f"\n{'='*60}")
        print(f"Processing {hops}-hop permutations")
        print('='*60)

        try:
            observed_df = load_observed_comparison(comp_dir, hops)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            continue

        print(f"  Observed comparison: {len(observed_df):,} triples", flush=True)

        # --- Apply focused filters if requested ---
        # These are a priori filters (independent of test statistic):
        #   1. Phenotype whitelist (biological scope)
        #   2. Shared coverage (data availability, per Bourgon et al. 2010)
        # When perm files are pre-filtered, these are redundant but harmless.
        if args.phenotype_filter:
            observed_df = apply_phenotype_filter(observed_df, config)
        if args.shared_only:
            observed_df = apply_shared_filter(observed_df, config)

        if len(observed_df) == 0:
            print(f"  WARNING: No triples remaining after filtering for {hops}-hop")
            continue

        print(f"  Testing {len(observed_df):,} triples", flush=True)

        # Theoretical FDR floor
        n_perm_files = len(discover_permutation_files(perm_dir, hops))
        if n_perm_files > 0:
            min_p = 1.0 / (n_perm_files + 1)
            fdr_floor = min_p * len(observed_df)
            print(f"  Theoretical FDR floor: {fdr_floor:.4f} "
                  f"(min p={min_p:.6f}, m={len(observed_df):,}, "
                  f"n_perms={n_perm_files})", flush=True)

        # Compute per-triple p-values
        pvalues_df = compute_per_triple_pvalues(observed_df, perm_dir, hops)

        if pvalues_df['perm_pvalue'].isna().all():
            print(f"  WARNING: No p-values computed for {hops}-hop")
            continue

        # Merge into comparison parquet (adds p-values to the FULL file)
        merged = merge_pvalues_into_comparison(comp_dir, pvalues_df, hops)

        # Save focused subset as separate file
        save_focused_subset(comp_dir, merged, hops, config)

        # Compute summary
        n_perms_actual = len(discover_permutation_files(perm_dir, hops))
        summary = compute_summary_stats(merged, n_perms_actual, config)
        overall_summary[f'{hops}hop'] = summary

        # Also compute focused-only summary
        if args.phenotype_filter or args.shared_only:
            focused_summary = compute_summary_stats(
                merged[merged['perm_pvalue'].notna()], n_perms_actual, config)
            overall_summary[f'{hops}hop_focused'] = focused_summary

        # Print summary
        print(f"\n  Summary ({hops}-hop):")
        print(f"    Permutations loaded:          {n_perms_actual}")
        print(f"    Triples tested:               {summary['n_testable']:,}")
        print(f"    Triples untestable:           {summary['n_untestable']:,}")
        print(f"    Significant (raw p<0.05):      {summary['n_significant_raw_005']:,}")
        print(f"    Significant (FDR q<0.05):      {summary['n_significant_fdr_005']:,}")
        print(f"    Significant (FDR q<0.10):      {summary['n_significant_fdr_010']:,}")
        print(f"    High diff & significant:       {summary['n_high_diff_and_sig']:,}")

    # Save summary JSON
    summary_path = perm_dir / 'permutation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    print(f"\n Saved summary to: {summary_path}")


if __name__ == '__main__':
    main()