#!/usr/bin/env python3
"""
RQ2 Enrichment Analysis — Over-Representation Analysis on LDA topic assignments.

For each comparison, independently per tissue:
  - Foreground: topic counts from tissue-matching edges
  - Background: topic counts from non-matching edges
  - Fisher's exact test per topic, BH FDR correction

Graceful degradation:
  - Missing topic files → skip enrichment, exit(0)
  - Missing LDA descriptions → use numeric IDs
  - Empty metapath data → skip that metapath

Usage:
    python enrichment.py --comparison subcut_vs_visceral --hops 2 --config config.yaml
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from utils import load_config

FDR_ALPHA = 0.05
MIN_TOPIC_COUNT = 5


def load_metapath_topics(path: Path) -> Optional[Dict]:
    """
    Load per-metapath topic counts from run_psr.py output.

    Handles both the current foreground/background format and legacy flat format.
    Returns None if the file doesn't exist.
    """
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    converted = {}
    for key, fields in data.items():
        entry = {}

        if 'foreground' in fields:
            entry['foreground'] = {}
            entry['background'] = {}
            for field in ['mechanisms', 'pathways']:
                fg = fields.get('foreground', {}).get(field, {})
                bg = fields.get('background', {}).get(field, {})
                entry['foreground'][field] = Counter({int(k): int(v) for k, v in fg.items()})
                entry['background'][field] = Counter({int(k): int(v) for k, v in bg.items()})
            entry['n_fg_edges'] = fields.get('n_fg_edges', 0)
            entry['n_bg_edges'] = fields.get('n_bg_edges', 0)
        else:
            # Legacy flat format
            entry['foreground'] = {'mechanisms': Counter(), 'pathways': Counter()}
            entry['background'] = {}
            for field in ['mechanisms', 'pathways']:
                if field in fields and fields[field]:
                    entry['background'][field] = Counter(
                        {int(k): int(v) for k, v in fields[field].items()})
                else:
                    entry['background'][field] = Counter()
            entry['n_fg_edges'] = 0
            entry['n_bg_edges'] = fields.get('n_edge_observations', 0)

        converted[key] = entry
    return converted


def load_topic_descriptions(lda_dir: Path) -> Dict[str, Dict[int, Dict]]:
    """Load LDA topic labels. Returns empty dicts if files don't exist."""
    descriptions = {}
    for field in ['mechanisms', 'pathways']:
        topics_path = lda_dir / field / 'topics.json'
        if topics_path.exists():
            with open(topics_path) as f:
                data = json.load(f)
            descriptions[field] = {
                t['topic_id']: {
                    'label': t.get('label', f"Topic {t['topic_id']}"),
                    'top_terms': [w['term'] for w in t.get('top_terms', [])[:5]],
                }
                for t in data.get('topics', [])
            }
        else:
            descriptions[field] = {}
    return descriptions


def fishers_test_topics(foreground: Counter, background: Counter,
                        min_count: int = MIN_TOPIC_COUNT) -> pd.DataFrame:
    """
    Fisher's exact test for each topic.

    2x2 table per topic:
                      Has topic   Doesn't have topic
    Foreground         a           b
    Background         c           d
    """
    all_topics = set(foreground.keys()) | set(background.keys())
    n_fg = sum(foreground.values())
    n_bg = sum(background.values())

    if n_fg == 0 or n_bg == 0:
        return pd.DataFrame()

    results = []
    for topic_id in sorted(all_topics):
        a = foreground.get(topic_id, 0)
        c = background.get(topic_id, 0)
        if a + c < min_count:
            continue

        b, d = n_fg - a, n_bg - c
        odds_ratio, p_value = scipy_stats.fisher_exact(
            np.array([[a, b], [c, d]]), alternative='two-sided')

        # Log2 fold change with +0.5 pseudocounts
        rate_fg = (a + 0.5) / (n_fg + 1)
        rate_bg = (c + 0.5) / (n_bg + 1)

        results.append({
            'topic_id': int(topic_id),
            'count_foreground': int(a),
            'count_background': int(c),
            'total_count': int(a + c),
            'n_foreground': int(n_fg),
            'n_background': int(n_bg),
            'pct_foreground': round(100 * a / n_fg, 2),
            'pct_background': round(100 * c / n_bg, 2),
            'odds_ratio': round(float(odds_ratio), 4) if not np.isinf(odds_ratio) else 999.0,
            'log2_fold_change': round(float(np.log2(rate_fg / rate_bg)), 4),
            'p_value': float(p_value),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Benjamini-Hochberg correction
    n_tests = len(df)
    sorted_idx = np.argsort(df['p_value'].values)
    sorted_p = df['p_value'].values[sorted_idx]

    fdr_q = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        rank = i + 1
        raw = sorted_p[i] * n_tests / rank
        fdr_q[sorted_idx[i]] = min(1.0, raw) if i == n_tests - 1 else min(fdr_q[sorted_idx[i + 1]], raw)

    df['fdr_q'] = np.round(fdr_q, 6)
    df['significant'] = df['fdr_q'] < FDR_ALPHA
    return df.sort_values('p_value').reset_index(drop=True)


def add_topic_labels(df: pd.DataFrame, topic_descs: Dict[int, Dict]) -> pd.DataFrame:
    """Add human-readable labels to enrichment results."""
    if df.empty:
        return df
    df['topic_label'] = [
        topic_descs.get(tid, {}).get('label', f'Topic {tid}')
        for tid in df['topic_id']
    ]
    df['top_terms'] = [
        ', '.join(topic_descs.get(tid, {}).get('top_terms', []))
        for tid in df['topic_id']
    ]
    return df


def run_enrichment_for_comparison(config: dict, comparison_name: str,
                                  hops: int) -> Dict:
    """
    Run independent enrichment per tissue for one comparison.

    Only tests triples whose target phenotype is in the configured whitelist.
    """
    output_dir = Path(config['paths']['output_dir'])
    inference_dir = output_dir / 'inference'

    comp_config = next(
        (c for c in config['comparisons'] if c['name'] == comparison_name), None)
    if comp_config is None:
        print(f"ERROR: Comparison '{comparison_name}' not found in config")
        sys.exit(1)

    tissue_A = comp_config['tissue_A']
    tissue_B = comp_config['tissue_B']

    print(f"\n{'=' * 80}")
    print(f"ENRICHMENT: {comparison_name} ({hops}-hop)")
    print(f"  Tissue A: {tissue_A}, Tissue B: {tissue_B}")
    print(f"{'=' * 80}")

    # Target phenotype whitelist
    enr_config = config.get('enrichment', {})
    target_phenotypes = set(enr_config.get('target_phenotypes', [
        'Inflammation', 'Insulin Resistance', 'Obesity',
        'Metabolic Diseases', 'Diabetes Mellitus, Type 2',
    ]))
    target_lower = {t.lower() for t in target_phenotypes}

    # Load topic counts
    topics_A = load_metapath_topics(
        inference_dir / f'{tissue_A}_{hops}hop_metapath_topics.json')
    topics_B = load_metapath_topics(
        inference_dir / f'{tissue_B}_{hops}hop_metapath_topics.json')

    if topics_A is None and topics_B is None:
        print("  WARNING: No metapath topic files found — skipping enrichment.")
        return {'status': 'skipped', 'reason': 'no_topic_data',
                'comparison': comparison_name, 'hops': hops}

    topics_A = topics_A or {}
    topics_B = topics_B or {}

    # Check for any foreground topics
    has_fg = any(
        entry.get('foreground', {}).get(f)
        for td in [topics_A, topics_B]
        for entry in td.values()
        for f in ['mechanisms', 'pathways']
    )
    if not has_fg:
        print("  WARNING: No foreground topic counts — skipping enrichment.")
        return {'status': 'skipped', 'reason': 'no_foreground_topics',
                'comparison': comparison_name, 'hops': hops}

    # Load LDA descriptions (optional)
    lda_dir = Path(config['paths'].get(
        'lda_output_dir', str(output_dir / 'lda_output')))
    topic_descs = load_topic_descriptions(lda_dir)

    SEP = '|||'

    def filter_keys_by_phenotype(topics_dict):
        filtered = set()
        for key in topics_dict:
            if SEP in key:
                parts = key.split(SEP)
                if len(parts) >= 3 and parts[1].lower() in target_lower:
                    filtered.add(key)
        return filtered

    keys_A = filter_keys_by_phenotype(topics_A)
    keys_B = filter_keys_by_phenotype(topics_B)
    print(f"  {tissue_A}: {len(keys_A)} triples matching phenotype filter")
    print(f"  {tissue_B}: {len(keys_B)} triples matching phenotype filter")

    # Run enrichment independently per tissue
    tissue_results = {}

    for tissue_name, topics_dict, keys in [
        (tissue_A, topics_A, keys_A),
        (tissue_B, topics_B, keys_B),
    ]:
        print(f"\n  --- {tissue_name.upper()} ---")
        tissue_results[tissue_name] = {}

        for field in ['mechanisms', 'pathways']:
            field_descs = topic_descs.get(field, {})
            field_results = []
            n_tested = 0

            for key in sorted(keys):
                entry = topics_dict.get(key, {})
                fg = entry.get('foreground', {}).get(field, Counter())
                bg = entry.get('background', {}).get(field, Counter())
                if not fg:
                    continue

                n_tested += 1
                parts = key.split(SEP)
                source_node = parts[0] if len(parts) >= 3 else None
                target_node = parts[1] if len(parts) >= 3 else None
                metapath_str = parts[2] if len(parts) >= 3 else key

                df = fishers_test_topics(fg, bg)
                if not df.empty:
                    df = add_topic_labels(df, field_descs)
                    df['tissue'] = tissue_name
                    df['source'] = source_node
                    df['target'] = target_node
                    df['metapath'] = metapath_str
                    df['n_fg_edges'] = entry.get('n_fg_edges', 0)
                    df['n_bg_edges'] = entry.get('n_bg_edges', 0)
                    field_results.append(df)

            if field_results:
                combined = pd.concat(field_results, ignore_index=True)
                tissue_results[tissue_name][field] = combined
                n_sig = int(combined['significant'].sum())
                print(f"    {field.upper()}: {n_tested} triples tested, "
                      f"{len(combined)} tests, {n_sig} significant (FDR < {FDR_ALPHA})")
            else:
                print(f"    {field.upper()}: No testable triples")

    # Save results
    enrichment_dir = output_dir / 'enrichment'
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'comparison': comparison_name,
        'tissue_A': tissue_A, 'tissue_B': tissue_B,
        'hops': hops, 'timestamp': datetime.now().isoformat(),
        'fdr_alpha': FDR_ALPHA, 'min_topic_count': MIN_TOPIC_COUNT,
        'target_phenotypes': sorted(target_phenotypes),
        'tissues': {},
    }

    for tissue_name in [tissue_A, tissue_B]:
        tissue_data = {'fields': {}}
        for field, df in tissue_results.get(tissue_name, {}).items():
            csv_path = (enrichment_dir /
                        f'{comparison_name}_{hops}hop_{tissue_name}_{field}_enrichment.csv')
            df.to_csv(csv_path, index=False)
            tissue_data['fields'][field] = {
                'n_tests': len(df),
                'n_significant': int(df['significant'].sum()),
            }
        output_data['tissues'][tissue_name] = tissue_data

    json_path = enrichment_dir / f'{comparison_name}_{hops}hop_enrichment.json'
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    output_data['status'] = 'completed'
    return output_data


def main():
    parser = argparse.ArgumentParser(description='RQ2 Enrichment Analysis')
    parser.add_argument('--comparison', type=str, required=True)
    parser.add_argument('--hops', type=int, required=True, choices=[2, 3])
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_enrichment_for_comparison(config, args.comparison, args.hops)

    if result.get('status') == 'skipped':
        print(f"\n⚠ Enrichment skipped: {result.get('reason')}")
        sys.exit(0)
    else:
        print(f"\n✓ Enrichment analysis complete")


if __name__ == '__main__':
    main()