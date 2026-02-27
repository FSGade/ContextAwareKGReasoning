#!/usr/bin/env python3
"""
RQ2 Report Generator - Matching RQ1 style
Now with Enrichment tab for LDA topic enrichment results.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from utils import load_config

import yaml
import numpy as np
import pandas as pd

MAX_GENE_LIST = 500  # Truncate intermediate gene lists to this length in HTML


def safe_json_dumps(obj):
    def convert(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            if np.isnan(o) or np.isinf(o):
                return None
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if pd.isna(o):
            return None
        return o
    return json.dumps(obj, default=convert, ensure_ascii=False)


def _truncate_gene_lists(record: dict) -> dict:
    """Truncate intermediate gene lists to avoid massive HTML files."""
    for key in ('intermediate_genes', 'intermediate_genes_A', 'intermediate_genes_B',
                'intermediate_genes_C'):
        val = record.get(key)
        if isinstance(val, list) and len(val) > MAX_GENE_LIST:
            record[key] = val[:MAX_GENE_LIST]
    return record


def load_tissue_results(inference_dir: Path, tissue: str, hops: int, max_rows: int = 50000) -> List[dict]:
    parquet_path = inference_dir / f"{tissue}_{hops}hop.parquet"
    if not parquet_path.exists():
        print(f"  WARNING: {parquet_path} not found")
        return []
    
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    
    if len(df) > max_rows:
        df = df.head(max_rows)
        print(f"  (limited to top {max_rows:,} of {total_rows:,} rows)")
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                record[col] = list(val) if hasattr(val, '__len__') else []
            elif isinstance(val, (np.integer, np.int64)):
                record[col] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                record[col] = None if (np.isnan(val) or np.isinf(val)) else float(val)
            elif pd.isna(val):
                record[col] = None
            else:
                record[col] = val
        
        relations = record.get('relationship_types', []) or []
        record['rel1'] = relations[0] if len(relations) > 0 else '?'
        record['rel2'] = relations[1] if len(relations) > 1 else '?'
        if hops == 3:
            record['rel3'] = relations[2] if len(relations) > 2 else '?'
        _truncate_gene_lists(record)
        records.append(record)
    
    print(f"  {tissue}: {len(records):,} results loaded")
    return records


def load_comparison_results(comp_dir: Path, hops: int, max_rows: int = 50000) -> List[dict]:
    parquet_path = comp_dir / f"comparison_{hops}hop.parquet"
    if not parquet_path.exists():
        print(f"  WARNING: {parquet_path} not found")
        return []
    
    df = pd.read_parquet(parquet_path)
    total_rows = len(df)
    
    if len(df) > max_rows:
        df['max_prob'] = df[['prob_A', 'prob_B']].max(axis=1)
        df['max_evidence'] = df[['evidence_A', 'evidence_B']].max(axis=1)
        if 'tissue_specificity' not in df.columns:
            df['tissue_specificity'] = df['diff_coverage'].abs()
        df = df.sort_values(['max_prob', 'max_evidence', 'tissue_specificity'], ascending=[False, False, False]).head(max_rows)
        df = df.drop(columns=['max_prob', 'max_evidence'])
        print(f"  (limited to top {max_rows:,} by probability of {total_rows:,} rows)")
    records = []
    for _, row in df.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (list, np.ndarray)):
                record[col] = list(val) if hasattr(val, '__len__') else []
            elif isinstance(val, (np.integer, np.int64)):
                record[col] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                record[col] = None if (np.isnan(val) or np.isinf(val)) else float(val)
            elif pd.isna(val):
                record[col] = None
            else:
                record[col] = val
        
        relations = record.get('relationship_types', []) or []
        record['rel1'] = relations[0] if len(relations) > 0 else '?'
        record['rel2'] = relations[1] if len(relations) > 1 else '?'
        if hops == 3:
            record['rel3'] = relations[2] if len(relations) > 2 else '?'
        _truncate_gene_lists(record)
        records.append(record)
    
    print(f"  Comparison: {len(records):,} results loaded")
    return records


def load_enrichment_results(output_dir: Path, comparison_name: str, hops: int) -> dict:
    """Load enrichment JSON and CSVs for inclusion in report."""
    enrichment_dir = output_dir / 'enrichment'
    json_path = enrichment_dir / f'{comparison_name}_{hops}hop_enrichment.json'
    
    if not json_path.exists():
        print(f"  No enrichment results found at {json_path}")
        return None
    
    with open(json_path) as f:
        summary = json.load(f)
    
    # New per-tissue format: load CSVs per tissue per field
    tissues = summary.get('tissues', {})
    for tissue_name, tissue_data in tissues.items():
        for field in ['mechanisms', 'pathways']:
            csv_path = enrichment_dir / f'{comparison_name}_{hops}hop_{tissue_name}_{field}_enrichment.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                records = []
                for _, row in df.iterrows():
                    rec = {}
                    for col in df.columns:
                        val = row[col]
                        if isinstance(val, (np.integer, np.int64)):
                            rec[col] = int(val)
                        elif isinstance(val, (np.floating, np.float64)):
                            rec[col] = None if (np.isnan(val) or np.isinf(val)) else float(val)
                        elif pd.isna(val):
                            rec[col] = None
                        else:
                            rec[col] = val
                    records.append(rec)
                
                if 'fields' not in tissue_data:
                    tissue_data['fields'] = {}
                if field not in tissue_data['fields']:
                    tissue_data['fields'][field] = {}
                tissue_data['fields'][field]['all_results'] = records
                print(f"  Enrichment {tissue_name}/{field}: {len(records)} tests loaded "
                      f"({sum(1 for r in records if r.get('significant'))} significant)")
            else:
                # Try legacy format (flat, not per-tissue)
                legacy_path = enrichment_dir / f'{comparison_name}_{hops}hop_{field}_enrichment.csv'
                if legacy_path.exists():
                    print(f"  NOTE: Using legacy enrichment CSV for {field}")
    
    # Also support legacy 'fields' key (old format) for backward compat
    if not tissues and 'fields' in summary:
        for field in ['mechanisms', 'pathways']:
            csv_path = enrichment_dir / f'{comparison_name}_{hops}hop_{field}_enrichment.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                records = []
                for _, row in df.iterrows():
                    rec = {}
                    for col in df.columns:
                        val = row[col]
                        if isinstance(val, (np.integer, np.int64)):
                            rec[col] = int(val)
                        elif isinstance(val, (np.floating, np.float64)):
                            rec[col] = None if (np.isnan(val) or np.isinf(val)) else float(val)
                        elif pd.isna(val):
                            rec[col] = None
                        else:
                            rec[col] = val
                    records.append(rec)
                summary['fields'][field]['all_results'] = records
    
    return summary


# ---------------------------------------------------------------------------
# LaTeX table generation — metapath & hop-comparison tables
# ---------------------------------------------------------------------------

TISSUE_ABBREVS = {
    "subcutaneous": "SC", "visceral": "VS", "white": "W", "brown": "B",
}

LATEX_PREAMBLE = (
    "% Auto-generated by generate_report.py\n"
    "% Requires: booktabs, adjustbox\n"
)


def _tissue_abbrev(name: str) -> str:
    return TISSUE_ABBREVS.get(name.lower(), name[:2].upper())


def _latex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    for char, repl in [("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                       ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(char, repl)
    return s


def _fmt_latex(v, decimals=4):
    if v is None:
        return "---"
    try:
        if np.isnan(v) or np.isinf(v):
            return "---"
    except (TypeError, ValueError):
        pass
    return f"{v:.{decimals}f}"


def _fmt_latex_int(v):
    if v is None:
        return "---"
    return f"{int(v):,}"


def _fmt_latex_pct(v, decimals=1):
    if v is None:
        return "---"
    return f"{v:.{decimals}f}\\%"


def generate_latex_metapath_table(metapath_data, tissue_A, tissue_B, hops,
                                  max_rows=20) -> str:
    """Generate LaTeX table for metapath analysis summary."""
    tA = tissue_A.replace('_', ' ').title()
    tB = tissue_B.replace('_', ' ').title()
    abA = _tissue_abbrev(tissue_A)
    abB = _tissue_abbrev(tissue_B)

    if not metapath_data or len(metapath_data) == 0:
        return f"% No metapath data available for {hops}-hop"

    shown = min(len(metapath_data), max_rows)

    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        rf"\caption[Metapath analysis ({hops}-hop)]"
        rf"{{\textbf{{Metapath analysis summary ({hops}-hop, {tA} vs {tB}).}} "
        rf"Top {shown} metapaths by count. "
        rf"$\overline{{\Delta\text{{Cov}}}}$ = mean differential coverage; "
        rf"\%\,{abA}/\%\,{abB} = fraction biased toward each tissue; "
        rf"\#\,Spec = tissue-specific paths ($|\Delta\text{{Cov}}| > 0.3$).}}",
        rf"\label{{tab:metapath_{hops}hop}}",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{lrrrrc}",
        r"\toprule",
        rf"\textbf{{Metapath}} & \textbf{{Count}} & "
        rf"\textbf{{$\overline{{\Delta\text{{Cov}}}}$}} & "
        rf"\textbf{{\%\,{abA}}} & \textbf{{\%\,{abB}}} & "
        rf"\textbf{{\#\,Spec}} \\",
        r"\midrule",
    ]

    for mp in metapath_data[:max_rows]:
        metapath_str = _latex_escape(str(mp.get('metapath', '?')))
        # Wrap in \texttt for monospace
        metapath_str = r"\texttt{" + metapath_str + "}"
        count = _fmt_latex_int(mp.get('count'))
        mean_diff = _fmt_latex(mp.get('mean_diff_coverage'), 4)
        pct_A = _fmt_latex_pct(mp.get('pct_A_biased'), 1)
        pct_B = _fmt_latex_pct(mp.get('pct_B_biased'), 1)
        n_spec = _fmt_latex_int(mp.get('n_tissue_specific'))

        lines.append(
            f"{metapath_str} & {count} & {mean_diff} & "
            f"{pct_A} & {pct_B} & {n_spec} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}",
              r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def generate_latex_hop_comparison_tables(hop_data, tissue_A, tissue_B) -> str:
    """Generate LaTeX tables for hop comparison (coverage decay, Jaccard, specificity)."""
    if not hop_data or not hop_data.get('coverage_decay'):
        return "% No hop comparison data available"

    tA = tissue_A.replace('_', ' ').title()
    tB = tissue_B.replace('_', ' ').title()
    abA = _tissue_abbrev(tissue_A)
    abB = _tissue_abbrev(tissue_B)

    sections = []

    # --- Coverage Decay Table ---
    cd = hop_data['coverage_decay']
    n_2 = hop_data.get('n_pairs_2hop')
    n_3 = hop_data.get('n_pairs_3hop')

    cap_extra = ""
    if n_2 is not None and n_3 is not None:
        cap_extra = f" 2-hop: {int(n_2):,} pairs; 3-hop: {int(n_3):,} pairs."

    lines = [
        r"\begin{table}[!htbp]", r"\centering",
        rf"\caption[Coverage decay (2-hop vs 3-hop)]"
        rf"{{\textbf{{Coverage decay across hop lengths ({tA} vs {tB}).}}"
        rf" Ratio = mean 3-hop / mean 2-hop coverage; lower values indicate "
        rf"greater information loss with longer paths.{cap_extra}}}",
        r"\label{tab:coverage_decay}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Tissue} & \textbf{2-hop} & \textbf{3-hop} & "
        r"\textbf{Ratio (3h/2h)} \\",
        r"\midrule",
        f"{tA} & {_fmt_latex(cd.get('mean_coverage_A_2hop'), 4)} & "
        f"{_fmt_latex(cd.get('mean_coverage_A_3hop'), 4)} & "
        f"{_fmt_latex(cd.get('decay_A'), 2)} \\\\",
        f"{tB} & {_fmt_latex(cd.get('mean_coverage_B_2hop'), 4)} & "
        f"{_fmt_latex(cd.get('mean_coverage_B_3hop'), 4)} & "
        f"{_fmt_latex(cd.get('decay_B'), 2)} \\\\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    sections.append("\n".join(lines))

    # --- Jaccard Overlap Table ---
    jaccard = hop_data.get('jaccard_overlap')
    if jaccard:
        lines = [
            r"\begin{table}[!htbp]", r"\centering",
            rf"\caption[Jaccard overlap of top source genes]"
            rf"{{\textbf{{Jaccard overlap of top source genes between "
            rf"2-hop and 3-hop ({tA} vs {tB}).}} "
            rf"Measures consistency of top-ranked genes across hop lengths.}}",
            r"\label{tab:jaccard_overlap}",
            r"\begin{tabular}{crrrr}",
            r"\toprule",
            r"\textbf{Top $K$} & \textbf{Jaccard} & \textbf{Shared} & "
            r"\textbf{Only 2-hop} & \textbf{Only 3-hop} \\",
            r"\midrule",
        ]
        for k, j in sorted(jaccard.items(), key=lambda x: int(x[0])):
            shared = j.get('shared') or j.get('intersection', 0)
            lines.append(
                f"{k} & {_fmt_latex(j.get('jaccard'), 3)} & "
                f"{int(shared)} & {int(j.get('only_2hop', 0))} & "
                f"{int(j.get('only_3hop', 0))} \\\\"
            )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        sections.append("\n".join(lines))

    # --- Tissue Specificity Table ---
    ts = hop_data.get('tissue_specificity')
    if ts:
        lines = [
            r"\begin{table}[!htbp]", r"\centering",
            rf"\caption[Tissue specificity (2-hop vs 3-hop)]"
            rf"{{\textbf{{Tissue specificity comparison ({tA} vs {tB}).}} "
            rf"High diff = $|\Delta\text{{Cov}}| > 0.3$.}}",
            r"\label{tab:tissue_specificity}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"& \textbf{2-hop} & \textbf{3-hop} \\",
            r"\midrule",
            f"Mean specificity & "
            f"{_fmt_latex(ts.get('mean_specificity_2hop'), 4)} & "
            f"{_fmt_latex(ts.get('mean_specificity_3hop'), 4)} \\\\",
            f"\\# High diff ($>0.3$) & "
            f"{_fmt_latex_int(ts.get('n_high_diff_2hop'))} & "
            f"{_fmt_latex_int(ts.get('n_high_diff_3hop'))} \\\\",
            r"\bottomrule", r"\end{tabular}", r"\end{table}",
        ]
        sections.append("\n".join(lines))

    return "\n\n\n".join(sections)


def save_latex_tables(metapath_data, hop_data, tissue_A, tissue_B,
                      hops, comparison_name, output_dir: Path):
    """Save metapath and hop-comparison tables as LaTeX files."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"report_{comparison_name}_{hops}hop"
    saved = []

    # Metapath table
    if metapath_data:
        tex = generate_latex_metapath_table(metapath_data, tissue_A, tissue_B, hops)
        out_path = tables_dir / f"{prefix}_metapath.tex"
        with open(out_path, "w") as f:
            f.write(LATEX_PREAMBLE + "\n" + tex + "\n")
        saved.append(out_path)
        print(f"    Saved: {out_path}")

    # Hop comparison tables (combined in one file)
    if hop_data:
        tex = generate_latex_hop_comparison_tables(hop_data, tissue_A, tissue_B)
        out_path = tables_dir / f"{prefix}_hop_comparison.tex"
        with open(out_path, "w") as f:
            f.write(LATEX_PREAMBLE + "\n" + tex + "\n")
        saved.append(out_path)
        print(f"    Saved: {out_path}")

    return saved


def generate_html_report(tissue_A_results, tissue_B_results, comparison_results,
                         tissue_A, tissue_B, hops, permutation_results=None,
                         metapath_analysis=None, hop_comparison=None, config=None,
                         enrichment_data=None) -> str:
    
    tissue_A_json = safe_json_dumps(tissue_A_results)
    tissue_B_json = safe_json_dumps(tissue_B_results)
    comparison_json = safe_json_dumps(comparison_results)
    permutation_json = safe_json_dumps(permutation_results) if permutation_results else 'null'
    metapath_json = safe_json_dumps(metapath_analysis) if metapath_analysis else 'null'
    hop_comparison_json = safe_json_dumps(hop_comparison) if hop_comparison else 'null'
    enrichment_json = safe_json_dumps(enrichment_data) if enrichment_data else 'null'
    
    n_total = len(comparison_results)
    n_tissue_specific = sum(1 for r in comparison_results if abs(r.get('diff_coverage', 0) or 0) > 0.3)
    n_A_biased = sum(1 for r in comparison_results if (r.get('diff_coverage', 0) or 0) > 0.1)
    n_B_biased = sum(1 for r in comparison_results if (r.get('diff_coverage', 0) or 0) < -0.1)
    n_fdr_sig = sum(1 for r in comparison_results if (r.get('perm_qvalue') or 1.0) < 0.05)
    
    tissue_A_label = tissue_A.replace('_', ' ').title()
    tissue_B_label = tissue_B.replace('_', ' ').title()
    
    if hops == 2:
        headers_tissue = '<tr><th></th><th>Rank</th><th>Source</th><th>Rel 1</th><th># Int</th><th>Rel 2</th><th>Phenotype</th><th>Prob</th><th>Evidence</th><th>Coverage</th><th>Corr</th><th>Intermediates</th><th>Enrichment</th></tr>'
        headers_comp = '<tr><th></th><th>Rank</th><th>Source</th><th>Rel 1</th><th># Int</th><th>Rel 2</th><th>Phenotype</th><th>Prob A</th><th>Prob B</th><th>Cov A</th><th>Cov B</th><th>Δ Cov</th><th>Log₂</th><th>p-val</th><th>q-val</th><th>Corr</th><th>Enr A</th><th>Enr B</th></tr>'
    else:
        headers_tissue = '<tr><th></th><th>Rank</th><th>Source</th><th>Rel 1</th><th># Int B</th><th>Rel 2</th><th># Int C</th><th>Rel 3</th><th>Phenotype</th><th>Prob</th><th>Evidence</th><th>Coverage</th><th>Corr</th><th>Int B</th><th>Int C</th><th>Enrichment</th></tr>'
        headers_comp = '<tr><th></th><th>Rank</th><th>Source</th><th>Rel 1</th><th># Int B</th><th>Rel 2</th><th># Int C</th><th>Rel 3</th><th>Phenotype</th><th>Prob A</th><th>Prob B</th><th>Cov A</th><th>Cov B</th><th>Δ Cov</th><th>Log₂</th><th>p-val</th><th>q-val</th><th>Corr</th><th>Enr A</th><th>Enr B</th></tr>'

    # Enrichment data loaded for inline display in tissue tabs
    enrichment_tab_btn = ''
    enrichment_tab_content = ''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RQ2: {tissue_A_label} vs {tissue_B_label} ({hops}-Hop)</title>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; font-size: 14px; }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        h1 {{ color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; }}
        h3 {{ color: #666; }}
        .tabs {{ display: flex; gap: 5px; border-bottom: 2px solid #dee2e6; }}
        .tab-btn {{ padding: 12px 24px; border: none; background: #e9ecef; cursor: pointer; border-radius: 6px 6px 0 0; font-size: 14px; font-weight: 500; }}
        .tab-btn:hover {{ background: #dee2e6; }}
        .tab-btn.active {{ background: white; color: #007bff; border-bottom: 2px solid white; margin-bottom: -2px; }}
        .tab-content {{ display: none; background: white; padding: 20px; border-radius: 0 0 8px 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .tab-content.active {{ display: block; }}
        .filters {{ display: flex; gap: 20px; align-items: center; margin-bottom: 15px; padding: 15px; background: #f8f9fa; border-radius: 6px; flex-wrap: wrap; }}
        .filter-group {{ display: flex; align-items: center; gap: 8px; }}
        .filter-group label {{ font-weight: 500; }}
        .filter-group select, .filter-group input {{ padding: 8px 12px; border-radius: 4px; border: 1px solid #ced4da; }}
        table.dataTable {{ width: 100% !important; font-size: 12px; }}
        table.dataTable thead th {{ background: #f8f9fa; font-weight: 600; font-size: 11px; }}
        .rel-badge {{ display: inline-block; background: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-family: monospace; }}
        .int-count {{ display: inline-block; background: #fff3e0; color: #e65100; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
        .intermediate-list {{ max-height: 60px; overflow-y: auto; font-family: monospace; font-size: 10px; background: #f8f9fa; padding: 4px 6px; border-radius: 4px; }}
        .corr-positive {{ color: #28a745; font-weight: bold; }}
        .corr-negative {{ color: #dc3545; font-weight: bold; }}
        .prob-high {{ color: #28a745; font-weight: 600; }}
        .prob-medium {{ color: #fd7e14; }}
        .cov-positive {{ color: #28a745; font-weight: 600; }}
        .cov-negative {{ color: #dc3545; font-weight: 600; }}
        .gene-name {{ font-weight: 600; color: #1a237e; }}
        .disease-name {{ color: #4a148c; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
        .stat-card.positive {{ border-left-color: #28a745; }}
        .stat-card.negative {{ border-left-color: #dc3545; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #007bff; }}
        .stat-card.positive .stat-value {{ color: #28a745; }}
        .stat-card.negative .stat-value {{ color: #dc3545; }}
        .stat-label {{ color: #666; font-size: 0.85em; margin-top: 5px; }}
        .perm-table {{ margin: 20px 0; border-collapse: collapse; }}
        .perm-table th, .perm-table td {{ padding: 10px 15px; border: 1px solid #dee2e6; }}
        .perm-table th {{ background: #f8f9fa; }}
        .sig-marker {{ color: #28a745; font-weight: bold; }}
        .metapath-table {{ width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 15px; }}
        .metapath-table th, .metapath-table td {{ padding: 8px 10px; border: 1px solid #dee2e6; }}
        .metapath-table th {{ background: #f8f9fa; }}
        .metapath-cell {{ font-family: monospace; font-size: 11px; }}
        .gene-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
        .gene-col {{ background: #f8f9fa; padding: 15px; border-radius: 8px; }}
        .gene-col h4 {{ margin: 0 0 10px 0; }}
        .gene-list {{ list-style: none; padding: 0; margin: 0; }}
        .gene-list li {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #eee; font-size: 13px; }}
        .gene-diff {{ font-family: monospace; font-weight: 600; }}
        .context-stats {{ display: flex; gap: 30px; margin-bottom: 15px; color: #666; font-size: 13px; }}
        .context-stats strong {{ color: #333; }}
    </style>
</head>
<body>
<div class="container">
    <h1>RQ2: {tissue_A_label} vs {tissue_B_label} ({hops}-Hop PSR Analysis)</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    
    <div class="tabs">
        <button class="tab-btn active" data-tab="comparison">Comparison</button>
        <button class="tab-btn" data-tab="tissue-a">{tissue_A_label}</button>
        <button class="tab-btn" data-tab="tissue-b">{tissue_B_label}</button>
        {enrichment_tab_btn}
        <button class="tab-btn" data-tab="summary">Summary & Stats</button>
    </div>
    
    <div id="tab-comparison" class="tab-content active">
        <h2>Tissue Comparison</h2>
        <div class="context-stats" id="stats-comparison"></div>
        <div class="filters">
            <div class="filter-group"><label>Phenotype:</label><select id="phenotype-filter-comparison"><option value="">All</option></select></div>
            <div class="filter-group"><label>Gene:</label><input type="text" id="gene-search-comparison" placeholder="Search..."></div>
            <div class="filter-group"><label>Source type:</label><select id="source-type-filter-comparison"><option value="">All</option></select></div>
            <div class="filter-group"><label>Min |Δ Cov|:</label><input type="number" id="diff-filter" min="0" max="1" step="0.1" value="0" style="width:70px;"></div>
            <div class="filter-group"><label>Max q-value:</label><input type="number" id="qval-filter" min="0" max="1" step="0.05" value="1" style="width:70px;"></div>
            <div class="filter-group"><label><input type="checkbox" id="enrich-filter-comparison"> Significant enrichment only</label></div>
            <div class="filter-group"><label>Mechanism:</label><select id="mech-topic-filter-comparison"><option value="">All</option></select></div>
            <div class="filter-group"><label>Pathway:</label><select id="path-topic-filter-comparison"><option value="">All</option></select></div>
        </div>
        <table id="table-comparison" class="display"><thead>{headers_comp}</thead><tbody></tbody></table>
    </div>
    
    <div id="tab-tissue-a" class="tab-content">
        <h2>{tissue_A_label} Results</h2>
        <div class="context-stats" id="stats-tissue-a"></div>
        <div class="filters">
            <div class="filter-group"><label>Phenotype:</label><select id="phenotype-filter-tissue-a"><option value="">All</option></select></div>
            <div class="filter-group"><label>Gene:</label><input type="text" id="gene-search-tissue-a" placeholder="Search..."></div>
            <div class="filter-group"><label>Source type:</label><select id="source-type-filter-tissue-a"><option value="">All</option></select></div>
            <div class="filter-group"><label><input type="checkbox" id="enrich-filter-tissue-a"> Significant enrichment only</label></div>
            <div class="filter-group"><label>Mechanism:</label><select id="mech-topic-filter-tissue-a"><option value="">All</option></select></div>
            <div class="filter-group"><label>Pathway:</label><select id="path-topic-filter-tissue-a"><option value="">All</option></select></div>
        </div>
        <table id="table-tissue-a" class="display"><thead>{headers_tissue}</thead><tbody></tbody></table>
    </div>
    
    <div id="tab-tissue-b" class="tab-content">
        <h2>{tissue_B_label} Results</h2>
        <div class="context-stats" id="stats-tissue-b"></div>
        <div class="filters">
            <div class="filter-group"><label>Phenotype:</label><select id="phenotype-filter-tissue-b"><option value="">All</option></select></div>
            <div class="filter-group"><label>Gene:</label><input type="text" id="gene-search-tissue-b" placeholder="Search..."></div>
            <div class="filter-group"><label>Source type:</label><select id="source-type-filter-tissue-b"><option value="">All</option></select></div>
            <div class="filter-group"><label><input type="checkbox" id="enrich-filter-tissue-b"> Significant enrichment only</label></div>
            <div class="filter-group"><label>Mechanism:</label><select id="mech-topic-filter-tissue-b"><option value="">All</option></select></div>
            <div class="filter-group"><label>Pathway:</label><select id="path-topic-filter-tissue-b"><option value="">All</option></select></div>
        </div>
        <table id="table-tissue-b" class="display"><thead>{headers_tissue}</thead><tbody></tbody></table>
    </div>
    
    {enrichment_tab_content}
    
    <div id="tab-summary" class="tab-content">
        <h2>Summary Statistics</h2>
        <div class="summary-grid">
            <div class="stat-card"><div class="stat-value">{n_total:,}</div><div class="stat-label">Total Pairs</div></div>
            <div class="stat-card"><div class="stat-value">{n_tissue_specific:,}</div><div class="stat-label">Tissue-Specific</div></div>
            <div class="stat-card positive"><div class="stat-value">{n_A_biased:,}</div><div class="stat-label">{tissue_A_label}-biased</div></div>
            <div class="stat-card negative"><div class="stat-value">{n_B_biased:,}</div><div class="stat-label">{tissue_B_label}-biased</div></div>
            <div class="stat-card"><div class="stat-value">{n_fdr_sig:,}</div><div class="stat-label">FDR q &lt; 0.05</div></div>
        </div>
        <h3>Permutation Test Results</h3>
        <div id="permutation-results"><p><em>Loading...</em></p></div>
        <h3>Top Tissue-Specific Heads</h3>
        <p>Source nodes with highest tissue specificity. Format: Name (Type)</p>
        <div class="gene-columns">
            <div class="gene-col"><h4 style="color:#28a745">{tissue_A_label}-biased (top 25)</h4><ul class="gene-list" id="genes-tissue-a"></ul></div>
            <div class="gene-col"><h4 style="color:#dc3545">{tissue_B_label}-biased (top 25)</h4><ul class="gene-list" id="genes-tissue-b"></ul></div>
        </div>
        <h3>Metapath Analysis</h3>
        <div id="metapath-results"><p><em>Loading...</em></p></div>
        <h3>Hop Comparison</h3>
        <div id="hop-comparison-results"><p><em>Available when both 2-hop and 3-hop exist.</em></p></div>
    </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script>
var tissueAResults = {tissue_A_json};
var tissueBResults = {tissue_B_json};
var comparisonResults = {comparison_json};
var permutationData = {permutation_json};
var metapathData = {metapath_json};
var hopComparisonData = {hop_comparison_json};
var enrichmentData = {enrichment_json};
var hops = {hops};
var tissueALabel = "{tissue_A_label}";
var tissueBLabel = "{tissue_B_label}";
'''

    # JS body uses [] and {} heavily - use raw string to avoid f-string issues
    html += r'''
var tables = {};
var initialized = {};

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        var tabId = this.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
        if (!initialized[tabId]) {
            if (tabId === 'comparison') initComparisonTable();
            else if (tabId === 'tissue-a') initTissueTable('tissue-a', tissueAResults);
            else if (tabId === 'tissue-b') initTissueTable('tissue-b', tissueBResults);
            else if (tabId === 'summary') initSummary();
            initialized[tabId] = true;
        }
    });
});

function fmt(v, d) { return v == null ? '-' : v.toFixed(d); }
function fmtCorr(v) { return v === 1 ? '<span class="corr-positive">▲</span>' : v === -1 ? '<span class="corr-negative">▼</span>' : '?'; }
function fmtProb(v) { if (v == null) return '-'; var f = v.toFixed(4); return v >= 0.1 ? '<span class="prob-high">' + f + '</span>' : v >= 0.01 ? '<span class="prob-medium">' + f + '</span>' : f; }
function fmtDiff(v) { if (v == null) return '-'; var f = v.toFixed(4); return v > 0.1 ? '<span class="cov-positive">+' + f + '</span>' : v < -0.1 ? '<span class="cov-negative">' + f + '</span>' : f; }
function fmtPval(v) { if (v == null || v === undefined) return '-'; if (v < 0.001) return '<span class="sig-marker">' + v.toExponential(1) + '</span>'; if (v < 0.01) return '<span class="sig-marker">' + v.toFixed(3) + '</span>'; if (v < 0.05) return '<span style="color:#e65100;font-weight:600">' + v.toFixed(3) + '</span>'; return v.toFixed(3); }
function fmtRel(r) { if (!r || r === '?') return '<span class="rel-badge">?</span>'; var d = r.length > 12 ? r.substring(0,12) + '..' : r; return '<span class="rel-badge" title="' + r + '">' + d + '</span>'; }
function fmtInt(n) { return '<span class="int-count">' + (n || 0) + '</span>'; }
function fmtGenes(g) { if (!g || !g.length) return '-'; var d = g.slice(0,10).join(', '); if (g.length > 10) d += ' (+' + (g.length-10) + ')'; return '<div class="intermediate-list">' + d + '</div>'; }

function fmtSource(name, metapath) {
    var nodeType = getSourceType(metapath);
    return '<span class="gene-name">' + (name||'-') + '</span> <small style="color:#888">(' + nodeType + ')</small>';
}

function getSourceType(metapath) {
    if (!metapath) return 'Unknown';
    var match = metapath.match(/^([^-\\[]+)/);
    return match ? match[1] : 'Unknown';
}

/* ======================== PER-TISSUE ENRICHMENT INDEX ======================== */
var enrichmentByTissue = {};
var tissueAName = enrichmentData ? (enrichmentData.tissue_A || '') : '';
var tissueBName = enrichmentData ? (enrichmentData.tissue_B || '') : '';
function _loadTissueEnrichment(tissueName, fieldsObj) {
    if (!fieldsObj) return;
    if (!enrichmentByTissue[tissueName]) enrichmentByTissue[tissueName] = {};
    var idx = enrichmentByTissue[tissueName];
    ['mechanisms', 'pathways'].forEach(function(field) {
        var results = (fieldsObj[field] || {}).all_results || [];
        results.forEach(function(r) {
            var key = (r.source || '') + '|||' + (r.target || '') + '|||' + (r.metapath || '');
            if (!idx[key]) idx[key] = [];
            idx[key].push({
                field: field, topic_id: r.topic_id,
                topic_label: r.topic_label || ('Topic ' + r.topic_id),
                top_terms: r.top_terms || '', tissue: tissueName,
                odds_ratio: r.odds_ratio, log2_fold_change: r.log2_fold_change,
                pct_foreground: r.pct_foreground, pct_background: r.pct_background,
                count_foreground: r.count_foreground, count_background: r.count_background,
                n_fg_edges: r.n_fg_edges, n_bg_edges: r.n_bg_edges,
                p_value: r.p_value, fdr_q: r.fdr_q, significant: r.significant
            });
        });
    });
}
if (enrichmentData && enrichmentData.tissues) {
    for (var tKey in enrichmentData.tissues) {
        _loadTissueEnrichment(tKey, (enrichmentData.tissues[tKey] || {}).fields || {});
    }
}

function getTissueEnrichmentSummary(source, target, metapath, tissueName) {
    var idx = enrichmentByTissue[tissueName] || {};
    var key = (source || '') + '|||' + (target || '') + '|||' + (metapath || '');
    var results = idx[key] || [];
    var sigResults = results.filter(function(r) { return r.significant; });
    if (sigResults.length === 0) return { html: '<span style="color:#999">-</span>', count: 0 };
    var mechSig = sigResults.filter(function(r) { return r.field === 'mechanisms'; });
    var pathSig = sigResults.filter(function(r) { return r.field === 'pathways'; });
    var parts = [];
    if (mechSig.length) parts.push('<span style="color:#007bff;font-weight:600">M:' + mechSig.length + '</span>');
    if (pathSig.length) parts.push('<span style="color:#6f42c1;font-weight:600">P:' + pathSig.length + '</span>');
    /* Hidden topic labels for filtering */
    var topicTags = [];
    mechSig.forEach(function(r) { topicTags.push('MT:' + r.topic_label); });
    pathSig.forEach(function(r) { topicTags.push('PT:' + r.topic_label); });
    var hidden = '<span style="display:none">' + topicTags.join('||') + '</span>';
    return { html: parts.join(' ') + hidden, count: sigResults.length };
}

function formatTissueEnrichmentDetail(source, target, metapath, tissueName) {
    var idx = enrichmentByTissue[tissueName] || {};
    var key = (source || '') + '|||' + (target || '') + '|||' + (metapath || '');
    var results = idx[key] || [];
    if (results.length === 0) return '<div style="padding:10px;color:#999">No enrichment data for this triple.</div>';
    var sigResults = results.filter(function(r) { return r.significant; });
    var html = '<div style="padding:10px 20px;background:#fafafa;border-left:3px solid #007bff">';
    html += '<strong>Topic Enrichment</strong> \u2014 ' + tissueName + ' vs background (' + sigResults.length + ' significant of ' + results.length + ' tested)<br><br>';
    ['mechanisms', 'pathways'].forEach(function(field) {
        var fieldResults = results.filter(function(r) { return r.field === field; });
        if (fieldResults.length === 0) return;
        var fieldSig = fieldResults.filter(function(r) { return r.significant; });
        html += '<strong>' + field.charAt(0).toUpperCase() + field.slice(1) + '</strong>';
        html += ' (' + fieldSig.length + '/' + fieldResults.length + ' significant)<br>';
        if (fieldSig.length > 0) {
            html += '<table style="font-size:11px;border-collapse:collapse;margin:5px 0 10px 0;width:100%">';
            html += '<tr style="background:#f0f0f0"><th style="padding:4px 8px;text-align:left">Topic</th><th style="padding:4px 8px;text-align:right">OR</th><th style="padding:4px 8px;text-align:right">log2FC</th><th style="padding:4px 8px;text-align:right">% Tissue</th><th style="padding:4px 8px;text-align:right">% Other</th><th style="padding:4px 8px;text-align:right">p-value</th><th style="padding:4px 8px;text-align:right">FDR q</th></tr>';
            fieldSig.sort(function(a,b) { return (a.fdr_q||1) - (b.fdr_q||1); }).forEach(function(r) {
                var fcColor = r.log2_fold_change > 0 ? '#28a745' : '#dc3545';
                html += '<tr style="border-bottom:1px solid #eee">';
                html += '<td style="padding:3px 8px;text-align:left" title="' + (r.top_terms || '') + '">' + r.topic_label + '</td>';
                html += '<td style="padding:3px 8px;text-align:right">' + fmt(r.odds_ratio, 2) + '</td>';
                html += '<td style="padding:3px 8px;text-align:right;color:' + fcColor + ';font-weight:600">' + fmt(r.log2_fold_change, 2) + '</td>';
                html += '<td style="padding:3px 8px;text-align:right">' + fmt(r.pct_foreground, 1) + '%</td>';
                html += '<td style="padding:3px 8px;text-align:right">' + fmt(r.pct_background, 1) + '%</td>';
                html += '<td style="padding:3px 8px;text-align:right">' + (r.p_value != null ? r.p_value.toExponential(2) : '-') + '</td>';
                html += '<td style="padding:3px 8px;text-align:right">' + (r.fdr_q != null ? r.fdr_q.toExponential(2) : '-') + '</td>';
                html += '</tr>';
            });
            html += '</table>';
        } else {
            html += '<div style="color:#999;margin:5px 0 10px 0">No significant enrichments</div>';
        }
    });
    html += '</div>';
    return html;
}

/* ======================== COMPARISON TABLE ======================== */
function initComparisonTable() {
    var data = comparisonResults || [];
    var sources = {}, phenos = {};
    data.forEach(r => { if(r.source_gene) sources[r.source_gene]=1; if(r.target_phenotype) phenos[r.target_phenotype]=1; });
    document.getElementById('stats-comparison').innerHTML = '<span><strong>' + data.length.toLocaleString() + '</strong> results</span><span><strong>' + Object.keys(sources).length + '</strong> sources</span><span><strong>' + Object.keys(phenos).length + '</strong> phenotypes</span>';
    var sel = document.getElementById('phenotype-filter-comparison');
    Object.keys(phenos).sort().forEach(p => { var o = document.createElement('option'); o.value = p; o.textContent = p; sel.appendChild(o); });
    
    var tableData = data.map((r, i) => {
        var enrA = getTissueEnrichmentSummary(r.source_gene, r.target_phenotype, r.metapath, tissueAName);
        var enrB = getTissueEnrichmentSummary(r.source_gene, r.target_phenotype, r.metapath, tissueBName);
        if (hops === 2) {
            return ['<span class="expand-btn" style="cursor:pointer;color:#007bff;font-weight:bold;font-size:16px" title="Show enrichment">+</span>',
                   i+1, fmtSource(r.source_gene, r.metapath), fmtRel(r.rel1), fmtInt(r.num_intermediates), fmtRel(r.rel2),
                   '<span class="disease-name">' + (r.target_phenotype||'-') + '</span>', 
                   fmtProb(r.prob_A), fmtProb(r.prob_B), fmt(r.coverage_A,4), fmt(r.coverage_B,4), 
                   fmtDiff(r.diff_coverage), fmt(r.log2_ratio,2), fmtPval(r.perm_pvalue), fmtPval(r.perm_qvalue), fmtCorr(r.correlation_A),
                   enrA.html, enrB.html];
        } else {
            return ['<span class="expand-btn" style="cursor:pointer;color:#007bff;font-weight:bold;font-size:16px" title="Show enrichment">+</span>',
                   i+1, fmtSource(r.source_gene, r.metapath), fmtRel(r.rel1), fmtInt(r.n_intermediates_B || r.num_intermediates || 0), fmtRel(r.rel2),
                   fmtInt(r.n_intermediates_C || 0), fmtRel(r.rel3),
                   '<span class="disease-name">' + (r.target_phenotype||'-') + '</span>', 
                   fmtProb(r.prob_A), fmtProb(r.prob_B), fmt(r.coverage_A,4), fmt(r.coverage_B,4), 
                   fmtDiff(r.diff_coverage), fmt(r.log2_ratio,2), fmtPval(r.perm_pvalue), fmtPval(r.perm_qvalue), fmtCorr(r.correlation_A),
                   enrA.html, enrB.html];
        }
    });
    
    var diffCol = hops === 2 ? 11 : 13;
    tables['comparison'] = $('#table-comparison').DataTable({ data: tableData, pageLength: 50, order: [[diffCol, 'desc']], dom: 'Bfrtip', buttons: ['csv'], deferRender: true, scrollX: true, columnDefs: [{ orderable: false, targets: 0, width: '20px' }] });
    
    /* Expand button click handler - shows enrichment detail for both tissues */
    $('#table-comparison tbody').on('click', '.expand-btn', function() {
        var tr = $(this).closest('tr');
        var row = tables['comparison'].row(tr);
        var rowIdx = row.index();
        var r = data[rowIdx];
        if (row.child.isShown()) {
            row.child.hide();
            $(this).html('+');
        } else {
            var detailA = formatTissueEnrichmentDetail(r.source_gene, r.target_phenotype, r.metapath, tissueAName);
            var detailB = formatTissueEnrichmentDetail(r.source_gene, r.target_phenotype, r.metapath, tissueBName);
            var combined = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">';
            combined += '<div><h4 style="margin:0 0 5px 0;color:#28a745">' + tissueALabel + '</h4>' + detailA + '</div>';
            combined += '<div><h4 style="margin:0 0 5px 0;color:#dc3545">' + tissueBLabel + '</h4>' + detailB + '</div>';
            combined += '</div>';
            row.child(combined).show();
            $(this).html('\u2212');
        }
    });
    
    var phenoCol = hops === 2 ? 6 : 8;
    var sourceCol = 2;
    
    /* Populate source type dropdown */
    var srcTypes = {};
    data.forEach(r => { var t = getSourceType(r.metapath); if (t) srcTypes[t] = 1; });
    var srcSel = document.getElementById('source-type-filter-comparison');
    Object.keys(srcTypes).sort().forEach(t => { var o = document.createElement('option'); o.value = t; o.textContent = t; srcSel.appendChild(o); });
    
    document.getElementById('phenotype-filter-comparison').addEventListener('change', function() { tables['comparison'].column(phenoCol).search(this.value ? '^' + this.value.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&') + '$' : '', true, false).draw(); });
    
    /* Combined gene name + source type filter on source column */
    function applySourceFilterComp() {
        var geneVal = document.getElementById('gene-search-comparison').value.toLowerCase();
        var typeVal = document.getElementById('source-type-filter-comparison').value;
        
        if (window['_srcFilterFn_comparison']) {
            $.fn.dataTable.ext.search.splice($.fn.dataTable.ext.search.indexOf(window['_srcFilterFn_comparison']), 1);
            window['_srcFilterFn_comparison'] = null;
        }
        tables['comparison'].column(sourceCol).search('', false, false);
        
        if (!geneVal && !typeVal) { tables['comparison'].draw(); return; }
        
        var fn = function(settings, rowData) {
            if (settings.nTable.id !== 'table-comparison') return true;
            var cell = (rowData[sourceCol] || '').toLowerCase();
            if (geneVal && cell.indexOf(geneVal) === -1) return false;
            if (typeVal && cell.indexOf('(' + typeVal.toLowerCase() + ')') === -1) return false;
            return true;
        };
        window['_srcFilterFn_comparison'] = fn;
        $.fn.dataTable.ext.search.push(fn);
        tables['comparison'].draw();
    }
    document.getElementById('gene-search-comparison').addEventListener('input', applySourceFilterComp);
    document.getElementById('source-type-filter-comparison').addEventListener('change', applySourceFilterComp);
    
    /* Diff coverage & q-value custom filter */
    var qvalCol = hops === 2 ? 14 : 16;
    function applyNumericFiltersComp() {
        var minDiff = parseFloat(document.getElementById('diff-filter').value) || 0;
        var maxQ = parseFloat(document.getElementById('qval-filter').value);
        if (isNaN(maxQ)) maxQ = 1;
        if (window['_numFilterFn_comparison']) {
            $.fn.dataTable.ext.search.splice($.fn.dataTable.ext.search.indexOf(window['_numFilterFn_comparison']), 1);
            window['_numFilterFn_comparison'] = null;
        }
        if (minDiff <= 0 && maxQ >= 1) { tables['comparison'].draw(); return; }
        var fn = function(settings, data, dataIndex) {
            if (settings.nTable.id !== 'table-comparison') return true;
            var row = comparisonResults[dataIndex];
            if (!row) return true;
            if (minDiff > 0 && Math.abs(row.diff_coverage || 0) < minDiff) return false;
            if (maxQ < 1 && (row.perm_qvalue == null || row.perm_qvalue > maxQ)) return false;
            return true;
        };
        window['_numFilterFn_comparison'] = fn;
        $.fn.dataTable.ext.search.push(fn);
        tables['comparison'].draw();
    }
    document.getElementById('diff-filter').addEventListener('input', applyNumericFiltersComp);
    document.getElementById('qval-filter').addEventListener('input', applyNumericFiltersComp);
    
    /* Enrichment dropdowns for comparison tab */
    var enrichColA = hops === 2 ? 16 : 18;
    var enrichColB = hops === 2 ? 17 : 19;
    var compMechTopics = {}, compPathTopics = {};
    [tissueAName, tissueBName].forEach(function(tn) {
        var tidx = enrichmentByTissue[tn] || {};
        for (var eKey in tidx) {
            tidx[eKey].forEach(function(r) {
                if (!r.significant) return;
                if (r.field === 'mechanisms') compMechTopics[r.topic_label] = 1;
                if (r.field === 'pathways') compPathTopics[r.topic_label] = 1;
            });
        }
    });
    var compMechSel = document.getElementById('mech-topic-filter-comparison');
    Object.keys(compMechTopics).sort().forEach(function(t) { var o = document.createElement('option'); o.value = t; o.textContent = t; compMechSel.appendChild(o); });
    var compPathSel = document.getElementById('path-topic-filter-comparison');
    Object.keys(compPathTopics).sort().forEach(function(t) { var o = document.createElement('option'); o.value = t; o.textContent = t; compPathSel.appendChild(o); });
    
    function applyEnrichFilterComp() {
        var sigOnly = document.getElementById('enrich-filter-comparison').checked;
        var mechVal = document.getElementById('mech-topic-filter-comparison').value;
        var pathVal = document.getElementById('path-topic-filter-comparison').value;
        
        if (window['_enrFilterFn_comparison']) {
            $.fn.dataTable.ext.search.splice(
                $.fn.dataTable.ext.search.indexOf(window['_enrFilterFn_comparison']), 1);
            window['_enrFilterFn_comparison'] = null;
        }
        
        if (!sigOnly && !mechVal && !pathVal) {
            tables['comparison'].draw();
            return;
        }
        
        var filterFn = function(settings, data, dataIndex) {
            if (settings.nTable.id !== 'table-comparison') return true;
            var cellA = data[enrichColA] || '';
            var cellB = data[enrichColB] || '';
            var combined = cellA + ' ' + cellB;
            if (sigOnly && !mechVal && !pathVal) {
                return combined.indexOf('M:') !== -1 || combined.indexOf('P:') !== -1;
            }
            var pass = true;
            if (mechVal && combined.indexOf('MT:' + mechVal) === -1) pass = false;
            if (pathVal && combined.indexOf('PT:' + pathVal) === -1) pass = false;
            if (pass && sigOnly && combined.indexOf('M:') === -1 && combined.indexOf('P:') === -1) pass = false;
            return pass;
        };
        
        window['_enrFilterFn_comparison'] = filterFn;
        $.fn.dataTable.ext.search.push(filterFn);
        tables['comparison'].draw();
    }
    
    document.getElementById('enrich-filter-comparison').addEventListener('change', applyEnrichFilterComp);
    document.getElementById('mech-topic-filter-comparison').addEventListener('change', applyEnrichFilterComp);
    document.getElementById('path-topic-filter-comparison').addEventListener('change', applyEnrichFilterComp);
}

/* ======================== TISSUE TABLE ======================== */
function initTissueTable(tabId, data) {
    if (!data || !data.length) { document.getElementById('stats-' + tabId).innerHTML = 'No data'; return; }
    var tissueName = tabId === 'tissue-a' ? tissueAName : tissueBName;
    var sources = {}, phenos = {};
    data.forEach(r => { if(r.source_gene) sources[r.source_gene]=1; if(r.target_phenotype) phenos[r.target_phenotype]=1; });
    document.getElementById('stats-' + tabId).innerHTML = '<span><strong>' + data.length.toLocaleString() + '</strong> results</span><span><strong>' + Object.keys(sources).length + '</strong> sources</span>';
    var sel = document.getElementById('phenotype-filter-' + tabId);
    Object.keys(phenos).sort().forEach(p => { var o = document.createElement('option'); o.value = p; o.textContent = p; sel.appendChild(o); });
    
    window['_tissueData_' + tabId] = data;
    
    var tableData = data.map((r, i) => {
        var enr = getTissueEnrichmentSummary(r.source_gene, r.target_phenotype, r.metapath, tissueName);
        if (hops === 2) {
            return ['<span class="expand-btn" style="cursor:pointer;color:#007bff;font-weight:bold;font-size:16px" title="Show enrichment">+</span>',
                   r.rank || i+1, fmtSource(r.source_gene, r.metapath), fmtRel(r.rel1), fmtInt(r.num_intermediates), fmtRel(r.rel2),
                   '<span class="disease-name">' + (r.target_phenotype||'-') + '</span>', 
                   fmtProb(r.probability), fmt(r.evidence_score,2), fmt(r.coverage,4), fmtCorr(r.correlation_type), 
                   fmtGenes(r.intermediate_genes), enr.html];
        } else {
            return ['<span class="expand-btn" style="cursor:pointer;color:#007bff;font-weight:bold;font-size:16px" title="Show enrichment">+</span>',
                   r.rank || i+1, fmtSource(r.source_gene, r.metapath), fmtRel(r.rel1), fmtInt(r.n_intermediates_B || r.num_intermediates || 0), fmtRel(r.rel2),
                   fmtInt(r.n_intermediates_C || 0), fmtRel(r.rel3),
                   '<span class="disease-name">' + (r.target_phenotype||'-') + '</span>', 
                   fmtProb(r.probability), fmt(r.evidence_score,2), fmt(r.coverage,4), fmtCorr(r.correlation_type), 
                   fmtGenes(r.intermediate_genes_B || r.intermediate_genes), fmtGenes(r.intermediate_genes_C), enr.html];
        }
    });
    
    tables[tabId] = $('#table-' + tabId).DataTable({
        data: tableData, pageLength: 50, order: [[1, 'asc']],
        dom: 'Bfrtip', buttons: ['csv'], deferRender: true, scrollX: true,
        columnDefs: [{ orderable: false, targets: 0, width: '20px' }]
    });
    
    $('#table-' + tabId + ' tbody').on('click', '.expand-btn', function() {
        var tr = $(this).closest('tr');
        var row = tables[tabId].row(tr);
        var rowIdx = row.index();
        var r = window['_tissueData_' + tabId][rowIdx];
        if (row.child.isShown()) {
            row.child.hide();
            $(this).html('+');
        } else {
            var detail = formatTissueEnrichmentDetail(r.source_gene, r.target_phenotype, r.metapath, tissueName);
            row.child(detail).show();
            $(this).html('\u2212');
        }
    });
    
    var phenoCol = hops === 2 ? 6 : 8;
    var enrichCol = hops === 2 ? 12 : 15;
    var sourceCol = 2;
    
    document.getElementById('phenotype-filter-' + tabId).addEventListener('change', function() { tables[tabId].column(phenoCol).search(this.value ? '^' + this.value.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&') + '$' : '', true, false).draw(); });
    
    /* Populate source type dropdown */
    var srcTypes = {};
    data.forEach(r => { var t = getSourceType(r.metapath); if (t) srcTypes[t] = 1; });
    var srcSel = document.getElementById('source-type-filter-' + tabId);
    Object.keys(srcTypes).sort().forEach(function(t) { var o = document.createElement('option'); o.value = t; o.textContent = t; srcSel.appendChild(o); });
    
    /* Combined gene name + source type filter */
    function applySourceFilter() {
        var geneVal = document.getElementById('gene-search-' + tabId).value.toLowerCase();
        var typeVal = document.getElementById('source-type-filter-' + tabId).value;
        
        if (window['_srcFilterFn_' + tabId]) {
            $.fn.dataTable.ext.search.splice($.fn.dataTable.ext.search.indexOf(window['_srcFilterFn_' + tabId]), 1);
            window['_srcFilterFn_' + tabId] = null;
        }
        tables[tabId].column(sourceCol).search('', false, false);
        
        if (!geneVal && !typeVal) { tables[tabId].draw(); return; }
        
        var tableId = 'table-' + tabId;
        var fn = function(settings, rowData) {
            if (settings.nTable.id !== tableId) return true;
            var cell = (rowData[sourceCol] || '').toLowerCase();
            if (geneVal && cell.indexOf(geneVal) === -1) return false;
            if (typeVal && cell.indexOf('(' + typeVal.toLowerCase() + ')') === -1) return false;
            return true;
        };
        window['_srcFilterFn_' + tabId] = fn;
        $.fn.dataTable.ext.search.push(fn);
        tables[tabId].draw();
    }
    document.getElementById('gene-search-' + tabId).addEventListener('input', applySourceFilter);
    document.getElementById('source-type-filter-' + tabId).addEventListener('change', applySourceFilter);
    
    /* Collect unique significant topic labels for dropdowns */
    var mechTopics = {}, pathTopics = {};
    var tissueIdx = enrichmentByTissue[tissueName] || {};
    for (var eKey in tissueIdx) {
        tissueIdx[eKey].forEach(function(r) {
            if (!r.significant) return;
            if (r.field === 'mechanisms') mechTopics[r.topic_label] = 1;
            if (r.field === 'pathways') pathTopics[r.topic_label] = 1;
        });
    }
    var mechSel = document.getElementById('mech-topic-filter-' + tabId);
    Object.keys(mechTopics).sort().forEach(function(t) { var o = document.createElement('option'); o.value = t; o.textContent = t; mechSel.appendChild(o); });
    var pathSel = document.getElementById('path-topic-filter-' + tabId);
    Object.keys(pathTopics).sort().forEach(function(t) { var o = document.createElement('option'); o.value = t; o.textContent = t; pathSel.appendChild(o); });
    
    /* Combined enrichment filter: checkbox + mechanism dropdown + pathway dropdown
       Uses custom search fn for AND semantics across filters */
    function applyEnrichFilter() {
        var sigOnly = document.getElementById('enrich-filter-' + tabId).checked;
        var mechVal = document.getElementById('mech-topic-filter-' + tabId).value;
        var pathVal = document.getElementById('path-topic-filter-' + tabId).value;
        
        /* Clear any previous column search on enrichment col */
        tables[tabId].column(enrichCol).search('', false, false);
        
        /* Remove previous custom filter if any */
        if (window['_enrFilterFn_' + tabId]) {
            $.fn.dataTable.ext.search.splice(
                $.fn.dataTable.ext.search.indexOf(window['_enrFilterFn_' + tabId]), 1);
            window['_enrFilterFn_' + tabId] = null;
        }
        
        if (!sigOnly && !mechVal && !pathVal) {
            tables[tabId].draw();
            return;
        }
        
        var filterFn = function(settings, data, dataIndex) {
            if (settings.nTable.id !== 'table-' + tabId) return true;
            var cell = data[enrichCol] || '';
            if (sigOnly && !mechVal && !pathVal) {
                return cell.indexOf('M:') !== -1 || cell.indexOf('P:') !== -1;
            }
            var pass = true;
            if (mechVal && cell.indexOf('MT:' + mechVal) === -1) pass = false;
            if (pathVal && cell.indexOf('PT:' + pathVal) === -1) pass = false;
            if (pass && sigOnly && cell.indexOf('M:') === -1 && cell.indexOf('P:') === -1) pass = false;
            return pass;
        };
        
        window['_enrFilterFn_' + tabId] = filterFn;
        $.fn.dataTable.ext.search.push(filterFn);
        tables[tabId].draw();
    }
    
    document.getElementById('enrich-filter-' + tabId).addEventListener('change', applyEnrichFilter);
    document.getElementById('mech-topic-filter-' + tabId).addEventListener('change', applyEnrichFilter);
    document.getElementById('path-topic-filter-' + tabId).addEventListener('change', applyEnrichFilter);
}

/* ======================== SUMMARY TAB ======================== */
function initSummary() {
    if (permutationData && permutationData.statistics) {
        var html = '<table class="perm-table"><tr><th>Statistic</th><th>Observed</th><th>Null Mean ± Std</th><th>Z-score</th><th>P-value</th></tr>';
        for (var stat in permutationData.statistics) {
            var s = permutationData.statistics[stat];
            var sig = s['significant_0.001'] ? '***' : s['significant_0.01'] ? '**' : s['significant_0.05'] ? '*' : '';
            html += '<tr><td>' + stat + '</td><td>' + (s.observed != null ? s.observed.toLocaleString() : '-') + '</td><td>' + fmt(s.null_mean,2) + ' ± ' + fmt(s.null_std,2) + '</td><td>' + fmt(s.z_score,2) + '</td><td>' + fmt(s.p_value,4) + ' <span class="sig-marker">' + sig + '</span></td></tr>';
        }
        html += '</table><p><small>* p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001</small></p>';
        if (permutationData.interpretation) html += '<p><em>' + permutationData.interpretation + '</em></p>';
        document.getElementById('permutation-results').innerHTML = html;
    } else { document.getElementById('permutation-results').innerHTML = '<p>No permutation results.</p>'; }
    
    var sortedA = comparisonResults.filter(r => r.diff_coverage > 0).sort((a,b) => b.diff_coverage - a.diff_coverage);
    var sortedB = comparisonResults.filter(r => r.diff_coverage < 0).sort((a,b) => a.diff_coverage - b.diff_coverage);
    var listA = document.getElementById('genes-tissue-a');
    sortedA.slice(0,25).forEach(r => { 
        var nodeType = getSourceType(r.metapath);
        var li = document.createElement('li'); 
        li.innerHTML = '<span>' + r.source_gene + ' <small style="color:#666">(' + nodeType + ')</small></span><span class="gene-diff" style="color:#28a745">+' + r.diff_coverage.toFixed(3) + '</span>'; 
        listA.appendChild(li); 
    });
    var listB = document.getElementById('genes-tissue-b');
    sortedB.slice(0,25).forEach(r => { 
        var nodeType = getSourceType(r.metapath);
        var li = document.createElement('li'); 
        li.innerHTML = '<span>' + r.source_gene + ' <small style="color:#666">(' + nodeType + ')</small></span><span class="gene-diff" style="color:#dc3545">' + r.diff_coverage.toFixed(3) + '</span>'; 
        listB.appendChild(li); 
    });
    
    if (metapathData && metapathData.length) {
        var html = '<table class="metapath-table"><tr><th>Metapath</th><th>Count</th><th>Mean Δ Cov</th><th>% ' + tissueALabel + '</th><th>% ' + tissueBLabel + '</th><th># Specific</th></tr>';
        metapathData.slice(0,20).forEach(mp => {
            html += '<tr><td class="metapath-cell">' + mp.metapath + '</td><td>' + (mp.count||0).toLocaleString() + '</td><td style="color:' + (mp.mean_diff_coverage > 0 ? '#28a745' : '#dc3545') + '">' + fmt(mp.mean_diff_coverage,4) + '</td><td>' + fmt(mp.pct_A_biased,1) + '%</td><td>' + fmt(mp.pct_B_biased,1) + '%</td><td>' + (mp.n_tissue_specific||0) + '</td></tr>';
        });
        html += '</table>';
        document.getElementById('metapath-results').innerHTML = html;
    } else { document.getElementById('metapath-results').innerHTML = '<p>No metapath data.</p>'; }
    
    if (hopComparisonData && hopComparisonData.coverage_decay) {
        var cd = hopComparisonData.coverage_decay;
        var html = '';
        if (hopComparisonData.n_pairs_2hop || hopComparisonData.n_pairs_3hop) {
            html += '<p><strong>Total pairs:</strong> 2-hop: ' + (hopComparisonData.n_pairs_2hop||0).toLocaleString() + ', 3-hop: ' + (hopComparisonData.n_pairs_3hop||0).toLocaleString() + '</p>';
        }
        html += '<h4>Coverage Decay</h4><p>Ratio of mean 3-hop coverage to mean 2-hop coverage (lower = more decay with longer paths).</p>';
        html += '<table class="perm-table"><tr><th></th><th>2-hop</th><th>3-hop</th><th>Ratio (3h/2h)</th></tr>';
        html += '<tr><td>' + tissueALabel + '</td><td>' + fmt(cd.mean_coverage_A_2hop,4) + '</td><td>' + fmt(cd.mean_coverage_A_3hop,4) + '</td><td>' + fmt(cd.decay_A,2) + '</td></tr>';
        html += '<tr><td>' + tissueBLabel + '</td><td>' + fmt(cd.mean_coverage_B_2hop,4) + '</td><td>' + fmt(cd.mean_coverage_B_3hop,4) + '</td><td>' + fmt(cd.decay_B,2) + '</td></tr></table>';
        if (hopComparisonData.jaccard_overlap) {
            html += '<h4>Jaccard Overlap of Top Source Genes</h4><p>Overlap of top genes by probability between 2-hop and 3-hop results.</p>';
            html += '<table class="perm-table"><tr><th>Top K</th><th>Jaccard</th><th>Shared</th><th>Only 2-hop</th><th>Only 3-hop</th></tr>';
            for (var k in hopComparisonData.jaccard_overlap) {
                var j = hopComparisonData.jaccard_overlap[k];
                html += '<tr><td>' + k + '</td><td>' + fmt(j.jaccard,3) + '</td><td>' + (j.shared||j.intersection||0) + '</td><td>' + (j.only_2hop||0) + '</td><td>' + (j.only_3hop||0) + '</td></tr>';
            }
            html += '</table>';
        }
        if (hopComparisonData.tissue_specificity) {
            var ts = hopComparisonData.tissue_specificity;
            html += '<h4>Tissue Specificity</h4>';
            html += '<table class="perm-table"><tr><th></th><th>2-hop</th><th>3-hop</th></tr>';
            html += '<tr><td>Mean specificity</td><td>' + fmt(ts.mean_specificity_2hop,4) + '</td><td>' + fmt(ts.mean_specificity_3hop,4) + '</td></tr>';
            html += '<tr><td># High diff (>0.3)</td><td>' + (ts.n_high_diff_2hop||0).toLocaleString() + '</td><td>' + (ts.n_high_diff_3hop||0).toLocaleString() + '</td></tr>';
            html += '</table>';
        }
        document.getElementById('hop-comparison-results').innerHTML = html;
    }
}

document.addEventListener('DOMContentLoaded', function() { initComparisonTable(); initialized['comparison'] = true; });
</script>
</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', required=True, choices=['subcut_vs_visceral', 'white_vs_brown'])
    parser.add_argument('--hops', type=int, required=True, choices=[2, 3])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(config['paths']['output_dir'])
    
    comp_cfg = next((c for c in config['comparisons'] if c['name'] == args.comparison), None)
    if not comp_cfg:
        print(f"ERROR: Comparison '{args.comparison}' not found"); sys.exit(1)
    
    tissue_A, tissue_B = comp_cfg['tissue_A'], comp_cfg['tissue_B']
    print(f"Generating report: {args.comparison} ({args.hops}-hop)")
    
    inference_dir = output_dir / 'inference'
    tissue_A_results = load_tissue_results(inference_dir, tissue_A, args.hops)
    tissue_B_results = load_tissue_results(inference_dir, tissue_B, args.hops)
    
    comp_dir = output_dir / 'comparisons' / args.comparison
    comparison_results = load_comparison_results(comp_dir, args.hops)
    
    # Filter to target phenotypes of interest
    enr_config = config.get('enrichment', {})
    target_phenotypes = enr_config.get('target_phenotypes', [
        'Inflammation', 'Inflamed', 'Insulin Resistance', 'Obesity',
        'Metabolic Diseases', 'Inflammatory Diseases',
        'Diabetes Mellitus 2', 'Diabetes Mellitus, Type 2',
    ])
    target_lower = {t.lower() for t in target_phenotypes}
    
    def filter_by_phenotype(results):
        return [r for r in results
                if (r.get('target_phenotype') or '').lower() in target_lower]
    
    n_before = (len(tissue_A_results), len(tissue_B_results), len(comparison_results))
    tissue_A_results = filter_by_phenotype(tissue_A_results)
    tissue_B_results = filter_by_phenotype(tissue_B_results)
    comparison_results = filter_by_phenotype(comparison_results)
    n_after = (len(tissue_A_results), len(tissue_B_results), len(comparison_results))
    
    print(f"  Phenotype filter: {sorted(target_phenotypes)}")
    print(f"  Tissue A: {n_before[0]:,} → {n_after[0]:,}")
    print(f"  Tissue B: {n_before[1]:,} → {n_after[1]:,}")
    print(f"  Comparison: {n_before[2]:,} → {n_after[2]:,}")
    
    # Load permutation results
    perm_path = output_dir / 'permutations' / args.comparison / 'permutation_summary.json'
    perm = None
    if perm_path.exists():
        try: perm = json.load(open(perm_path)); print("  Loaded permutation results")
        except: pass
    
    # Load metapath analysis
    meta_path = comp_dir / f'metapath_analysis_{args.hops}hop.json'
    meta = None
    if meta_path.exists():
        try: meta = json.load(open(meta_path)); print("  Loaded metapath analysis")
        except: pass
    
    # Load hop comparison (produced by compare.py)
    hop_path = comp_dir / 'hop_comparison.json'
    hop = None
    if hop_path.exists():
        try: hop = json.load(open(hop_path)); print("  Loaded hop comparison")
        except: pass
    
    # Load enrichment results
    enrichment = load_enrichment_results(output_dir, args.comparison, args.hops)
    
    # Filter enrichment to only triples in the report + only significant results
    # This prevents 500k+ enrichment records from bloating the HTML
    if enrichment and enrichment.get('tissues'):
        # Build set of triple keys present in report data
        report_triples = set()
        for r in tissue_A_results:
            report_triples.add((r.get('source_gene', ''), r.get('target_phenotype', ''), r.get('metapath', '')))
        for r in tissue_B_results:
            report_triples.add((r.get('source_gene', ''), r.get('target_phenotype', ''), r.get('metapath', '')))
        for r in comparison_results:
            report_triples.add((r.get('source_gene', ''), r.get('target_phenotype', ''), r.get('metapath', '')))
        
        total_before = 0
        total_after = 0
        for tissue_name, tissue_data in enrichment['tissues'].items():
            fields_data = tissue_data.get('fields', {})
            for field in ['mechanisms', 'pathways']:
                all_results = fields_data.get(field, {}).get('all_results', [])
                total_before += len(all_results)
                filtered = [r for r in all_results
                            if r.get('significant')
                            and (r.get('source', ''), r.get('target', ''), r.get('metapath', ''))
                            in report_triples]
                total_after += len(filtered)
                if field in fields_data:
                    fields_data[field]['all_results'] = filtered
        
        print(f"  Enrichment filter: {total_before:,} → {total_after:,} records "
              f"(significant + matching {len(report_triples):,} report triples)")
    
    html = generate_html_report(
        tissue_A_results, tissue_B_results, comparison_results,
        tissue_A, tissue_B, args.hops, perm, meta, hop, config,
        enrichment_data=enrichment)
    
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / f'report_{args.comparison}_{args.hops}hop.html'
    with open(output_path, 'w') as f:
        f.write(html)
    
    # Report file size
    size_mb = output_path.stat().st_size / 1e6
    print(f"\n✓ Report saved: {output_path} ({size_mb:.1f} MB)")
    
    # Generate LaTeX tables for metapath analysis and hop comparison
    if meta or hop:
        print("\nGenerating LaTeX tables...")
        save_latex_tables(
            metapath_data=meta,
            hop_data=hop,
            tissue_A=tissue_A,
            tissue_B=tissue_B,
            hops=args.hops,
            comparison_name=args.comparison,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    main()