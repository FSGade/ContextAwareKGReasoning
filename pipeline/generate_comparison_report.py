#!/usr/bin/env python3
"""
RQ2 Pipeline Comparison Report — Generate HTML report from inference parquets.

Reads output parquets from both pipelines:
  - Aggregated: inference/{tissue}_{hops}hop.parquet       (from run_psr.py --method all)
  - Raw:        inference/{tissue}_{hops}hop_raw.parquet    (from run_psr_raw.py --method all)

Pivots the wide probability columns (prob_bdd, prob_mc, prob_path_noisy_or)
into a long-format DataFrame with one row per (source, target, metapath, method),
then generates an interactive HTML report with:
  - Per-method DataTables (searchable, filterable)
  - Spearman/Pearson correlation heatmaps
  - Top-k Jaccard overlap heatmaps
  - Error vs reference bar charts
  - Pipeline comparison table (aggregated vs raw ratios)

Runs as a lightweight Slurm job after inference completes.

Usage:
    python generate_comparison_report.py \
        --tissue subcutaneous --hops 2 --config config.yaml

    # Custom file paths:
    python generate_comparison_report.py \
        --agg-parquet results/inference/subcutaneous_2hop.parquet \
        --raw-parquet results/inference/subcutaneous_2hop_raw.parquet \
        --output-dir results/pipeline_comparison

    # Only one pipeline available:
    python generate_comparison_report.py \
        --agg-parquet results/inference/subcutaneous_2hop.parquet \
        --output-dir results/pipeline_comparison
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Load and pivot parquets
# ---------------------------------------------------------------------------
PROB_COLUMNS = {
    'prob_path_noisy_or': 'path_noisy_or',
    'prob_bdd':           'bdd_exact',
    'prob_mc':            'monte_carlo',
}

# Columns to carry through from the wide parquet
KEEP_COLUMNS = [
    'source_gene', 'source_gene_id', 'target_phenotype', 'target_id',
    'metapath', 'coverage', 'evidence_score', 'correlation_type',
    'num_paths', 'num_intermediates', 'intermediate_genes',
    'relationship_types', 'rank',
]


def load_and_pivot(parquet_path: Path, pipeline_label: str) -> pd.DataFrame:
    """
    Load a parquet with wide probability columns and pivot to long format.

    If the parquet has prob_bdd / prob_mc / prob_path_noisy_or columns
    (from --method all), each becomes a separate row with method label.
    If only 'probability' exists (single method), uses the 'method' column.
    """
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {parquet_path.name}: {len(df):,} rows, "
          f"{len(df.columns)} columns")

    # Determine which probability columns exist
    available_prob_cols = {
        col: name for col, name in PROB_COLUMNS.items()
        if col in df.columns and df[col].notna().any()
    }

    keep = [c for c in KEEP_COLUMNS if c in df.columns]

    if available_prob_cols:
        # Wide format → melt
        rows = []
        for _, row in df.iterrows():
            base = {c: row[c] for c in keep}
            # Use source_gene_id as canonical source key for joins
            base['source'] = str(row.get('source_gene_id', row.get('source_gene', '')))
            base['target'] = str(row.get('target_id', row.get('target_phenotype', '')))
            base['source_name'] = row.get('source_gene', '')
            base['target_name'] = row.get('target_phenotype', '')

            for col, method_name in available_prob_cols.items():
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    r = base.copy()
                    r['method'] = f'{pipeline_label}_{method_name}'
                    r['probability'] = float(val)
                    rows.append(r)

        result = pd.DataFrame(rows)
    else:
        # Single method — use the 'probability' and 'method' columns
        result = df[keep + ['probability']].copy()
        result['source'] = df.get('source_gene_id', df.get('source_gene', '')).astype(str)
        result['target'] = df.get('target_id', df.get('target_phenotype', '')).astype(str)
        result['source_name'] = df.get('source_gene', '')
        result['target_name'] = df.get('target_phenotype', '')

        base_method = df.get('method', pd.Series(['unknown'] * len(df)))
        result['method'] = pipeline_label + '_' + base_method.astype(str)

    print(f"    → {len(result):,} rows after pivot, "
          f"methods: {sorted(result['method'].unique())}")
    return result


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def compute_pairwise_corr(pivot: pd.DataFrame, method: str = "spearman"):
    methods = list(pivot.columns)
    m = len(methods)
    corr = np.full((m, m), np.nan, dtype=float)
    n_common = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            a, b = pivot.iloc[:, i], pivot.iloc[:, j]
            mask = a.notna() & b.notna()
            n = int(mask.sum())
            n_common[i, j] = n
            if n < 2:
                continue
            x, y = a[mask].astype(float), b[mask].astype(float)
            if method == "spearman":
                x, y = x.rank(method="average"), y.rank(method="average")
            corr[i, j] = float(np.corrcoef(x, y)[0, 1])

    return methods, corr, n_common


def compute_jaccard_topk(df: pd.DataFrame, k_values=(50, 100, 250, 500, 1000)):
    out = {}
    methods = sorted(df["method"].unique().tolist())
    grouped = {m: d.sort_values("probability", ascending=False)
               for m, d in df.groupby("method")}
    for k in k_values:
        sets = {}
        for m in methods:
            top = grouped[m].head(k) if m in grouped else pd.DataFrame()
            sets[m] = set(zip(
                top["source"].astype(str), top["target"].astype(str)
            )) if len(top) > 0 else set()

        mat = np.zeros((len(methods), len(methods)), dtype=float)
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                union = len(sets[mi] | sets[mj])
                mat[i, j] = len(sets[mi] & sets[mj]) / union if union else np.nan
        out[int(k)] = {"methods": methods, "matrix": mat.tolist()}
    return out


def compute_error_vs_reference(pivot: pd.DataFrame, reference: str):
    if reference not in pivot.columns:
        return None
    ref = pivot[reference]
    metrics = []
    for m in pivot.columns:
        if m == reference:
            continue
        a = pivot[m]
        mask = ref.notna() & a.notna()
        n = int(mask.sum())
        if n == 0:
            metrics.append({"method": m, "n": 0,
                            "mae": None, "rmse": None, "mean_diff": None})
            continue
        diff = (a[mask].astype(float) - ref[mask].astype(float)).values
        metrics.append({
            "method": m, "n": n,
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "mean_diff": float(np.mean(diff)),
        })
    metrics.sort(key=lambda d: (np.inf if d["mae"] is None else d["mae"]))
    return metrics


def compute_pipeline_comparison(pivot: pd.DataFrame) -> Dict:
    """Compare agg_X vs raw_X for each base method."""
    comparison = {}
    for base in ['bdd_exact', 'monte_carlo', 'path_noisy_or']:
        agg_col = f'agg_{base}'
        raw_col = f'raw_{base}'
        if agg_col not in pivot.columns or raw_col not in pivot.columns:
            continue
        mask = pivot[agg_col].notna() & pivot[raw_col].notna()
        n = int(mask.sum())
        if n == 0:
            continue
        agg_v = pivot[agg_col][mask]
        raw_v = pivot[raw_col][mask]
        ratio = agg_v / raw_v.clip(lower=1e-15)
        diff = agg_v - raw_v
        comparison[base] = {
            'n_common': n,
            'mean_ratio_agg_over_raw': float(ratio.mean()),
            'median_ratio': float(ratio.median()),
            'mean_abs_diff': float(diff.abs().mean()),
            'pct_agg_lower': float((agg_v < raw_v).mean() * 100),
            'pct_agg_higher': float((agg_v > raw_v).mean() * 100),
            'pct_equal': float((agg_v == raw_v).mean() * 100),
            'spearman': float(agg_v.rank().corr(raw_v.rank())),
        }
    return comparison


def compute_summary(df: pd.DataFrame) -> Dict:
    methods = sorted(df["method"].unique().tolist())
    pivot = df.pivot_table(
        index=["source", "target"],
        columns="method", values="probability", aggfunc="first",
    )

    sp_m, sp_mat, sp_n = compute_pairwise_corr(pivot, "spearman")
    pe_m, pe_mat, pe_n = compute_pairwise_corr(pivot, "pearson")

    # Pick best reference: prefer raw_bdd_exact > agg_bdd_exact > first
    for candidate in ['raw_bdd_exact', 'agg_bdd_exact']:
        if candidate in pivot.columns:
            reference = candidate
            break
    else:
        reference = methods[0] if methods else None

    errors = compute_error_vs_reference(pivot, reference) if reference else None
    jaccard = compute_jaccard_topk(df)
    pipeline_comp = compute_pipeline_comparison(pivot)

    def _safe(v):
        if isinstance(v, (np.floating, np.integer)):
            f = float(v)
            return None if np.isnan(f) else f
        return v

    coverage_stats = []
    for m in methods:
        sub = df[df["method"] == m]
        coverage_stats.append({
            "method": m,
            "n_pairs": int(sub[["source", "target"]].drop_duplicates().shape[0]),
            "n_rows": int(len(sub)),
            "mean_prob": float(sub["probability"].mean()),
            "median_prob": float(sub["probability"].median()),
            "mean_coverage": float(sub["coverage"].mean()) if "coverage" in sub else None,
        })

    return {
        "methods": methods,
        "n_rows": int(len(df)),
        "n_pairs": int(df[["source", "target"]].drop_duplicates().shape[0]),
        "coverage": coverage_stats,
        "reference_method": reference,
        "spearman": {
            "methods": sp_m,
            "matrix": [[_safe(v) for v in row] for row in sp_mat.tolist()],
            "n_common": sp_n.tolist(),
        },
        "pearson": {
            "methods": pe_m,
            "matrix": [[_safe(v) for v in row] for row in pe_mat.tolist()],
            "n_common": pe_n.tolist(),
        },
        "jaccard": jaccard,
        "errors_vs_reference": errors,
        "pipeline_comparison": pipeline_comp,
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def safe_json_dumps(obj):
    return json.dumps(obj, default=str, ensure_ascii=False)


def generate_html(
    title: str,
    method_results: Dict[str, list],
    summary: Dict,
) -> str:
    methods = summary["methods"]

    # Tabs
    tab_buttons = []
    tab_panels = []

    headers = ("<tr><th>Rank</th><th>Source</th><th>Target</th>"
               "<th>Metapath</th><th>Probability</th><th>Coverage</th>"
               "<th>Evidence</th><th>#Paths</th><th>#Interm.</th></tr>")

    for m in methods:
        mid = "".join(c if (c.isalnum() or c in "_-") else "_" for c in m)
        tab_buttons.append(
            f'<button class="tab-btn" data-tab="{mid}">{m}</button>')
        tab_panels.append(f"""
        <div id="tab-{mid}" class="tab-content">
          <h2><code>{m}</code></h2>
          <div class="fl">
            <div class="fg"><label>Min prob:</label>
              <input type="number" step="0.0001" id="mp-{mid}" value="0"></div>
            <div class="fg"><label>Search:</label>
              <input type="text" id="sr-{mid}" placeholder="gene / phenotype..."></div>
          </div>
          <table id="tb-{mid}" class="display" style="width:100%">
            <thead>{headers}</thead><tbody></tbody>
          </table>
        </div>""")

    # Summary tab
    tab_buttons.append('<button class="tab-btn" data-tab="summary">Summary</button>')

    # Pipeline comparison table
    pc = summary.get("pipeline_comparison", {})
    pc_html = ""
    if pc:
        pc_html = """<h3>Pipeline effect (aggregated ÷ raw)</h3>
        <p>Ratio &lt; 1 means aggregation <em>lowers</em> probability (evidence lost from best-type selection).<br>
        Ratio &gt; 1 means aggregation <em>raises</em> it (rare; can happen when type selection
        removes low-probability competing arcs).</p>
        <table class="ct"><tr><th>Method</th><th>N common</th>
        <th>Mean ratio</th><th>Median ratio</th><th>Spearman ρ</th>
        <th>% agg lower</th><th>% agg higher</th></tr>"""
        for bm, v in pc.items():
            pc_html += (f"<tr><td>{bm}</td><td>{v['n_common']:,}</td>"
                        f"<td>{v['mean_ratio_agg_over_raw']:.4f}</td>"
                        f"<td>{v['median_ratio']:.4f}</td>"
                        f"<td>{v['spearman']:.4f}</td>"
                        f"<td>{v['pct_agg_lower']:.1f}%</td>"
                        f"<td>{v['pct_agg_higher']:.1f}%</td></tr>")
        pc_html += "</table>"

    # Method stats table
    cs_html = "<h3>Method statistics</h3><table class='ct'>"
    cs_html += ("<tr><th>Method</th><th>Pairs</th><th>Rows</th>"
                "<th>Mean prob</th><th>Median prob</th><th>Mean cov</th></tr>")
    for c in summary.get("coverage", []):
        mc = c.get('mean_coverage')
        mc_s = f"{mc:.4f}" if mc is not None else "-"
        cs_html += (f"<tr><td>{c['method']}</td><td>{c['n_pairs']:,}</td>"
                    f"<td>{c['n_rows']:,}</td>"
                    f"<td>{c['mean_prob']:.6f}</td>"
                    f"<td>{c['median_prob']:.6f}</td>"
                    f"<td>{mc_s}</td></tr>")
    cs_html += "</table>"

    tab_panels.append(f"""
    <div id="tab-summary" class="tab-content">
      <h2>Summary</h2>
      <div class="sg">
        <div class="sc"><div class="sv">{summary['n_pairs']:,}</div>
          <div class="sl">Unique (source, target) pairs</div></div>
        <div class="sc"><div class="sv">{len(methods)}</div>
          <div class="sl">Method × Pipeline combinations</div></div>
      </div>
      {cs_html}
      {pc_html}
      <h3>Top-k Jaccard overlap</h3>
      <div class="fl"><div class="fg"><label>k:</label>
        <select id="jk"></select></div></div>
      <div class="hb"><div id="jp" style="width:100%;height:500px"></div></div>
      <h3>Spearman correlation</h3>
      <div class="hb"><div id="sp" style="width:100%;height:520px"></div></div>
      <h3>Error vs reference (<code>{summary.get('reference_method')}</code>)</h3>
      <div class="hb"><div id="eb" style="width:100%;height:420px"></div></div>
    </div>""")

    mr_json = safe_json_dumps(method_results)
    sm_json = safe_json_dumps(summary)
    dt = "".join(c if (c.isalnum() or c in "_-") else "_"
                 for c in (methods[0] if methods else "summary"))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  margin:0;padding:20px;background:#f5f5f5;font-size:14px}}
.ctn{{max-width:1800px;margin:0 auto}}
h1{{color:#333;border-bottom:3px solid #007bff;padding-bottom:10px}}
.tabs{{display:flex;gap:4px;flex-wrap:wrap;border-bottom:2px solid #dee2e6}}
.tab-btn{{padding:9px 13px;border:none;background:#e9ecef;cursor:pointer;
  border-radius:6px 6px 0 0;font-size:12px}}
.tab-btn.active{{background:#fff;color:#007bff;border-bottom:2px solid #fff;margin-bottom:-2px}}
.tab-content{{display:none;background:#fff;padding:20px;border-radius:0 0 8px 8px;
  box-shadow:0 2px 4px rgba(0,0,0,.1)}}
.tab-content.active{{display:block}}
.fl{{display:flex;gap:16px;align-items:center;margin-bottom:12px;padding:10px;
  background:#f8f9fa;border-radius:6px;flex-wrap:wrap}}
.fg{{display:flex;align-items:center;gap:6px}}
.fg input,.fg select{{padding:5px 8px;border:1px solid #ced4da;border-radius:4px}}
table.dataTable{{width:100%!important;font-size:12px}}
table.dataTable thead th{{background:#f8f9fa;font-weight:600;font-size:11px;padding:8px 6px}}
.ph{{color:#28a745;font-weight:600}}.pm{{color:#fd7e14}}.pl{{color:#6c757d}}
.sg{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px}}
.sc{{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;border-left:4px solid #007bff}}
.sv{{font-size:2em;font-weight:bold;color:#007bff}}.sl{{color:#666;font-size:.9em;margin-top:4px}}
.hb{{background:#f8f9fa;padding:12px;border-radius:6px;margin-top:8px}}
.ct{{border-collapse:collapse;width:100%;margin:8px 0}}
.ct th,.ct td{{border:1px solid #dee2e6;padding:7px 10px;text-align:left;font-size:13px}}
.ct th{{background:#f8f9fa}}
code{{background:#f1f3f5;padding:2px 5px;border-radius:4px}}
</style></head>
<body><div class="ctn">
<h1>{title}</h1>
<div class="tabs">{''.join(tab_buttons)}</div>
{''.join(tab_panels)}
</div>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
<script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
<script>
var MR={mr_json};var SM={sm_json};
var tbls={{}},init={{}};
function fp(v){{if(v==null)return'-';var n=+v;if(isNaN(n))return'-';var s=n.toFixed(6);
  return n>=.1?'<span class="ph">'+s+'</span>':n>=.01?'<span class="pm">'+s+'</span>':'<span class="pl">'+s+'</span>'}}
function initT(mid){{if(init[mid])return;var rows=MR[mid]||[];
  var d=rows.map(r=>[r.rank||'-',r.source_gene||r.source_name||r.source||'-',
    r.target_phenotype||r.target_name||r.target||'-',r.metapath||'-',fp(r.probability),
    r.coverage!=null?(+r.coverage).toFixed(4):'-',
    r.evidence_score!=null?(+r.evidence_score).toFixed(2):'-',
    r.num_paths||'-',r.num_intermediates||'-']);
  $.fn.dataTable.ext.search.push(function(s,dd,idx){{
    if(s.nTable.id!=='tb-'+mid)return true;
    var min=+($('#mp-'+mid).val()||0);var raw=rows[idx]?+rows[idx].probability:NaN;
    return!isNaN(raw)&&raw>=min}});
  tbls[mid]=$('#tb-'+mid).DataTable({{data:d,pageLength:50,order:[[0,'asc']],
    dom:'Bfrtip',buttons:['csv'],deferRender:true,scrollX:true}});
  $('#mp-'+mid).on('input',function(){{tbls[mid].draw()}});
  $('#sr-'+mid).on('input',function(){{tbls[mid].search(this.value).draw()}});
  init[mid]=true}}
function actTab(id){{$('.tab-btn').removeClass('active');$('.tab-content').removeClass('active');
  $('.tab-btn[data-tab="'+id+'"]').addClass('active');$('#tab-'+id).addClass('active');
  if(id!=='summary')initT(id);if(id==='summary')renderSum()}}
function rJac(){{var k=$('#jk').val();var j=SM.jaccard[k];if(!j)return;
  var L=j.methods,Z=j.matrix,A=[];
  for(var i=0;i<L.length;i++)for(var j2=0;j2<L.length;j2++){{
    var v=Z[i][j2];A.push({{x:L[j2],y:L[i],
      text:(v==null||isNaN(v))?'—':(+v).toFixed(2),showarrow:false,
      font:{{color:(v!=null&&!isNaN(v)&&v>.5)?'#fff':'#000',size:9}}}})}}
  Plotly.newPlot('jp',[{{z:Z,x:L,y:L,type:'heatmap',colorscale:'Blues',zmin:0,zmax:1}}],
    {{title:'Jaccard overlap (k='+k+')',annotations:A,margin:{{l:150,r:20,t:50,b:170}}}})}}
function rSpe(){{var s=SM.spearman;if(!s)return;var L=s.methods,Z=s.matrix,N=s.n_common,A=[];
  for(var i=0;i<L.length;i++)for(var j=0;j<L.length;j++){{
    var v=Z[i][j];A.push({{x:L[j],y:L[i],
      text:(v==null||isNaN(v))?'—':(+v).toFixed(3)+'\\n(n='+N[i][j]+')',showarrow:false,
      font:{{color:(v!=null&&!isNaN(v)&&Math.abs(v)>.5)?'#fff':'#000',size:8}}}})}}
  var Zp=Z.map(r=>r.map(v=>(v==null||isNaN(v))?0:v));
  Plotly.newPlot('sp',[{{z:Zp,x:L,y:L,type:'heatmap',colorscale:'RdBu',zmin:-1,zmax:1,
    reversescale:true}}],{{title:'Spearman ρ',annotations:A,margin:{{l:160,r:20,t:50,b:170}}}})}}
function rErr(){{var E=SM.errors_vs_reference;if(!E){{Plotly.newPlot('eb',[]);return}}
  var ms=E.map(e=>e.method),mae=E.map(e=>e.mae),rmse=E.map(e=>e.rmse);
  Plotly.newPlot('eb',[{{x:ms,y:mae,type:'bar',name:'MAE'}},
    {{x:ms,y:rmse,type:'bar',name:'RMSE'}}],
    {{title:'Error vs '+SM.reference_method,barmode:'group',
      margin:{{l:60,r:20,t:50,b:190}}}})}}
var _sr=false;function renderSum(){{if(_sr)return;
  var ks=Object.keys(SM.jaccard||{{}}).sort((a,b)=>+a- +b);
  var sel=$('#jk');ks.forEach(k=>sel.append('<option>'+k+'</option>'));
  sel.on('change',rJac);if(ks.length)sel.val(ks[0]);
  rJac();rSpe();rErr();_sr=true}}
$('.tab-btn').on('click',function(){{actTab($(this).data('tab'))}});
actTab('{dt}');
</script></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate pipeline comparison report from inference parquets')

    # Option A: specify tissue + hops (derives paths from config)
    parser.add_argument('--tissue', type=str, default=None,
                        choices=['subcutaneous', 'visceral', 'white', 'brown'])
    parser.add_argument('--hops', type=int, default=None, choices=[2, 3])
    parser.add_argument('--config', type=str, default=None)

    # Option B: specify paths directly
    parser.add_argument('--agg-parquet', type=Path, default=None,
                        help='Path to aggregated pipeline parquet')
    parser.add_argument('--raw-parquet', type=Path, default=None,
                        help='Path to raw pipeline parquet')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory (overrides config)')

    parser.add_argument('--raw-suffix', type=str, default='_raw',
                        help='Filename suffix for raw pipeline parquet (default: _raw)')
    parser.add_argument('--max-rows-per-method', type=int, default=5000)
    args = parser.parse_args()

    # Resolve file paths
    if args.agg_parquet or args.raw_parquet:
        # Direct path mode
        agg_path = args.agg_parquet
        raw_path = args.raw_parquet
        out_dir = args.output_dir or Path('.')
        title_parts = []
        if agg_path:
            title_parts.append(agg_path.stem)
        if raw_path:
            title_parts.append(raw_path.stem)
        title = "Pipeline Comparison: " + " vs ".join(title_parts)
        out_stem = "comparison"
    elif args.tissue and args.hops and args.config:
        # Config mode
        from utils import load_config
        config = load_config(args.config)
        inference_dir = Path(config['paths']['output_dir']) / 'inference'
        agg_path = inference_dir / f'{args.tissue}_{args.hops}hop.parquet'
        raw_path = inference_dir / f'{args.tissue}_{args.hops}hop{args.raw_suffix}.parquet'
        out_dir = args.output_dir or (
            Path(config['paths']['output_dir']) / 'pipeline_comparison')
        title = f"Pipeline Comparison: {args.tissue} ({args.hops}-hop)"
        out_stem = f"comparison_{args.tissue}_{args.hops}hop"
    else:
        parser.error("Provide either (--tissue, --hops, --config) or "
                      "(--agg-parquet and/or --raw-parquet)")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load available parquets
    dfs = []

    if agg_path and agg_path.exists():
        print(f"\nLoading aggregated pipeline output:")
        dfs.append(load_and_pivot(agg_path, 'agg'))
    elif agg_path:
        print(f"  WARNING: aggregated parquet not found: {agg_path}")

    if raw_path and raw_path.exists():
        print(f"\nLoading raw pipeline output:")
        dfs.append(load_and_pivot(raw_path, 'raw'))
    elif raw_path:
        print(f"  WARNING: raw parquet not found: {raw_path}")

    if not dfs:
        print("ERROR: No parquet files found. Run inference first.")
        sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(df_all):,} rows, "
          f"methods: {sorted(df_all['method'].unique())}")

    # Compute summary
    print("Computing summary statistics...")
    summary = compute_summary(df_all)

    # Build method records for DataTables
    method_records = {}
    for method, g in df_all.groupby("method"):
        g = g.sort_values("probability", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)
        if args.max_rows_per_method:
            g = g.head(args.max_rows_per_method)
        method_records[method] = g.to_dict('records')

    # Generate report
    print("Generating HTML report...")
    html = generate_html(title, method_records, summary)

    html_path = out_dir / f'{out_stem}.html'
    html_path.write_text(html, encoding='utf-8')
    print(f"  Saved: {html_path}")

    # Save combined parquet
    parquet_path = out_dir / f'{out_stem}.parquet'
    df_all.to_parquet(parquet_path, index=False)
    print(f"  Saved: {parquet_path}")

    # Save summary JSON
    summary_path = out_dir / f'{out_stem}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved: {summary_path}")

    # Print pipeline comparison to stdout
    pc = summary.get("pipeline_comparison", {})
    if pc:
        print(f"\n{'='*60}")
        print("PIPELINE EFFECT (aggregated ÷ raw)")
        print(f"{'='*60}")
        for bm, v in pc.items():
            print(f"  {bm:20s}  ratio={v['mean_ratio_agg_over_raw']:.4f}  "
                  f"ρ={v['spearman']:.4f}  "
                  f"agg_lower={v['pct_agg_lower']:.1f}%  "
                  f"agg_higher={v['pct_agg_higher']:.1f}%  "
                  f"(n={v['n_common']:,})")

    print(f"\n✓ Report complete: {html_path}")


if __name__ == '__main__':
    main()
