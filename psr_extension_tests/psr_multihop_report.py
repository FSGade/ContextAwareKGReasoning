#!/usr/bin/env python3
"""
Generate interactive HTML reports comparing multi-hop probability methods.

- Tabs: one per method (bdd_exact, monte_carlo, hierarchical, path_noisy_or, ...)
- Summary tab: Spearman/Pearson heatmaps, Top-k Jaccard overlap heatmaps, error-vs-reference
- Exact-length hop results (2-hop / 3-hop)
- Separate report files for consider_undirected=False/True (optional)

Usage:
  python generate_multihop_report.py --results-dir /path/to/results --output-dir /path/to/reports
  python generate_multihop_report.py --results-dir ... --output-dir ... --hops 2
  python generate_multihop_report.py --results-dir ... --output-dir ... --consider-undirected both
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def sanitize_id(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)

def safe_json_dumps(obj):
    return json.dumps(obj, default=str, ensure_ascii=False)

def to_py(v):
    # robust conversion for numpy/pandas scalars
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def compute_pairwise_corr(pivot: pd.DataFrame, method: str = "spearman"):
    """
    Pairwise correlation matrix with varying overlap.
    Spearman implemented via rank-transform + Pearson.
    Returns (methods, matrix, n_common).
    """
    methods = list(pivot.columns)
    m = len(methods)
    corr = np.full((m, m), np.nan, dtype=float)
    n_common = np.zeros((m, m), dtype=int)

    for i in range(m):
        for j in range(m):
            a = pivot.iloc[:, i]
            b = pivot.iloc[:, j]
            mask = a.notna() & b.notna()
            n = int(mask.sum())
            n_common[i, j] = n
            if n < 2:
                continue
            x = a[mask].astype(float)
            y = b[mask].astype(float)
            if method == "spearman":
                x = x.rank(method="average")
                y = y.rank(method="average")
            # pearson correlation
            corr[i, j] = float(np.corrcoef(x, y)[0, 1])

    return methods, corr, n_common


def compute_jaccard_topk(df: pd.DataFrame, k_values=(50, 100, 250, 500)):
    """
    Jaccard overlap of top-k (source,target) pairs by probability across methods.
    Returns dict: {k: {"methods": [...], "matrix": [[...]]}}
    """
    out = {}
    methods = sorted(df["method"].unique().tolist())
    # Pre-sort per method
    grouped = {m: d.sort_values("probability", ascending=False) for m, d in df.groupby("method")}
    for k in k_values:
        sets = {}
        for m in methods:
            if m not in grouped:
                sets[m] = set()
                continue
            top = grouped[m].head(k)
            sets[m] = set(zip(top["source"].astype(str), top["target"].astype(str)))

        mat = np.zeros((len(methods), len(methods)), dtype=float)
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                A = sets[mi]
                B = sets[mj]
                if not A and not B:
                    mat[i, j] = np.nan
                else:
                    inter = len(A & B)
                    union = len(A | B)
                    mat[i, j] = inter / union if union else np.nan

        out[int(k)] = {"methods": methods, "matrix": mat.tolist()}
    return out


def compute_error_vs_reference(pivot: pd.DataFrame, reference: str):
    """
    Compute error metrics vs reference on overlapping pairs.
    Returns list of dicts.
    """
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
            metrics.append({"method": m, "n": 0, "mae": None, "rmse": None, "mean_diff": None})
            continue
        diff = (a[mask].astype(float) - ref[mask].astype(float)).values
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mean_diff = float(np.mean(diff))
        metrics.append({"method": m, "n": n, "mae": mae, "rmse": rmse, "mean_diff": mean_diff})
    # sort by MAE
    metrics.sort(key=lambda d: (np.inf if d["mae"] is None else d["mae"]))
    return metrics


# -----------------------------
# Load + prepare
# -----------------------------
def load_results_parquet(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    required = {"hops", "consider_undirected", "method", "source", "target", "probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} is missing required columns: {sorted(missing)}")

    # normalize types
    df = df.copy()
    df["method"] = df["method"].astype(str)
    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    df["probability"] = df["probability"].astype(float)

    return df


def split_by_method_records(
    df: pd.DataFrame,
    max_rows_per_method: int | None = 5000,
) -> dict:
    """
    Return {method: [record,...]} with ranks computed within each method.
    """
    out = {}
    for method, g in df.groupby("method"):
        g = g.sort_values("probability", ascending=False).reset_index(drop=True)
        g["rank"] = np.arange(1, len(g) + 1)

        if max_rows_per_method is not None:
            g = g.head(max_rows_per_method)

        records = []
        for _, row in g.iterrows():
            rec = {c: to_py(row[c]) for c in g.columns}
            # prefer names if present
            rec["source_display"] = rec.get("source_name") or rec.get("source")
            rec["target_display"] = rec.get("target_name") or rec.get("target")
            records.append(rec)

        out[method] = records
    return out


def compute_summary(df: pd.DataFrame) -> dict:
    methods = sorted(df["method"].unique().tolist())
    n_total = int(len(df))
    n_pairs = int(df[["source", "target"]].drop_duplicates().shape[0])

    pivot = df.pivot_table(
        index=["source", "target"],
        columns="method",
        values="probability",
        aggfunc="first",
    )

    spearman_methods, spearman_mat, spearman_n = compute_pairwise_corr(pivot, method="spearman")
    pearson_methods, pearson_mat, pearson_n = compute_pairwise_corr(pivot, method="pearson")

    # Choose a reference method for error (prefer bdd_exact)
    reference = "bdd_exact" if "bdd_exact" in pivot.columns else (methods[0] if methods else None)
    errors = compute_error_vs_reference(pivot, reference) if reference else None

    jaccard = compute_jaccard_topk(df, k_values=(50, 100, 250, 500))

    # coverage per method
    coverage = []
    for m in methods:
        n_m = int(df[df["method"] == m][["source", "target"]].drop_duplicates().shape[0])
        coverage.append({"method": m, "n_pairs": n_m})

    return {
        "methods": methods,
        "n_rows": n_total,
        "n_pairs": n_pairs,
        "coverage": coverage,
        "reference_method": reference,
        "spearman": {
            "methods": spearman_methods,
            "matrix": spearman_mat.tolist(),
            "n_common": spearman_n.tolist(),
        },
        "pearson": {
            "methods": pearson_methods,
            "matrix": pearson_mat.tolist(),
            "n_common": pearson_n.tolist(),
        },
        "jaccard": jaccard,
        "errors_vs_reference": errors,
    }


# -----------------------------
# HTML generation
# -----------------------------
def generate_html(
    hops: int,
    consider_undirected: bool,
    method_results: dict,
    summary: dict,
) -> str:
    methods = summary["methods"]
    method_tabs_html = []
    method_panels_html = []

    # union columns to display across all methods
    # (keep stable order; show '-' when missing)
    table_headers = """
        <tr>
            <th>Rank</th>
            <th>Source</th>
            <th>Source type</th>
            <th>Target</th>
            <th>Target type</th>
            <th>Probability</th>
            <th>#Paths</th>
            <th>#Vars</th>
            <th>BDD nodes</th>
            <th>Trunc</th>
            <th>MC n</th>
            <th>MC CI</th>
            <th>#B</th>
            <th>#C</th>
        </tr>
    """

    for m in methods:
        mid = sanitize_id(m)
        method_tabs_html.append(f'<button class="tab-btn" data-tab="{mid}">{m}</button>')

        method_panels_html.append(f"""
        <div id="tab-{mid}" class="tab-content">
            <h2>Method: <code>{m}</code></h2>
            <div class="filters">
                <div class="filter-group">
                    <label>Min prob:</label>
                    <input type="number" step="0.0001" id="minprob-{mid}" value="0">
                </div>
                <div class="filter-group">
                    <label>Search:</label>
                    <input type="text" id="search-{mid}" placeholder="source/target/type...">
                </div>
            </div>
            <table id="table-{mid}" class="display" style="width:100%">
                <thead>{table_headers}</thead>
                <tbody></tbody>
            </table>
        </div>
        """)

    # Summary tab (one)
    method_tabs_html.append('<button class="tab-btn" data-tab="summary">Summary</button>')
    method_panels_html.append(f"""
    <div id="tab-summary" class="tab-content">
        <h2>Summary</h2>

        <div class="summary-grid">
            <div class="stat-card">
                <div class="stat-value">{summary["n_pairs"]:,}</div>
                <div class="stat-label">Unique (source,target) pairs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(methods)}</div>
                <div class="stat-label">Methods</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{hops}</div>
                <div class="stat-label">Exact hop length</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{str(bool(consider_undirected))}</div>
                <div class="stat-label">consider_undirected</div>
            </div>
        </div>

        <h3>Top-k Jaccard overlap</h3>
        <div class="filters">
            <div class="filter-group">
                <label>k:</label>
                <select id="jaccard-k"></select>
            </div>
        </div>
        <div class="heatmap-box">
            <div id="jaccard-plot" style="width:100%; height:450px;"></div>
        </div>

        <h3>Spearman correlation (probabilities)</h3>
        <div class="heatmap-box">
            <div id="spearman-plot" style="width:100%; height:480px;"></div>
        </div>

        <h3>Error vs reference</h3>
        <p>Reference method: <code>{summary.get("reference_method")}</code></p>
        <div class="heatmap-box">
            <div id="error-bar" style="width:100%; height:420px;"></div>
        </div>
    </div>
    """)

    # Embed JSON data
    method_results_json = safe_json_dumps(method_results)
    summary_json = safe_json_dumps(summary)

    # Make first method active by default
    default_method = "bdd_exact" if "bdd_exact" in methods else (methods[0] if methods else None)
    default_tab = sanitize_id(default_method) if default_method else "summary"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Multi-hop PSR Method Comparison ({hops}-hop, undirected={bool(consider_undirected)})</title>

  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
  <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; font-size: 14px; }}
    .container {{ max-width: 1800px; margin: 0 auto; }}
    h1 {{ color:#333; border-bottom:3px solid #007bff; padding-bottom:10px; }}

    .tabs {{ display:flex; gap:5px; flex-wrap:wrap; border-bottom:2px solid #dee2e6; }}
    .tab-btn {{ padding:10px 16px; border:none; background:#e9ecef; cursor:pointer; border-radius:6px 6px 0 0; }}
    .tab-btn.active {{ background:white; color:#007bff; border-bottom:2px solid white; margin-bottom:-2px; }}

    .tab-content {{ display:none; background:white; padding:20px; border-radius:0 0 8px 8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); }}
    .tab-content.active {{ display:block; }}

    .filters {{ display:flex; gap:20px; align-items:center; margin-bottom:15px; padding:12px; background:#f8f9fa; border-radius:6px; flex-wrap:wrap; }}
    .filter-group {{ display:flex; align-items:center; gap:8px; }}
    .filter-group input, .filter-group select {{ padding:6px 10px; border:1px solid #ced4da; border-radius:4px; }}

    table.dataTable {{ width:100% !important; font-size:12px; }}
    table.dataTable thead th {{ background:#f8f9fa; font-weight:600; font-size:11px; padding:8px 6px; }}

    .prob-high {{ color:#28a745; font-weight:600; }}
    .prob-medium {{ color:#fd7e14; }}
    .prob-low {{ color:#6c757d; }}

    .summary-grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:15px; margin-bottom:20px; }}
    .stat-card {{ background:#f8f9fa; padding:18px; border-radius:8px; text-align:center; border-left:4px solid #007bff; }}
    .stat-value {{ font-size:2em; font-weight:bold; color:#007bff; }}
    .stat-label {{ color:#666; font-size:0.9em; margin-top:5px; }}

    .heatmap-box {{ background:#f8f9fa; padding:15px; border-radius:6px; margin-top:10px; }}
    code {{ background:#f1f3f5; padding:2px 6px; border-radius:4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Multi-hop method comparison ({hops}-hop, consider_undirected={bool(consider_undirected)})</h1>

    <div class="tabs" id="tabs">
      {''.join(method_tabs_html)}
    </div>

    {''.join(method_panels_html)}
  </div>

  <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
  <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>

  <script>
    var methodResults = {method_results_json};
    var summary = {summary_json};

    var methods = summary.methods || [];
    var tables = {{}};
    var initialized = {{}};

    function formatProbability(val) {{
      if (val === null || val === undefined) return '-';
      var v = Number(val);
      if (isNaN(v)) return '-';
      var s = v.toFixed(6);
      if (v >= 0.1) return '<span class="prob-high">' + s + '</span>';
      if (v >= 0.01) return '<span class="prob-medium">' + s + '</span>';
      return '<span class="prob-low">' + s + '</span>';
    }}

    function formatBool(val) {{
      if (val === true) return 'Y';
      if (val === false) return 'N';
      return '-';
    }}

    function formatCI(lo, hi) {{
      if (lo === null || lo === undefined || hi === null || hi === undefined) return '-';
      var a = Number(lo), b = Number(hi);
      if (isNaN(a) || isNaN(b)) return '-';
      return a.toFixed(4) + '–' + b.toFixed(4);
    }}

    function initTable(methodName) {{
      var mid = methodName;
      if (initialized[mid]) return;

      var rows = methodResults[methodName] || [];
      var tableData = rows.map(function(r) {{
        return [
          r.rank || '-',
          r.source_display || r.source || '-',
          r.source_type || '-',
          r.target_display || r.target || '-',
          r.target_type || '-',
          formatProbability(r.probability),
          (r.num_paths !== undefined && r.num_paths !== null) ? r.num_paths : '-',
          (r.num_vars !== undefined && r.num_vars !== null) ? r.num_vars : '-',
          (r.bdd_nodes !== undefined && r.bdd_nodes !== null) ? r.bdd_nodes : '-',
          formatBool(r.truncated),
          (r.n_samples !== undefined && r.n_samples !== null) ? r.n_samples : '-',
          formatCI(r.ci_low, r.ci_high),
          (r.num_B !== undefined && r.num_B !== null) ? r.num_B : '-',
          (r.num_C !== undefined && r.num_C !== null) ? r.num_C : '-'
        ];
      }});

      var probCol = 5; // formatted HTML but still searchable; we use custom filter by parsing the raw from rows via hidden mapping
      // Custom min-prob filter per table
      $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {{
        if (settings.nTable.id !== 'table-' + mid) return true;
        var minVal = Number(document.getElementById('minprob-' + mid).value || 0);
        var raw = rows[dataIndex] ? Number(rows[dataIndex].probability) : NaN;
        if (isNaN(raw)) return false;
        return raw >= minVal;
      }});

      tables[mid] = $('#table-' + mid).DataTable({{
        data: tableData,
        pageLength: 50,
        order: [[0, 'asc']],
        dom: 'Bfrtip',
        buttons: ['csv'],
        deferRender: true,
        scrollX: true
      }});

      document.getElementById('minprob-' + mid).addEventListener('input', function() {{
        tables[mid].draw();
      }});

      document.getElementById('search-' + mid).addEventListener('input', function() {{
        tables[mid].search(this.value).draw();
      }});

      initialized[mid] = true;
    }}

    function activateTab(tabId) {{
      // buttons
      document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
      document.querySelectorAll('.tab-content').forEach(function(c) {{ c.classList.remove('active'); }});

      var btn = document.querySelector('.tab-btn[data-tab="' + tabId + '"]');
      if (btn) btn.classList.add('active');
      var panel = document.getElementById('tab-' + tabId);
      if (panel) panel.classList.add('active');

      if (tabId !== 'summary') initTable(tabId);
      if (tabId === 'summary') renderSummary();
    }}

    function renderJaccard() {{
      var sel = document.getElementById('jaccard-k');
      var k = sel.value;
      var j = summary.jaccard[k];
      if (!j) return;

      var labels = j.methods;
      var z = j.matrix;

      var annotations = [];
      for (var i=0; i<labels.length; i++) {{
        for (var j2=0; j2<labels.length; j2++) {{
          var v = z[i][j2];
          var txt = (v === null || isNaN(v)) ? 'N/A' : Number(v).toFixed(2);
          annotations.push({{
            x: labels[j2],
            y: labels[i],
            text: txt,
            showarrow: false,
            font: {{ color: (v !== null && !isNaN(v) && v > 0.5) ? 'white' : 'black', size: 11 }}
          }});
        }}
      }}

      Plotly.newPlot('jaccard-plot', [{{
        z: z,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'Blues',
        zmin: 0,
        zmax: 1
      }}], {{
        title: 'Jaccard overlap of top-k pairs (k=' + k + ')',
        annotations: annotations,
        margin: {{ l: 120, r: 20, t: 60, b: 120 }}
      }});
    }}

    function renderSpearman() {{
      var sp = summary.spearman;
      if (!sp) return;
      var labels = sp.methods;
      var z = sp.matrix;
      var n = sp.n_common;

      var annotations = [];
      for (var i=0; i<labels.length; i++) {{
        for (var j=0; j<labels.length; j++) {{
          var v = z[i][j];
          var txt = (v === null || isNaN(v)) ? 'N/A' : Number(v).toFixed(3) + '\\n(n=' + n[i][j] + ')';
          annotations.push({{
            x: labels[j],
            y: labels[i],
            text: txt,
            showarrow: false,
            font: {{ color: (v !== null && !isNaN(v) && Math.abs(v) > 0.5) ? 'white' : 'black', size: 10 }}
          }});
        }}
      }}

      // Replace NaN with 0 for plotting but keep annotations
      var zplot = z.map(function(row) {{
        return row.map(function(v) {{
          return (v === null || isNaN(v)) ? 0 : v;
        }});
      }});

      Plotly.newPlot('spearman-plot', [{{
        z: zplot,
        x: labels,
        y: labels,
        type: 'heatmap',
        colorscale: 'RdBu',
        zmin: -1,
        zmax: 1,
        reversescale: true
      }}], {{
        title: 'Spearman correlation (pairwise overlap)',
        annotations: annotations,
        margin: {{ l: 140, r: 20, t: 60, b: 140 }}
      }});
    }}

    function renderErrorBar() {{
      var errs = summary.errors_vs_reference;
      if (!errs) {{
        Plotly.newPlot('error-bar', [], {{ title: 'No reference errors available' }});
        return;
      }}
      var methods = errs.map(e => e.method);
      var mae = errs.map(e => (e.mae === null ? null : e.mae));
      var rmse = errs.map(e => (e.rmse === null ? null : e.rmse));
      var n = errs.map(e => e.n);

      Plotly.newPlot('error-bar', [
        {{
          x: methods, y: mae, type: 'bar', name: 'MAE',
          hovertemplate: 'MAE=%{{y:.6f}}<br>n=%{{customdata}}<extra></extra>',
          customdata: n
        }},
        {{
          x: methods, y: rmse, type: 'bar', name: 'RMSE',
          hovertemplate: 'RMSE=%{{y:.6f}}<br>n=%{{customdata}}<extra></extra>',
          customdata: n
        }}
      ], {{
        title: 'Error vs reference (' + summary.reference_method + ')',
        barmode: 'group',
        margin: {{ l: 60, r: 20, t: 60, b: 160 }}
      }});
    }}

    var summaryRendered = false;
    function renderSummary() {{
      if (summaryRendered) return;

      // populate k selector
      var ks = Object.keys(summary.jaccard || {{}}).sort(function(a,b){{return Number(a)-Number(b);}});
      var sel = document.getElementById('jaccard-k');
      ks.forEach(function(k) {{
        var opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k;
        sel.appendChild(opt);
      }});
      sel.addEventListener('change', renderJaccard);

      // render default
      if (ks.length > 0) sel.value = ks[0];
      renderJaccard();
      renderSpearman();
      renderErrorBar();

      summaryRendered = true;
    }}

    // tab wiring
    document.querySelectorAll('.tab-btn').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        activateTab(this.getAttribute('data-tab'));
      }});
    }});

    // activate default tab
    activateTab('{default_tab}');
  </script>
</body>
</html>
"""
    return html


# -----------------------------
# Main
# -----------------------------
def find_results_file(results_dir: Path, hops: int, consider_undirected: bool) -> Path | None:
    tag = "1" if consider_undirected else "0"
    p = results_dir / f"results_{hops}hop_undirected{tag}.parquet"
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True, help="Directory containing results_?hop_undirected?.parquet")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory to write HTML reports")
    ap.add_argument("--hops", type=int, default=None, help="Generate report for a specific hop length (2 or 3). Default: both.")
    ap.add_argument("--consider-undirected", type=str, default="false", choices=["false", "true", "both"],
                    help="Whether to generate undirected reports. Default: false.")
    ap.add_argument("--max-rows-per-method", type=int, default=5000, help="Cap rows per method embedded into HTML.")
    args = ap.parse_args()

    results_dir = args.results_dir
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    hops_list = [args.hops] if args.hops else [2, 3]
    undirected_list = [False, True] if args.consider_undirected == "both" else [args.consider_undirected == "true"]

    for hops in hops_list:
        for und in undirected_list:
            f = find_results_file(results_dir, hops, und)
            if not f:
                print(f"SKIP: missing results file for hops={hops}, undirected={und}")
                continue

            print(f"Loading: {f}")
            df = load_results_parquet(f)

            # sanity: enforce the file matches the requested condition (but allow mixed if user did that)
            df_cond = df[(df["hops"] == hops) & (df["consider_undirected"] == und)]
            if df_cond.empty:
                print(f"WARNING: no rows match hops={hops}, undirected={und} inside {f}. Using all rows.")
                df_cond = df

            method_results = split_by_method_records(df_cond, max_rows_per_method=args.max_rows_per_method)
            summary = compute_summary(df_cond)

            html = generate_html(hops=hops, consider_undirected=und, method_results=method_results, summary=summary)

            outpath = outdir / f"report_{hops}hop_undirected{int(und)}.html"
            outpath.write_text(html, encoding="utf-8")
            print(f"Wrote: {outpath}")

    print("Done.")


if __name__ == "__main__":
    main()
