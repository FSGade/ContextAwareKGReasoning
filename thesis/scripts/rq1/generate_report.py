#!/usr/bin/env python3
"""
RQ1 Step 4: Generate interactive HTML reports.

This script generates interactive HTML reports with:
- Visual chain format: Gene → [Rel1] → (# Int) → [Rel2] → Disease
- Detailed per-tissue tables showing all Gene→Disease results
- Expandable intermediate gene lists
- Disease filtering
- Summary tab with Jaccard heatmaps and tissue-exclusive genes

Usage:
    python generate_report.py --config config.yaml
    python generate_report.py --input-dir /path/to/comparisons --output-dir /path/to/reports
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def has_metapath_columns(df: pd.DataFrame) -> bool:
    """Check if DataFrame has metapath grouping columns."""
    return 'metapath_name' in df.columns


def extract_relation_sequence(row: dict, hops: int) -> dict:
    """
    Extract relation sequence from a result row for display.
    
    Returns dict with rel1, rel2, (rel3 for 3-hop) and intermediate info.
    """
    relations = row.get('relation_types', []) or row.get('relation_sequence', [])
    if not relations:
        relations = []
    
    if hops == 2:
        rel1 = relations[0] if len(relations) > 0 else '?'
        rel2 = relations[1] if len(relations) > 1 else '?'
        return {
            'rel1': rel1,
            'rel2': rel2,
            'n_int': row.get('num_intermediates', 0),
        }
    else:  # 3-hop
        rel1 = relations[0] if len(relations) > 0 else '?'
        rel2 = relations[1] if len(relations) > 1 else '?'
        rel3 = relations[2] if len(relations) > 2 else '?'
        
        # For 3-hop, we might have separate B and C intermediate counts
        int_b = row.get('intermediate_B_genes', [])
        int_c = row.get('intermediate_C_genes', [])
        n_b = row.get('n_unique_intermediates_B', len(int_b) if int_b else 0)
        n_c = row.get('n_unique_intermediates_C', len(int_c) if int_c else 0)
        
        return {
            'rel1': rel1,
            'rel2': rel2,
            'rel3': rel3,
            'n_B': n_b,
            'n_C': n_c,
        }


def load_context_results(psr_results_dir: Path, hops: int) -> tuple:
    """
    Load PSR results for all contexts.
    
    Returns tuple: (dict: context_name -> list of result dicts, bool: has_metapaths)
    """
    contexts = ['baseline', 'adipose', 'nonadipose', 'liver']
    results = {}
    has_metapaths = False
    
    # Only include these metabolic diseases
    ALLOWED_DISEASES = {
        'Inflammation',
        'Diabetes Mellitus, Type 2',
        'Diabetes Mellitus, Type 1',
        'Diabetes Mellitus',
        'Diabetes',
        'Diabetes, Gestational',
        'Insulin Resistance',
        'Obesity',
        'Overweight',
        'Metabolic Syndrome',
        'Metabolic Diseases',
    }
    
    for ctx in contexts:
        parquet_path = psr_results_dir / f"{ctx}_{hops}hop_results.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            
            # Check for metapath columns
            if has_metapath_columns(df):
                has_metapaths = True
            
            # Filter to only allowed diseases
            df = df[df['target'].isin(ALLOWED_DISEASES)]
            
            # Convert ALL results to records (no limit)
            records = []
            for _, row in df.iterrows():
                record = {}
                for col in df.columns:
                    val = row[col]
                    # Handle arrays/lists FIRST (before pd.isna check)
                    if isinstance(val, (list, np.ndarray)):
                        # Convert numpy array to list
                        record[col] = list(val) if hasattr(val, '__len__') else []
                    elif np.isscalar(val) and pd.isna(val):
                        record[col] = None
                    elif isinstance(val, (np.integer, np.int64)):
                        record[col] = int(val)
                    elif isinstance(val, (np.floating, np.float64)):
                        record[col] = float(val)
                    else:
                        record[col] = val
                
                # Add extracted relation sequence for easier display
                rel_info = extract_relation_sequence(record, hops)
                record.update(rel_info)
                
                records.append(record)
            results[ctx] = records
            mp_str = " (with metapaths)" if has_metapaths else ""
            print(f"  {ctx}: {len(results[ctx]):,} results loaded{mp_str} (filtered to metabolic diseases)")
        else:
            print(f"  {ctx}: NOT FOUND ({parquet_path})")
            results[ctx] = []
    
    return results, has_metapaths


def generate_html_report(
    comparisons_dir: Path,
    psr_results_dir: Path,
    output_dir: Path,
    hops: int,
    analysis_config: dict
):
    """
    Generate an interactive HTML report for the specified hop length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load per-context results
    print(f"\nLoading {hops}-hop results...")
    context_results, has_metapaths = load_context_results(psr_results_dir, hops)
    
    if not any(context_results.values()):
        print(f"No results found for {hops}-hop")
        return None
    
    print(f"  Has metapath grouping: {has_metapaths}")
    
    # Load comparison results for summary tab
    spearman_path = comparisons_dir / f"spearman_{hops}hop.json"
    jaccard_path = comparisons_dir / f"jaccard_{hops}hop.json"
    exclusive_path = comparisons_dir / f"tissue_exclusive_{hops}hop.json"
    diagnostics_path = comparisons_dir / f"diagnostics_{hops}hop.json"
    metapath_path = comparisons_dir / f"metapath_analysis_{hops}hop.json"
    
    def _load_json(path):
        with open(path) as f:
            return json.load(f)
    
    spearman_data = _load_json(spearman_path) if spearman_path.exists() else None
    jaccard_data = _load_json(jaccard_path) if jaccard_path.exists() else None
    exclusive_data = _load_json(exclusive_path) if exclusive_path.exists() else None
    diagnostics_data = _load_json(diagnostics_path) if diagnostics_path.exists() else None
    metapath_data = _load_json(metapath_path) if metapath_path.exists() else None
    
    # Get unique diseases across all contexts
    all_diseases = set()
    for ctx_results in context_results.values():
        for r in ctx_results:
            all_diseases.add(r.get('target', ''))
    diseases = sorted(all_diseases)
    
    # Count total results
    total_results = sum(len(r) for r in context_results.values())
    
    # Generate HTML
    html_content = generate_html_template(
        hops=hops,
        context_results=context_results,
        spearman_data=spearman_data,
        jaccard_data=jaccard_data,
        exclusive_data=exclusive_data,
        diagnostics_data=diagnostics_data,
        metapath_data=metapath_data,
        diseases=diseases,
        total_results=total_results,
        has_metapaths=has_metapaths,
    )
    
    # Save
    output_path = output_dir / f"report_{hops}hop.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated report: {output_path}")
    return output_path


def safe_json_dumps(obj):
    """Safely serialize object to JSON, escaping for HTML embedding."""
    return json.dumps(obj, default=str, ensure_ascii=False)


def generate_html_template(
    hops: int,
    context_results: dict,
    spearman_data: dict,
    jaccard_data: dict,
    exclusive_data: dict,
    diagnostics_data: dict,
    metapath_data: dict,
    diseases: list,
    total_results: int,
    has_metapaths: bool = False,
) -> str:
    """Generate the HTML content for the report."""
    
    # Convert data to JSON for JavaScript - escape properly
    context_results_json = safe_json_dumps(context_results)
    spearman_json = safe_json_dumps(spearman_data) if spearman_data else 'null'
    jaccard_json = safe_json_dumps(jaccard_data) if jaccard_data else 'null'
    exclusive_json = safe_json_dumps(exclusive_data) if exclusive_data else 'null'
    diagnostics_json = safe_json_dumps(diagnostics_data) if diagnostics_data else 'null'
    metapath_json = safe_json_dumps(metapath_data) if metapath_data else 'null'
    diseases_json = safe_json_dumps(diseases[:100])
    
    # Generate table headers based on hop length
    if hops == 2:
        table_headers = '''<tr>
                        <th>Rank</th>
                        <th>Gene</th>
                        <th>Rel 1</th>
                        <th># Int</th>
                        <th>Rel 2</th>
                        <th>Disease</th>
                        <th>Prob</th>
                        <th>Evidence</th>
                        <th>Corr</th>
                        <th>Intermediates</th>
                    </tr>'''
    else:  # 3-hop
        table_headers = '''<tr>
                        <th>Rank</th>
                        <th>Gene</th>
                        <th>Rel 1</th>
                        <th># B</th>
                        <th>Rel 2</th>
                        <th># C</th>
                        <th>Rel 3</th>
                        <th>Disease</th>
                        <th>Prob</th>
                        <th>Evidence</th>
                        <th>Corr</th>
                        <th>Intermediates B</th>
                        <th>Intermediates C</th>
                    </tr>'''
    
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQ1: Tissue Context Analysis - ''' + str(hops) + '''-Hop Results</title>
    
    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
    <link rel="stylesheet" href="https://cdn.datatables.net/buttons/2.4.2/css/buttons.dataTables.min.css">
    
    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        * { box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
            font-size: 14px;
        }
        
        .container { max-width: 1800px; margin: 0 auto; }
        
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h2 { color: #555; margin-top: 20px; margin-bottom: 15px; }
        h3 { color: #666; margin-top: 15px; margin-bottom: 10px; }
        
        /* Tabs */
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 0;
            flex-wrap: wrap;
            border-bottom: 2px solid #dee2e6;
        }
        
        .tab-btn {
            padding: 12px 24px;
            border: none;
            background: #e9ecef;
            cursor: pointer;
            border-radius: 6px 6px 0 0;
            font-size: 14px;
            font-weight: 500;
            color: #495057;
            transition: all 0.2s;
        }
        
        .tab-btn:hover { background: #dee2e6; }
        
        .tab-btn.active {
            background: white;
            color: #007bff;
            border-bottom: 2px solid white;
            margin-bottom: -2px;
        }
        
        .tab-content {
            display: none;
            background: white;
            padding: 20px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 500px;
        }
        
        .tab-content.active { display: block; }
        
        /* Filters */
        .filters {
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
            flex-wrap: wrap;
        }
        
        .filter-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .filter-group label {
            font-weight: 500;
            color: #495057;
        }
        
        .filter-group select, .filter-group input {
            padding: 8px 12px;
            font-size: 14px;
            border-radius: 4px;
            border: 1px solid #ced4da;
        }
        
        .filter-group select { min-width: 200px; }
        
        /* Tables */
        table.dataTable {
            width: 100% !important;
            font-size: 12px;
        }
        
        table.dataTable thead th {
            background: #f8f9fa;
            font-weight: 600;
            font-size: 11px;
            padding: 8px 6px;
        }
        
        table.dataTable tbody td {
            vertical-align: top;
            padding: 6px 8px;
        }
        
        /* Relation badges */
        .rel-badge {
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-family: monospace;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        
        /* Intermediate count badge */
        .int-count {
            display: inline-block;
            background: #fff3e0;
            color: #e65100;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            font-weight: 600;
        }
        
        /* Intermediate genes display */
        .intermediate-list {
            max-height: 60px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 10px;
            background: #f8f9fa;
            padding: 4px 6px;
            border-radius: 4px;
            line-height: 1.3;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        /* Correlation badges */
        .corr-positive { 
            color: #28a745; 
            font-weight: bold;
        }
        .corr-negative { 
            color: #dc3545; 
            font-weight: bold;
        }
        .corr-unknown { 
            color: #6c757d; 
        }
        
        /* Probability formatting */
        .prob-high { color: #28a745; font-weight: 600; }
        .prob-medium { color: #fd7e14; }
        .prob-low { color: #6c757d; }
        
        /* Summary cards */
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        /* Heatmap container */
        .heatmap-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .heatmap-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        
        /* Exclusive genes */
        .exclusive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
        }
        
        .exclusive-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }
        
        .exclusive-card h4 {
            margin: 0 0 10px 0;
            color: #333;
            text-transform: capitalize;
        }
        
        .gene-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        
        .gene-chip {
            background: #e9ecef;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-family: monospace;
        }
        
        /* Context stats */
        .context-stats {
            display: flex;
            gap: 30px;
            margin-bottom: 15px;
            color: #666;
            font-size: 13px;
        }
        
        .context-stats span {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .context-stats strong {
            color: #333;
        }
        
        /* Loading indicator */
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        /* Gene name styling */
        .gene-name {
            font-weight: 600;
            color: #1a237e;
        }
        
        /* Disease name styling */
        .disease-name {
            color: #4a148c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RQ1: Does Tissue Context Matter? (''' + str(hops) + '''-Hop PSR Analysis)</h1>
        
        <div class="tabs">
            <button class="tab-btn active" data-tab="baseline">Baseline</button>
            <button class="tab-btn" data-tab="adipose">Adipose</button>
            <button class="tab-btn" data-tab="nonadipose">Non-Adipose</button>
            <button class="tab-btn" data-tab="liver">Liver</button>
            <button class="tab-btn" data-tab="summary">Summary</button>
        </div>
        
        <!-- Baseline Tab -->
        <div id="tab-baseline" class="tab-content active">
            <h2>Baseline (All Tissue-Annotated Edges)</h2>
            <div class="context-stats" id="stats-baseline"></div>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="disease-filter-baseline">
                        <option value="">All Diseases</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search Gene:</label>
                    <input type="text" id="gene-search-baseline" placeholder="Gene name...">
                </div>
            </div>
            <table id="table-baseline" class="display" style="width:100%">
                <thead>
                    ''' + table_headers + '''
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <!-- Adipose Tab -->
        <div id="tab-adipose" class="tab-content">
            <h2>Adipose Tissue Context</h2>
            <div class="context-stats" id="stats-adipose"></div>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="disease-filter-adipose">
                        <option value="">All Diseases</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search Gene:</label>
                    <input type="text" id="gene-search-adipose" placeholder="Gene name...">
                </div>
            </div>
            <table id="table-adipose" class="display" style="width:100%">
                <thead>
                    ''' + table_headers + '''
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <!-- Non-Adipose Tab -->
        <div id="tab-nonadipose" class="tab-content">
            <h2>Non-Adipose Tissue Context</h2>
            <div class="context-stats" id="stats-nonadipose"></div>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="disease-filter-nonadipose">
                        <option value="">All Diseases</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search Gene:</label>
                    <input type="text" id="gene-search-nonadipose" placeholder="Gene name...">
                </div>
            </div>
            <table id="table-nonadipose" class="display" style="width:100%">
                <thead>
                    ''' + table_headers + '''
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <!-- Liver Tab -->
        <div id="tab-liver" class="tab-content">
            <h2>Liver Tissue Context</h2>
            <div class="context-stats" id="stats-liver"></div>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="disease-filter-liver">
                        <option value="">All Diseases</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search Gene:</label>
                    <input type="text" id="gene-search-liver" placeholder="Gene name...">
                </div>
            </div>
            <table id="table-liver" class="display" style="width:100%">
                <thead>
                    ''' + table_headers + '''
                </thead>
                <tbody></tbody>
            </table>
        </div>
        
        <!-- Summary Tab -->
        <div id="tab-summary" class="tab-content">
            <h2>Cross-Context Summary</h2>
            
            <div class="summary-grid">
                <div class="stat-card">
                    <div class="stat-value">''' + f"{total_results:,}" + '''</div>
                    <div class="stat-label">Total Gene-Disease Pairs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">''' + str(hops) + '''</div>
                    <div class="stat-label">Hop Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">4</div>
                    <div class="stat-label">Tissue Contexts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="n-diseases">-</div>
                    <div class="stat-label">Unique Diseases</div>
                </div>
            </div>
            
            <h3>Jaccard Overlap (Top-k Gene Sets)</h3>
            <p>How similar are the top-ranked genes across tissue contexts?</p>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="jaccard-disease-select">
                        <option value="">Select disease...</option>
                    </select>
                </div>
            </div>
            <div class="heatmap-container" id="jaccard-heatmaps"></div>
            
            <h3>Spearman Rank Correlation</h3>
            <p>How correlated are gene rankings across tissue contexts? (Only genes present in both contexts are compared)</p>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="spearman-disease-select">
                        <option value="">Select disease...</option>
                    </select>
                </div>
            </div>
            <div class="heatmap-container" id="spearman-heatmaps"></div>
            
            <h3>Tissue-Exclusive Genes</h3>
            
            <h4>vs Baseline</h4>
            <p>Genes in top-100 of a tissue context but NOT in top-500 of baseline.</p>
            <div class="filters">
                <div class="filter-group">
                    <label>Disease:</label>
                    <select id="exclusive-disease-select">
                        <option value="">Select disease...</option>
                    </select>
                </div>
            </div>
            <div class="exclusive-grid" id="exclusive-genes"></div>
            
            <h4 style="margin-top: 30px;">vs Other Tissues</h4>
            <p>Genes in top-100 of a tissue context but NOT in top-500 of ANY other tissue context (including baseline).</p>
            <div class="exclusive-grid" id="exclusive-genes-cross-tissue"></div>
        </div>
    </div>
    
    <!-- jQuery and DataTables -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/dataTables.buttons.min.js"></script>
    <script src="https://cdn.datatables.net/buttons/2.4.2/js/buttons.html5.min.js"></script>
    
    <script>
        // Data embedded from Python
        var contextResults = ''' + context_results_json + ''';
        var jaccardData = ''' + jaccard_json + ''';
        var spearmanData = ''' + spearman_json + ''';
        var exclusiveData = ''' + exclusive_json + ''';
        var hops = ''' + str(hops) + ''';
        
        // State
        var tables = {};
        var initialized = {};
        var contexts = ['baseline', 'adipose', 'nonadipose', 'liver'];
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var tabId = this.getAttribute('data-tab');
                
                // Update button states
                document.querySelectorAll('.tab-btn').forEach(function(b) {
                    b.classList.remove('active');
                });
                this.classList.add('active');
                
                // Update content visibility
                document.querySelectorAll('.tab-content').forEach(function(content) {
                    content.classList.remove('active');
                });
                document.getElementById('tab-' + tabId).classList.add('active');
                
                // Initialize table if needed
                if (contexts.indexOf(tabId) !== -1 && !initialized[tabId]) {
                    initTable(tabId);
                }
            });
        });
        
        // Format correlation - more visible
        function formatCorrelation(val) {
            if (val === 1) return '<span class="corr-positive" title="Positive/Pro-disease">▲ Pro</span>';
            if (val === -1) return '<span class="corr-negative" title="Negative/Anti-disease">▼ Anti</span>';
            return '<span class="corr-unknown" title="Unknown">? Unk</span>';
        }
        
        // Format probability
        function formatProbability(val) {
            if (val === null || val === undefined) return '-';
            var formatted = val.toFixed(4);
            if (val >= 0.1) return '<span class="prob-high">' + formatted + '</span>';
            if (val >= 0.01) return '<span class="prob-medium">' + formatted + '</span>';
            return '<span class="prob-low">' + formatted + '</span>';
        }
        
        // Format relation as badge
        function formatRelation(rel) {
            if (!rel || rel === '?') return '<span class="rel-badge">?</span>';
            // Truncate long relation names
            var display = rel.length > 15 ? rel.substring(0, 15) + '...' : rel;
            return '<span class="rel-badge" title="' + rel + '">' + display + '</span>';
        }
        
        // Format intermediate count
        function formatIntCount(n) {
            return '<span class="int-count">' + (n || 0) + '</span>';
        }
        
        // Format intermediate genes list
        function formatIntermediates(genes) {
            if (!genes || genes.length === 0) return '-';
            
            var display = genes.slice(0, 15).join(', ');
            if (genes.length > 15) {
                display += ' ... (+' + (genes.length - 15) + ')';
            }
            
            return '<div class="intermediate-list">' + display + '</div>';
        }
        
        // Initialize table for a context
        function initTable(ctx) {
            console.log('Initializing table for:', ctx);
            
            var data = contextResults[ctx] || [];
            console.log('Data length:', data.length);
            
            if (data.length === 0) {
                document.getElementById('stats-' + ctx).innerHTML = '<span>No results available</span>';
                initialized[ctx] = true;
                return;
            }
            
            // Populate disease filter
            var diseaseSet = {};
            data.forEach(function(r) {
                if (r.target) diseaseSet[r.target] = true;
            });
            var diseaseList = Object.keys(diseaseSet).sort();
            
            var diseaseSelect = document.getElementById('disease-filter-' + ctx);
            diseaseList.forEach(function(d) {
                var opt = document.createElement('option');
                opt.value = d;
                opt.textContent = d;
                diseaseSelect.appendChild(opt);
            });
            
            // Show context stats
            var uniqueGenes = {};
            data.forEach(function(r) {
                if (r.source_gene) uniqueGenes[r.source_gene] = true;
            });
            
            var statsDiv = document.getElementById('stats-' + ctx);
            statsDiv.innerHTML = 
                '<span><strong>' + data.length.toLocaleString() + '</strong> results</span>' +
                '<span><strong>' + Object.keys(uniqueGenes).length.toLocaleString() + '</strong> unique genes</span>' +
                '<span><strong>' + diseaseList.length + '</strong> diseases</span>';
            
            // Build table data based on hop length
            var tableData;
            
            if (hops === 2) {
                // 2-hop: Gene | Rel1 | #Int | Rel2 | Disease | Prob | Evid | Corr | Intermediates
                tableData = data.map(function(row, idx) {
                    return [
                        row.rank || (idx + 1),
                        '<span class="gene-name">' + (row.source_gene || '-') + '</span>',
                        formatRelation(row.rel1),
                        formatIntCount(row.n_int || row.num_intermediates),
                        formatRelation(row.rel2),
                        '<span class="disease-name">' + (row.target || '-') + '</span>',
                        formatProbability(row.path_probability),
                        (row.evidence_score || 0).toFixed(2),
                        formatCorrelation(row.correlation_type),
                        formatIntermediates(row.intermediate_genes)
                    ];
                });
            } else {
                // 3-hop: Gene | Rel1 | #B | Rel2 | #C | Rel3 | Disease | Prob | Evid | Corr | IntB | IntC
                tableData = data.map(function(row, idx) {
                    return [
                        row.rank || (idx + 1),
                        '<span class="gene-name">' + (row.source_gene || '-') + '</span>',
                        formatRelation(row.rel1),
                        formatIntCount(row.n_B),
                        formatRelation(row.rel2),
                        formatIntCount(row.n_C),
                        formatRelation(row.rel3),
                        '<span class="disease-name">' + (row.target || '-') + '</span>',
                        formatProbability(row.path_probability),
                        (row.evidence_score || 0).toFixed(2),
                        formatCorrelation(row.correlation_type),
                        formatIntermediates(row.intermediate_B_genes),
                        formatIntermediates(row.intermediate_C_genes)
                    ];
                });
            }
            
            // Column definitions based on hop length
            var columnDefs;
            if (hops === 2) {
                columnDefs = [
                    { targets: [9], orderable: false }  // Intermediates column
                ];
            } else {
                columnDefs = [
                    { targets: [11, 12], orderable: false }  // Intermediates columns
                ];
            }
            
            // Initialize DataTable
            tables[ctx] = $('#table-' + ctx).DataTable({
                data: tableData,
                pageLength: 50,
                order: [[0, 'asc']],
                dom: 'Bfrtip',
                buttons: ['csv'],
                columnDefs: columnDefs,
                deferRender: true,
                scrollX: true
            });
            
            // Disease column index depends on hop length
            var diseaseColIdx = (hops === 2) ? 5 : 7;
            var geneColIdx = 1;
            
            // Add filter handlers
            document.getElementById('disease-filter-' + ctx).addEventListener('change', function() {
                var disease = this.value;
                tables[ctx].column(diseaseColIdx).search(disease ? '^' + disease.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + '$' : '', true, false).draw();
            });
            
            document.getElementById('gene-search-' + ctx).addEventListener('input', function() {
                tables[ctx].column(geneColIdx).search(this.value).draw();
            });
            
            initialized[ctx] = true;
            console.log('Table initialized for:', ctx);
        }
        
        // Update Jaccard heatmaps
        function updateJaccardHeatmaps() {
            var disease = document.getElementById('jaccard-disease-select').value;
            if (!disease || !jaccardData || !jaccardData.results) return;
            
            var result = jaccardData.results.find(function(r) {
                return r.disease.toLowerCase() === disease.toLowerCase();
            });
            if (!result) return;
            
            var container = document.getElementById('jaccard-heatmaps');
            container.innerHTML = '';
            
            var kValues = result.k_values || [50, 100, 250, 500];
            var ctxLabels = result.contexts || contexts;
            
            kValues.forEach(function(k) {
                if (!result.jaccard || !result.jaccard[k]) return;
                
                var div = document.createElement('div');
                div.className = 'heatmap-box';
                var plotId = 'jaccard-plot-' + k;
                div.innerHTML = '<div id="' + plotId + '" style="width:100%; height:350px;"></div>';
                container.appendChild(div);
                
                var matrix = result.jaccard[k].matrix;
                
                var annotations = [];
                for (var i = 0; i < ctxLabels.length; i++) {
                    for (var j = 0; j < ctxLabels.length; j++) {
                        annotations.push({
                            x: ctxLabels[j],
                            y: ctxLabels[i],
                            text: matrix[i][j].toFixed(2),
                            showarrow: false,
                            font: { color: matrix[i][j] > 0.5 ? 'white' : 'black' }
                        });
                    }
                }
                
                Plotly.newPlot(plotId, [{
                    z: matrix,
                    x: ctxLabels,
                    y: ctxLabels,
                    type: 'heatmap',
                    colorscale: 'Blues',
                    zmin: 0,
                    zmax: 1
                }], {
                    title: 'Jaccard @ k=' + k,
                    annotations: annotations,
                    margin: { l: 100, r: 20, t: 50, b: 80 }
                });
            });
        }
        
        // Update Spearman heatmaps
        function updateSpearmanHeatmaps() {
            var disease = document.getElementById('spearman-disease-select').value;
            if (!disease || !spearmanData || !spearmanData.results) return;
            
            var result = spearmanData.results.find(function(r) {
                return r.disease.toLowerCase() === disease.toLowerCase();
            });
            if (!result) return;
            
            var container = document.getElementById('spearman-heatmaps');
            container.innerHTML = '';
            
            var ctxLabels = result.contexts || contexts;
            var matrix = result.correlation_matrix;
            var nCommon = result.n_common_genes;
            
            if (!matrix) return;
            
            // Create single heatmap for Spearman correlation
            var div = document.createElement('div');
            div.className = 'heatmap-box';
            var plotId = 'spearman-plot';
            div.innerHTML = '<div id="' + plotId + '" style="width:100%; height:400px;"></div>';
            container.appendChild(div);
            
            var annotations = [];
            for (var i = 0; i < ctxLabels.length; i++) {
                for (var j = 0; j < ctxLabels.length; j++) {
                    var rho = matrix[i][j];
                    var n = nCommon ? nCommon[i][j] : '?';
                    var text = (rho === null || isNaN(rho)) ? 'N/A' : rho.toFixed(3) + '\\n(n=' + n + ')';
                    annotations.push({
                        x: ctxLabels[j],
                        y: ctxLabels[i],
                        text: text,
                        showarrow: false,
                        font: { 
                            color: (rho === null || isNaN(rho)) ? 'gray' : (rho > 0.5 ? 'white' : 'black'),
                            size: 11
                        }
                    });
                }
            }
            
            // Replace NaN with 0 for plotting
            var plotMatrix = matrix.map(function(row) {
                return row.map(function(val) {
                    return (val === null || isNaN(val)) ? 0 : val;
                });
            });
            
            Plotly.newPlot(plotId, [{
                z: plotMatrix,
                x: ctxLabels,
                y: ctxLabels,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmin: -1,
                zmax: 1,
                reversescale: true
            }], {
                title: 'Spearman Rank Correlation (ρ)',
                annotations: annotations,
                margin: { l: 100, r: 20, t: 50, b: 80 }
            });
        }
        
        // Update exclusive genes
        function updateExclusiveGenes() {
            var disease = document.getElementById('exclusive-disease-select').value;
            if (!disease || !exclusiveData || !exclusiveData.results) return;
            
            var result = exclusiveData.results.find(function(r) {
                return r.disease.toLowerCase() === disease.toLowerCase();
            });
            if (!result || !result.exclusive_genes) return;
            
            // Row 1: vs Baseline only
            var container = document.getElementById('exclusive-genes');
            container.innerHTML = '';
            
            ['adipose', 'nonadipose', 'liver'].forEach(function(ctx) {
                var info = result.exclusive_genes[ctx];
                if (!info) return;
                
                var card = document.createElement('div');
                card.className = 'exclusive-card';
                
                var chips = info.genes.slice(0, 30).map(function(g) {
                    return '<span class="gene-chip">' + g + '</span>';
                }).join('');
                
                var moreText = info.genes.length > 30 ? '<p style="margin-top:10px;color:#666;">+ ' + (info.genes.length - 30) + ' more...</p>' : '';
                
                card.innerHTML = 
                    '<h4>' + ctx + ' vs baseline (' + info.count + ' genes)</h4>' +
                    '<div class="gene-chips">' + chips + '</div>' +
                    moreText;
                
                container.appendChild(card);
            });
            
            // Row 2: Cross-tissue comparison (exclusive to one tissue vs ALL others)
            var crossContainer = document.getElementById('exclusive-genes-cross-tissue');
            crossContainer.innerHTML = '';
            
            // We need to compute this from the context results
            // Get top-100 genes for each context for this disease
            var topGenes = {};
            var allContexts = ['baseline', 'adipose', 'nonadipose', 'liver'];
            
            allContexts.forEach(function(ctx) {
                var ctxData = contextResults[ctx] || [];
                var diseaseData = ctxData.filter(function(r) {
                    return r.target && r.target.toLowerCase() === disease.toLowerCase();
                });
                // Already sorted by rank, take top 100
                topGenes[ctx] = {};
                diseaseData.slice(0, 100).forEach(function(r) {
                    if (r.source_gene) topGenes[ctx][r.source_gene] = true;
                });
            });
            
            // Also get top-500 for exclusion check
            var top500Genes = {};
            allContexts.forEach(function(ctx) {
                var ctxData = contextResults[ctx] || [];
                var diseaseData = ctxData.filter(function(r) {
                    return r.target && r.target.toLowerCase() === disease.toLowerCase();
                });
                top500Genes[ctx] = {};
                diseaseData.slice(0, 500).forEach(function(r) {
                    if (r.source_gene) top500Genes[ctx][r.source_gene] = true;
                });
            });
            
            // For each tissue, find genes in its top-100 but NOT in top-500 of ANY other context
            ['adipose', 'nonadipose', 'liver'].forEach(function(ctx) {
                var otherContexts = allContexts.filter(function(c) { return c !== ctx; });
                
                var exclusiveGenes = [];
                Object.keys(topGenes[ctx] || {}).forEach(function(gene) {
                    var inOther = otherContexts.some(function(otherCtx) {
                        return top500Genes[otherCtx] && top500Genes[otherCtx][gene];
                    });
                    if (!inOther) {
                        exclusiveGenes.push(gene);
                    }
                });
                
                var card = document.createElement('div');
                card.className = 'exclusive-card';
                
                if (exclusiveGenes.length === 0) {
                    card.innerHTML = 
                        '<h4>' + ctx + ' vs all others (0 genes)</h4>' +
                        '<p style="color:#666;">No genes unique to this tissue context.</p>';
                } else {
                    var chips = exclusiveGenes.slice(0, 30).map(function(g) {
                        return '<span class="gene-chip" style="background:#d4edda;">' + g + '</span>';
                    }).join('');
                    
                    var moreText = exclusiveGenes.length > 30 ? '<p style="margin-top:10px;color:#666;">+ ' + (exclusiveGenes.length - 30) + ' more...</p>' : '';
                    
                    card.innerHTML = 
                        '<h4>' + ctx + ' vs all others (' + exclusiveGenes.length + ' genes)</h4>' +
                        '<div class="gene-chips">' + chips + '</div>' +
                        moreText;
                }
                
                crossContainer.appendChild(card);
            });
        }
        
        // Initialize on DOM ready
        document.addEventListener('DOMContentLoaded', function() {
            console.log('DOM ready, initializing...');
            console.log('Context results keys:', Object.keys(contextResults));
            console.log('Hop length:', hops);
            
            // Initialize baseline table
            initTable('baseline');
            
            // Populate summary disease dropdowns from both jaccard and spearman data
            var summaryDiseases = {};
            if (jaccardData && jaccardData.results) {
                jaccardData.results.forEach(function(r) {
                    summaryDiseases[r.disease] = true;
                });
            }
            if (spearmanData && spearmanData.results) {
                spearmanData.results.forEach(function(r) {
                    summaryDiseases[r.disease] = true;
                });
            }
            
            var jaccardSelect = document.getElementById('jaccard-disease-select');
            var spearmanSelect = document.getElementById('spearman-disease-select');
            var exclusiveSelect = document.getElementById('exclusive-disease-select');
            
            Object.keys(summaryDiseases).sort().forEach(function(d) {
                var opt1 = document.createElement('option');
                opt1.value = d;
                opt1.textContent = d;
                jaccardSelect.appendChild(opt1);
                
                var opt2 = document.createElement('option');
                opt2.value = d;
                opt2.textContent = d;
                spearmanSelect.appendChild(opt2);
                
                var opt3 = document.createElement('option');
                opt3.value = d;
                opt3.textContent = d;
                exclusiveSelect.appendChild(opt3);
            });
            
            // Add event listeners for summary selects
            jaccardSelect.addEventListener('change', updateJaccardHeatmaps);
            spearmanSelect.addEventListener('change', updateSpearmanHeatmaps);
            exclusiveSelect.addEventListener('change', updateExclusiveGenes);
            
            // Update disease count
            document.getElementById('n-diseases').textContent = Object.keys(summaryDiseases).length;
            
            console.log('Initialization complete');
        });
    </script>
</body>
</html>
'''
    
    return html_template


def main():
    parser = argparse.ArgumentParser(description='Generate HTML reports for RQ1 analysis')
    parser.add_argument('--config', type=Path, help='Path to config.yaml')
    parser.add_argument('--input-dir', type=Path, help='Directory with comparison results')
    parser.add_argument('--output-dir', type=Path, help='Directory for HTML reports')
    parser.add_argument('--hops', type=int, default=None, help='Generate report for specific hop length')
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        base_output = Path(config['paths']['output_dir'])
        comparisons_dir = base_output / 'comparisons'
        psr_results_dir = base_output / 'psr_results'
        output_dir = Path(args.output_dir or base_output / 'reports')
        analysis_config = config.get('analysis', {})
    else:
        if not args.input_dir or not args.output_dir:
            parser.error("Either --config or both --input-dir and --output-dir are required")
        comparisons_dir = args.input_dir
        psr_results_dir = args.input_dir.parent / 'psr_results'
        output_dir = args.output_dir
        analysis_config = {
            'disease_focus': ['inflammation', 'obesity', 'insulin resistance', 'type 2 diabetes']
        }
    
    print("=" * 80)
    print("RQ1 STEP 4: GENERATING HTML REPORTS")
    print("=" * 80)
    print(f"\nPSR results directory: {psr_results_dir}")
    print(f"Comparisons directory: {comparisons_dir}")
    print(f"Output directory: {output_dir}")
    
    # Generate reports
    hops_to_generate = [args.hops] if args.hops else [2, 3]
    
    for hops in hops_to_generate:
        print(f"\n{'='*40}")
        print(f"Generating {hops}-hop report...")
        print(f"{'='*40}")
        
        report_path = generate_html_report(
            comparisons_dir=comparisons_dir,
            psr_results_dir=psr_results_dir,
            output_dir=output_dir,
            hops=hops,
            analysis_config=analysis_config
        )
        
        if report_path:
            print(f"  Created: {report_path}")
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nReports saved to: {output_dir}")


if __name__ == '__main__':
    main()