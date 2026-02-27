#!/usr/bin/env python3
"""
Systematic evaluation of PubMed search strategies for KG subsetting.

This script runs a comprehensive set of experiments to evaluate:
1. Impact of including/excluding reviews
2. Impact of disease filters
3. Impact of strict vs permissive mode
4. Impact of k-hop expansion

Results are saved to CSV for analysis.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
from contextlib import redirect_stdout

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph import KnowledgeGraph, print_kg_stats
from pubmed.pubmed_cache import PubMedBatchCache
from Bio import Entrez

# Configure Entrez
Entrez.email = "s233139@dtu.dk"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchConfig:
    """Configuration for a single search experiment."""
    name: str
    keywords: List[str]
    cooccur_terms: List[str]
    disease_terms: Optional[List[str]] = None
    journal_filter: str = "no_reviews"  # "no_reviews", "all", "reviews_only"
    strict_mode: bool = True
    k_hop: int = 0
    max_results: int = 10000
    start_year: int = 2000
    end_year: int = 2023


# =============================================================================
# Search Functions
# =============================================================================

def search_pubmed_cooccurrence(
    key_terms: List[str],
    adipose_terms: List[str],
    disease_terms: Optional[List[str]] = None,
    max_results: int = 10000,
    start_year: int = 2000,
    end_year: int = 2023,
    journal_filter: str = "no_reviews"
) -> set:
    """
    Search PubMed with flexible filtering.
    
    Args:
        key_terms: Main search terms (e.g., ["inflammation"])
        adipose_terms: Co-occurrence terms
        disease_terms: Optional disease filter terms
        journal_filter: "no_reviews", "all", or "reviews_only"
    
    Returns:
        Set of PMIDs
    """
    all_pmids = set()
    
    for key_term in key_terms:
        # Build co-occurrence query
        adipose_query = " OR ".join(f'"{term}"' for term in adipose_terms)
        query = f'("{key_term}") AND ({adipose_query})'
        
        # Add disease filter if provided
        if disease_terms:
            disease_query = " OR ".join(f'"{term}"' for term in disease_terms)
            query += f' AND ({disease_query})'
        
        # Add date range
        query += f' AND {start_year}:{end_year}[dp]'
        
        # Add article type filter
        if journal_filter == "no_reviews":
            query += ' AND "journal article"[pt] NOT "review"[pt]'
        elif journal_filter == "reviews_only":
            query += ' AND "review"[pt]'
        # else: "all" - no filter
        
        logging.info(f"Searching: {query}")
        
        try:
            # Execute search
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            pmids = set(record['IdList'])
            logging.info(f"  Found {len(pmids)} PMIDs for '{key_term}'")
            all_pmids.update(pmids)
            
        except Exception as e:
            logging.error(f"Error searching for '{key_term}': {e}")
            continue
    
    logging.info(f"Total unique PMIDs: {len(all_pmids)}")
    return all_pmids


def subset_graph_by_pmids(kg, pmids: set, k_hop: int = 0):
    """
    Subset graph to edges with document_id in pmids.
    
    Args:
        kg: KnowledgeGraph
        pmids: Set of PubMed IDs
        k_hop: Number of hops for expansion (0 = no expansion)
    
    Returns:
        Subgraph
    """
    subgraph = KnowledgeGraph(schema=kg.schema)
    
    # Add edges with matching PMIDs
    for u, v, data in kg.edges(data=True):
        doc_id = data.get('document_id')
        if doc_id and doc_id in pmids:
            # Add nodes if not present
            if u not in subgraph:
                subgraph.add_node(u, **kg.nodes[u])
            if v not in subgraph:
                subgraph.add_node(v, **kg.nodes[v])
            # Add edge
            subgraph.add_edge(u, v, **data)
    
    return subgraph


def add_all_edges_between_nodes(kg, subgraph):
    """
    Add all edges from kg between nodes in subgraph.
    
    Args:
        kg: Full KnowledgeGraph
        subgraph: Subset KnowledgeGraph
    
    Returns:
        New subgraph with all edges
    """
    # Create new graph with same nodes
    new_subgraph = KnowledgeGraph(schema=kg.schema)
    for node in subgraph.nodes():
        new_subgraph.add_node(node, **subgraph.nodes[node])
    
    # Add ALL edges from kg between these nodes
    nodes_set = set(subgraph.nodes())
    for u, v, data in kg.edges(data=True):
        if u in nodes_set and v in nodes_set:
            new_subgraph.add_edge(u, v, **data)
    
    return new_subgraph


def expand_subgraph_k_hops(kg, subgraph, k: int):
    """
    Expand subgraph by k hops.
    
    Args:
        kg: Full KnowledgeGraph
        subgraph: Starting subgraph
        k: Number of hops
    
    Returns:
        Expanded subgraph
    """
    if k == 0:
        return subgraph
    
    # Start with current nodes
    current_nodes = set(subgraph.nodes())
    
    # Expand k times
    for _ in range(k):
        new_nodes = set()
        for node in current_nodes:
            if node in kg:
                neighbors = set(kg.neighbors(node))
                new_nodes.update(neighbors)
        current_nodes.update(new_nodes)
    
    # Create expanded subgraph
    expanded = KnowledgeGraph(schema=kg.schema)
    for node in current_nodes:
        if node in kg:
            expanded.add_node(node, **kg.nodes[node])
    
    # Add edges
    for u, v, data in kg.edges(data=True):
        if u in current_nodes and v in current_nodes:
            expanded.add_edge(u, v, **data)
    
    return expanded


def collect_pmids_from_subgraph(subgraph) -> set:
    """Extract all unique PMIDs from subgraph edges."""
    pmids = set()
    for u, v, data in subgraph.edges(data=True):
        doc_id = data.get('document_id')
        if doc_id:
            pmids.add(doc_id)
    return pmids


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_graph_metrics(subgraph) -> Dict:
    """
    Compute comprehensive metrics for a subgraph.
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'n_nodes': subgraph.number_of_nodes(),
        'n_edges': subgraph.number_of_edges(),
    }
    
    if metrics['n_nodes'] > 0:
        degrees = [subgraph.degree(n) for n in subgraph.nodes()]
        metrics['avg_degree'] = float(np.mean(degrees))
        metrics['median_degree'] = float(np.median(degrees))
        metrics['max_degree'] = int(np.max(degrees))
        metrics['std_degree'] = float(np.std(degrees))
        metrics['min_degree'] = int(np.min(degrees))
        
        # Density: actual edges / possible edges
        n = metrics['n_nodes']
        possible_edges = n * (n - 1) / 2
        metrics['density'] = metrics['n_edges'] / possible_edges if possible_edges > 0 else 0.0
        
        # Degree distribution quartiles
        metrics['degree_q25'] = float(np.percentile(degrees, 25))
        metrics['degree_q75'] = float(np.percentile(degrees, 75))
    else:
        metrics.update({
            'avg_degree': 0.0,
            'median_degree': 0.0,
            'max_degree': 0,
            'std_degree': 0.0,
            'min_degree': 0,
            'density': 0.0,
            'degree_q25': 0.0,
            'degree_q75': 0.0,
        })
    
    # Count unique PMIDs
    pmids = collect_pmids_from_subgraph(subgraph)
    metrics['n_pmids'] = len(pmids)
    
    return metrics


# =============================================================================
# Experiment Runner
# =============================================================================

class SubsettingExperiment:
    """Orchestrates systematic experiments for KG subsetting strategies."""
    
    def __init__(self, base_path: Path, results_path: Path, cache_dir: Path):
        self.base_path = base_path
        self.results_path = results_path
        self.cache_dir = cache_dir
        self.results = []
        
        # Create directories
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_dir = results_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"subsetting_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to: {log_file}")
        self.logger.info(f"Results will be saved to: {results_path}")
        self.logger.info(f"Caches will be saved to: {cache_dir}")
        
    def _get_cache_path(self, graph_name: str, config: SearchConfig) -> Path:
        """
        Get cache database path for this specific experiment.
        
        Creates separate cache per (graph, search_strategy, article_type)
        to keep databases manageable.
        """
        # Create cache filename
        cache_name = f"{graph_name}_{config.name.split('_')[0]}_{config.journal_filter}.db"
        cache_path = self.cache_dir / cache_name
        
        return cache_path
        
    def run_single_experiment(
        self,
        graph_name: str,
        kg,
        config: SearchConfig
    ) -> Dict:
        """
        Run a single experiment with given configuration.
        
        Returns:
            Dictionary with all metrics
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Experiment: {graph_name} - {config.name}")
        self.logger.info(f"{'='*80}")
        
        # Get dedicated cache for this experiment
        cache_path = self._get_cache_path(graph_name, config)
        self.logger.info(f"Using cache: {cache_path}")
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'graph_name': graph_name,
            'experiment_name': config.name,
            'search_strategy': config.name.split('_')[0],  # Extract base strategy
            'article_type': config.journal_filter,
            'mode': 'strict' if config.strict_mode and config.k_hop == 0 else
                   'permissive' if not config.strict_mode else f'{config.k_hop}hop',
            'n_keywords': len(config.keywords),
            'n_cooccur_terms': len(config.cooccur_terms),
            'has_disease_filter': config.disease_terms is not None,
            'strict_mode': config.strict_mode,
            'k_hop': config.k_hop,
            'cache_db': str(cache_path.name),  # Store cache name for reference
        }
        
        try:
            # Step 1: Search PubMed
            self.logger.info("Step 1: Searching PubMed...")
            initial_pmids = search_pubmed_cooccurrence(
                key_terms=config.keywords,
                adipose_terms=config.cooccur_terms,
                disease_terms=config.disease_terms,
                max_results=config.max_results,
                start_year=config.start_year,
                end_year=config.end_year,
                journal_filter=config.journal_filter
            )
            
            result['pmids_initial'] = len(initial_pmids)
            
            if len(initial_pmids) == 0:
                self.logger.warning("No PMIDs found! Skipping...")
                result['status'] = 'no_pmids'
                return result
            
            # Step 2: Cache abstracts
            self.logger.info(f"Step 2: Caching {len(initial_pmids)} PMIDs...")
            n_cached = self._cache_pmids(initial_pmids, cache_path)
            result['abstracts_cached'] = n_cached
            
            # Step 3: Initial subset
            self.logger.info("Step 3: Creating initial subset...")
            subgraph = subset_graph_by_pmids(kg, initial_pmids, k_hop=0)
            metrics_initial = compute_graph_metrics(subgraph)
            
            # Add initial metrics to result
            for key, value in metrics_initial.items():
                result[f'{key}_initial'] = value
            
            # Step 4: Add all edges (if permissive mode)
            if not config.strict_mode:
                self.logger.info("Step 4: Adding all edges between nodes...")
                subgraph_all = add_all_edges_between_nodes(kg, subgraph)
                metrics_all = compute_graph_metrics(subgraph_all)
                
                all_pmids = collect_pmids_from_subgraph(subgraph_all)
                new_pmids = all_pmids - initial_pmids
                
                # Add permissive metrics
                for key, value in metrics_all.items():
                    result[f'{key}_all'] = value
                
                result['pmids_discovered'] = len(new_pmids)
                result['pmids_total_after_all'] = len(all_pmids)
                
                # Cache newly discovered PMIDs
                if len(new_pmids) > 0:
                    self.logger.info(f"Caching {len(new_pmids)} newly discovered PMIDs...")
                    self._cache_pmids(new_pmids, cache_path)
            
            # Step 5: K-hop expansion (if applicable)
            if config.k_hop > 0:
                self.logger.info(f"Step 5: Performing {config.k_hop}-hop expansion...")
                subgraph_khop = expand_subgraph_k_hops(kg, subgraph, config.k_hop)
                metrics_khop = compute_graph_metrics(subgraph_khop)
                
                # Add k-hop metrics
                for key, value in metrics_khop.items():
                    result[f'{key}_khop'] = value
                
                # Cache PMIDs from k-hop expansion
                khop_pmids = collect_pmids_from_subgraph(subgraph_khop)
                new_khop_pmids = khop_pmids - initial_pmids
                if len(new_khop_pmids) > 0:
                    self.logger.info(f"Caching {len(new_khop_pmids)} PMIDs from k-hop expansion...")
                    self._cache_pmids(new_khop_pmids, cache_path)
            
            result['status'] = 'success'
            self.logger.info(f"✓ Completed successfully")
            
        except Exception as e:
            self.logger.error(f"✗ Error: {e}", exc_info=True)
            result['status'] = 'error'
            result['error_message'] = str(e)
        
        return result
    
    def _cache_pmids(self, pmids: set, cache_path: Path) -> int:
        """
        Cache PMIDs to specified database and return number successfully cached.
        
        Args:
            pmids: Set of PMIDs to cache
            cache_path: Path to cache database
        
        Returns:
            Number of abstracts successfully cached
        """
        if len(pmids) == 0:
            return 0
        
        # Check cache size before adding
        if cache_path.exists():
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Current cache size: {cache_size_mb:.2f} MB")
            
        cache = PubMedBatchCache(db_path=str(cache_path), email=Entrez.email)
        try:
            cache.fetch_batch(list(pmids), batch_size=200, rate_limiting=0.4)
            abstracts = cache.get_abstracts(list(pmids))
            
            # Report cache size after adding
            cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Cache size after adding: {cache_size_mb:.2f} MB")
            
            return len(abstracts)
        except Exception as e:
            self.logger.error(f"Error caching PMIDs: {e}")
            return 0
        finally:
            cache.close()
    
    def run_all_experiments(self, graphs: Dict, experiments: List[SearchConfig]):
        """
        Run all experiments across all graphs.
        
        Args:
            graphs: Dictionary of {graph_name: KnowledgeGraph}
            experiments: List of SearchConfig objects
        """
        total = len(graphs) * len(experiments)
        current = 0
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"RUNNING {total} TOTAL EXPERIMENTS")
        self.logger.info(f"  Graphs: {len(graphs)}")
        self.logger.info(f"  Configs per graph: {len(experiments)}")
        self.logger.info(f"{'='*80}\n")
        
        for graph_name, graph in graphs.items():
            self.logger.info(f"\n{'#'*80}")
            self.logger.info(f"# GRAPH: {graph_name}")
            self.logger.info(f"# Nodes: {graph.number_of_nodes():,}, Edges: {graph.number_of_edges():,}")
            self.logger.info(f"{'#'*80}\n")
            
            for config in experiments:
                current += 1
                self.logger.info(f"\nProgress: {current}/{total}")
                
                result = self.run_single_experiment(graph_name, graph, config)
                self.results.append(result)
                
                # Save intermediate results after each experiment
                self._save_results()
        
        # Final save and summary
        self._save_results()
        self._generate_summary_report()
    
    def _load_all_graphs(self) -> Dict:
        """Load all graph variants to test."""
        graphs = {}
        
        graph_configs = [
            ('ikraph_full', self.base_path / "data/graphs/ikraph.pkl"),
            ('ikraph_pubmed', self.base_path / "data/graphs/subsets/ikraph_pubmed.pkl"),
            ('ikraph_pubmed_human', self.base_path / "data/graphs/subsets/ikraph_pubmed_human.pkl"),
        ]
        
        for name, path in graph_configs:
            if path.exists():
                self.logger.info(f"Loading {name} from {path}...")
                try:
                    graphs[name] = KnowledgeGraph.import_graph(str(path))
                    self.logger.info(f"  ✓ Loaded: {graphs[name].number_of_nodes():,} nodes, "
                                   f"{graphs[name].number_of_edges():,} edges")
                except Exception as e:
                    self.logger.error(f"  ✗ Failed to load {name}: {e}")
            else:
                self.logger.warning(f"  ⊘ {name} not found at {path}")
        
        return graphs
    
    def _save_results(self):
        """Save results to CSV."""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        output_path = self.results_path / "subsetting_experiments.csv"
        df.to_csv(output_path, index=False)
        self.logger.info(f"💾 Results saved to {output_path}")
    
    def _generate_summary_report(self):
        """Generate summary statistics and reports."""
        if not self.results:
            return
            
        df = pd.DataFrame(self.results)
        
        # Filter successful experiments only
        df_success = df[df['status'] == 'success'].copy()
        
        if len(df_success) == 0:
            self.logger.warning("No successful experiments to summarize")
            return
        
        # Summary by graph and article type
        summary_cols = [
            'pmids_initial', 'n_nodes_initial', 'n_edges_initial',
            'avg_degree_initial', 'density_initial'
        ]
        
        summary = df_success.groupby(['graph_name', 'search_strategy', 'article_type', 'mode'])[summary_cols].agg(['mean', 'std'])
        
        summary_path = self.results_path / "subsetting_summary.csv"
        summary.to_csv(summary_path)
        self.logger.info(f"📊 Summary saved to {summary_path}")
        
        # Generate comparison report
        self._generate_comparison_report(df_success)
        
        # Report on cache sizes
        self._report_cache_sizes()
    
    def _report_cache_sizes(self):
        """Report sizes of all cache databases."""
        self.logger.info("\n" + "="*80)
        self.logger.info("CACHE DATABASE SIZES")
        self.logger.info("="*80)
        
        total_size = 0
        cache_files = list(self.cache_dir.glob("*.db"))
        
        if not cache_files:
            self.logger.info("No cache files found")
            return
        
        for cache_file in sorted(cache_files):
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            total_size += size_mb
            self.logger.info(f"  {cache_file.name}: {size_mb:.2f} MB")
        
        self.logger.info(f"\nTotal cache size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    def _generate_comparison_report(self, df):
        """Generate text report comparing article types."""
        report_path = self.results_path / "subsetting_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SUBSETTING EXPERIMENTS COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            for graph in df['graph_name'].unique():
                f.write(f"\n{'='*80}\n")
                f.write(f"GRAPH: {graph}\n")
                f.write(f"{'='*80}\n\n")
                
                for search in df['search_strategy'].unique():
                    for mode in df['mode'].unique():
                        subset = df[
                            (df['graph_name'] == graph) &
                            (df['search_strategy'] == search) &
                            (df['mode'] == mode)
                        ]
                        
                        if len(subset) == 0:
                            continue
                        
                        f.write(f"\nSearch: {search}, Mode: {mode}\n")
                        f.write("-"*80 + "\n")
                        
                        for article_type in ['no_reviews', 'all', 'reviews_only']:
                            row = subset[subset['article_type'] == article_type]
                            if len(row) == 0:
                                continue
                            row = row.iloc[0]
                            
                            f.write(f"\n  {article_type}:\n")
                            f.write(f"    PMIDs: {row['pmids_initial']}\n")
                            f.write(f"    Nodes: {row['n_nodes_initial']}\n")
                            f.write(f"    Edges: {row['n_edges_initial']}\n")
                            f.write(f"    Avg Degree: {row['avg_degree_initial']:.2f}\n")
                            f.write(f"    Density: {row['density_initial']:.6f}\n")
                            if 'cache_db' in row:
                                f.write(f"    Cache: {row['cache_db']}\n")
                        
                        f.write("\n")
        
        self.logger.info(f"📄 Report saved to {report_path}")


# =============================================================================
# Experiment Configuration
# =============================================================================

def generate_experiments() -> List[SearchConfig]:
    """Generate all experiment configurations."""
    
    # Define search strategies
    search_strategies = [
        {
            "keywords": ["inflammation"],
            "cooccur_terms": [
                "adipose tissue", "adipocyte", "fat tissue",
                "white adipose tissue", "brown adipose tissue",
                "subcutaneous fat", "visceral fat",
                "obesity", "adipogenesis"
            ],
            "disease_terms": None,
            "name": "baseline"
        },
        {
            "keywords": ["inflammation"],
            "cooccur_terms": [
                "adipose tissue", "adipocyte", "fat tissue",
                "white adipose tissue", "brown adipose tissue",
                "subcutaneous fat", "visceral fat",
                "adipogenesis"
            ],
            "disease_terms": ["obesity"],
            "name": "obesity"
        },
        {
            "keywords": ["inflammation"],
            "cooccur_terms": [
                "adipose tissue", "adipocyte", "fat tissue",
                "white adipose tissue", "brown adipose tissue",
                "subcutaneous fat", "visceral fat",
                "obesity", "adipogenesis"
            ],
            "disease_terms": ["obesity", "diabetes", "metabolic syndrome"],
            "name": "multidisease"
        },
    ]
    
    # Article types
    article_types = ["no_reviews", "all", "reviews_only"]
    
    # Modes: (strict, k_hop, name)
    modes = [
        (True, 0, "strict"),
        (False, 0, "permissive"),
        (True, 1, "1hop"),
    ]
    
    # Generate all combinations
    experiments = []
    for search in search_strategies:
        for article_type in article_types:
            for strict, k_hop, mode_name in modes:
                experiments.append(
                    SearchConfig(
                        name=f"{search['name']}_{article_type}_{mode_name}",
                        keywords=search["keywords"],
                        cooccur_terms=search["cooccur_terms"],
                        disease_terms=search["disease_terms"],
                        journal_filter=article_type,
                        strict_mode=strict,
                        k_hop=k_hop
                    )
                )
    
    return experiments


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all subsetting experiments."""
    
    print("\n" + "="*80)
    print("SYSTEMATIC SUBSETTING EXPERIMENTS")
    print("="*80 + "\n")
    
    # Paths
    base_path = Path("/home/projects2/ContextAwareKGReasoning")
    results_path = base_path / "results/search_subset"
    cache_dir = base_path / "data/pubmed_cache"
    
    # Initialize experiment runner
    runner = SubsettingExperiment(base_path, results_path, cache_dir)
    
    # Load graphs
    print("Loading graphs...")
    all_graphs = runner._load_all_graphs()
    
    # Check if we should run only specific graph (from environment variable)
    graph_filter = os.environ.get('GRAPH_TO_RUN', None)
    
    if graph_filter:
        print(f"\nRunning experiments for: {graph_filter} only")
        if graph_filter in all_graphs:
            graphs = {graph_filter: all_graphs[graph_filter]}
        else:
            print(f"ERROR: Graph '{graph_filter}' not found!")
            return
    else:
        graphs = all_graphs
    
    if not graphs:
        print("ERROR: No graphs loaded!")
        return
    
    print(f"\nLoaded {len(graphs)} graph(s):")
    for name, graph in graphs.items():
        print(f"  {name}: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
    
    # Generate experiments
    print("\nGenerating experiment configurations...")
    experiments = generate_experiments()
    print(f"Generated {len(experiments)} experiments per graph")
    
    # Show cache strategy
    print(f"\nCache strategy:")
    print(f"  Separate cache DB per (graph × search_strategy × article_type)")
    print(f"  Total cache files: ~{len(graphs) * 3 * 3} databases")
    
    # Confirm before running (skip in batch mode)
    total = len(graphs) * len(experiments)
    print(f"\nTotal experiments to run: {total}")
    
    if not graph_filter:  # Only ask for confirmation if running interactively
        response = input("\nProceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted.")
            return
    else:
        print("Running in batch mode (auto-confirmed)")
    
    # Run all experiments
    print("\nStarting experiments...\n")
    runner.run_all_experiments(graphs, experiments)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  CSV: {results_path}/subsetting_experiments.csv")
    print(f"  Summary: {results_path}/subsetting_summary.csv")
    print(f"  Report: {results_path}/subsetting_report.txt")
    print(f"\nCaches saved to: {cache_dir}")
    print(f"  Total cache files: {len(list(cache_dir.glob('*.db')))}")


if __name__ == "__main__":
    main()