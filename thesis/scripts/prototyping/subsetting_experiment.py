#!/usr/bin/env python3
"""
Enhanced systematic evaluation of knowledge graph subsetting strategies.

Saves:
- Subgraphs as pickle files
- Comprehensive stats using print_kg_stats()
- Results CSV for each graph

Only runs multi_disease strategy per discussion.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph import KnowledgeGraph, print_kg_stats
from pubmed.pubmed_cache import PubMedBatchCache
from Bio import Entrez

# Import the core search/subsetting functions from search_utils
from search_utils import (
    search_pubmed_cooccurrence,
    subset_graph_by_pmids,
    add_all_edges_between_nodes,
    collect_pmids_from_subgraph,
    expand_subgraph_k_hops,
)

# Configure Entrez
Entrez.email = "s233139@dtu.dk"


class PubMedSearcher:
    """Wrapper around Bio.Entrez for PubMed searching."""
    
    def __init__(self, email: str = "s233139@dtu.dk"):
        """Initialize with email."""
        Entrez.email = email
    
    def search(self, query: str, max_results: int = 10000) -> list:
        """Search PubMed and return list of PMIDs."""
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            return record.get('IdList', [])
        except Exception as e:
            logging.error(f"Search error: {e}")
            return []


def setup_logging(results_dir: Path) -> logging.Logger:
    """Setup logging to both file and console."""
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"subsetting_enhanced_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("subsetting_enhanced")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    
    return logger


def load_existing_results(results_file: Path) -> list:
    """Load existing results from CSV if it exists."""
    if not results_file.exists():
        return []
    
    results = []
    try:
        with open(results_file, "r") as f:
            lines = f.readlines()
            if len(lines) <= 1:  # Only header or empty
                return []
            
            header = lines[0].strip().split(",")
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    values = line.strip().split(",")
                    result = dict(zip(header, values))
                    results.append(result)
    except Exception as e:
        logging.error(f"Error loading existing results: {e}")
        return []
    
    return results


def load_graphs(base_dir: Path, logger: logging.Logger, graph_filter: str = None):
    """
    Load knowledge graphs.
    
    Args:
        base_dir: Base directory containing graph files
        logger: Logger instance
        graph_filter: If provided, only load this specific graph
    
    Returns:
        dict: {graph_name: KnowledgeGraph}
    """
    graph_paths = {
        "ikraph_full": base_dir / "graphs/ikraph.pkl",
        "ikraph_pubmed": base_dir / "graphs/subsets/ikraph_pubmed.pkl", 
        "ikraph_pubmed_human": base_dir / "graphs/subsets/ikraph_pubmed_human.pkl",
    }
    
    # Filter graphs if specified
    if graph_filter:
        if graph_filter not in graph_paths:
            raise ValueError(
                f"Unknown graph: {graph_filter}. "
                f"Valid options: {list(graph_paths.keys())}"
            )
        graph_paths = {graph_filter: graph_paths[graph_filter]}
        logger.info(f"Running experiments for: {graph_filter} only")
    
    graphs = {}
    for name, path in graph_paths.items():
        logger.info(f"Loading {name} from {path}...")
        kg = KnowledgeGraph.import_graph(str(path))
        logger.info(f"  ✓ Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
        graphs[name] = kg
    
    return graphs


def define_search_strategy():
    """Define the multi_disease search strategy (only one we're using)."""
    return {
        "disease_terms": ["inflammation"],
        "tissue_terms": [
            "adipose tissue", "adipocyte", "fat tissue",
            "white adipose tissue", "brown adipose tissue",
            "subcutaneous fat", "visceral fat", "adipogenesis",
            "browning", "beiging"
        ],
        "disease_filters": ["obesity", "diabetes", "metabolic syndrome"],
    }


def run_single_experiment(
    graph_name: str,
    kg: KnowledgeGraph,
    strategy_name: str,
    strategy: dict,
    article_type: str,
    mode: str,
    cache_dir: Path,
    graph_results_dir: Path,
    subgraph_dir: Path,
    stats_dir: Path,
    searcher: PubMedSearcher,
    logger: logging.Logger,
    start_year: int = 2018,
    end_year: int = 2023,
):
    """Run a single experiment configuration and save subgraph + stats."""
    logger.info("=" * 80)
    logger.info(f"EXPERIMENT: {graph_name} | {strategy_name} | {article_type} | {mode}")
    logger.info("=" * 80)
    
    # Create cache for this configuration
    cache_file = cache_dir / f"{graph_name}_{strategy_name}_{article_type}.db"
    cache = PubMedBatchCache(db_path=str(cache_file), email="s233139@dtu.dk")
    
    # Define output filenames
    config_name = f"{strategy_name}_{article_type}_{mode}"
    subgraph_file = subgraph_dir / f"{config_name}.pkl"
    stats_file = stats_dir / f"{config_name}_stats.txt"
    
    # Check if already done
    if subgraph_file.exists() and stats_file.exists():
        logger.info(f"Subgraph and stats already exist for {config_name}, skipping...")
        
        # Load to get stats
        try:
            subset_kg = KnowledgeGraph.import_graph(str(subgraph_file))
            n_nodes = subset_kg.number_of_nodes()
            n_edges = subset_kg.number_of_edges()
            
            if n_nodes > 1:
                density = n_edges / (n_nodes * (n_nodes - 1) / 2)
                avg_degree = 2 * n_edges / n_nodes
            else:
                density = 0
                avg_degree = 0
            
            return {
                "graph_name": graph_name,
                "strategy": strategy_name,
                "article_type": article_type,
                "mode": mode,
                "status": "loaded_existing",
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "density": density,
                "avg_degree": avg_degree,
            }
        except Exception as e:
            logger.warning(f"Could not load existing subgraph: {e}. Rerunning...")
    
    try:
        # Build query
        query_parts = []
        
        # Disease terms
        if strategy["disease_terms"]:
            disease_query = " OR ".join(f'"{term}"' for term in strategy["disease_terms"])
            query_parts.append(f"({disease_query})")
        
        # Tissue terms
        if strategy["tissue_terms"]:
            tissue_query = " OR ".join(f'"{term}"' for term in strategy["tissue_terms"])
            query_parts.append(f"({tissue_query})")
        
        # Disease filters
        if strategy["disease_filters"]:
            disease_filter = " OR ".join(f'"{disease}"' for disease in strategy["disease_filters"])
            query_parts.append(f"({disease_filter})")
        
        # Combine with AND
        query = " AND ".join(query_parts)
        
        # Add date filter
        query += f" AND ({start_year}:{end_year}[edat])"
        
        # Article type filters
        if article_type == "no_reviews":
            query += ' NOT ("Review"[Publication Type])'
        elif article_type == "reviews_only":
            query += ' AND ("Review"[Publication Type])'
        
        logger.info(f"Query: {query}")
        
        # Search PubMed
        logger.info("Searching PubMed...")
        pmids = searcher.search(query, max_results=100000)
        logger.info(f"Found {len(pmids)} PMIDs")
        
        if len(pmids) == 0:
            logger.warning("No PMIDs found - skipping experiment")
            return {
                "graph_name": graph_name,
                "strategy": strategy_name,
                "article_type": article_type,
                "mode": mode,
                "status": "no_pmids",
                "pmids_found": 0,
            }
        
        # Fetch abstracts using PubMedBatchCache
        logger.info("Fetching abstracts with PubMedBatchCache...")
        cache.fetch_batch(pmids, batch_size=200, rate_limiting=0.4)
        abstracts = cache.get_abstracts(pmids)
        logger.info(f"Retrieved {len(abstracts)} abstracts from cache")
        
        # Convert PMIDs to set for faster lookup
        pmids_set = set(pmids)
        
        # Extract subset based on mode
        logger.info(f"Extracting subset (mode: {mode})...")
        
        if mode == "strict":
            # Only edges mentioned in these PMIDs
            subset_edges = []
            for u, v, key, data in kg.edges(keys=True, data=True):
                doc_id = data.get("document_id", "")
                if doc_id:
                    pmid = doc_id.split(".")[0]
                    if pmid in pmids_set:
                        subset_edges.append((u, v, key, data))
            
            logger.info(f"Found {len(subset_edges)} edges in strict mode")
            
        elif mode == "permissive":
            # Start with strict edges, then add all edges between entities mentioned
            mentioned_entities = set()
            
            for u, v, key, data in kg.edges(keys=True, data=True):
                doc_id = data.get("document_id", "")
                if doc_id:
                    pmid = doc_id.split(".")[0]
                    if pmid in pmids_set:
                        mentioned_entities.add(u)
                        mentioned_entities.add(v)
            
            logger.info(f"Strict: {len(mentioned_entities)} entities")
            
            # Add all edges between mentioned entities
            subset_edges = []
            for u, v, key, data in kg.edges(keys=True, data=True):
                if u in mentioned_entities and v in mentioned_entities:
                    subset_edges.append((u, v, key, data))
            
            logger.info(f"Permissive: {len(subset_edges)} total edges")
            
        elif mode == "1hop":
            # Permissive + 1-hop expansion
            mentioned_entities = set()
            for u, v, key, data in kg.edges(keys=True, data=True):
                doc_id = data.get("document_id", "")
                if doc_id:
                    pmid = doc_id.split(".")[0]
                    if pmid in pmids_set:
                        mentioned_entities.add(u)
                        mentioned_entities.add(v)
            
            # 1-hop neighbors
            expanded_entities = set(mentioned_entities)
            for entity in mentioned_entities:
                if entity in kg:
                    neighbors = list(kg.neighbors(entity))
                    expanded_entities.update(neighbors)
            
            logger.info(f"Expanded from {len(mentioned_entities)} to {len(expanded_entities)} entities")
            
            # Extract edges
            subset_edges = []
            for u, v, key, data in kg.edges(keys=True, data=True):
                if u in expanded_entities and v in expanded_entities:
                    subset_edges.append((u, v, key, data))
            
            logger.info(f"1-hop: {len(subset_edges)} total edges")
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Create subgraph
        logger.info("Creating subgraph...")
        subset_kg = KnowledgeGraph(schema=kg.schema)
        
        # Collect unique nodes
        nodes_in_subset = set()
        for u, v, key, _ in subset_edges:
            nodes_in_subset.add(u)
            nodes_in_subset.add(v)
        
        # Add nodes
        for node in nodes_in_subset:
            if node in kg:
                subset_kg.add_node(node, **kg.nodes[node])
        
        # Add edges (with keys for MultiDiGraph)
        for u, v, key, data in subset_edges:
            # For MultiDiGraph, we can use add_edge which will auto-generate keys
            # Or we could preserve the original key, but add_edge is safer
            subset_kg.add_edge(u, v, **data)
        
        n_nodes = subset_kg.number_of_nodes()
        n_edges = subset_kg.number_of_edges()
        
        logger.info(f"Subset: {n_nodes} nodes, {n_edges} edges")
        
        # Calculate metrics
        if n_nodes > 1:
            density = n_edges / (n_nodes * (n_nodes - 1) / 2)
            avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        else:
            density = 0
            avg_degree = 0
        
        logger.info(f"Density: {density:.6f}, Avg degree: {avg_degree:.2f}")
        
        # Save subgraph (for detailed analysis later)
        logger.info(f"Saving subgraph to {subgraph_file}...")
        subset_kg.export_graph(str(subgraph_file))
        
        # Save lightweight node/edge lists for overlap analysis (much more memory efficient!)
        logger.info(f"Saving lightweight node/edge lists for overlap analysis...")
        node_list_dir = graph_results_dir / "node_lists"
        edge_list_dir = graph_results_dir / "edge_lists"
        node_list_dir.mkdir(parents=True, exist_ok=True)
        edge_list_dir.mkdir(parents=True, exist_ok=True)
        
        node_list_file = node_list_dir / f"{config_name}_nodes.json"
        edge_list_file = edge_list_dir / f"{config_name}_edges.json"
        
        # Export nodes (just identifiers)
        node_ids = []
        for node in nodes_in_subset:
            # Store a simple representation of the node
            if hasattr(node, 'name'):
                node_ids.append({'name': node.name, 'type': node.type})
            else:
                node_ids.append(str(node))
        
        with open(node_list_file, 'w') as f:
            json.dump(node_ids, f)
        
        # Export edges (source-target-key tuples for MultiDiGraph)
        edge_list = []
        for u, v, key, data in subset_kg.edges(keys=True, data=True):
            # Store node identifiers
            if hasattr(u, 'name') and hasattr(v, 'name'):
                u_id = (u.name, u.type)
                v_id = (v.name, v.type)
            else:
                u_id = str(u)
                v_id = str(v)
            
            # For MultiDiGraph, we need to track each edge separately
            # Store as [u, v, key] to distinguish parallel edges
            # Canonicalize for undirected comparison: sort by string representation
            if str(u_id) <= str(v_id):
                edge_list.append([u_id, v_id, key])
            else:
                edge_list.append([v_id, u_id, key])
        
        # Convert tuples to lists for JSON serialization
        edge_list_json = []
        for e in edge_list:
            edge_list_json.append([
                list(e[0]) if isinstance(e[0], tuple) else e[0],
                list(e[1]) if isinstance(e[1], tuple) else e[1],
                e[2]  # key
            ])
        
        with open(edge_list_file, 'w') as f:
            json.dump(edge_list_json, f)
        
        logger.info(f"  Saved node list: {len(node_ids):,} nodes")
        logger.info(f"  Saved edge list: {len(edge_list_json):,} edges (including parallel edges)")
        
        # Save comprehensive stats using print_kg_stats
        logger.info(f"Saving stats to {stats_file}...")
        with open(stats_file, "w") as f:
            with redirect_stdout(f):
                print(f"Configuration: {config_name}")
                print(f"Graph: {graph_name}")
                print(f"Strategy: {strategy_name}")
                print(f"Article Type: {article_type}")
                print(f"Mode: {mode}")
                print(f"PMIDs Found: {len(pmids)}")
                print(f"Query: {query}")
                print("\n" + "="*80)
                print("KNOWLEDGE GRAPH STATISTICS")
                print("="*80 + "\n")
                print_kg_stats(subset_kg)
        
        result = {
            "graph_name": graph_name,
            "strategy": strategy_name,
            "article_type": article_type,
            "mode": mode,
            "status": "success",
            "pmids_found": len(pmids),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "avg_degree": avg_degree,
        }
        
        logger.info("✓ Experiment complete")
        return result
        
    except Exception as e:
        logger.error(f"✗ Experiment failed: {e}", exc_info=True)
        return {
            "graph_name": graph_name,
            "strategy": strategy_name,
            "article_type": article_type,
            "mode": mode,
            "status": "error",
            "error": str(e),
        }
    finally:
        # Always close the cache connection
        cache.close()


def main():
    """Run all experiments."""
    # Check for graph filter from environment
    graph_filter = os.environ.get("GRAPH_TO_RUN", None)
    
    if graph_filter:
        print(f"Running experiments for: {graph_filter}")
    else:
        print("Running experiments for all graphs")
    
    # Setup paths
    base_dir = Path("/home/projects2/ContextAwareKGReasoning/data")
    results_dir = Path("/home/projects2/ContextAwareKGReasoning/results/search_subset_2")
    cache_dir = Path("/home/projects2/ContextAwareKGReasoning/data/pubmed_cache_2")
    
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(results_dir)
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"Caches will be saved to: {cache_dir}")
    
    # Define time period to match iKGraph
    START_YEAR = 2000
    END_YEAR = 2023
    logger.info(f"Time period filter: {START_YEAR}-{END_YEAR}")
    
    # Load graphs
    try:
        graphs = load_graphs(base_dir, logger, graph_filter)
    except Exception as e:
        logger.error(f"Failed to load graphs: {e}", exc_info=True)
        sys.exit(1)
    
    # Initialize searcher
    searcher = PubMedSearcher()
    
    # Define experiment configurations - ONLY multi_disease
    strategy = define_search_strategy()
    strategy_name = "multi_disease"
    article_types = ["no_reviews", "all_articles", "reviews_only"]
    modes = ["strict", "permissive", "1hop"]
    
    # Run experiments FOR EACH GRAPH
    for graph_name, kg in graphs.items():
        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING EXPERIMENTS FOR GRAPH: {graph_name}")
        logger.info("=" * 80)
        
        # Create output directories for this graph
        graph_results_dir = results_dir / graph_name
        subgraph_dir = graph_results_dir / "subgraphs"
        stats_dir = graph_results_dir / "stats"
        
        subgraph_dir.mkdir(parents=True, exist_ok=True)
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        # Each graph gets its own CSV file
        results_file = graph_results_dir / f"subsetting_results_{graph_name}.csv"
        
        # Load existing results for this graph
        results = load_existing_results(results_file)
        
        if results:
            logger.info(f"Loaded {len(results)} existing results from {results_file}")
        
        # Figure out which experiments are already done for this graph
        completed_experiments = set()
        for r in results:
            key = (r.get("article_type"), r.get("mode"))
            completed_experiments.add(key)
        
        logger.info(f"Already completed: {len(completed_experiments)} experiments for {graph_name}")
        
        experiments_per_graph = len(article_types) * len(modes)
        experiment_num = len(results)  # Start counting from existing results
        
        for article_type in article_types:
            for mode in modes:
                # Skip if already completed
                exp_key = (article_type, mode)
                if exp_key in completed_experiments:
                    logger.info(f"Skipping already completed: {exp_key}")
                    continue
                
                experiment_num += 1
                logger.info(f"\n{graph_name}: Experiment {experiment_num}/{experiments_per_graph}")
                
                result = run_single_experiment(
                    graph_name=graph_name,
                    kg=kg,
                    strategy_name=strategy_name,
                    strategy=strategy,
                    article_type=article_type,
                    mode=mode,
                    cache_dir=cache_dir,
                    graph_results_dir=graph_results_dir,
                    subgraph_dir=subgraph_dir,
                    stats_dir=stats_dir,
                    searcher=searcher,
                    logger=logger,
                    start_year=START_YEAR,
                    end_year=END_YEAR,
                )
                
                results.append(result)
                
                # Save results for THIS GRAPH only
                with open(results_file, "w") as f:
                    if results:
                        keys = results[0].keys()
                        f.write(",".join(keys) + "\n")
                        
                        for r in results:
                            values = [str(r.get(k, "")) for k in keys]
                            f.write(",".join(values) + "\n")
        
        logger.info(f"\n✓ Completed all experiments for {graph_name}")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Subgraphs saved to: {subgraph_dir}")
        logger.info(f"Stats saved to: {stats_dir}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ALL EXPERIMENTS COMPLETE FOR ALL GRAPHS")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()