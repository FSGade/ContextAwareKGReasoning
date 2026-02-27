#!/usr/bin/env python3
"""
Run script for subsetting iKGraph by PubMed abstracts.

USAGE:
1. Edit the CONFIG section below with your parameters
2. Run: python3 run_subset_ikraph.py
3. Results will be saved to the paths you specify

This approach is better for reproducibility - all parameters are visible
and version-controlled in this file.
"""

import sys
import logging
from pathlib import Path

# Add project path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge_graph import KnowledgeGraph, print_kg_stats
from pubmed.pubmed_cache import PubMedBatchCache
from Bio import Entrez

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # PATHS (Required)
    # -------------------------------------------------------------------------
    "kg_path": "/home/projects2/ContextAwareKGReasoning/data/graphs/ikraph.pkl",           # Input iKGraph
    "output_path": "/home/projects2/ContextAwareKGReasoning/data/graphs/subsets/ikraph_pmid_subset.pkl",  # Output subset graph
    "cache_db_path": "/home/projects2/ContextAwareKGReasoning/data/pubmed_cache/pubmed_cache_subset.db",  # PubMed cache DB

    
    # -------------------------------------------------------------------------
    # MODE: Choose ONE of these
    # -------------------------------------------------------------------------
    # Option 1: Search PubMed
    "mode": "search",  # "search", "file", or "default"
    
    # Option 2: Use existing PMID file
    # "mode": "file",
    # "pmids_file": "my_pmids.txt",
    
    # Option 3: Default (built-in terms)
    # "mode": "default",
    
    # -------------------------------------------------------------------------
    # SEARCH PARAMETERS (only used if mode="search")
    # -------------------------------------------------------------------------
    "search_keywords": [
        "inflammation",
    ],
    
    "cooccur_terms": [
        "adipose tissue",
        "adipocyte",
        "fat tissue",
        "white adipose tissue",
        "brown adipose tissue",
        "subcutaneous fat",
        "visceral fat",
        "obesity",
        "adipogenesis",
    ],
    
    # -------------------------------------------------------------------------
    # SUBSETTING PARAMETERS
    # -------------------------------------------------------------------------
    "strict_mode": True,              # True = only search PMIDs, False = all edges
    "k_hop": 0,                        # Number of expansion hops (0 = no expansion)
    
    # -------------------------------------------------------------------------
    # PUBMED PARAMETERS
    # -------------------------------------------------------------------------
    "email": "s233139@dtu.dk",  # Required by NCBI
    "max_results_per_query": 10000,    # Max PMIDs per search term
    "start_year": 2000,                 # Publication start year
    "end_year": 2023,                   # Publication end year
    "apply_journal_filter": True,       # False = include reviews, editorials, etc.
    
    # -------------------------------------------------------------------------
    # CACHE PARAMETERS
    # -------------------------------------------------------------------------
    "fetch_batch_size": 200,            # PMIDs per batch
    "rate_limiting": 0.4,               # Seconds between batches
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


# Import the core subsetting functions
from search_utils import (
    search_pubmed_cooccurrence,
    subset_graph_by_pmids,
    add_all_edges_between_nodes,
    collect_pmids_from_subgraph,
    expand_subgraph_k_hops,
)


def main():
    """Run the subsetting workflow with the CONFIG above."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Update Entrez
    Entrez.email = CONFIG["email"]
    
    logging.info("=" * 80)
    logging.info("iKGRAPH SUBSETTING BY PUBMED")
    logging.info("=" * 80)
    logging.info(f"\nConfiguration loaded from: {__file__}")
    
    # =========================================================================
    # STEP 1: Get initial PMIDs
    # =========================================================================
    
    initial_pmids = set()
    search_metadata = {}
    
    if CONFIG["mode"] == "search":
        logging.info("\nMode: SEARCH PubMed")
        logging.info(f"  Keywords: {len(CONFIG['search_keywords'])} terms")
        logging.info(f"  Co-occur: {len(CONFIG['cooccur_terms'])} terms")
        logging.info(f"  Years: {CONFIG['start_year']}-{CONFIG['end_year']}")
        
        initial_pmids = search_pubmed_cooccurrence(
            key_terms=CONFIG["search_keywords"],
            adipose_terms=CONFIG["cooccur_terms"],
            max_results=CONFIG["max_results_per_query"],
            start_year=CONFIG["start_year"],
            end_year=CONFIG["end_year"],
            apply_journal_filter=CONFIG["apply_journal_filter"],
        )
        
        search_metadata = {
            "source": "pubmed_search",
            "key_terms": CONFIG["search_keywords"],
            "cooccur_terms": CONFIG["cooccur_terms"],
            "start_year": CONFIG["start_year"],
            "end_year": CONFIG["end_year"],
        }
        
    elif CONFIG["mode"] == "file":
        logging.info(f"\nMode: Load PMIDs from file: {CONFIG['pmids_file']}")
        
        with open(CONFIG["pmids_file"], 'r') as f:
            content = f.read()
            if ',' in content:
                initial_pmids = set(p.strip() for p in content.split(',') if p.strip())
            else:
                initial_pmids = set(p.strip() for p in content.split('\n') if p.strip())
        
        search_metadata = {
            "source": "file",
            "pmids_file": CONFIG["pmids_file"],
        }
        
    elif CONFIG["mode"] == "default":
        logging.info("\nMode: DEFAULT (built-in terms)")
        
        key_terms = [
            "inflammation", "IL-6", "TNF-alpha", "IL-1beta", "CRP",
            "adipokine", "cytokine", "leptin", "adiponectin", "resistin",
        ]
        cooccur_terms = ["adipose tissue", "adipocyte", "obesity"]
        
        initial_pmids = search_pubmed_cooccurrence(
            key_terms=key_terms,
            adipose_terms=cooccur_terms,
            max_results=CONFIG["max_results_per_query"],
            start_year=CONFIG["start_year"],
            end_year=CONFIG["end_year"],
            apply_journal_filter=CONFIG["apply_journal_filter"],
        )
        
        search_metadata = {
            "source": "default_search",
            "key_terms": key_terms,
            "cooccur_terms": cooccur_terms,
        }
    
    else:
        logging.error(f"Invalid mode: {CONFIG['mode']}")
        logging.error("Must be 'search', 'file', or 'default'")
        sys.exit(1)
    
    logging.info(f"\nInitial PMIDs: {len(initial_pmids)}")
    
    # =========================================================================
    # STEP 2: Cache initial PMIDs
    # =========================================================================
    
    logging.info("\nCaching initial PMIDs...")
    cache = PubMedBatchCache(db_path=CONFIG["cache_db_path"], email=CONFIG["email"])
    try:
        cache.fetch_batch(
            list(initial_pmids),
            batch_size=CONFIG["fetch_batch_size"],
            rate_limiting=CONFIG["rate_limiting"]
        )
        abstracts = cache.get_abstracts(list(initial_pmids))
        logging.info(f"Successfully cached {len(abstracts)} abstracts")
    finally:
        cache.close()
    
    # =========================================================================
    # STEP 3: Load iKGraph
    # =========================================================================
    
    logging.info(f"\nLoading iKGraph from: {CONFIG['kg_path']}")
    kg = KnowledgeGraph.import_graph(CONFIG["kg_path"])
    logging.info(f"Original graph: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    # =========================================================================
    # STEP 4: Initial subset
    # =========================================================================
    
    logging.info(f"\nSubsetting graph with {len(initial_pmids)} initial PMIDs...")
    subgraph = subset_graph_by_pmids(kg, initial_pmids, k_hop=0)
    logging.info(f"Initial subset: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")
    
    # =========================================================================
    # STEP 5: Add edges (strict vs permissive)
    # =========================================================================
    
    if not CONFIG["strict_mode"]:
        logging.info("\nMode: PERMISSIVE (adding all edges between nodes)")
        logging.info("This may discover PMIDs from different biological contexts")
        
        subgraph = add_all_edges_between_nodes(kg, subgraph)
        logging.info(f"After adding all edges: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")
        
        # Cache discovered PMIDs
        all_pmids_so_far = collect_pmids_from_subgraph(subgraph)
        new_pmids = all_pmids_so_far - initial_pmids
        
        if new_pmids:
            logging.info(f"Discovered {len(new_pmids)} new PMIDs from edges between nodes")
            logging.info("Caching newly discovered PMIDs...")
            cache = PubMedBatchCache(db_path=CONFIG["cache_db_path"], email=CONFIG["email"])
            try:
                cache.fetch_batch(
                    list(new_pmids),
                    batch_size=CONFIG["fetch_batch_size"],
                    rate_limiting=CONFIG["rate_limiting"]
                )
            finally:
                cache.close()
    else:
        logging.info("\nMode: STRICT (only edges with PMIDs from search/file)")
        logging.info("Skipping edges from other biological contexts")
        all_pmids_so_far = initial_pmids.copy()
    
    # =========================================================================
    # STEP 6: K-hop expansion
    # =========================================================================
    
    if CONFIG["k_hop"] > 0:
        logging.info(f"\nPerforming {CONFIG['k_hop']}-hop expansion...")
        subgraph = expand_subgraph_k_hops(kg, subgraph, CONFIG["k_hop"])
        logging.info(f"After {CONFIG['k_hop']}-hop expansion: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")
        
        # Cache PMIDs from expansion
        all_pmids_final = collect_pmids_from_subgraph(subgraph)
        new_pmids_from_expansion = all_pmids_final - all_pmids_so_far
        
        if new_pmids_from_expansion:
            logging.info(f"Discovered {len(new_pmids_from_expansion)} new PMIDs from k-hop expansion")
            logging.info("Caching PMIDs from expansion...")
            cache = PubMedBatchCache(db_path=CONFIG["cache_db_path"], email=CONFIG["email"])
            try:
                cache.fetch_batch(
                    list(new_pmids_from_expansion),
                    batch_size=CONFIG["fetch_batch_size"],
                    rate_limiting=CONFIG["rate_limiting"]
                )
            finally:
                cache.close()
    
    # =========================================================================
    # STEP 7: Finalize and save
    # =========================================================================
    
    final_pmids = collect_pmids_from_subgraph(subgraph)
    discovered_pmids = final_pmids - initial_pmids
    
    logging.info("\n" + "=" * 80)
    logging.info("PMID SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Mode: {'STRICT' if CONFIG['strict_mode'] else 'PERMISSIVE'}")
    logging.info(f"Initial PMIDs: {len(initial_pmids)}")
    logging.info(f"Discovered PMIDs: {len(discovered_pmids)}")
    logging.info(f"Total PMIDs: {len(final_pmids)}")
    
    # Store metadata
    subgraph.graph['metadata'] = {
        'cache_db_path': str(Path(CONFIG["cache_db_path"]).absolute()),
        'initial_pmids': len(initial_pmids),
        'discovered_pmids': len(discovered_pmids),
        'total_pmids': len(final_pmids),
        'k_hop': CONFIG["k_hop"],
        'strict_mode': CONFIG["strict_mode"],
        'config_file': __file__,
        **search_metadata
    }
    
    logging.info("\n" + "=" * 80)
    logging.info("SUBGRAPH STATISTICS")
    logging.info("=" * 80)
    print_kg_stats(subgraph)
    
    # Save
    output_path = Path(CONFIG["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"\nSaving subgraph to: {output_path}")
    subgraph.export_graph(output_path)
    
    logging.info("\n" + "=" * 80)
    logging.info("COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Subgraph saved: {output_path}")
    logging.info(f"Abstracts cached: {CONFIG['cache_db_path']}")
    logging.info(f"Original: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    logging.info(f"Subset: {subgraph.number_of_nodes():,} nodes, {subgraph.number_of_edges():,} edges")
    
    if kg.number_of_nodes() > 0:
        logging.info(f"Reduction: {100 * (1 - subgraph.number_of_nodes() / kg.number_of_nodes()):.1f}% nodes, "
                     f"{100 * (1 - subgraph.number_of_edges() / kg.number_of_edges()):.1f}% edges")


if __name__ == "__main__":
    main()