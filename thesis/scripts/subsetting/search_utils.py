"""
Core functions for subsetting iKGraph by PubMed abstracts.

This module contains the main logic for:
- Searching PubMed
- Subsetting graphs by PMIDs
- Adding edges between nodes
- K-hop expansion
- PMID collection

These functions are used by run_subset_ikraph.py (config-based)
and subset_ikraph_by_pubmed.py (argparse-based).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from collections import Counter
from typing import Set, List
from urllib.error import HTTPError

from Bio import Entrez
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph


# Journal type filter
JOURNAL_TYPE_FILTER = (
    'AND "journal article"[pt] NOT (preprint[pt] OR review[pt] OR editorial[pt] '
    'OR "clinical trial protocol"[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] '
    'OR "letter"[pt] OR "comment"[pt])'
)


def search_pubmed_cooccurrence(
    key_terms: List[str],
    adipose_terms: List[str],
    max_results: int = 10000,
    start_year: int = 2000,
    end_year: int = 2025,
    apply_journal_filter: bool = True,
) -> Set[str]:
    """
    Search PubMed for co-occurrence of key terms with adipose tissue terms.
    
    Parameters
    ----------
    key_terms : List[str]
        Key biological terms (e.g., inflammation markers, cytokines)
    adipose_terms : List[str]
        Adipose tissue related terms (e.g., adipose, adipocyte, fat)
    max_results : int
        Maximum number of results per query
    start_year : int
        Start year for publication date filter
    end_year : int
        End year for publication date filter
    apply_journal_filter : bool
        Whether to apply journal type filter
        
    Returns
    -------
    Set[str]
        Set of unique PMIDs
    """
    all_pmids = set()
    query_results = {}
    
    # Create adipose query component
    adipose_query = " OR ".join([f'"{term}"' for term in adipose_terms])
    
    logging.info(f"Searching PubMed with {len(key_terms)} key terms...")
    logging.info(f"Adipose terms: {', '.join(adipose_terms)}")
    
    for key_term in tqdm(key_terms, desc="Searching PubMed"):
        # Build query: (key_term) AND (adipose terms) AND date range
        query = f'("{key_term}") AND ({adipose_query}) AND ({start_year}:{end_year}[edat])'
        
        if apply_journal_filter:
            query = f"{query} {JOURNAL_TYPE_FILTER}"
        
        try:
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            
            record_count = int(record["Count"])
            id_list = record["IdList"]
            
            query_results[key_term] = len(id_list)
            
            logging.info(
                f"  '{key_term}': {record_count} total, {len(id_list)} retrieved"
            )
            
            all_pmids.update(id_list)
            
        except Exception as e:
            logging.error(f"Error searching for '{key_term}': {e}")
            continue
    
    # Log summary
    logging.info(f"\nQuery results summary:")
    for term, count in sorted(query_results.items(), key=lambda x: -x[1])[:10]:
        logging.info(f"  {term}: {count} PMIDs")
    
    return all_pmids


def subset_graph_by_pmids(
    kg: KnowledgeGraph,
    pmids: Set[str],
    k_hop: int = 0,
) -> KnowledgeGraph:
    """
    Subset knowledge graph based on PubMed IDs.
    
    Parameters
    ----------
    kg : KnowledgeGraph
        Input knowledge graph
    pmids : Set[str]
        Set of PMIDs to filter by
    k_hop : int
        Number of hops to expand the subgraph (0 = no expansion)
        
    Returns
    -------
    KnowledgeGraph
        Subsetted knowledge graph
    """
    logging.info(f"\nSubsetting graph with {len(pmids)} PMIDs...")
    
    # Create new graph with same schema
    subgraph = KnowledgeGraph(schema=kg.schema)
    
    # Find edges with matching PMIDs
    matching_edges = []
    matching_nodes = set()
    pmid_counts = Counter()
    
    for u, v, key, data in tqdm(
        kg.edges(keys=True, data=True),
        desc="Finding matching edges",
        total=kg.number_of_edges()
    ):
        if data.get("source") == "PubMed":
            doc_id = data.get("document_id", "")
            if doc_id:
                # Extract PMID (first part before '.')
                pmid = doc_id.split(".")[0]
                if pmid in pmids:
                    matching_edges.append((u, v, key, data))
                    matching_nodes.add(u)
                    matching_nodes.add(v)
                    pmid_counts[pmid] += 1
    
    logging.info(f"Found {len(matching_edges)} matching edges from {len(pmid_counts)} PMIDs")
    logging.info(f"Found {len(matching_nodes)} nodes in initial subset")
    logging.info(f"Top 10 PMIDs by edge count: {pmid_counts.most_common(10)}")
    
    # Add nodes
    for node in tqdm(matching_nodes, desc="Adding nodes"):
        if node in kg.nodes:
            subgraph.add_node(node, **kg.nodes[node].copy())
    
    # Add edges
    for u, v, key, data in tqdm(matching_edges, desc="Adding edges"):
        subgraph.add_edge(u, v, **data.copy())
    
    return subgraph


def collect_pmids_from_subgraph(subgraph: KnowledgeGraph) -> Set[str]:
    """
    Collect all PMIDs from PubMed edges in the subgraph.
    
    Parameters
    ----------
    subgraph : KnowledgeGraph
        The subgraph to collect PMIDs from
        
    Returns
    -------
    Set[str]
        Set of all PMIDs found in the subgraph
    """
    pmids = set()
    for u, v, data in subgraph.edges(data=True):
        if data.get("source") == "PubMed":
            doc_id = data.get("document_id", "")
            if doc_id:
                pmid = doc_id.split(".")[0]
                pmids.add(pmid)
    return pmids


def add_all_edges_between_nodes(
    original_kg: KnowledgeGraph,
    subgraph: KnowledgeGraph
) -> KnowledgeGraph:
    """
    Add ALL edges between nodes in the subgraph from the original graph.
    
    This captures edges that weren't found in the initial PMID search
    but connect the relevant nodes.
    
    Parameters
    ----------
    original_kg : KnowledgeGraph
        Original full graph
    subgraph : KnowledgeGraph
        Current subgraph
        
    Returns
    -------
    KnowledgeGraph
        Subgraph with all edges between its nodes
    """
    logging.info("\nAdding all edges between subgraph nodes...")
    
    nodes_in_subgraph = set(subgraph.nodes())
    edges_added = 0
    
    for u in tqdm(nodes_in_subgraph, desc="Finding all edges"):
        if u not in original_kg.nodes:
            continue
            
        # Get all neighbors of u in original graph
        try:
            for v in original_kg.neighbors(u):
                if v in nodes_in_subgraph:
                    # Add all edges between u and v
                    if original_kg.has_edge(u, v):
                        for key, data in original_kg[u][v].items():
                            if not subgraph.has_edge(u, v, key):
                                subgraph.add_edge(u, v, **data.copy())
                                edges_added += 1
        except:
            continue
    
    logging.info(f"Added {edges_added} additional edges between nodes")
    return subgraph


def expand_subgraph_k_hops(
    original_kg: KnowledgeGraph,
    subgraph: KnowledgeGraph,
    k: int
) -> KnowledgeGraph:
    """
    Expand subgraph by k hops in the original graph.
    
    Parameters
    ----------
    original_kg : KnowledgeGraph
        Original full graph
    subgraph : KnowledgeGraph
        Current subgraph to expand
    k : int
        Number of hops to expand
        
    Returns
    -------
    KnowledgeGraph
        Expanded subgraph
    """
    current_nodes = set(subgraph.nodes())
    
    for hop in range(k):
        logging.info(f"Hop {hop + 1}/{k}...")
        new_nodes = set()
        
        # Find neighbors of current nodes in original graph
        for node in tqdm(current_nodes, desc=f"Finding neighbors (hop {hop + 1})"):
            if node not in original_kg.nodes:
                continue
                
            # Get neighbors in original graph
            try:
                neighbors = set(original_kg.neighbors(node))
                new_nodes.update(neighbors - current_nodes)
            except:
                continue
        
        logging.info(f"Found {len(new_nodes)} new nodes in hop {hop + 1}")
        
        if not new_nodes:
            logging.info("No new nodes found, stopping expansion")
            break
        
        # Add new nodes
        for node in new_nodes:
            if node not in subgraph.nodes:
                subgraph.add_node(node, **original_kg.nodes[node].copy())
        
        # Add edges to/from new nodes
        edges_added = 0
        for node in tqdm(new_nodes, desc=f"Adding edges (hop {hop + 1})"):
            # Get all edges involving this node in original graph
            try:
                for neighbor in original_kg.neighbors(node):
                    if neighbor in subgraph.nodes:
                        # Add all edges between node and neighbor (MultiDiGraph)
                        if original_kg.has_edge(node, neighbor):
                            for key, data in original_kg[node][neighbor].items():
                                if not subgraph.has_edge(node, neighbor, key):
                                    subgraph.add_edge(node, neighbor, **data.copy())
                                    edges_added += 1
            except:
                continue
        
        logging.info(f"Added {edges_added} new edges in hop {hop + 1}")
        
        # Update current nodes for next iteration
        current_nodes = set(subgraph.nodes())
    
    return subgraph