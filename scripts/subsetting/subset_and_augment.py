#!/usr/bin/env python3
"""
Subset and augment a knowledge graph based on PubMed search.

Creates an augmented KG subset:
1. Search PubMed → PMIDs (with abstract caching)
2. Extract strict subset (edges matching PMIDs)
3. Extract permissive subset (all edges between strict nodes)
4. Augment with non-association edges from non-hub nodes

Sampling strategies for augmentation:
- greedy: Take highest probability edges first
- random: Uniform random sampling from candidates
- weighted: Probability-weighted random sampling

NOTE: This version INCLUDES reviews (systematic reviews, meta-analyses).
The graph is a MultiDiGraph - edges have keys and we must iterate correctly.
"""

import argparse
import json
import logging
import random
import socket
import sys
import gc
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional

import numpy as np
from Bio import Entrez
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from knowledge_graph import KnowledgeGraph, print_kg_stats
from pubmed.pubmed_cache import PubMedBatchCache


socket.setdefaulttimeout(60)


# =============================================================================
# Sampling Strategy
# =============================================================================

class SamplingStrategy(Enum):
    GREEDY = "greedy"
    RANDOM = "random"
    WEIGHTED = "weighted"


# =============================================================================
# Article Type Filters - INCLUDING REVIEWS
# =============================================================================

JOURNAL_ARTICLE_WITH_REVIEWS = (
    'AND ("journal article"[pt] OR "review"[pt] OR "systematic review"[pt] OR "meta-analysis"[pt]) '
    'NOT (preprint[pt] OR editorial[pt] OR "clinical trial protocol"[pt] '
    'OR "letter"[pt] OR "comment"[pt])'
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SearchConfig:
    disease_terms: List[str] = field(default_factory=lambda: ["inflammation"])
    tissue_terms: List[str] = field(default_factory=lambda: [
        "adipose tissue", "adipocyte", "fat tissue",
        "white adipose tissue", "brown adipose tissue",
        "subcutaneous fat", "visceral fat", "adipogenesis",
        "browning", "beiging"
    ])
    disease_filters: List[str] = field(default_factory=lambda: [
        "obesity", "diabetes", "metabolic syndrome"
    ])
    start_year: int = 2000
    end_year: int = 2023
    max_results: int = 10000


@dataclass
class AugmentConfig:
    max_total_edges: int = 100000
    top_percentile_excluded: float = 0.10
    excluded_edge_types: List[str] = field(default_factory=lambda: ["Association"])
    sampling_strategy: SamplingStrategy = SamplingStrategy.WEIGHTED
    random_seed: Optional[int] = 42
    only_directed: bool = False  # iKraph: direction == "1"


# =============================================================================
# PubMed Search
# =============================================================================

def build_pubmed_query(config: SearchConfig) -> str:
    parts = []

    if config.disease_terms:
        parts.append("(" + " OR ".join(f'"{t}"' for t in config.disease_terms) + ")")
    if config.tissue_terms:
        parts.append("(" + " OR ".join(f'"{t}"' for t in config.tissue_terms) + ")")
    if config.disease_filters:
        parts.append("(" + " OR ".join(f'"{t}"' for t in config.disease_filters) + ")")

    query = " AND ".join(parts)
    query += f" AND ({config.start_year}:{config.end_year}[edat])"
    query += " " + JOURNAL_ARTICLE_WITH_REVIEWS
    return query


def search_pubmed(query: str, max_results: int, email: str) -> List[str]:
    Entrez.email = email

    logging.info(f"Searching PubMed with query:\n{query}")

    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    total_count = int(record["Count"])
    pmids = record["IdList"]
    
    logging.info(f"Found {total_count} total results, retrieved {len(pmids)} PMIDs")
    
    if total_count > max_results:
        logging.warning(
            f"Query returned {total_count} results but only {max_results} retrieved. "
            "Consider increasing max_results or refining query."
        )

    return record.get("IdList", [])


def cache_abstracts(pmids: List[str], cache_path: Path, email: str) -> PubMedBatchCache:
    
    logging.info(f"Caching abstracts to {cache_path}")

    cache = PubMedBatchCache(db_path=str(cache_path), email=email)
    cache.fetch_batch(pmids, batch_size=200, rate_limiting=0.4)

    # Report cache statistics
    cached_abstracts = cache.get_abstracts(pmids)
    logging.info(f"Successfully cached {len(cached_abstracts)} abstracts")

    return cache


# =============================================================================
# Graph Subsetting
# =============================================================================

def extract_strict_subset(kg: KnowledgeGraph, pmids: Set[str]) -> KnowledgeGraph:
    
    logging.info(f"Extracting strict subset with {len(pmids)} PMIDs...")
    
    subgraph = KnowledgeGraph(schema=kg.schema)
    nodes = set()
    edges = []
    pmid_counts = Counter()

    for u, v, k, d in tqdm(kg.edges(keys=True, data=True), total=kg.number_of_edges(), desc="Finding matching edges"):
        doc_id = d.get("document_id", "")
        if doc_id and doc_id.split(".")[0] in pmids:
            nodes.update([u, v])
            ed = d.copy()
            ed["source_subset"] = "strict"
            edges.append((u, v, ed))
            pmid_counts[doc_id.split(".")[0]] += 1


    logging.info(f"Found {len(edges)} matching edges")
    logging.info(f"Found {len(nodes)} nodes")
    logging.info(f"Edges from {len(pmid_counts)} unique PMIDs")

    for n in nodes:
        subgraph.add_node(n, **kg.nodes[n].copy())
    
    for u, v, d in edges:
        subgraph.add_edge(u, v, **d)

    return subgraph


def extract_permissive_subset(kg: KnowledgeGraph, nodes: Set) -> KnowledgeGraph:
    
    logging.info(f"Extracting permissive subset with {len(nodes)} nodes...")
    
    subgraph = KnowledgeGraph(schema=kg.schema)
    for n in nodes:
        subgraph.add_node(n, **kg.nodes[n].copy())

    for u, v, k, d in tqdm(kg.edges(keys=True, data=True), total=kg.number_of_edges(), desc="Finding permissive edges"):
        if u in nodes and v in nodes:
            ed = d.copy()
            ed["source_subset"] = "permissive"
            subgraph.add_edge(u, v, **ed)

    logging.info(f"Permissive subset has {subgraph.number_of_edges()} edges")

    return subgraph


# =============================================================================
# Augmentation
# =============================================================================

def compute_high_degree_nodes(graph: KnowledgeGraph, top_percentile: float) -> Set:
    degrees = dict(graph.degree())
    if not degrees:
        return set()

    vals = sorted(degrees.values(), reverse=True)
    cutoff = max(1, int(len(vals) * top_percentile))
    threshold = vals[cutoff - 1]

    high_degree_nodes = {n for n, d in degrees.items() if d >= threshold}

    logging.info(f"Degree threshold for top {top_percentile*100:.0f}%: {threshold}")
    logging.info(f"High-degree nodes: {len(high_degree_nodes)}")

    # Log top 10 high-degree nodes
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    logging.info("Top 10 high-degree nodes:")
    for n, d in top_nodes:
        logging.info(f"  Node: {n}, Degree: {d}")

    return high_degree_nodes


def get_augmentation_candidates(
    kg: KnowledgeGraph,
    strict_graph: KnowledgeGraph,
    nodes: Set,
    high_degree_nodes: Set,
    excluded_edge_types: List[str],
    only_directed: bool,
) -> List[Tuple]:

    logging.info("Collecting augmentation candidates...")
    logging.info(f"Excluding edge types: {excluded_edge_types}")
    logging.info(f"Only directed edges: {only_directed}")
    logging.info(f"Number of high-degree nodes excluded: {len(high_degree_nodes)}")

    strict_keys = {(u, v, k) for u, v, k in strict_graph.edges(keys=True)}
    candidates = []

    for u, v, k, d in tqdm(kg.edges(keys=True, data=True), total=kg.number_of_edges(), desc="Collecting augmentation candidates"):
        if u not in nodes or v not in nodes:
            continue
        if (u, v, k) in strict_keys:
            continue
        if d.get("type") in excluded_edge_types:
            continue
        if u in high_degree_nodes or v in high_degree_nodes:
            continue
        if only_directed and d.get("direction") != "1":
            continue

        candidates.append((u, v, k, d))

    logging.info(f"Found {len(candidates)} augmentation candidates")

    candidates.sort(key=lambda x: x[3].get("probability", 0.0), reverse=True)
    return candidates


def sample_edges(
    candidates: List[Tuple],
    n_edges: int,
    strategy: SamplingStrategy,
    seed: Optional[int],
) -> List[Tuple]:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if not candidates:
        return []

    n = min(n_edges, len(candidates))

    if strategy == SamplingStrategy.GREEDY:
        return candidates[:n]

    if strategy == SamplingStrategy.RANDOM:
        return random.sample(candidates, n)

    if strategy == SamplingStrategy.WEIGHTED:
        probs = np.array([c[3].get("probability", 0.0) for c in candidates])
        if probs.sum() == 0:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs = probs / probs.sum()
        idx = np.random.choice(len(candidates), size=n, replace=False, p=probs)
        return [candidates[i] for i in idx]


    raise ValueError(strategy)


def augment_graph(
    strict_graph: KnowledgeGraph,
    candidates: List[Tuple],
    config: AugmentConfig,
) -> Tuple[KnowledgeGraph, Dict]:

    logging.info(f"Augmenting graph (max {config.max_total_edges} total edges)...")

    augmented = KnowledgeGraph(schema=strict_graph.schema)

    for n in strict_graph.nodes():
        augmented.add_node(n, **strict_graph.nodes[n].copy())

    strict_edges = 0
    for u, v, k, d in strict_graph.edges(keys=True, data=True):
        augmented.add_edge(u, v, **d.copy())
        strict_edges += 1

    logging.info(f"Strict graph has {strict_edges} edges")

    budget = config.max_total_edges - strict_edges
    selected = sample_edges(
        candidates,
        budget,
        config.sampling_strategy,
        config.random_seed,
    )

    added = 0
    types = Counter()

    for u, v, k, d in selected:
        ed = d.copy()
        ed["source_subset"] = "augmented"
        augmented.add_edge(u, v, **ed)
        added += 1
        types[d.get("type", "unknown")] += 1

    stats = {
        "strict_edges": strict_edges,
        "augmented_edges": added,
        "total_edges": strict_edges + added,
        "sampling_strategy": config.sampling_strategy.value,
    }

    return augmented, stats


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Subset and augment a knowledge graph based on PubMed search."
    )

    parser.add_argument("--base-graph", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--email", type=str, required=True)

    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--max-results", type=int, default=10000)

    parser.add_argument("--max-edges", type=int, default=100000)
    parser.add_argument("--top-percentile", type=float, default=0.10)
    parser.add_argument("--excluded-types", nargs="+", default=["Association"])

    parser.add_argument(
        "--sampling-strategy",
        choices=["greedy", "random", "weighted"],
        default="weighted",
    )
    parser.add_argument("--only-directed", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    search_config = SearchConfig(
        start_year=args.start_year,
        end_year=args.end_year,
        max_results=args.max_results,
    )

    augment_config = AugmentConfig(
        max_total_edges=args.max_edges,
        top_percentile_excluded=args.top_percentile,
        excluded_edge_types=args.excluded_types,
        sampling_strategy=SamplingStrategy(args.sampling_strategy),
        random_seed=args.random_seed,
        only_directed=args.only_directed,
    )

    # ------------------------------------------------------------------
    # 1. Build PubMed query
    # ------------------------------------------------------------------
    logging.info("Building PubMed query")
    query = build_pubmed_query(search_config)

    with open(args.output_dir / "query.txt", "w") as f:
        f.write(query)

    # ------------------------------------------------------------------
    # 2. Search PubMed
    # ------------------------------------------------------------------
    logging.info("Searching PubMed")
    pmids = search_pubmed(query, search_config.max_results, args.email)
    pmids_set = set(pmids)

    with open(args.output_dir / "search_pmids.txt", "w") as f:
        for pmid in pmids:
            f.write(pmid + "\n")

    logging.info(f"Retrieved {len(pmids_set)} PMIDs")

    # ------------------------------------------------------------------
    # 3. Cache abstracts
    # ------------------------------------------------------------------
    logging.info("Caching abstracts")
    cache = cache_abstracts(
        pmids,
        args.output_dir / "abstracts_cache.db",
        args.email,
    )
    cache.close()

    # ------------------------------------------------------------------
    # 4. Load base graph
    # ------------------------------------------------------------------
    logging.info("Loading base graph")
    kg = KnowledgeGraph.import_graph(str(args.base_graph))
    logging.info(
        f"Base graph: {kg.number_of_nodes():,} nodes, "
        f"{kg.number_of_edges():,} edges"
    )

    # ------------------------------------------------------------------
    # 5. Extract strict subset
    # ------------------------------------------------------------------
    logging.info("Extracting strict subset")
    strict = extract_strict_subset(kg, pmids_set)

    strict.export_graph(str(args.output_dir / "strict_graph.pkl"))
    with open(args.output_dir / "strict_stats.txt", "w") as f:
        with redirect_stdout(f):
            print_kg_stats(strict)

    # ------------------------------------------------------------------
    # 6. Extract permissive subset
    # ------------------------------------------------------------------
    logging.info("Extracting permissive subset")
    strict_nodes = set(strict.nodes())
    permissive = extract_permissive_subset(kg, strict_nodes)

    permissive.export_graph(str(args.output_dir / "permissive_graph.pkl"))
    with open(args.output_dir / "permissive_stats.txt", "w") as f:
        with redirect_stdout(f):
            print_kg_stats(permissive)

    # ------------------------------------------------------------------
    # 7. Augmentation
    # ------------------------------------------------------------------
    logging.info("Computing high-degree nodes")
    hubs = compute_high_degree_nodes(
        strict, augment_config.top_percentile_excluded
    )

    logging.info("Collecting augmentation candidates")
    candidates = get_augmentation_candidates(
        kg=permissive,
        strict_graph=strict,
        nodes=strict_nodes,
        high_degree_nodes=hubs,
        excluded_edge_types=augment_config.excluded_edge_types,
        only_directed=augment_config.only_directed,
    )

    del kg
    gc.collect()

    logging.info("Augmenting graph")
    augmented, augment_stats = augment_graph(
        strict,
        candidates,
        augment_config,
    )

    augmented.export_graph(str(args.output_dir / "augmented_graph.pkl"))
    with open(args.output_dir / "augmented_stats.txt", "w") as f:
        with redirect_stdout(f):
            print_kg_stats(augmented)

    with open(args.output_dir / "augment_stats.json", "w") as f:
        json.dump(augment_stats, f, indent=2)

    logging.info("Pipeline complete")

if __name__ == "__main__":
    main()