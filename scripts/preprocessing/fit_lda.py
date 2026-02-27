#!/usr/bin/env python3
"""
Fit LDA Topic Models on Mechanisms and Pathways

Runs on the graph AFTER normalization (normalize_mechanisms_pathways.py).
Discovers latent themes in mechanism and pathway edge attributes using LDA.

Two modes:
  --explore   Sweep topic counts (k=5..30), compute coherence, produce
              diagnostic plots and per-k topic listings for manual inspection.
  
  --fit       Fit final LDA model with chosen k, assign topic IDs to every
              edge, save augmented graph.

Each edge's normalized mechanisms/pathways field (semicolon-delimited) is
treated as a short document. Individual terms become tokens.

Outputs:
  Explore mode:
    - coherence_vs_k_{field}.png         Elbow plot
    - topics_k{k}_{field}.txt           Human-readable topic listings per k
    - explore_summary_{field}.json       All metrics in machine-readable form
  
  Fit mode:
    - augmented graph (pickle)           Edges gain mechanism_topic_id,
                                         mechanism_topic_dist, pathway_topic_id,
                                         pathway_topic_dist attributes
    - lda_model_{field}.pkl              Saved sklearn model for reuse
    - lda_vocabulary_{field}.json        Term list
    - lda_topics_{field}.json            Topic descriptions

Usage:
    # Explore: find good k
    python fit_lda.py --mode explore --input graph.pkl --output-dir lda_output/

    # Fit: augment graph with chosen k
    python fit_lda.py --mode fit --input graph.pkl --output graph_with_topics.pkl \\
                      --k-mechanisms 15 --k-pathways 12
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the tokenizer from the normalization script
from normalize_graph import tokenize_semicolon_field


# =============================================================================
# CONSTANTS
# =============================================================================

FIELDS = ['mechanisms', 'pathways']
DEFAULT_K_RANGE = [5, 6, 7, 8, 9, 10, 11, 12, 13]
MIN_DF = 2        # Minimum document frequency for a term to be included
MAX_DF_FRAC = 0.50 # Maximum document frequency fraction (ignore ubiquitous terms)
MIN_TOKENS = 1     # Minimum vocabulary tokens per document after filtering;
                   # documents with fewer are excluded from LDA and get topic_id = -1


# =============================================================================
# DOCUMENT EXTRACTION
# =============================================================================

def extract_documents(kg, field: str) -> Tuple[List[List[str]], List[int]]:
    """
    Extract tokenized documents from edge attributes.
    
    Each edge with a non-empty field value becomes one document,
    consisting of the individual terms after semicolon splitting.
    
    Args:
        kg: KnowledgeGraph
        field: 'mechanisms' or 'pathways'
    
    Returns:
        Tuple of:
        - documents: List of token lists (one per edge with data)
        - edge_indices: List of edge indices for mapping back
    """
    documents = []
    edge_indices = []
    
    for idx, (u, v, data) in enumerate(kg.edges(data=True)):
        raw = data.get(field)
        if raw is None:
            # Try context dict
            if 'context' in data and isinstance(data['context'], dict):
                raw = data['context'].get(field.capitalize(), data['context'].get(field))
        
        if raw is None:
            continue
        
        raw_str = str(raw).strip()
        if raw_str.rstrip('.').lower() in ('', 'none', 'not specified', 'nan',
                                           'unknown', 'n/a', 'na', 'not available',
                                           'not applicable', 'null', 'unspecified'):
            continue
        
        terms = tokenize_semicolon_field(raw)
        if not terms:
            continue
        
        # Lowercase for LDA vocabulary consistency
        terms_lower = [t.lower().strip() for t in terms if t.strip()]
        if terms_lower:
            documents.append(terms_lower)
            edge_indices.append(idx)
    
    return documents, edge_indices


def build_vocabulary(documents: List[List[str]], 
                      min_df: int = MIN_DF, 
                      max_df_frac: float = MAX_DF_FRAC) -> Tuple[Dict[str, int], List[str]]:
    """
    Build vocabulary from documents with frequency filtering.
    
    Returns:
        Tuple of (term_to_index dict, index_to_term list)
    """
    # Count document frequency (how many documents contain each term)
    doc_freq = Counter()
    for doc in documents:
        for term in set(doc):  # set() to count each term once per document
            doc_freq[term] += 1
    
    n_docs = len(documents)
    max_df = int(max_df_frac * n_docs)
    
    # Filter
    vocab_terms = sorted([
        term for term, freq in doc_freq.items()
        if freq >= min_df and freq <= max_df
    ])
    
    term_to_idx = {term: i for i, term in enumerate(vocab_terms)}
    
    return term_to_idx, vocab_terms


def documents_to_matrix(documents: List[List[str]], 
                         term_to_idx: Dict[str, int]) -> 'scipy.sparse.csr_matrix':
    """
    Convert documents to a document-term count matrix.
    """
    from scipy.sparse import lil_matrix
    
    n_docs = len(documents)
    n_terms = len(term_to_idx)
    
    mat = lil_matrix((n_docs, n_terms), dtype=np.int32)
    
    for doc_idx, terms in enumerate(documents):
        for term in terms:
            if term in term_to_idx:
                mat[doc_idx, term_to_idx[term]] += 1
    
    return mat.tocsr()


def filter_sparse_documents(documents: List[List[str]], edge_indices: List[int],
                             term_to_idx: Dict[str, int],
                             min_tokens: int = MIN_TOKENS
                             ) -> Tuple[List[List[str]], List[int],
                                        List[int]]:
    """
    Remove documents that have fewer than min_tokens vocabulary terms.

    After vocabulary filtering (min_df / max_df), many documents may have
    zero or one surviving token.  These carry no discriminative information
    for LDA and get dumped into a catch-all topic, degrading model quality.

    Args:
        documents:  Original token lists (one per edge).
        edge_indices: Parallel list of edge indices.
        term_to_idx: Vocabulary mapping (only terms that passed filtering).
        min_tokens:  Minimum number of *vocabulary* tokens a document must
                     have to be kept for LDA.

    Returns:
        Tuple of:
        - kept_documents:  Filtered document list.
        - kept_edge_indices: Corresponding edge indices.
        - dropped_edge_indices: Edge indices of removed documents (these
          will receive topic_id = -1 in fit mode).
    """
    kept_docs = []
    kept_idx = []
    dropped_idx = []

    for doc, eidx in zip(documents, edge_indices):
        n_vocab_tokens = sum(1 for t in doc if t in term_to_idx)
        if n_vocab_tokens >= min_tokens:
            kept_docs.append(doc)
            kept_idx.append(eidx)
        else:
            dropped_idx.append(eidx)

    return kept_docs, kept_idx, dropped_idx


# =============================================================================
# COHERENCE COMPUTATION
# =============================================================================

def compute_umass_coherence(topic_term_matrix: np.ndarray,
                             doc_term_matrix,
                             vocab: List[str],
                             top_n: int = 10) -> float:
    """
    Compute UMass coherence score for a set of topics.
    
    UMass coherence measures how often the top words in a topic
    co-occur in the same documents. Higher (less negative) is better.
    
    Args:
        topic_term_matrix: (n_topics, n_terms) matrix from LDA
        doc_term_matrix: (n_docs, n_terms) sparse count matrix
        vocab: Vocabulary list
        top_n: Number of top terms per topic to use
    
    Returns:
        Mean UMass coherence across all topics
    """
    # Convert to binary (presence/absence)
    binary_dtm = (doc_term_matrix > 0).astype(np.float64)
    n_docs = binary_dtm.shape[0]
    
    # Document frequency per term
    df = np.array(binary_dtm.sum(axis=0)).flatten()
    
    # Co-document frequency (pairwise)
    # For efficiency, we only compute for top terms per topic
    topic_coherences = []
    
    for topic_idx in range(topic_term_matrix.shape[0]):
        top_indices = topic_term_matrix[topic_idx].argsort()[-top_n:][::-1]
        
        coherence = 0.0
        n_pairs = 0
        
        for i in range(1, len(top_indices)):
            for j in range(i):
                idx_i = top_indices[i]
                idx_j = top_indices[j]
                
                # Co-occurrence count
                co_doc = binary_dtm[:, idx_i].multiply(binary_dtm[:, idx_j]).sum()
                
                # UMass: log( (D(wi, wj) + 1) / D(wj) )
                d_j = df[idx_j]
                if d_j > 0:
                    coherence += np.log((co_doc + 1.0) / d_j)
                    n_pairs += 1
        
        if n_pairs > 0:
            topic_coherences.append(coherence / n_pairs)
    
    return np.mean(topic_coherences) if topic_coherences else 0.0


# =============================================================================
# LDA FITTING
# =============================================================================

def fit_lda_single(doc_term_matrix, n_topics: int, random_state: int = 42):
    """
    Fit a single LDA model.
    
    Returns:
        Tuple of (model, doc_topic_matrix, perplexity)
    """
    from sklearn.decomposition import LatentDirichletAllocation
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=50,
        learning_method='batch',
        n_jobs=-1,
    )
    
    doc_topic_matrix = lda.fit_transform(doc_term_matrix)
    perplexity = lda.perplexity(doc_term_matrix)
    
    return lda, doc_topic_matrix, perplexity


def extract_topic_descriptions(lda_model, vocab: List[str], 
                                top_n: int = 15) -> List[Dict]:
    """
    Extract human-readable topic descriptions from fitted LDA model.
    """
    topics = []
    
    for topic_idx in range(lda_model.n_components):
        term_weights = lda_model.components_[topic_idx]
        total_weight = term_weights.sum()
        top_indices = term_weights.argsort()[-top_n:][::-1]
        
        top_terms = []
        for idx in top_indices:
            top_terms.append({
                'term': vocab[idx],
                'weight': float(term_weights[idx] / total_weight),
                'raw_weight': float(term_weights[idx]),
            })
        
        topics.append({
            'topic_id': topic_idx,
            'top_terms': top_terms,
            'label': top_terms[0]['term'] if top_terms else f'Topic {topic_idx}',
        })
    
    return topics


# =============================================================================
# EXPLORE MODE
# =============================================================================

def run_explore(kg, field: str, output_dir: Path, k_range: List[int],
                min_df: int = MIN_DF, max_df_frac: float = MAX_DF_FRAC,
                min_tokens: int = MIN_TOKENS):
    """
    Sweep topic counts, compute metrics, produce diagnostics.
    """
    print(f"\n{'='*80}")
    print(f"EXPLORING LDA FOR: {field.upper()}")
    print(f"{'='*80}")
    
    # Extract documents
    documents, edge_indices = extract_documents(kg, field)
    print(f"  Documents (edges with {field}): {len(documents):,}")
    
    if len(documents) < 50:
        print(f"  WARNING: Too few documents for meaningful LDA. Skipping.")
        return
    
    # Build vocabulary
    term_to_idx, vocab = build_vocabulary(documents, min_df, max_df_frac)
    print(f"  Vocabulary size (min_df={min_df}, max_df<{max_df_frac}): {len(vocab):,}")
    
    if len(vocab) < 10:
        print(f"  WARNING: Vocabulary too small. Try lowering min_df.")
        return
    
    # Filter sparse documents
    kept_docs, kept_idx, dropped_idx = filter_sparse_documents(
        documents, edge_indices, term_to_idx, min_tokens
    )
    n_original = len(documents)
    n_kept = len(kept_docs)
    n_dropped = len(dropped_idx)
    print(f"  Documents after min_tokens={min_tokens} filter: {n_kept:,} "
          f"(dropped {n_dropped:,}, {100*n_dropped/n_original:.1f}%)")
    
    if n_kept < 50:
        print(f"  WARNING: Too few documents after filtering. Try lowering min_tokens or min_df.")
        return
    
    # Show vocabulary stats
    doc_freq = Counter()
    for doc in kept_docs:
        for term in set(doc):
            doc_freq[term] += 1
    
    all_terms_count = Counter()
    for doc in documents:  # Use original docs for "before" stats
        for term in doc:
            all_terms_count[term] += 1
    
    print(f"  Total unique terms (before filtering): {len(all_terms_count):,}")
    print(f"  Terms filtered out by min_df/max_df: {len(all_terms_count) - len(vocab):,}")
    print(f"\n  Top 20 terms in vocabulary:")
    for term in sorted(vocab, key=lambda t: -doc_freq.get(t, 0))[:20]:
        print(f"    {doc_freq[term]:>6,} docs  {term}")
    
    # Build document-term matrix from KEPT documents only
    dtm = documents_to_matrix(kept_docs, term_to_idx)
    print(f"\n  Document-term matrix: {dtm.shape}")
    
    # Sweep k values
    results = []
    
    # Filter k_range to valid values
    max_k = min(len(vocab) // 2, n_kept // 5)
    valid_k_range = [k for k in k_range if k <= max_k and k >= 2]
    
    if not valid_k_range:
        print(f"  WARNING: No valid k values. max_k={max_k}")
        valid_k_range = [min(5, max_k)]
    
    print(f"\n  Testing k values: {valid_k_range}")
    
    for k in valid_k_range:
        print(f"\n  --- k={k} ---")
        
        lda, doc_topic, perplexity = fit_lda_single(dtm, k)
        coherence = compute_umass_coherence(lda.components_, dtm, vocab)
        
        topics = extract_topic_descriptions(lda, vocab)
        
        # Topic size distribution (dominant topic per document)
        dominant_topics = doc_topic.argmax(axis=1)
        topic_sizes = Counter(dominant_topics.tolist())
        
        result = {
            'k': k,
            'perplexity': float(perplexity),
            'coherence_umass': float(coherence),
            'topics': topics,
            'topic_sizes': {int(t): int(c) for t, c in topic_sizes.items()},
        }
        results.append(result)
        
        print(f"    Perplexity: {perplexity:.1f}")
        print(f"    Coherence (UMass): {coherence:.4f}")
        
        # Show topics
        for topic in topics:
            tid = topic['topic_id']
            size = topic_sizes.get(tid, 0)
            top3 = ', '.join(t['term'] for t in topic['top_terms'][:3])
            print(f"    Topic {tid} ({size:,} docs): {top3}")
    
    # Find best k by coherence (highest = least negative for UMass)
    best_result = max(results, key=lambda r: r['coherence_umass'])
    print(f"\n  Best k by UMass coherence: {best_result['k']} "
          f"(coherence={best_result['coherence_umass']:.4f})")
    
    # Save results
    field_dir = output_dir / field
    field_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Summary JSON
    summary = {
        'field': field,
        'n_documents_raw': n_original,
        'n_documents_kept': n_kept,
        'n_documents_dropped': n_dropped,
        'pct_dropped': round(100 * n_dropped / n_original, 1),
        'n_vocabulary': len(vocab),
        'min_df': min_df,
        'max_df_frac': max_df_frac,
        'min_tokens': min_tokens,
        'k_range_tested': valid_k_range,
        'best_k_by_coherence': best_result['k'],
        'results': [
            {
                'k': r['k'],
                'perplexity': r['perplexity'],
                'coherence_umass': r['coherence_umass'],
                'topic_sizes': r['topic_sizes'],
            }
            for r in results
        ],
    }
    
    summary_path = field_dir / 'explore_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_path}")
    
    # 2. Detailed topic listings per k
    for result in results:
        k = result['k']
        topic_path = field_dir / f'topics_k{k}.txt'
        
        with open(topic_path, 'w') as f:
            f.write(f"LDA Topics for {field} (k={k})\n")
            f.write(f"Documents: {len(documents):,}, Vocabulary: {len(vocab):,}\n")
            f.write(f"Perplexity: {result['perplexity']:.1f}, "
                    f"Coherence (UMass): {result['coherence_umass']:.4f}\n")
            f.write(f"{'='*80}\n\n")
            
            for topic in result['topics']:
                tid = topic['topic_id']
                size = result['topic_sizes'].get(tid, 0)
                f.write(f"Topic {tid} ({size:,} dominant documents)\n")
                f.write(f"{'-'*60}\n")
                
                for t in topic['top_terms']:
                    bar = '█' * int(t['weight'] * 100)
                    f.write(f"  {t['weight']:6.3f}  {bar:<20s}  {t['term']}\n")
                
                f.write(f"\n")
        
        print(f"  Saved topic listing: {topic_path}")
    
    # 3. Coherence plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ks = [r['k'] for r in results]
        coherences = [r['coherence_umass'] for r in results]
        perplexities = [r['perplexity'] for r in results]
        
        # Coherence plot
        ax1.plot(ks, coherences, 'bo-', linewidth=2, markersize=8)
        best_idx = coherences.index(max(coherences))
        ax1.plot(ks[best_idx], coherences[best_idx], 'r*', markersize=15, 
                 label=f'Best: k={ks[best_idx]}')
        ax1.set_xlabel('Number of Topics (k)', fontsize=12)
        ax1.set_ylabel('UMass Coherence (higher is better)', fontsize=12)
        ax1.set_title(f'{field.capitalize()}: Coherence vs Number of Topics', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Perplexity plot
        ax2.plot(ks, perplexities, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Topics (k)', fontsize=12)
        ax2.set_ylabel('Perplexity (lower is better)', fontsize=12)
        ax2.set_title(f'{field.capitalize()}: Perplexity vs Number of Topics', fontsize=13)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = field_dir / f'coherence_vs_k.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved plot: {plot_path}")
        
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plot")
    
    # 4. Save vocabulary for reference
    vocab_path = field_dir / 'vocabulary.json'
    vocab_with_freq = [
        {'term': term, 'doc_freq': doc_freq.get(term, 0)}
        for term in vocab
    ]
    vocab_with_freq.sort(key=lambda x: -x['doc_freq'])
    with open(vocab_path, 'w') as f:
        json.dump(vocab_with_freq, f, indent=2, ensure_ascii=False)
    print(f"  Saved vocabulary: {vocab_path}")
    
    return results


# =============================================================================
# FIT MODE
# =============================================================================

def run_fit(kg, field: str, n_topics: int, output_dir: Path,
            min_df: int = MIN_DF, max_df_frac: float = MAX_DF_FRAC,
            min_tokens: int = MIN_TOKENS):
    """
    Fit final LDA model with chosen k, assign topic IDs to edges.
    
    Adds to each edge:
        {field}_topic_id:   int, dominant topic index (or -1 if unclassifiable)
        {field}_topic_dist: list of floats, full topic distribution (or None)
    
    Edges with fewer than min_tokens vocabulary terms after filtering
    get topic_id = -1 (unclassifiable — too sparse for reliable assignment).
    
    Returns:
        Tuple of (lda_model, vocab, topics, edge_indices, doc_topic_matrix)
    """
    print(f"\n{'='*80}")
    print(f"FITTING LDA FOR: {field.upper()} (k={n_topics})")
    print(f"{'='*80}")
    
    # Extract documents
    documents, edge_indices = extract_documents(kg, field)
    print(f"  Documents: {len(documents):,}")
    
    # Build vocabulary
    term_to_idx, vocab = build_vocabulary(documents, min_df, max_df_frac)
    print(f"  Vocabulary: {len(vocab):,}")
    
    # Filter sparse documents
    kept_docs, kept_idx, dropped_idx = filter_sparse_documents(
        documents, edge_indices, term_to_idx, min_tokens
    )
    n_original = len(documents)
    n_kept = len(kept_docs)
    n_dropped = len(dropped_idx)
    print(f"  Documents after min_tokens={min_tokens} filter: {n_kept:,} "
          f"(dropped {n_dropped:,}, {100*n_dropped/n_original:.1f}%)")
    
    # Build matrix from kept documents only
    dtm = documents_to_matrix(kept_docs, term_to_idx)
    print(f"  DTM shape: {dtm.shape}")
    
    # Fit
    print(f"  Fitting LDA with k={n_topics}...")
    lda, doc_topic, perplexity = fit_lda_single(dtm, n_topics)
    coherence = compute_umass_coherence(lda.components_, dtm, vocab)
    
    print(f"  Perplexity: {perplexity:.1f}")
    print(f"  Coherence (UMass): {coherence:.4f}")
    
    # Extract topic descriptions
    topics = extract_topic_descriptions(lda, vocab)
    
    # Assign topic IDs to edges
    print(f"  Assigning topic IDs to {len(kept_idx):,} classified + "
          f"{len(dropped_idx):,} unclassified edges...")
    
    # Build edge list for indexed access
    edge_list = list(kg.edges(data=True))
    
    # Classified edges: assign dominant topic
    edges_assigned = 0
    for doc_idx, edge_idx in enumerate(kept_idx):
        u, v, data = edge_list[edge_idx]
        
        dominant_topic = int(doc_topic[doc_idx].argmax())
        topic_dist = doc_topic[doc_idx].tolist()
        
        data[f'{field}_topic_id'] = dominant_topic
        data[f'{field}_topic_dist'] = [round(p, 4) for p in topic_dist]
        
        edges_assigned += 1
    
    # Unclassified edges: assign -1
    for edge_idx in dropped_idx:
        u, v, data = edge_list[edge_idx]
        data[f'{field}_topic_id'] = -1
        data[f'{field}_topic_dist'] = None
    
    print(f"  Assigned topics to {edges_assigned:,} edges")
    print(f"  Marked {len(dropped_idx):,} edges as unclassified (topic_id=-1)")
    
    # Topic size distribution
    dominant_topics = doc_topic.argmax(axis=1)
    topic_sizes = Counter(dominant_topics.tolist())
    
    print(f"\n  Topic summary:")
    for topic in topics:
        tid = topic['topic_id']
        size = topic_sizes.get(tid, 0)
        top5 = ', '.join(t['term'] for t in topic['top_terms'][:5])
        print(f"    Topic {tid:2d} ({size:>5,} edges): {top5}")
    
    # Save model and metadata
    field_dir = output_dir / field
    field_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model_path = field_dir / f'lda_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': lda,
            'term_to_idx': term_to_idx,
            'vocab': vocab,
            'n_topics': n_topics,
            'min_df': min_df,
            'max_df_frac': max_df_frac,
        }, f)
    print(f"  Saved model: {model_path}")
    
    # Vocabulary
    vocab_path = field_dir / 'vocabulary.json'
    doc_freq = Counter()
    for doc in documents:
        for term in set(doc):
            doc_freq[term] += 1
    
    vocab_with_freq = [
        {'term': term, 'doc_freq': doc_freq.get(term, 0)}
        for term in vocab
    ]
    vocab_with_freq.sort(key=lambda x: -x['doc_freq'])
    with open(vocab_path, 'w') as f:
        json.dump(vocab_with_freq, f, indent=2, ensure_ascii=False)
    
    # Topics
    topics_output = {
        'field': field,
        'n_topics': n_topics,
        'n_documents_raw': n_original,
        'n_documents_kept': n_kept,
        'n_documents_dropped': n_dropped,
        'n_vocabulary': len(vocab),
        'min_df': min_df,
        'max_df_frac': max_df_frac,
        'min_tokens': min_tokens,
        'perplexity': float(perplexity),
        'coherence_umass': float(coherence),
        'topics': [
            {
                **topic,
                'n_dominant_edges': topic_sizes.get(topic['topic_id'], 0),
            }
            for topic in topics
        ],
    }
    
    topics_path = field_dir / 'topics.json'
    with open(topics_path, 'w') as f:
        json.dump(topics_output, f, indent=2, ensure_ascii=False)
    print(f"  Saved topics: {topics_path}")
    
    # Human-readable topic listing
    txt_path = field_dir / 'topics.txt'
    with open(txt_path, 'w') as f:
        f.write(f"LDA Topics for {field} (k={n_topics})\n")
        f.write(f"Documents: {n_kept:,} kept / {n_dropped:,} dropped / {n_original:,} total\n")
        f.write(f"Vocabulary: {len(vocab):,} (min_df={min_df}, max_df<{max_df_frac}, min_tokens={min_tokens})\n")
        f.write(f"Perplexity: {perplexity:.1f}, Coherence (UMass): {coherence:.4f}\n")
        f.write(f"{'='*80}\n\n")
        
        for topic in topics:
            tid = topic['topic_id']
            size = topic_sizes.get(tid, 0)
            f.write(f"Topic {tid} ({size:,} dominant edges)\n")
            f.write(f"{'-'*60}\n")
            for t in topic['top_terms']:
                bar = '█' * int(t['weight'] * 100)
                f.write(f"  {t['weight']:6.3f}  {bar:<20s}  {t['term']}\n")
            f.write(f"\n")
    print(f"  Saved readable topics: {txt_path}")
    
    return lda, vocab, topics, edge_indices, doc_topic


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fit LDA topic models on mechanisms and pathways edge attributes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore different topic counts (produces plots + topic listings)
  python fit_lda.py --mode explore --input normalized_graph.pkl --output-dir lda_explore/
  
  # Fit final model with chosen k values
  python fit_lda.py --mode fit --input normalized_graph.pkl --output augmented_graph.pkl \\
                    --k-mechanisms 15 --k-pathways 12
  
  # Fit only mechanisms
  python fit_lda.py --mode fit --input normalized_graph.pkl --output augmented_graph.pkl \\
                    --k-mechanisms 15 --fields mechanisms
        """
    )
    
    parser.add_argument('--mode', type=str, required=True, choices=['explore', 'fit'],
                        help='explore: sweep k values with diagnostics; fit: augment graph')
    parser.add_argument('--input', type=str, required=True,
                        help='Input graph pickle (should be post-normalization)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output graph pickle (fit mode only)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for explore results / fit metadata')
    parser.add_argument('--fields', type=str, nargs='+', default=FIELDS,
                        choices=FIELDS, help='Which fields to process')
    
    # Explore mode parameters
    parser.add_argument('--k-range', type=int, nargs='+', default=DEFAULT_K_RANGE,
                        help='Topic counts to test in explore mode')
    
    # Fit mode parameters
    parser.add_argument('--k-mechanisms', type=int, default=None,
                        help='Number of topics for mechanisms (fit mode)')
    parser.add_argument('--k-pathways', type=int, default=None,
                        help='Number of topics for pathways (fit mode)')
    
    # Vocabulary parameters
    parser.add_argument('--min-df', type=int, default=MIN_DF,
                        help=f'Minimum document frequency for terms (default: {MIN_DF})')
    parser.add_argument('--max-df-frac', type=float, default=MAX_DF_FRAC,
                        help=f'Maximum document frequency fraction (default: {MAX_DF_FRAC})')
    parser.add_argument('--min-tokens', type=int, default=MIN_TOKENS,
                        help=f'Minimum vocabulary tokens per document (default: {MIN_TOKENS}). '
                             f'Documents with fewer are excluded from LDA and get topic_id=-1.')
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        output_dir = Path(args.output).parent / 'lda_output'
    else:
        output_dir = Path('lda_output')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    from knowledge_graph import KnowledgeGraph
    
    print(f"Loading graph from: {args.input}")
    kg = KnowledgeGraph.import_graph(args.input)
    print(f"Loaded: {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")
    
    if args.mode == 'explore':
        # =============================================
        # EXPLORE MODE
        # =============================================
        print(f"\nMode: EXPLORE")
        print(f"K range: {args.k_range}")
        print(f"Fields: {args.fields}")
        print(f"Output directory: {output_dir}")
        
        all_results = {}
        for field in args.fields:
            results = run_explore(kg, field, output_dir, args.k_range,
                                  args.min_df, args.max_df_frac, args.min_tokens)
            all_results[field] = results
        
        print(f"\n{'='*80}")
        print("EXPLORATION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Inspect topic listings in {output_dir}/{{field}}/topics_k*.txt")
        print(f"  2. Check coherence plots in {output_dir}/{{field}}/coherence_vs_k.png")
        print(f"  3. Choose k values and run:")
        print(f"     python fit_lda.py --mode fit --input {args.input} \\")
        print(f"       --output <output_graph.pkl> \\")
        print(f"       --k-mechanisms <k> --k-pathways <k>")
    
    elif args.mode == 'fit':
        # =============================================
        # FIT MODE
        # =============================================
        if args.output is None:
            if args.input.endswith('.pkl'):
                args.output = args.input.replace('.pkl', '_with_topics.pkl')
            else:
                args.output = args.input + '_with_topics.pkl'
        
        print(f"\nMode: FIT")
        print(f"Fields: {args.fields}")
        print(f"Output graph: {args.output}")
        print(f"Output directory: {output_dir}")
        
        k_values = {}
        if 'mechanisms' in args.fields:
            if args.k_mechanisms is None:
                # Try to load from explore results
                explore_path = output_dir / 'mechanisms' / 'explore_summary.json'
                if explore_path.exists():
                    with open(explore_path) as f:
                        explore = json.load(f)
                    k_values['mechanisms'] = explore['best_k_by_coherence']
                    print(f"  Using best k from explore for mechanisms: {k_values['mechanisms']}")
                else:
                    print("ERROR: --k-mechanisms required (or run explore first)")
                    sys.exit(1)
            else:
                k_values['mechanisms'] = args.k_mechanisms
        
        if 'pathways' in args.fields:
            if args.k_pathways is None:
                explore_path = output_dir / 'pathways' / 'explore_summary.json'
                if explore_path.exists():
                    with open(explore_path) as f:
                        explore = json.load(f)
                    k_values['pathways'] = explore['best_k_by_coherence']
                    print(f"  Using best k from explore for pathways: {k_values['pathways']}")
                else:
                    print("ERROR: --k-pathways required (or run explore first)")
                    sys.exit(1)
            else:
                k_values['pathways'] = args.k_pathways
        
        # Fit models
        for field in args.fields:
            k = k_values[field]
            run_fit(kg, field, k, output_dir, args.min_df, args.max_df_frac, args.min_tokens)
        
        # Save augmented graph
        print(f"\nSaving augmented graph to: {args.output}")
        kg.export_graph(args.output)
        
        # Save metadata
        meta = {
            'timestamp': datetime.now().isoformat(),
            'input_graph': args.input,
            'output_graph': args.output,
            'fields': args.fields,
            'k_values': k_values,
            'min_df': args.min_df,
            'max_df_frac': args.max_df_frac,
            'min_tokens': args.min_tokens,
        }
        meta_path = output_dir / 'fit_metadata.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n{'='*80}")
        print("FIT COMPLETE")
        print(f"{'='*80}")
        print(f"Augmented graph: {args.output}")
        print(f"Model artifacts: {output_dir}")
        print(f"\nNew edge attributes added:")
        for field in args.fields:
            print(f"  {field}_topic_id   (int: dominant topic index)")
            print(f"  {field}_topic_dist (list: full topic distribution)")


if __name__ == '__main__':
    main()