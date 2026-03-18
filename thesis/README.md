# ContextAwareKGReasoning
Code repository for the M.Sc. thesis, Enhancing Probabilistic Semantic Reasoning with Context-Aware Knowledge Graphs to Unravel Adipose Tissue's Role in Inflammation DTU (2025).

This project extends Probabilistic Semantic Reasoning (PSR) — the inference framework introduced alongside the [iKraph](https://doi.org/10.1038/s42256-025-01014-w) biomedical knowledge graph (Zhang *et al.*, 2025) — with multi-hop generalization, metapath-based aggregation, and tissue coverage propagation, and applies the resulting framework to uncover depot-specific mechanisms in adipose tissue inflammation.

## Motivation

Biomedical knowledge graphs like iKraph encode millions of literature-derived relationships with probabilistic confidence scores, but lack a critical context dimension: the biological setting in which a relationship was observed. An edge stating that *TNFα promotes insulin resistance* may originate from a study on visceral adipose tissue, hepatic inflammation in mice, or isolated adipocytes — yet the graph records only the relationship itself. This context-blindness has concrete consequences: in adipose tissue biology, the same molecule (e.g., IL-6) can play opposing roles depending on the tissue depot, meaning that context-agnostic inference conflates contradictory signals.

## Research Questions

**RQ1.** Can PSR be extended to *k*-hop paths, and what are the interpretational and statistical trade-offs of 3-hop versus 2-hop inference?

**RQ2.** Can LLM-based re-extraction of tissue context from source abstracts meaningfully enrich knowledge graph edges at scale?

**RQ3.** Does a tissue-aware coverage metric, integrated into PSR ranking, surface biologically relevant depot-specific gene associations?

**RQ4.** Do 2-hop and 3-hop inference provide complementary or redundant biological insights?

## Pipeline Overview

The pipeline maps onto three context paradigms adapted from recommender systems (Adomavicius *et al.*, 2011):

1. **Graph representation and data loading** — Parsing raw iKraph JSON into a NetworkX `MultiDiGraph` via the `KnowledgeGraph` class.
2. **Contextual prefiltering** — PubMed-guided subgraph extraction: filtering to human, literature-derived edges and subsetting around adipose-inflammation seed entities.
3. **LLM-based contextual re-extraction** — Re-reading source PubMed abstracts with an LLM to extract structured tissue, cell type, organism, and experimental condition annotations at the edge level.
4. **Normalization and topic discovery** — Graph normalization and LDA-based topic modeling on mechanism and pathway edge attributes.
5. **Contextual modeling (PSR extensions)** — Multi-hop generalization (arbitrary *k*-hop), metapath-based aggregation preserving mechanistic distinctions, and tissue coverage propagation alongside probability and evidence during inference.
6. **Contextual postfiltering** — Tissue comparison, permutation testing, and enrichment analysis to interpret and validate depot-specific results.


## Repository Structure

```
.
├── knowledge_graph/               # Core KG library
│   ├── core/                      #   KnowledgeGraph class and schema definitions
│   ├── convert/                   #   Converters (PyKEEN, PyTorch Geometric)
│   ├── io/                        #   Import/export and dataset adapters (iKraph, ULTRA)
│   └── utils/                     #   Filtering and statistics utilities
│
├── pubmed/                        # PubMed integration
│   ├── pubmed_cache.py            #   SQLite-backed Entrez API cache with batch retrieval
│   ├── pubmed_extraction.py       #   Structured PubMed queries for subgraph extraction 
│   └── old/                       #   Old veriosns of the scripts extraction pipline
│
├── scripts/
│   ├── preprocessing/             # Stage 1: Raw iKGraph → clean, normalized KG
│   ├── subsetting/                # Stage 2: Full KG → disease-relevant subset + augmentation
│   ├── psr/                       # PSR algorithm (standalone 2-hop and 3-hop + aggregation)
│   ├── rq1/                       # RQ1 pipeline: multi-hop PSR across tissue contexts (not included in thesis)
│   ├── rq2/                       # RQ2 pipeline: tissue-aware PSR + permutation testing
│   ├── analyze_and_vis/           # EDA, graph characterisation, and plotting
│   ├── prototyping/               # Exploratory subsetting experiments
│   └── utils/                     # Shared utilities (metapath grouping)
│
├── slurm/                         # Slurm job submission scripts
└── README.md
```

### Key Components

#### `knowledge_graph/`

A Python library wrapping NetworkX's `MultiDiGraph` for biomedical KG operations. Nodes are `(official_name, entity_type)` named tuples with attribute dictionaries (BioKDE ID, subtype, external IDs). Edges carry relation type, correlation sign, directionality, NLP probability, evidence score, source PMID, and species annotation. Developed by Frederik Steensgaard Gade (fzsg@novonordisk.com or fstga@dtu.dk).

#### `pubmed/`

Handles PubMed API interaction for the contextual prefiltering stage. `PubMedBatchCache` wraps NCBI Entrez with batch fetching and rate limiting, storing titles, abstracts, authors, journals, and DOIs in a local SQLite database to ensure reproducibility and provide abstract texts for the LLM re-extraction pipeline.

#### `scripts/psr/`

Implements Probabilistic Semantic Reasoning,  and multi-hop generalization using chained noisy-OR aggregation in log-space for numerical stability. Also integrates metapath-based grouping. These were later adapted into scripts/rq2/ with RQ2-specific modifications (tissue coverage integration, lda topics proapagation, config-driven parameters).

### Script Reference

#### Preprocessing (`scripts/preprocessing/`)

| Script | Purpose |
|---|---|
| `dataset_ikraph.py` | Load raw iKraph JSON data into the `KnowledgeGraph` format |
| `filter.py` | Filter to PubMed-sourced and human-relevant edges |
| `normalize_graph.py` | Normalize tissue, mechanism, and pathway fields (single-pass) |
| `qa_check.py` | BioRED-based quality assurance checks |
| `fit_lda.py` | Fit LDA topic models on mechanism and pathway edge attributes |
| `exploration.py` | Interactive graph exploration and statistics |

#### Subsetting (`scripts/subsetting/`)

| Script | Purpose |
|---|---|
| `search_utils.py` | Core library: PubMed search, PMID-based subsetting, *k*-hop expansion |
| `search_subset.py` | Config-driven runner for `search_utils` |
| `subset_and_augment.py` | Full pipeline: PubMed search → strict subset → augmentation with non-association edges |

#### PSR Algorithm (`scripts/psr/`)

| Script | Purpose |
|---|---|
| `psr.py` | 2-hop PSR inference with metapath grouping |
| `three_hop_psr.py` | 3-hop PSR inference with metapath grouping |
| `aggregate.py` | Edge aggregation: groups parallel edges by (source, target, type), computes evidence scores |

#### RQ1 Pipeline (`scripts/rq1/`)/Sankey

| Script | Purpose |
|---|---|
| `filter_graphs.py` | Create tissue-specific subgraphs (adipose, liver, non-adipose, baseline) |
| `aggregate_graphs.py` | Aggregate filtered graphs, compute evidence scores |
| `run_psr.py` | Run 2-hop and 3-hop PSR inference on tissue-filtered graphs |
| `compare_contexts.py` | Cross-context comparison of inference results |
| `generate_report.py` | HTML report generation |
| `orchestrator.py` | Slurm pipeline manager |
| `config.yaml` | Pipeline configuration |

#### RQ2 Pipeline (`scripts/rq2/`)

| Script | Purpose |
|---|---|
| `preprocess.py` | Graph preprocessing (node-type filtering, edge expansion) |
| `aggregate.py` | Per-tissue aggregation with coverage as tiebreaker |
| `run_psr.py` | PSR inference with tissue coverage propagation |
| `compare.py` | Tissue pair comparison (e.g., subcutaneous vs. visceral) |
| `enrichment.py` | Over-representation analysis on LDA topics per metapath |
| `run_permutation.py` | Permutation testing worker (batched, Slurm-parallelised) |
| `aggregate_permutations.py` | Per-triple empirical p-values with BH FDR correction |
| `generate_report.py` | HTML report with enrichment tab |
| `plot_results.py` | Publication-ready multi-panel figures |
| `plot_volcano.py` | Volcano plots (effect size vs. evidence) |
| `tissue_mapping.py` | Tissue group definitions and coverage computation |
| `utils.py` | RQ2-specific helper functions |
| `orchestrator.py` | Slurm pipeline manager |
| `config.yaml` | Pipeline configuration |

#### Analysis and Visualization (`scripts/analyze_and_vis/`)

Standalone scripts for characterising intermediate outputs. Not part of the main pipeline.

| Script | Purpose |
|---|---|
| `analyze_augmented_kg.py` | Augmented KG statistics and edge-type dist/Sankeyributions |
| `analyze_kg_sources.py` | Edge source and provenance analysis |
| `analyze_subgraph_overlap.py` | Subgraph overlap analysis (Venn diagrams) |
| `augment_analysis.py` | Augmentation impact analysis |
| `audit_direction.py` | Edge direction consistency audit |
| `viz_graph.py` | KG statistics visualizations (degree, probability, node/edge types) |
| `viz_helper.py` | Static graph rendering (direct, 2-hop, 3-hop edge styling) |
| `plot_alluvial.py` | Alluvial diagrams |
| `plot_distributions_simple.py` | Edge probability distributions |
| `plot_lda.py` | LDA topic visualizations |
| `plot_stats.py` | Barplots of node/edge type distributions |

## Data

This project operates on the **iKraph** knowledge graph (Zhang *et al.*, 2025): 12M nodes across 12 entity types and 83M edges across 53 relation types, constructed via NLP extraction from the full PubMed corpus supplemented by 40 curated databases. The raw data comprises four JSON files: `NER_ID_dict_cap_final.json` (entity dictionary), `RelTypeInt.json` (relation schema), `PubMedList.json` (literature-derived relations), and `DBRelations.json` (database-curated relations).

Graph data and PubMed caches are not included in this repository due to their size. The expected data layout is documented in `scripts/rq2/config.yaml` under the `paths:` section. iKraph data is available from the [original publication](https://doi.org/10.1038/s42256-025-01014-w).

## Setup

Python 3.10+ is required.

```bash
pip install -r requirements.txt
```

The `knowledge_graph` and `pubmed` packages are included in this repository and do not need separate installation. All scripts add the project root to `sys.path` at runtime.

## Running the Pipelines

**RQ1** — managed by the orchestrator, submitted to Slurm:

```bash
cd scripts/rq1
python orchestrator.py --config config.yaml --submit --wait
```

**RQ2** — same pattern:

```bash
cd scripts/rq2
python orchestrator.py --config config.yaml --submit --wait
```

Individual steps can be run standalone; see each script's `--help` for arguments. The preprocessing and subsetting stages are run individually via their corresponding Slurm scripts in `slurm/`.

## Acknowledgements

The `KnowledgeGraph` class and `dataset_ikraph.py` loader were developed by Frederik Steensgaard Gade as part of his ongoing doctoral research and shared with this project.

## License

See [LICENSE](LICENSE) for details.

## References

- Y. Zhang et al., “A comprehensive large-scale biomedical knowledge graph for AI-powered data-driven biomedical research,” Nat Mach Intell, vol. 7, no. 4, pp. 602–614, Mar. 2025, doi: [10.1038/s42256-025-01014-w](https://doi.org/10.1038/s42256-025-01014-w).

- G. Adomavicius, B. Mobasher, F. Ricci, and A. Tuzhilin, “Context-Aware Recommender Systems,” AI Magazine, vol. 32, no. 3, pp. 67–80, 2011, doi: [10.1609/aimag.v32i3.2364](https://doi.org/10.1609/aimag.v32i3.2364).

