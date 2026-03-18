"""PubMed context extraction and KG processing with stored journal scores and context caching."""

import logging
import os
from collections import Counter
from collections.abc import Iterable
from typing import Any

import fire
from knowledge_graph import KnowledgeGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pubmed_cache import PubMedBatchCache
from tqdm.auto import tqdm


def extract_context(
    u: Any,
    v: Any,
    d: dict[str, Any],
    abstract: str,
    title: str,
    llm_model: str = "openai_gpt5_chat",
    base_url: str = "https://api.marketplace.novo-genai.com/v1",
) -> dict[str, Any]:
    """Use an LLM to extract structured biomedical contextual information for a given edge."""
    system_prompt: str = """
    You are tasked with extracting detailed contextual information about the **{relationship} relationship** between a {u_type} and a {v_type}, as described in a biomedical abstract.

    Granularity rules:
    - Use "general" fields for broad categories (e.g., Tissue, Organism, Cell_Type).
    - Use "detailed" fields for finer resolution (e.g., specific substructures, strains, named subtypes).
    - If multiple values apply, return a list of strings.
    - If a field is not present, use "Not specified." Do not infer beyond the text.
    - Keep entries concise and evidence-based.

    Extract and organize the following fields:

    1. Tissue: General tissue associated with the relationship.
    Detailed_Tissue: Specific tissue substructures or microanatomy (e.g., muscle fibers in skeletal muscle, alveolar epithelium, glomeruli).

    2. Organism: The organism studied (e.g., human, mouse).
    Detailed_Organism: More specific details (e.g., strain, sex, age, disease model; e.g., C57BL/6J male mice, human pediatric cohort).

    Note: "Organism" refers to the biological species under study. "Model_System" (field 9) describes the experimental context (e.g., animal model, cell line, organoid, clinical cohort).

    3. Cell_Type: General cell type involved (e.g., T cell, neuron, fibroblast).
    Detailed_Cell_Type: Specific named subtypes or states (e.g., CD4+ T cells, dopaminergic neurons, myofibroblasts).
    Cell_Line: Specific cell line(s) (e.g., HEK293, A549).

    4. Target_Disease_Role: The functional role or relationship of the gene/protein with the disease (e.g., causal, risk factor, biomarker, protective, therapeutic target, modulator).

    5. Mechanisms: Specific molecular, cellular, or systemic mechanisms described in the target-disease relationship.
    Examples include:
    - Molecular: receptor–ligand interactions, kinase activation/inhibition, transcriptional regulation, epigenetic modification (e.g., DNA methylation, histone acetylation), miRNA-mediated regulation, protein aggregation, post-translational modification.
    - Cellular: apoptosis, autophagy, proliferation, differentiation, immune cell infiltration or activation, metabolic reprogramming, synaptic plasticity, fibrosis.
    - System/Physiology: inflammatory signaling, endocrine modulation, vascular remodeling, neuronal circuit changes, organ dysfunction.

    6. Gene_Regulation: Other genes or regulatory elements influenced by the target, or that influence the target (e.g., "Target X upregulates IL6; Target X is suppressed by miR-21").

    7. Pathways: Molecular or biological pathways connected to the relationship (e.g., NF-κB, PI3K/AKT, MAPK/ERK, Wnt/β-catenin, TGF-β).

    8. Study_Type: Type of study (e.g., experimental, observational, clinical, computational, meta-analysis).

    9. Model_System: Experimental or study model used (e.g., mouse disease model, human clinical cohort, cell line culture, organoids, ex vivo tissue).

    10. Assays_Techniques: Experimental assays or techniques used (e.g., RNA-seq, qPCR, Western blot, immunohistochemistry, flow cytometry, CRISPR, reporter assays, ELISA, imaging modalities).

    11. Statistical_Methodology: Statistical methods described (e.g., t-test, ANOVA, multiple-comparison correction, regression, p-values, confidence intervals, effect sizes). If not reported, use "Not specified."

    12. Physical_Phenotype: Observable physiological or morphological changes (e.g., weight loss, motor deficits, tissue hypertrophy, improved survival).

    13. Molecular_Phenotype: Molecular-level observations (e.g., gene/protein expression changes, metabolite levels, cytokine profiles).

    14. Disease_Progression_Stage: Stage, severity, subtype, or timepoint of disease (e.g., early-stage, advanced, acute flare, remission).

    15. Temporal_Response: Time-course information (e.g., changes at 24 h vs. 7 days; longitudinal trends).

    16. Population_Demographics: Details for clinical populations (e.g., age, sex, sample size, ethnicity, ancestry, geographic region).

    17. Organ: Organ-level context (e.g., liver, heart, brain, lung).
    
    18. System: System-level context (e.g., cardiovascular, nervous, immune, musculoskeletal).

    Output format (json, strict):
    {{
        "Tissue": "[Extracted general tissue]",
        "Detailed_Tissue": "[Extracted specific tissue]",
        "Organism": "[Extracted organism]",
        "Detailed_Organism": "[Detailed organism information]",
        "Cell_Type": "[Extracted general cell type]",
        "Detailed_Cell_Type": "[Extracted specific cell type]",
        "Cell_Line": "[Cell line name]",
        "Target_Disease_Role": "[Functional role in disease]",
        "Mechanisms": "[Biological/molecular mechanisms]",
        "Gene_Regulation": "[Genes or regulatory elements linked]",
        "Pathways": "[Pathways impacted by the relationship]",
        "Study_Type": "[Type of study]",
        "Model_System": "[Experimental system]",
        "Assays_Techniques": "[Techniques described]",
        "Statistical_Methodology": "[Statistical details]",
        "Physical_Phenotype": "[Physiological description]",
        "Molecular_Phenotype": "[Molecular findings]",
        "Disease_Progression_Stage": "[Disease stage]",
        "Temporal_Response": "[Time-course if described]",
        "Population_Demographics": "[Population demographics]",
        "Organ": "[Organ-level description]",
        "System": "[System-level description]"
    }}
    """
    user_prompt: str = """For the following abstract, extract the fields defined above for the {relationship} relationship between {u_name} ({u_type}) and {v_name} ({v_type}).
    {title}
    {abstract}"""
    llm = ChatOpenAI(model=llm_model, base_url=base_url)
    llm = llm.bind(response_format={"type": "json_object"})
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    response: dict[str, Any] = (prompt | llm | parser).invoke(
        {
            "u_name": getattr(u, "name", str(u)),
            "u_type": getattr(u, "type", "Unknown"),
            "v_name": getattr(v, "name", str(v)),
            "v_type": getattr(v, "type", "Unknown"),
            "relationship": d.get("type", "Association"),
            "abstract": abstract or "",
            "title": title or "",
        }
    )
    return response


def _normalize_issn(issn: str | None) -> str | None:
    """Normalize ISSN to uppercase with trimmed whitespace."""
    if not issn:
        return None
    return issn.strip().upper()

def initialize_pubmed_cache(db_path: str, email: str) -> "PubMedBatchCache":
    cache = PubMedBatchCache(db_path=db_path, email=email)
    cache.ensure_tables()
    return cache


def collect_pubmed_pmids(kg: KnowledgeGraph) -> Iterable[str]:
    """Collect unique PubMed PMIDs from edges in the knowledge graph."""
    pmids: set[str] = set()
    for _u, _v, d in kg.edges(data=True):
        if d.get("source") == "PubMed":
            doc_id: str = d.get("document_id", "") or ""
            if doc_id:
                pmid: str = doc_id.split(".", 1)[0]
                pmids.add(str(pmid))
    return pmids


# --- Helper functions ---


def load_knowledge_graph(kg_path: str, logger: logging.Logger) -> "KnowledgeGraph":
    """Load and return a KnowledgeGraph from the given file path."""
    logger.info("Loading KnowledgeGraph from: %s", kg_path)
    return KnowledgeGraph.import_graph(kg_path)


def collect_pubmed_pmids_list(
    kg: "KnowledgeGraph", logger: logging.Logger
) -> list[str]:
    """Collect PubMed PMIDs from the KG and return them as a list."""
    pmids: Iterable[str] = collect_pubmed_pmids(kg)
    pmid_list = list(pmids)
    logger.info("Total PubMed PMIDs found: %s", len(pmid_list))
    return pmid_list


def fetch_and_prepare_cache(
    cache: "PubMedBatchCache",
    pmid_list: list[str],
    journals_csv_path: str,
    fetch_batch_size: int,
    rate_limiting: float,
) -> None:
    """Fetch records into the cache, load the journal index, and recompute journal scores for numeric PMIDs."""
    cache.fetch_batch(
        pmid_list, batch_size=fetch_batch_size, rate_limiting=rate_limiting
    )
    cache.load_or_build_journal_index(journals_csv_path)
    # only numeric PMIDs are passed to recompute
    numeric_pmids = [int(p) for p in pmid_list if p.isdigit()]
    cache.recompute_journal_scores_for_pmids(numeric_pmids)


def log_publication_year_distribution(
    records: dict[int, dict], logger: logging.Logger
) -> None:
    """Log the top publication year distribution extracted from cached records."""
    pub_date_counter: Counter = Counter(
        [
            (data.get("pub_date") or "").split("-")[0]
            for data in records.values()
            if data and data.get("pub_date")
        ]
    )
    logger.info(
        "Publication year distribution (top 20): %s",
        pub_date_counter.most_common(20),
    )


def build_pmid_info_map(records: dict[int, dict]) -> dict[int, dict]:
    """Build and return a mapping from PMID (int) to a small info dict (year, score, title, abstract)."""
    pmid_to_info: dict[int, dict[str, Any]] = {}
    for pmid_int, rec in records.items():
        year_part: str = (rec.get("selected_year") or rec.get("pub_date") or "").split(
            "-"
        )[0]
        pmid_to_info[pmid_int] = {
            "selected_year": (
                rec.get("selected_year") or (year_part if year_part else "-1")
            ),
            "score": (rec.get("journal_score") or "-1"),
            "title": rec.get("title", ""),
            "abstract": rec.get("abstract", ""),
        }
    return pmid_to_info


def list_pubmed_edges(
    kg: "KnowledgeGraph",
) -> list[tuple[Any, Any, Any, dict[str, Any]]]:
    """Return a list of edges from the KG that have source='PubMed'."""
    return [
        (u, v, k, d)
        for u, v, k, d in kg.edges(data=True, keys=True)
        if d.get("source") == "PubMed"
    ]


def _year_key(y: str) -> int:
    """Convert a year string into an int key for sorting, placing missing/non-numeric years last."""
    if not y or not str(y).isdigit():
        return 10**9
    return int(y)


def annotate_edges_with_scores(
    pubmed_edges: list[tuple[Any, Any, Any, dict[str, Any]]],
    pmid_to_info: dict[int, dict[str, Any]],
) -> tuple[dict[tuple[Any, Any], tuple[Any, str, str, int]], dict[str, int]]:
    """Annotate PubMed edges with journal_score and year, and compute highest-score edge per (u,v)."""
    # highest_journal_score maps (u,v) -> (best_key, best_score_str, best_selected_year, best_pmid_int)
    highest_journal_score: dict[tuple[Any, Any], tuple[Any, str, str, int]] = {}
    all_journal_scores: dict[str, int] = {}

    for u, v, k, d in tqdm(pubmed_edges, desc="Annotating journal scores"):
        pmid_str: str = (d.get("document_id", "") or "").split(".", 1)[0]
        if not pmid_str.isdigit():
            d["journal_score"] = "-1"
            d["year"] = "-1"
            all_journal_scores["-1"] = all_journal_scores.get("-1", 0) + 1
            continue

        pmid_int: int = int(pmid_str)
        info: dict[str, Any] | None = pmid_to_info.get(pmid_int)
        if not info:
            d["journal_score"] = "-1"
            d["year"] = "-1"
            all_journal_scores["-1"] = all_journal_scores.get("-1", 0) + 1
            continue

        sel_year: str = str(info["selected_year"])
        score_str: str = str(info["score"])
        d["journal_score"] = score_str
        d["year"] = sel_year
        all_journal_scores[score_str] = all_journal_scores.get(score_str, 0) + 1

        current = highest_journal_score.get((u, v))
        candidate = (k, score_str, sel_year, pmid_int)

        if current is None:
            highest_journal_score[(u, v)] = candidate
        else:
            cur_score = int(current[1]) if str(current[1]).lstrip("-").isdigit() else -1
            cand_score = int(score_str) if str(score_str).lstrip("-").isdigit() else -1
            if cand_score > cur_score:
                highest_journal_score[(u, v)] = candidate
            elif cand_score == cur_score:
                cur_year = _year_key(current[2])
                cand_year = _year_key(sel_year)
                if cand_year < cur_year or (
                    cand_year == cur_year and pmid_int < current[3]
                ):
                    highest_journal_score[(u, v)] = candidate

    return highest_journal_score, all_journal_scores


def perform_context_extraction_for_edges(
    kg: "KnowledgeGraph",
    pmid_to_info: dict[int, dict[str, Any]],
    cache: "PubMedBatchCache",
    llm_model: str,
    llm_base_url: str,
    logger,
    idx_list: list[int] | None = None,
    only_cached: bool = False
) -> None:
    logger.info("Starting context extraction for top journal score edges...")
    
    kg_edges = list(kg.edges(data=True, keys=True))
    
    if idx_list is not None:
        selected_edges = [kg_edges[i] for i in idx_list]
    else:
        selected_edges = kg_edges

    for u, v, k, d in tqdm(selected_edges, desc="Context extraction"):
        pmid_str: str = (d.get("document_id", "") or "").split(".", 1)[0]
        if not pmid_str.isdigit():
            continue
        pmid_int = int(pmid_str)
        info = pmid_to_info.get(pmid_int)
        if not info:
            continue

        title = str(info["title"])
        abstract_text = str(info["abstract"])
        score_str = str(d.get("journal_score", "-1"))

        # Only extract for top journals (score 1 or 2)
        if True: #score_str in ("1", "2"):
            edge_key = PubMedBatchCache.make_edge_key(u, v, k)

            cached = cache.get_cached_edge_context(edge_key, pmid_int, llm_model)
            if cached is not None:
                d["context"] = cached
                continue
            
            if only_cached:
                continue

            try:
                context = extract_context(
                    u,
                    v,
                    d,
                    abstract=abstract_text,
                    title=title,
                    llm_model=llm_model,
                    base_url=llm_base_url,
                )
                d["context"] = context
                cache.store_edge_context(edge_key, pmid_int, llm_model, context)
            except Exception as e:
                logger.warning(
                    f"Context extraction failed for edge ({u}, {v}, {k}), PMID {pmid_int}: {e}"
                )


def prune_non_highest_edges(
    kg: "KnowledgeGraph", highest_journal_score: dict, logger: logging.Logger
) -> None:
    """Remove edges from the KG that are not the highest journal-score for their (u,v) pair."""
    logger.info("Removing non-highest journal score edges...")
    edges_snapshot = list(kg.edges(data=True, keys=True))
    for u, v, k, _d in tqdm(edges_snapshot, desc="Pruning edges"):
        best = highest_journal_score.get((u, v))
        if best and k != best[0]:
            kg.remove_edge(u, v, key=k)

def prune_no_context_edges(
    kg: "KnowledgeGraph", logger: logging.Logger
) -> None:
    """Remove edges from the KG that are not the highest journal-score for their (u,v) pair."""
    logger.info("Removing no-context edges...")
    edges_snapshot = list(kg.edges(data=True, keys=True))
    for u, v, k, d in tqdm(edges_snapshot, desc="Pruning edges"):
        if "context" not in d:
            kg.remove_edge(u, v, key=k)


def determine_export_path(original_path: str, provided_export_path: str | None, optional_i: int | None) -> str:
    """Return the final export path, using a default suffix if none was provided."""
    if optional_i is None:
        i_str = ""
    else:
        i_str = str(optional_i)
    if provided_export_path is not None:
        return provided_export_path
    if original_path.endswith(".pkl"):
        return original_path[:-4] + f"_annotated{i_str}.pkl"
    return original_path + f"_annotated{i_str}.pkl"


def log_journal_score_distribution(
    all_journal_scores: dict[str, int], logger: logging.Logger
) -> None:
    """Log the distribution of journal scores observed across annotated edges."""
    sorted_scores = sorted(all_journal_scores.items(), key=lambda x: -x[1])
    logger.info("Journal scores distribution: %s", sorted_scores)


def process_kg(
    kg_path: str,
    context_extraction: bool = False,
    remove_nonhighest_journal_score: bool = False,
    remove_no_context_edges: bool = False,
    journals_csv_path: str = "/novo/users/fzsg/PhDProjects/GraphAnalysis/Pipeline/data/2025-10-06 Scientific Journals and Series.csv",
    export_path: str | None = None,
    llm_model: str = "openai_gpt5_chat",
    llm_base_url: str = "https://api.marketplace.novo-genai.com/v1",
    db_path: str = "/scratch/fzsg/pubmed_abstracts_context.db",
    email: str = "fzsg@novonordisk.com",
    fetch_batch_size: int = 200,
    rate_limiting: float = 0.4,
    process_batch_i: int | None = None,
    process_batch_n: int | None = None,
    only_cached: bool = False
) -> None:
    """Process a knowledge graph.

    Steps:
    - Import the KG
    - Fetch PubMed records for PubMed-sourced edges
    - Annotate edges with journal_score and publication year from cached table
    - Optionally extract context via LLM for top-journal-score edges
    - Optionally remove edges not having the highest journal score per (u, v)
    - Export the updated KG.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kg_processor")

    export_path = determine_export_path(kg_path, export_path, process_batch_i)
    if os.path.exists(export_path):
        logger.error("Export path %s already exists. Exiting" % export_path)
        return

    kg = load_knowledge_graph(kg_path, logger)

    pmid_list = collect_pubmed_pmids_list(kg, logger)
    
    if process_batch_i is not None and process_batch_n is not None:
        N = len(kg.edges())
        k = process_batch_i % 10
        db_path = f"{db_path[:-3]}{k}.db"

    cache = initialize_pubmed_cache(db_path=db_path, email=email)
    try:
        fetch_and_prepare_cache(
            cache, pmid_list, journals_csv_path, fetch_batch_size, rate_limiting
        )
        records = cache.get_records(pmid_list)
    finally:
        # Keep cache open during the run for possible context persistence
        pass

    logger.info("Successfully fetched records: %s", len(records))

    log_publication_year_distribution(records, logger)

    pmid_to_info = build_pmid_info_map(records)

    pubmed_edges = list_pubmed_edges(kg)

    highest_journal_score, all_journal_scores = annotate_edges_with_scores(
        pubmed_edges, pmid_to_info
    )

    if context_extraction:
        idx_list = None
        
        if process_batch_i is not None and process_batch_n is not None:
            idx_list = list(range(process_batch_i,  N, process_batch_n))
            print(f"Selected {len(idx_list)} edges for context extraction in batch {process_batch_i} of {process_batch_n}.")
        
        perform_context_extraction_for_edges(
            kg,
            pmid_to_info,
            cache,
            llm_model,
            llm_base_url,
            logger,
            idx_list=idx_list,
            only_cached=only_cached
        )

    if remove_nonhighest_journal_score:
        prune_non_highest_edges(kg, highest_journal_score, logger)

    if remove_no_context_edges:
        prune_no_context_edges(kg, logger)
    
    kg.export_graph(export_path)
    logger.info("Exported updated KnowledgeGraph to: %s", export_path)

    log_journal_score_distribution(all_journal_scores, logger)

    cache.close()


if __name__ == "__main__":
    fire.Fire(process_kg)
