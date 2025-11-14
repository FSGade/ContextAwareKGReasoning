"""PubMed context extraction"""

from collections import Counter
from typing import Dict, Tuple, Any, Optional, Iterable
import logging
import os
import pickle

import pandas as pd
import fire
from tqdm.auto import tqdm

from knowledge_graph import KnowledgeGraph
from pubmed_cache import PubMedBatchCache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


def extract_context(
    u,
    v,
    d: Dict[str, Any],
    abstract: str,
    title: str,
    llm_model: str = "openai_gpt5_chat",
    base_url: str = "",
) -> Dict[str, Any]:
    """
    Use an LLM to extract structured biomedical contextual information for a given edge
    (u, v) based on the PubMed abstract and title.

    Parameters
    ----------
    u : Any
        Source node object of the KnowledgeGraph (expected to have 'name' and 'type' attributes).
    v : Any
        Target node object of the KnowledgeGraph (expected to have 'name' and 'type' attributes).
    d : dict
        Edge attribute dictionary; must include key 'type' describing the relationship.
    abstract : str
        The PubMed abstract text.
    title : str
        The PubMed article title.
    llm_model : str, default "openai_gpt5_chat"
        The LLM model identifier for the ChatOpenAI client.
    base_url : str, default "https://api.marketplace.novo-genai.com/v1"
        Base URL for the custom OpenAI-compatible API endpoint.

    Returns
    -------
    dict
        Parsed JSON object containing the requested contextual fields.
    """
    system_prompt = """
    You are tasked with extracting detailed contextual information about the **{relationship} relationship** between a {u_type} and a {v_type}, as described in a biomedical abstract. Please follow the fields listed below, organizing the extracted data with appropriate granularity. Ensure the information is accurate and concise and identify fields that are not available in the text using "Not specified.". If multiple values apply, please create a list of strings, where relevant. The fields to extract are specified as follows:
    ---
    1. **Tissue**: General tissue associated with the relationship.
    **Detailed_Tissue**: Specific tissue, if described.
    2. **Organism**: The organism studied in the abstract.
    **Detailed_Organism**: More specific details about the organism (e.g., strain or model).
    3. **Cell_Type**: General cell type involved.
    **Detailed_Cell_Type**: Specific cell types or substructures (e.g., muscle fibers in a particular tissue).
    **Cell_Line**: Specific cell line name (e.g., HEK-293).
    4. **Target_Disease_Role**: The role or type of relationship the gene/protein has with the disease.
    5. **Mechanisms**: Specific molecular, cellular, or systemic mechanisms described in the target-disease relationship.
    6. **Gene_Regulation**: Other genes or regulatory elements influenced by the target gene, or that influence the target gene.
    7. **Pathways**: Molecular or biological pathways mentioned in connection with the relationship.
    8. **Study_Type**: The type of study conducted (e.g., experimental, computational, observational).
    9. **Model_System**: The experimental model (e.g., human trials, animal models, cell lines).
    10. **Assays_Techniques**: Experimental assays or techniques used to assess the relationship.
    11. **Statistical_Methodology**: Statistical methods described in the study.
    12. **Physical_Phenotype**: Observable physical traits or changes noted.
    13. **Molecular_Phenotype**: Molecular-level observations, such as gene expression changes.
    14. **Disease_Progression_Stage**: The stage, severity, or subtype of the disease being studied.
    15. **Temporal_Response**: Any information about how the relationship or changes progress over time.
    16. **Population_Demographics**: Further details about the population studied for clinical studies.
    17. **Organ**: Organ-level description associated with the tissue or system (if relevant).
        **System**: Larger system-level context (e.g., musculoskeletal, cardiovascular).
    ---
    The output should follow this structure strictly:
    {
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
    }
    ---
    """

    user_prompt = """For the following abstract, extract the fields defined above for the Association relationship between {u_name} ({u_type}) and {v_name} ({v_type}).
    {title}
    {abstract}"""

    llm = ChatOpenAI(model=llm_model, base_url=base_url)
    llm = llm.bind(response_format={"type": "json_object"})
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", user_prompt)]
    )
    chain = prompt | llm | parser

    response = chain.invoke(
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


def _normalize_issn(issn: Optional[str]) -> Optional[str]:
    """
    Normalize ISSN strings for consistent lookups:
    - Strip whitespace
    - Uppercase
    - Keep hyphen if present (canonical form)

    Returns None if issn is falsy.
    """
    if not issn:
        return None
    return issn.strip().upper()


def _journal_pickle_path(journals_csv_path: str) -> str:
    """
    Return the pickle path with the same base name as the CSV, but .pkl extension.
    """
    base, _ext = os.path.splitext(journals_csv_path)
    return f"{base}.pkl"


def load_or_build_journal_index(journals_csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load a precomputed journal index from pickle if present, otherwise build it
    from the CSV and save the pickle.

    The returned dict maps ISSN -> {
        "Established": Optional[str],
        "years": Dict[str, str],  # year -> score (raw string from CSV)
        "first_available_year_after_established": Optional[str],  # earliest >= max(Established+2, 2004) with non-NA score
    }

    Parameters
    ----------
    journals_csv_path : str
        Path to the "Scientific Journals and Series" CSV file.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Precomputed index for fast lookups.
    """
    pkl_path = _journal_pickle_path(journals_csv_path)
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    df = pd.read_csv(journals_csv_path, sep=";", dtype=str)

    # Identify year columns (Level YYYY -> YYYY)
    level_columns = [col for col in df.columns if col.startswith("Level ")]
    year_columns = [col[len("Level ") :] for col in level_columns]

    # Merge Print and Online ISSN to one ISSN per row via stacking
    merged = df[["Print ISSN", "Online ISSN"]].stack().reset_index(level=1, drop=True)
    merged.name = "ISSN"
    df = df.drop(columns=["Print ISSN", "Online ISSN"]).join(merged)
    df = df.dropna(subset=["ISSN"])
    df = df[["ISSN", "Established"] + level_columns].rename(
        columns={f"Level {y}": y for y in year_columns}
    )

    index: Dict[str, Dict[str, Any]] = {}

    # Precompute per-ISSN dicts
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Precomputing journal index"):
        issn = _normalize_issn(row["ISSN"])
        if not issn:
            continue

        established = row.get("Established")
        established = established if isinstance(established, str) and established.strip() else None

        # Build year->score map
        years_map: Dict[str, str] = {}
        for y in year_columns:
            val = row.get(y)
            years_map[y] = val

        # Compute first available year after established (+2) with caveat scores start in 2004
        first_available_year = None
        try:
            start_year = 2004
            if established and established.isdigit():
                start_year = max(start_year, int(established) + 2)
            for yr in range(start_year, max([int(y) for y in year_columns] + [start_year]) + 1):
                ystr = str(yr)
                if ystr in years_map:
                    val = years_map[ystr]
                    if val is not None and not pd.isna(val):
                        first_available_year = ystr
                        break
        except Exception:
            first_available_year = None

        index[issn] = {
            "Established": established,
            "years": years_map,
            "first_available_year_after_established": first_available_year,
        }

        # Optional: add hyphenless alias for robustness
        if "-" in issn:
            alias = issn.replace("-", "")
            index.setdefault(alias, index[issn])

    # Save pickle for reuse
    with open(pkl_path, "wb") as f:
        pickle.dump(index, f)

    return index


def pick_year_and_score(index_entry: Dict[str, Any], year: str) -> Tuple[str, str]:
    """
    Given a journal index entry and a desired publication year, select the year and score.

    Logic:
    - If desired year exists, use it.
    - Else, if Established is set and desired year > Established+1, use
      first_available_year_after_established (precomputed, respecting the 2004 start).
    - If no score available, return "-1".

    Returns
    -------
    (str, str)
        (selected_year, score_str) where score_str is "-1" if not available or NA.
    """
    years_map: Dict[str, Any] = index_entry.get("years", {})
    established = index_entry.get("Established", None)
    first_after = index_entry.get("first_available_year_after_established", None)

    selected_year = year
    score = years_map.get(selected_year)

    def _normalize_score(s):
        if s is None or pd.isna(s) or s == "NA":
            return "-1"
        return str(s)

    if score is None:
        try:
            if established and int(year) > int(established) + 1:
                if first_after:
                    selected_year = first_after
                    score = years_map.get(selected_year)
                else:
                    score = None
            else:
                score = None
        except Exception:
            score = None

    return selected_year, _normalize_score(score)


def collect_pubmed_pmids(kg) -> Iterable[str]:
    """
    Collect unique PubMed PMIDs from edges in the knowledge graph.

    Parameters
    ----------
    kg : KnowledgeGraph
        The graph instance.

    Returns
    -------
    Iterable[str]
        Set of PMIDs as strings.
    """
    pmids = set()
    for _u, _v, d in kg.edges(data=True):
        if d.get("source") == "PubMed":
            doc_id = d.get("document_id", "")
            if doc_id:
                pmid = doc_id.split(".", 1)[0]
                pmids.add(str(pmid))
    return pmids


def process_kg(
    kg_path: str,
    context_extraction: bool = False,
    remove_nonhighest_journal_score: bool = False,
    journals_csv_path: str = "2025-10-06 Scientific Journals and Series.csv",
    export_path: Optional[str] = None,
    llm_model: str = "openai_gpt5_chat",
    llm_base_url: str = "",
    # PubMedBatchCache parameters:
    db_path: str = "pubmed_abstracts.db",
    email: str = "example@example.com",
    # fetch_batch parameters:
    fetch_batch_size: int = 200,
    rate_limiting: float = 0.4,
) -> None:
    """
    Process a knowledge graph by:
    - Importing the KG
    - Fetching PubMed abstracts for PubMed-sourced edges
    - Assigning journal scores and publication year to edges using a precomputed journals index (CSV -> pickle)
    - Optionally extracting context via LLM for top-journal-score edges
    - Optionally removing edges not having the highest journal score per (u, v)
    - Exporting the updated KG

    Parameters
    ----------
    kg_path : str
        Path to the pickled KnowledgeGraph to import via KnowledgeGraph.import_graph.
    context_extraction : bool, default False
        If True, extracts contextual information (LLM) for edges with the highest journal score per (u, v).
    remove_nonhighest_journal_score : bool, default False
        If True, removes edges that do not have the highest journal score per (u, v).
    journals_csv_path : str, default "...Scientific Journals and Series.csv"
        Path to the journals-level CSV file. A pickle with the same base name and .pkl extension
        will be used/created to speed up future runs.
    export_path : str, optional
        Path to export the updated KnowledgeGraph. If None, derives a default name from kg_path.
    batch_size : int, default 100
        Batch size used internally by this script (not the PubMed fetch batch size).
    llm_model : str, default "openai_gpt5_chat"
        Model name for ChatOpenAI used by context extraction.
    llm_base_url : str, default ""
        Base URL for the OpenAI-compatible endpoint.
    db_path : str, default "pubmed_abstracts.db"
        Path for the PubMedBatchCache SQLite DB file.
    email : str, default "example@example.com"
        Email address used for PubMed utilities in PubMedBatchCache.
    fetch_batch_size : int, default 200
        Batch size used by PubMedBatchCache.fetch_batch.
    rate_limiting : float, default 0.4
        Seconds to wait between batches in PubMedBatchCache.fetch_batch.

    Returns
    -------
    None
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("kg_processor")

    # Load KG
    logger.info(f"Loading KnowledgeGraph from: {kg_path}")
    kg = KnowledgeGraph.import_graph(kg_path)

    # Collect PMIDs from PubMed edges
    pmids = collect_pubmed_pmids(kg)
    logger.info(f"Total PubMed PMIDs found: {len(pmids)}")

    # Fetch abstracts with user-configured cache settings and rate limits
    cache = PubMedBatchCache(db_path=db_path, email=email)
    try:
        # PubMedBatchCache implements batching and rate limiting internally
        cache.fetch_batch(list(pmids), batch_size=fetch_batch_size, rate_limiting=rate_limiting)
        abstracts = cache.get_abstracts([str(pmid) for pmid in pmids])
    finally:
        cache.close()

    logger.info(f"Successfully fetched abstracts: {len(abstracts)}")

    # Summarize publication years
    pub_date_counter = Counter(
        [
            (data.get("pub_date") or "").split("-")[0]
            for data in abstracts.values()
            if data and data.get("pub_date")
        ]
    )
    logger.info(f"Publication year distribution (top 20): {pub_date_counter.most_common(20)}")

    # Load precomputed journal index (CSV -> pickle)
    journal_index = load_or_build_journal_index(journals_csv_path)

    all_journal_scores: Dict[str, int] = {}
    highest_journal_score: Dict[Tuple[Any, Any], Tuple[Any, str]] = {}

    # Annotate edges with journal_score and year
    pubmed_edges = [(u, v, k, d) for u, v, k, d in kg.edges(data=True, keys=True) if d.get("source") == "PubMed"]
    for u, v, k, d in tqdm(pubmed_edges, desc="Annotating journal scores"):
        pmid = (d.get("document_id", "") or "").split(".", 1)[0]
        if not pmid or pmid not in abstracts:
            d["journal_score"] = "-1"
            d["year"] = "-1"
            all_journal_scores["-1"] = all_journal_scores.get("-1", 0) + 1
            continue

        abstract_data = abstracts[pmid] or {}
        year = (abstract_data.get("pub_date") or "").split("-")[0]
        issn = _normalize_issn(abstract_data.get("issn"))

        if not year:
            d["journal_score"] = "-1"
            d["year"] = "-1"
            all_journal_scores["-1"] = all_journal_scores.get("-1", 0) + 1
            continue

        # Determine journal score
        journal_score_str = "-1"
        if issn and issn in journal_index:
            try:
                year, journal_score_str = pick_year_and_score(journal_index[issn], year)
            except Exception:
                journal_score_str = "-1"
        else:
            journal_score_str = "-1"

        # Track highest per (u, v)
        try:
            current_best = highest_journal_score.get((u, v))
            if current_best is None or int(journal_score_str) > int(current_best[1]):
                highest_journal_score[(u, v)] = (k, journal_score_str)
        except Exception:
            if (u, v) not in highest_journal_score:
                highest_journal_score[(u, v)] = (k, journal_score_str)

        # Annotate edge
        d["journal_score"] = journal_score_str
        d["year"] = year

        # Count score distribution
        all_journal_scores[journal_score_str] = all_journal_scores.get(journal_score_str, 0) + 1

    # Optional LLM-based context extraction for highest journal score edges
    if context_extraction:
        logger.info("Starting context extraction for top journal score edges...")
        # Prepare edges to process
        top_edges = []
        for u, v, k, d in kg.edges(data=True, keys=True):
            top_k, _score = highest_journal_score.get((u, v), (None, None))
            if top_k is not None and k == top_k:
                top_edges.append((u, v, k, d))

        for u, v, k, d in tqdm(top_edges, desc="Context extraction"):
            pmid = (d.get("document_id", "") or "").split(".", 1)[0]
            if not pmid or pmid not in abstracts:
                continue

            abstract_data = abstracts[pmid] or {}
            title = abstract_data.get("title", "")
            abstract_text = abstract_data.get("abstract", "")
            score_str = d.get("journal_score", "-1")

            # Heuristic: only extract for journal score 1 or 2
            if score_str in ("1", "2"):
                try:
                    context = extract_context(
                        u, v, d, abstract=abstract_text, title=title, llm_model=llm_model, base_url=llm_base_url
                    )
                    d["context"] = context
                except Exception as e:
                    logger.warning(f"Context extraction failed for PMID {pmid}: {e}")

        # Flatten 'context' fields into the edge attributes
        # for _u, _v, _k, d in tqdm(top_edges, desc="Merging context into edges"):
        #     if "context" in d and isinstance(d["context"], dict):
        #         d.update(d["context"])

    # Optionally remove edges that are not the highest journal score per (u, v)
    if remove_nonhighest_journal_score:
        logger.info("Removing non-highest journal score edges...")
        edges_snapshot = list(kg.edges(data=True, keys=True))
        for u, v, k, _d in tqdm(edges_snapshot, desc="Pruning edges"):
            top_k, _score = highest_journal_score.get((u, v), (None, None))
            if top_k is not None and k != top_k:
                kg.remove_edge(u, v, key=k)

    # Export graph
    if export_path is None:
        if kg_path.endswith(".pkl"):
            export_path = kg_path[:-4] + "_annotated.pkl"
        else:
            export_path = kg_path + "_annotated.pkl"

    kg.export_graph(export_path)
    logger.info(f"Exported updated KnowledgeGraph to: {export_path}")

    # Log journal score distribution
    sorted_scores = sorted(all_journal_scores.items(), key=lambda x: -x[1])
    logger.info(f"Journal scores distribution: {sorted_scores}")


if __name__ == "__main__":
    fire.Fire(process_kg)
