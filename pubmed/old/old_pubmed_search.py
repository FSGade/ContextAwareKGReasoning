from __future__ import annotations

import os
import pickle
import re
import sys
import time
from urllib.error import HTTPError

import numpy as np
from Bio import Entrez

# import xml.etree.ElementTree as et

MONTH_DICT = {
    "Jan": "01",
    "Feb": "02",
    "Mar": "03",
    "Apr": "04",
    "May": "05",
    "Jun": "06",
    "Jul": "07",
    "Aug": "08",
    "Sep": "09",
    "Oct": "10",
    "Nov": "11",
    "Dec": "12",
}

JOURNAL_TYPE_FILTER = 'AND "journal article"[pt] NOT (preprint[pt] OR review[pt] OR editorial[pt] OR "clinical trial protocol"[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR "letter"[pt] OR "comment"[pt])'
T2D_SPECIFICATION_FILTER = "AND (diabetes OR metabolism)"
GENERAL_SPECIFICATION_FILTER = 'AND ("protein" OR "gene" OR "pathway" OR "metabolite" OR "chemical" OR "phenotype")'


("inflammation") AND ("adipose tissue" OR "adipocyte" OR "fat tissue" OR "white adipose tissue" OR "brown adipose tissue" OR "subcutaneous fat" OR "visceral fat" OR "adipogenesis" OR "browning" OR "beiging") AND ("obesity" OR "diabetes" OR "metabolic syndrome") AND (2000:2023[edat]) NOT ("Review"[Publication Type])

("inflammation") AND ("adipose tissue" OR "adipocyte" OR "fat tissue" OR "white adipose tissue" OR "brown adipose tissue" OR "subcutaneous fat" OR "visceral fat" OR "adipogenesis" OR "browning" OR "beiging") AND ("obesity" OR "diabetes" OR "metabolic syndrome") AND (2000:2023[edat]) AND "journal article"[pt] NOT (preprint[pt] OR review[pt] OR editorial[pt] OR "clinical trial protocol"[pt] OR "systematic review"[pt] OR "meta-analysis"[pt] OR "letter"[pt] OR "comment"[pt])'


def search_keyphrase(
    keyphrase,
    key_index,
    corpus_name,
    max_count=1000,
    start_year=2023,
    end_year=2050,
    type_filter=True,
):
    # Entry Date [edat] - Date used for PubMed processing, such as “Most Recent” sort order.
    if type_filter:
        keyphrase = (
            f"{keyphrase} AND ({start_year}:{end_year}[edat]) {JOURNAL_TYPE_FILTER}"
        )
    else:
        keyphrase = f"{keyphrase} AND ({start_year}:{end_year}[edat])"

    # Do the search
    handle = Entrez.esearch(
        db="pubmed", term=keyphrase, retmax=max_count
    )  # default sort is relevance
    record = Entrez.read(handle)
    record_count = int(record["Count"])

    if record_count > max_count:
        keyphrase = f"{keyphrase} {GENERAL_SPECIFICATION_FILTER}"

        if "T2D" in corpus_name:
            keyphrase = f"{keyphrase} {T2D_SPECIFICATION_FILTER}"

        print(
            f"record_count exceeds max_count ({max_count}), retrying with a more scoped query ({keyphrase})."
        )

        handle = Entrez.esearch(db="pubmed", term=keyphrase, retmax=max_count)
        record = Entrez.read(handle)
        record_count = int(record["Count"])

        if record_count > max_count:
            print("The more scoped query also exceeded max_count. Fetching max_count.")
    elif record_count == 0:
        print("Empty ID list; Nothing to store")
        return []

    # Fetch details. If they were more than 10000 ids, use epost instead of efetch (https://stackoverflow.com/questions/46579694/biopythons-esearch-does-not-give-me-full-idlist).
    # So for everything we'll just use epost
    id_list = record["IdList"]
    post_xml = Entrez.epost("pubmed", id=",".join(id_list))

    try:
        search_results = Entrez.read(post_xml)
    except RuntimeError as e:
        print(f"An error occured while trying to fetch search results: {e}")
        sys.exit(1)

    # Download the entries now with efetch batch by batch
    webenv = search_results["WebEnv"]
    query_key = search_results["QueryKey"]

    BATCH_SIZE = 10000

    loop_max = min(max_count, record_count)

    corpus_list = []  # pd.DataFrame(columns = ['Title', 'PMID', 'DOI', 'AbstractID',
    #                        'PublicationTypeList', 'JournalArticle', 'Review',
    #                        'RCT', 'Languages', 'Year', 'Month', 'Day'])

    for start in range(0, loop_max, BATCH_SIZE):
        end = min(loop_max, start + BATCH_SIZE)

        entrez_filename = f"{corpus_name}/keyphrase{key_index}_{start + 1}_{end}.pkl"

        if os.path.isfile(entrez_filename):
            print(f"Loading record {start + 1} to {end}. Already downloaded")
            with open(entrez_filename, "rb") as pkl:
                data = pickle.load(pkl)
        else:
            print(f"Downloading record {start + 1} to {end}")
            # Batch-wise fetch and processing
            attempt = 0
            while attempt < 3:
                attempt += 1
                try:
                    print(f"Attempt {attempt} of 3")
                    fetch_handle = Entrez.efetch(
                        db="pubmed",
                        retstart=start,
                        retmax=BATCH_SIZE,
                        webenv=webenv,
                        query_key=query_key,
                        retmode="xml",
                    )
                except HTTPError as err:
                    if 500 <= err.code <= 599:
                        print(f"Received error from server {err}")
                        time.sleep(15)
                    else:
                        raise
                else:
                    attempt = 3

            data = Entrez.read(fetch_handle)
            fetch_handle.close()

        for i, article_data in enumerate(data["PubmedArticle"]):
            article = article_data["MedlineCitation"]["Article"]

            article_title = article["ArticleTitle"]
            article_title = re.sub("<[^>]*>", "", article_title)

            article_elocations = article["ELocationID"]
            article_doi = np.nan
            for eloc in article_elocations:
                if eloc.attributes["EIdType"] == "doi":
                    article_doi = str(eloc)

            article_pmid = str(article_data["MedlineCitation"]["PMID"])
            article_languages = article["Language"]
            # journal_medline_info = article_data['MedlineCitation']['MedlineJournalInfo']
            journal = article["Journal"]
            if "ISSN" in journal:
                journal_issn = str(journal["ISSN"])
            else:
                journal_issn = ""
            journal_title = journal["Title"]

            ## Article type
            article_type_ui = [
                pub_type_element.attributes["UI"]
                for pub_type_element in article["PublicationTypeList"]
            ]
            article_type = [
                str(pub_type_element)
                for pub_type_element in article["PublicationTypeList"]
            ]

            pub_type_journal_article = False
            pub_type_review = False
            pub_type_rct = False
            pub_type_editorial_letter_or_comment = False

            editorial_letter_or_comment_uis = {
                "D016420",  # Comment
                "D016421",  # Editorial
                "D016422",  # Letter
            }

            if "D016428" in article_type_ui:
                pub_type_journal_article = True
            if "D016454" in article_type_ui:
                pub_type_review = True
            if "D016449" in article_type_ui:
                pub_type_rct = True
            if editorial_letter_or_comment_uis & set(article_type_ui):  # if intersect
                pub_type_editorial_letter_or_comment = True

            article_date = article["ArticleDate"]
            if article_date == []:
                article_date = article["Journal"]["JournalIssue"]["PubDate"]
                if "Month" in article_date and article_date["Month"] in MONTH_DICT:
                    article_date["Month"] = MONTH_DICT[article_date["Month"]]
            else:
                article_date = dict(article_date[0])

            if "Year" not in article_date:
                article_date["Year"] = np.nan
            if "Month" not in article_date:
                article_date["Month"] = np.nan
            if "Day" not in article_date:
                article_date["Day"] = np.nan

            ## Parse abstracts
            abstract_id = -1

            if "Abstract" in article:
                abstract_parts = []
                # TODO: Consider removing select parts such as "ETHICS AND DISSEMINATION"
                for abstract_part in article["Abstract"]["AbstractText"]:
                    if (
                        hasattr(abstract_part, "attributes")
                        and "NlmCategory" in abstract_part.attributes
                    ):
                        # if abstract_part.attributes['NlmCategory'] == "UNASSIGNED":
                        #     part_prefix = abstract_part.attributes['Label']
                        # else:
                        #     part_prefix = abstract_part.attributes['NlmCategory']
                        # abstract_parts.append(part_prefix + ": " + str(abstract_part))
                        abstract_parts.append(
                            abstract_part.attributes["Label"]
                            + ": "
                            + str(abstract_part)
                        )
                    else:
                        abstract_parts.append(str(abstract_part))
                article_abstract = "\n".join(abstract_parts)
                article_abstract = re.sub("<[^>]*>", "", article_abstract)

                abstract_id = i + start

                with open(
                    f"{corpus_name}/abstracts/{key_index}_{abstract_id}.txt",
                    "w",
                    encoding="utf-8",
                ) as f:
                    print(article_abstract, file=f)

            corpus_list.append(
                [
                    article_title,  # Title
                    article_pmid,  # PMID
                    article_doi,  # DOI
                    f"{key_index}_{abstract_id}",  # AbstractID
                    article_type,  # PublicationTypeList
                    journal_issn,  # ISSN
                    journal_title,
                    pub_type_journal_article,  # JournalArticle
                    pub_type_review,  # Review
                    pub_type_rct,  # RCT
                    pub_type_editorial_letter_or_comment,  # EditorialLetterOrComment
                    article_languages,  # Languages
                    article_date["Year"],  # Year
                    article_date["Month"],  # Month
                    article_date["Day"],  # Day
                ]
            )

        with open(
            f"{corpus_name}/keyphrase{key_index}_{start + 1}_{end}.pkl", "wb"
        ) as pkl:
            pickle.dump(data, pkl)

    return corpus_list
