#!/usr/bin/env python3
# https://github.com/MLZM-lab/NLPamr/blob/master/1-DataRetrieval.py
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
from Bio import Entrez
from utils.pubmed_search import search_keyphrase

############# Entrez part #########
Entrez.email = "email@example.com"

KEYWORD_PATH = "keyword/path/"
REVIEW_PATH = "review/path/"

parser = argparse.ArgumentParser(
    prog="AbstractRetriever",
    description="Retrieves abstracts from PubMed using list of keyphrases.",
)

parser.add_argument("corpus_name")  # option that takes a value
parser.add_argument("--start_year", default=2023, type=int)
parser.add_argument("--end_year", default=2050, type=int)
parser.add_argument("--from_reviews", action="store_true")

args = parser.parse_args()

CORPUS_NAME = args.corpus_name  # "DiseasesPre2023"
FROM_REVIEWS = args.from_reviews  # False
START_YEAR = args.start_year  # 2010
END_YEAR = args.end_year  # 2023

try:
    os.mkdir(CORPUS_NAME)
    os.mkdir(f"{CORPUS_NAME}/abstracts")
    os.mkdir(f"{CORPUS_NAME}/kg_information")
except FileExistsError:
    print("WARNING: Output directory already exists.")
except OSError as e:
    print(f"An unsolved error occured when creating output folder: {e}. Exiting.")
    sys.exit(1)

if FROM_REVIEWS:
    review_article_paths = glob.glob(f"{REVIEW_PATH}/*.pdf")
    review_article_ids = [
        os.path.basename(review_article_path).replace(".pdf", "")
        for review_article_path in review_article_paths
    ]

    keyphrases_all = []
    for review_id in review_article_ids:
        with open(f"{REVIEW_PATH}/{review_id}.txt", encoding="utf-8") as f:
            keyphrases_all.append(f.read().strip().split("\n"))

    # Merge keyphrase lists
    from itertools import chain

    keyphrases = list(set(chain(*keyphrases_all)))
else:
    with open(KEYWORD_PATH, encoding="utf-8") as f:
        keyphrases = [line.strip() for line in f]
    keyphrases = list(dict.fromkeys(keyphrases))  # Keeps ordering as opposed to set.

corpus_list = []

for key_index, keyphrase in enumerate(keyphrases):
    print(f"{keyphrase} ({key_index})")
    corpus_list.extend(
        search_keyphrase(
            keyphrase,
            key_index,
            CORPUS_NAME,
            start_year=START_YEAR,
            end_year=END_YEAR,
        )
    )

corpus_df = pd.DataFrame(
    corpus_list,
    columns=[
        "Title",
        "PMID",
        "DOI",
        "AbstractID",
        "PublicationTypeList",
        "ISSN",
        "Journal",
        "JournalArticle",
        "Review",
        "RCT",
        "EditorialLetterOrComment",
        "Languages",
        "Year",
        "Month",
        "Day",
    ],
)
 

corpus_df.drop_duplicates("PMID", inplace=True)
corpus_df.drop_duplicates("DOI", inplace=True)
corpus_df = corpus_df[~corpus_df["AbstractID"].str.contains("-1")]
corpus_df.to_csv(f"{CORPUS_NAME}/corpus.tsv", sep="\t", index=False)
corpus_df.to_pickle(f"{CORPUS_NAME}/corpus.pkl")
