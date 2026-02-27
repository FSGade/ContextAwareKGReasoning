"""PubMed cache."""

import logging
import sqlite3
import time
from datetime import datetime

from Bio import Entrez


class PubMedBatchCache:
    """PubMed batch cache."""

    def __init__(
        self,
        db_path="pubmed_abstracts.db",
        email="example@example.com",
    ):
        """PubMedBatchCache.

        Args:
            db_path (str, optional): _description_. Defaults to "pubmed_abstracts.db".
            email (str, optional): _description_. Defaults to "example@example.com".

        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._create_table()

        # Configure Entrez
        Entrez.email = email
        Entrez.api_key = None  # Add your NCBI API key if you have one

    def _create_table(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS abstracts (
                pmid TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,
                journal TEXT,
                issn TEXT,
                pub_date TEXT,
                doi TEXT,
                date_fetched TEXT,
                fetch_success BOOLEAN
            )
        """
        )
        self.conn.commit()

    def fetch_batch(self, pmids: list[str], batch_size: int = 200, rate_limiting: float = 0.4):
        """Fetch abstracts in batches to respect rate limits."""
        missing_pmids = self.get_missing_pmids(pmids)

        if not missing_pmids:
            logging.info("All PMIDs already cached")
            return

        logging.info(f"Fetching {len(missing_pmids)} missing abstracts")

        for i in range(0, len(missing_pmids), batch_size):
            batch = missing_pmids[i : i + batch_size]
            self._fetch_and_store_batch(batch)

            # Rate limiting: 3 requests/second without API key
            time.sleep(rate_limiting)  # Conservative rate limiting

    def _fetch_and_store_batch(self, pmids: list[str]):
        """Fetch a single batch and store in database."""
        try:
            # Fetch summaries
            handle = Entrez.efetch(
                db="pubmed", id=pmids, rettype="medline", retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            # Process and store each record
            for record in records["PubmedArticle"]:
                self._store_record(record)

        except Exception as e:
            logging.exception(f"Error fetching batch {pmids[:3]}...: {e}")
            # Store failed attempts to avoid re-fetching
            for pmid in pmids:
                self._store_failed_record(pmid)

    def _extract_authors(self, article) -> str:
        """Extract author list as comma-separated string."""
        try:
            authors = []
            if "AuthorList" in article:
                for author in article["AuthorList"]:
                    if "LastName" in author and "ForeName" in author:
                        authors.append(f"{author['LastName']}, {author['ForeName']}")
                    elif "LastName" in author:
                        authors.append(author["LastName"])
                    elif "CollectiveName" in author:
                        authors.append(author["CollectiveName"])
            return "; ".join(authors)
        except Exception as e:
            logging.warning(f"Error extracting authors: {e}")
            return ""

    def _extract_date(self, article) -> str:
        """Extract publication date."""
        try:
            # Try different date fields in order of preference
            date_fields = [
                ("Journal", "JournalIssue", "PubDate"),
                ("ArticleDate",),
            ]

            for field_path in date_fields:
                current = article
                for field in field_path:
                    if field in current:
                        current = current[field]
                    else:
                        break
                else:
                    # Successfully navigated the path
                    if isinstance(current, list) and len(current) > 0:
                        current = current[0]  # Take first date if multiple

                    if not isinstance(current, dict):
                        break

                    # Extract date components
                    year = current.get("Year", "")
                    month = current.get("Month", "")
                    day = current.get("Day", "")

                    # Format date
                    date_parts = [year, month, day]
                    date_parts = [str(part) for part in date_parts if part]
                    if date_parts:
                        return "-".join(date_parts)

            return ""
        except Exception as e:
            logging.warning(article)
            logging.warning(f"Error extracting date: {e}")
            return ""

    def _extract_doi(self, record) -> str:
        """Extract DOI from article identifiers."""
        try:
            # Check PubmedData section first
            if "PubmedData" in record:
                article_ids = record["PubmedData"].get("ArticleIdList", [])
                for article_id in article_ids:
                    if (
                        hasattr(article_id, "attributes")
                        and article_id.attributes.get("IdType") == "doi"
                    ):
                        return str(article_id)

            # Check MedlineCitation section
            article = record.get("MedlineCitation", {}).get("Article", {})
            if "ELocationID" in article:
                elocation_ids = article["ELocationID"]
                if not isinstance(elocation_ids, list):
                    elocation_ids = [elocation_ids]

                for elocation_id in elocation_ids:
                    if (
                        hasattr(elocation_id, "attributes")
                        and elocation_id.attributes.get("EIdType") == "doi"
                    ):
                        return str(elocation_id)

            return ""
        except Exception as e:
            logging.warning(f"Error extracting DOI: {e}")
            return ""

    def _store_failed_record(self, pmid: str):
        """Store a record that failed to fetch."""
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO abstracts
                (pmid, title, abstract, authors, journal, issn, pub_date, doi, date_fetched, fetch_success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pmid,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    datetime.now().isoformat(),
                    False,
                ),
            )
            self.conn.commit()
        except Exception as e:
            logging.exception(f"Error storing failed record for PMID {pmid}: {e}")

    def _store_record(self, record):
        """Extract and store record data."""
        try:
            pmid = str(record["MedlineCitation"]["PMID"])
            article = record["MedlineCitation"]["Article"]

            # Extract fields
            title = article.get("ArticleTitle", "")
            abstract = self._extract_abstract(article)
            authors = self._extract_authors(article)
            journal = article.get("Journal", {}).get("Title", "")
            issn = article.get("Journal", {}).get("ISSN", "")
            pub_date = self._extract_date(article)
            doi = self._extract_doi(record)

            # Store in database
            self.conn.execute(
                """
                INSERT OR REPLACE INTO abstracts
                (pmid, title, abstract, authors, journal, issn, pub_date, doi, date_fetched, fetch_success)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pmid,
                    title,
                    abstract,
                    authors,
                    journal,
                    issn,
                    pub_date,
                    doi,
                    datetime.now().isoformat(),
                    True,
                ),
            )

            self.conn.commit()

        except Exception as e:
            logging.exception(f"Error storing record: {e}")

    def _extract_abstract(self, article) -> str:
        """Extract abstract text with structured sections."""
        try:
            abstract_parts = []
            if "Abstract" in article and "AbstractText" in article["Abstract"]:
                abstract_texts = article["Abstract"]["AbstractText"]

                # Handle single abstract text or list
                if not isinstance(abstract_texts, list):
                    abstract_texts = [abstract_texts]

                for abstract_text in abstract_texts:
                    text_content = str(abstract_text)

                    # Check if it has a label (structured abstract)
                    if (
                        hasattr(abstract_text, "attributes")
                        and "Label" in abstract_text.attributes
                    ):
                        label = abstract_text.attributes["Label"]
                        abstract_parts.append(f"{label}: {text_content}")
                    else:
                        abstract_parts.append(text_content)

            return " ".join(abstract_parts)
        except Exception as e:
            logging.warning(f"Error extracting abstract: {e}")
            return ""

    def get_abstracts(self, pmids: list[str]) -> dict[str, dict]:
        """Get abstracts for given PMIDs."""
        if not pmids:
            return {}

        placeholders = ",".join("?" * len(pmids))
        cursor = self.conn.execute(
            f"""
            SELECT * FROM abstracts
            WHERE pmid IN ({placeholders}) AND fetch_success = 1
        """,
            tuple(pmids),
        )  # Convert to tuple

        return {row["pmid"]: dict(row) for row in cursor.fetchall()}

    # def get_missing_pmids(self, pmids: list[str]) -> list[str]:
    #     """Return PMIDs that aren't in the cache."""
    #     if not pmids:
    #         return []

    #     placeholders = ",".join("?" * len(pmids))
    #     cursor = self.conn.execute(
    #         f"SELECT pmid FROM abstracts WHERE pmid IN ({placeholders})",
    #         tuple(pmids),  # Convert to tuple
    #     )
    #     cached_pmids = {row["pmid"] for row in cursor.fetchall()}
    #     return [pmid for pmid in pmids if pmid not in cached_pmids]
    
    def get_missing_pmids(self, pmids: list[str]) -> list[str]:
        """Return PMIDs that aren't in the cache."""
        if not pmids:
            return []
        
        # SQLite has a limit of 999 variables per query
        # Split into batches to avoid "too many SQL variables" error
        BATCH_SIZE = 900  # Use 900 to be safe (limit is 999)
        
        cached_pmids = set()
        
        # Process in batches
        for i in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[i:i + BATCH_SIZE]
            placeholders = ",".join("?" * len(batch))
            cursor = self.conn.execute(
                f"SELECT pmid FROM abstracts WHERE pmid IN ({placeholders})",
                tuple(batch),
            )
            cached_pmids.update(row["pmid"] for row in cursor.fetchall())
        
        # Return PMIDs not found in cache
        return [pmid for pmid in pmids if pmid not in cached_pmids]

    def close(self):
        """Close connection."""
        self.conn.close()
