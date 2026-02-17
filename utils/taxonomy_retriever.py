import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import pandas as pd
from abc import ABC, abstractmethod
from collections import namedtuple
import time

import taxopy
from tqdm import tqdm

# base feature retriever

"""
Base class for feature retrievers.

This module provides an abstract base class for all feature retrievers.
"""



ProteinFeatures = namedtuple("ProteinFeatures", ["identifier", "features"])


class BaseFeatureRetriever(ABC):
    """Abstract base class for all feature retrievers."""

    def __init__(self, headers: list[str] = None, features: list = None):
        """
        Initialize the retriever.

        Args:
            headers: List of protein identifiers/accessions
            features: List of features to retrieve (retriever-specific)
        """
        self.headers = headers if headers else []
        self.features = features

    @abstractmethod
    def fetch_features(self) -> list[ProteinFeatures]:
        """
        Fetch features from the data source.

        Returns:
            List of ProteinFeatures containing identifier and features dict

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement fetch_features()")


# uniprot id to taxon id
import requests

class UniProtTaxonomyRetriever(BaseFeatureRetriever):
    """Retrieves taxonomy lineages for UniProt protein accessions."""

    def __init__(self, uniprot_ids: list[str], features: list = None):
        """
        Args:
            uniprot_ids: List of UniProt accessions (e.g. ['P12345', 'Q9Y2T3'])
            features: Taxonomic ranks to retrieve (default = all)
        """
        super().__init__(headers=uniprot_ids, features=features or TAXONOMY_FEATURES)
        self.uniprot_ids = uniprot_ids
        self.tax_retriever = None  # Will be initialized once we have taxon IDs

    def fetch_features(self) -> dict[str, dict[str, Any]]:
        """
        Fetch taxonomy lineage information for each UniProt ID.
        """
        print(f"Processing {len(self.uniprot_ids)} UniProt IDs...")
        
        # Step 1. Get mapping: UniProt ID → taxon ID
        uniprot_to_taxid = self._get_taxon_ids_from_uniprot(self.uniprot_ids)
        
        n_with_taxid = sum(1 for tid in uniprot_to_taxid.values() if tid is not None)
        n_missing_taxid = len(uniprot_to_taxid) - n_with_taxid
        print(f"Retrieved taxon IDs: {n_with_taxid} successful, {n_missing_taxid} missing/not found.")

        # Step 2. Initialize the taxonomy retriever with the unique taxon IDs
        unique_taxids = list({tid for tid in uniprot_to_taxid.values() if tid is not None})
        print(f"Unique taxon IDs to look up: {len(unique_taxids)}")
        
        self.tax_retriever = TaxonomyRetriever(unique_taxids, features=self.features)
        tax_data = self.tax_retriever.fetch_features()

        # Step 3. Map taxonomy info back to UniProt IDs (include taxon_id)
        result = {}
        n_with_features = 0
        n_missing_features = 0
        
        for uid, taxid in uniprot_to_taxid.items():
            entry = {"taxon_id": taxid}
            if taxid in tax_data:
                entry["features"] = tax_data[taxid]["features"]
                # Check if features are non-empty (at least one field has a value)
                if any(v for v in entry["features"].values() if isinstance(v, str) and v):
                    n_with_features += 1
                else:
                    n_missing_features += 1
            else:
                entry["features"] = dict.fromkeys(self.features, "")
                n_missing_features += 1
            result[uid] = entry

        print(f"Taxonomy retrieval complete: {n_with_features} entries with taxonomy data, "
              f"{n_missing_features} entries with missing/empty taxonomy data.")
        
        return result

    def _get_taxon_ids_from_uniprot(self, uniprot_ids: list[str]) -> dict[str, int | None]:
        """
        Retrieve taxonomy IDs in small, resilient batches using UniProt's TSV search.
        No large local mapping files, no heavy POST jobs.
        """
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        batch_size = 100  # keep URL short & stable
        result: dict[str, int | None] = {}
        total_batches = (len(uniprot_ids) - 1) // batch_size + 1
        print(f"Fetching taxon IDs from UniProt in {total_batches} batches (batch size: {batch_size})...")

        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i + batch_size]
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            params = {
                "query": query,
                "fields": "accession,organism_id",
                "format": "tsv",
                "size": batch_size,
            }

            batch_num = i // batch_size + 1
            for attempt in range(3):
                try:
                    r = requests.get(base_url, params=params, timeout=30)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    logger.warning(f"Batch {batch_num}/{total_batches}: {e}, retrying...")
                    time.sleep(2 ** attempt)
            else:
                for uid in batch:
                    result[uid] = None
                tqdm.write(f"Batch {batch_num}/{total_batches}: Failed after 3 attempts, marking as missing.")
                continue

            lines = r.text.strip().splitlines()
            n_found_in_batch = 0
            for line in lines[1:]:  # Skip header
                if not line.strip():  # Skip empty lines
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    # Malformed line, skip it
                    continue
                acc, taxid = parts[0], parts[1]
                result[acc] = int(taxid) if taxid.isdigit() else None
                if result[acc] is not None:
                    n_found_in_batch += 1

            # fill any IDs not returned
            for uid in batch:
                result.setdefault(uid, None)

            tqdm.write(
                f"Batch {batch_num}/{total_batches}: Found {n_found_in_batch}/{len(batch)} taxon IDs"
            )
            time.sleep(0.3)  # polite pause

        return result


# taxonomy retriever
logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Taxonomy features
TAXONOMY_FEATURES = [
    "root",
    "domain",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
]


class TaxonomyRetriever(BaseFeatureRetriever):
    """Retrieves taxonomy lineage data from NCBI."""

    def __init__(self, taxon_ids: list[int], features: list = None):
        # Don't call super().__init__() as we use taxon_ids instead of headers
        self.taxon_ids = self._validate_taxon_ids(taxon_ids)
        self.features = features
        self.taxdb = self._initialize_taxdb()

    def fetch_features(self) -> dict[int, dict[str, Any]]:
        print(f"Fetching taxonomy features for {len(self.taxon_ids)} taxon IDs...")
        result = {}

        with tqdm(
            total=len(self.taxon_ids), desc="Fetching taxonomy features", unit="taxon"
        ) as pbar:
            taxonomies_info = self._get_taxonomy_info(self.taxon_ids)

            n_successful = 0
            n_failed = 0
            
            for taxon_id in self.taxon_ids:
                if taxon_id in taxonomies_info:
                    result[taxon_id] = {"features": taxonomies_info[taxon_id]}
                    # Check if we got valid data (not all empty)
                    features = taxonomies_info[taxon_id]
                    if any(v for k, v in features.items() if k != "tax_array" and v):
                        n_successful += 1
                    else:
                        n_failed += 1
                else:
                    result[taxon_id] = {"features": dict.fromkeys(self.features, "")}
                    n_failed += 1
                pbar.update(1)

        print(f"Taxonomy lookup complete: {n_successful} successful, {n_failed} failed/missing.")
        return result

    def _validate_taxon_ids(self, taxon_ids: list[int]) -> list[int]:
        for taxon_id in taxon_ids:
            if not isinstance(taxon_id, int):
                raise ValueError(f"Taxon ID {taxon_id} is not an integer")

        return taxon_ids

    def _get_taxonomy_info(self, taxon_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Return rank names, IDs, and a final array [phylum, class, order, family, genus, species] of numeric taxon IDs."""
        result = {}

        for taxon_id in taxon_ids:
            try:
                taxon = taxopy.Taxon(taxon_id, self.taxdb)

                rank_name_dict = taxon.rank_name_dictionary
                ranks = taxon.rank_lineage
                taxids = taxon.taxid_lineage
                rank_id_dict = {r: tid for r, tid in zip(ranks, taxids) if r}

                # Handle alternate roots/domains
                root_name = rank_name_dict.get("cellular root", "") or rank_name_dict.get("acellular root", "")
                root_id = rank_id_dict.get("cellular root", "") or rank_id_dict.get("acellular root", "")
                domain_name = rank_name_dict.get("domain", "") or rank_name_dict.get("realm", "")
                domain_id = rank_id_dict.get("domain", "") or rank_id_dict.get("realm", "")

                # Construct the complete info dictionary
                full_taxonomy_info = {
                    "root": root_name,
                    "root_id": root_id,
                    "domain": domain_name,
                    "domain_id": domain_id,
                    "kingdom": rank_name_dict.get("kingdom", ""),
                    "kingdom_id": rank_id_dict.get("kingdom", 0),
                    "phylum": rank_name_dict.get("phylum", ""),
                    "phylum_id": rank_id_dict.get("phylum", 0),
                    "class": rank_name_dict.get("class", ""),
                    "class_id": rank_id_dict.get("class", 0),
                    "order": rank_name_dict.get("order", ""),
                    "order_id": rank_id_dict.get("order", 0),
                    "family": rank_name_dict.get("family", ""),
                    "family_id": rank_id_dict.get("family", 0),
                    "genus": rank_name_dict.get("genus", ""),
                    "genus_id": rank_id_dict.get("genus", 0),
                    "species": rank_name_dict.get("species", ""),
                    "species_id": rank_id_dict.get("species", 0),
                }

                # Create numeric array [phylum_id, class_id, order_id, family_id, genus_id, species_id]
                ranks_of_interest = ["phylum", "class", "order", "family", "genus", "species"]
                tax_array = [int(rank_id_dict.get(r, 0)) for r in ranks_of_interest]
                full_taxonomy_info["tax_array"] = tax_array

                result[taxon_id] = full_taxonomy_info

            except Exception as e:
                logger.error(f"Failed to get taxonomy for {taxon_id}: {e}")
                empty_info = {f: "" for f in TAXONOMY_FEATURES}
                for f in TAXONOMY_FEATURES:
                    empty_info[f + "_id"] = 0
                empty_info["tax_array"] = [0, 0, 0, 0, 0, 0]
                result[taxon_id] = empty_info

        return result

    def _initialize_taxdb(self):
        # Allow overriding cache directory via environment variable
        # Defaults to ~/.cache/taxopy_db
        env_override = os.environ.get("PROTSPACE_TAXDB_DIR")
        db_dir = (
            Path(env_override).expanduser()
            if env_override
            else Path.home() / ".cache" / "taxopy_db"
        )
        db_dir.mkdir(parents=True, exist_ok=True)
        nodes_file = db_dir / "nodes.dmp"
        names_file = db_dir / "names.dmp"
        merged_file = db_dir / "merged.dmp"
        timestamp_file = db_dir / ".download_timestamp"

        # Determine if this is a first-time setup (missing required files)
        first_time_setup = not (nodes_file.exists() and names_file.exists())

        # Check if cache needs refresh based on timestamp file
        needs_refresh = False
        if timestamp_file.exists():
            try:
                with open(timestamp_file) as f:
                    download_time = datetime.fromisoformat(f.read().strip())
                one_week_ago = datetime.now() - timedelta(weeks=1)

                if download_time < one_week_ago:
                    logger.info(
                        "Your taxonomy dataset is more than one week old. Refreshing cache..."
                    )
                    print(
                        "Your taxonomy dataset is more than one week old. Refreshing cache..."
                    )
                    needs_refresh = True
            except (ValueError, OSError) as e:
                logger.warning(
                    f"Could not read timestamp file: {e}. Will refresh cache."
                )
                print(f"Could not read timestamp file: {e}. Will refresh cache.")
                needs_refresh = True
        else:
            # No timestamp file: if DB files exist, create timestamp; otherwise we need a fresh download
            if first_time_setup:
                needs_refresh = True
            else:
                try:
                    with open(timestamp_file, "w") as f:
                        f.write(datetime.now().isoformat())
                except OSError as e:
                    logger.warning(
                        f"Failed to create timestamp file at first-time detection: {e}"
                    )

        existing_db_present = nodes_file.exists() and names_file.exists()

        # Load or download the database with a safe refresh strategy
        if existing_db_present:
            if needs_refresh:
                logger.info(
                    "Taxonomy cache is stale. Attempting safe refresh without deleting existing cache."
                )
                temp_dir_path = None
                try:
                    # Download into a temporary directory first
                    temp_dir_path = Path(tempfile.mkdtemp(prefix="taxopy_tmp_"))
                    taxopy.TaxDb(taxdb_dir=str(temp_dir_path), keep_files=True)

                    # Move refreshed files into place atomically
                    for src_name, dst_path in [
                        ("nodes.dmp", nodes_file),
                        ("names.dmp", names_file),
                        ("merged.dmp", merged_file),
                    ]:
                        src_path = temp_dir_path / src_name
                        if src_path.exists():
                            # Replace destination with the new file
                            shutil.move(str(src_path), str(dst_path))

                    # Update timestamp only after a successful refresh
                    with open(timestamp_file, "w") as f:
                        f.write(datetime.now().isoformat())

                except Exception as e:
                    logger.warning(
                        f"Failed to refresh taxonomy database: {e}. Falling back to existing cached database."
                    )
                    # Fall back: keep using the existing DB files
                finally:
                    if temp_dir_path and temp_dir_path.exists():
                        shutil.rmtree(temp_dir_path, ignore_errors=True)

            # Load existing (potentially refreshed) DB files
            logger.info(f"Loading taxopy database from {db_dir}")
            try:
                taxdb = taxopy.TaxDb(
                    nodes_dmp=str(nodes_file),
                    names_dmp=str(names_file),
                    merged_dmp=str(merged_file) if merged_file.exists() else None,
                )
            except Exception as e:
                logger.error(
                    f"Failed to load existing taxonomy database from cache: {e}"
                )
                raise
        else:
            # First-time setup: must download
            logger.info(f"Downloading taxopy database to {db_dir}")
            try:
                taxdb = taxopy.TaxDb(taxdb_dir=str(db_dir), keep_files=True)
                # Create/update timestamp file after successful download
                with open(timestamp_file, "w") as f:
                    f.write(datetime.now().isoformat())
            except Exception as e:
                logger.error(
                    f"Failed to initialize taxopy database (first-time setup): {e}"
                )
                raise

        return taxdb


def annotate_csv_with_taxonomy(input_csv: str, output_csv: str):
    """
    Reads a CSV with column 'identifier' (UniProt IDs),
    fetches taxonomy data, and writes results to a new CSV.
    """
    print(f"\n{'='*60}")
    print(f"Taxonomy Annotation Pipeline")
    print(f"{'='*60}")
    print(f"Reading input CSV: {input_csv}")
    
    df = pd.read_csv(input_csv)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns.")

    if "identifier" not in df.columns:
        raise ValueError("CSV must contain a column named 'identifier' (UniProt ID).")

    uniprot_ids = df["identifier"].dropna().astype(str).tolist()
    print(f"Found {len(uniprot_ids)} non-null identifiers (out of {len(df)} total rows).")
    
    retriever = UniProtTaxonomyRetriever(uniprot_ids)
    taxonomy_results = retriever.fetch_features()

    # Convert the result dict to DataFrame (include taxon_id)
    tax_df = pd.DataFrame.from_dict(
        {
            uid: {**{"taxon_id": data["taxon_id"]}, **data["features"]}
            for uid, data in taxonomy_results.items()
        },
        orient="index"
    ).reset_index().rename(columns={"index": "identifier"})

    # Merge taxonomy info back into original CSV
    merged = df.merge(tax_df, on="identifier", how="left")
    
    # Statistics on final output
    n_with_taxid = merged["taxon_id"].notna().sum()
    n_with_species = merged["species"].notna().sum() if "species" in merged.columns else 0
    n_with_genus = merged["genus"].notna().sum() if "genus" in merged.columns else 0
    n_with_phylum = merged["phylum"].notna().sum() if "phylum" in merged.columns else 0
    
    print(f"\n{'='*60}")
    print(f"Final Statistics:")
    print(f"{'='*60}")
    print(f"Total rows in output: {len(merged)}")
    print(f"Rows with taxon_id: {n_with_taxid} ({100*n_with_taxid/len(merged):.1f}%)")
    print(f"Rows with species: {n_with_species} ({100*n_with_species/len(merged):.1f}%)")
    print(f"Rows with genus: {n_with_genus} ({100*n_with_genus/len(merged):.1f}%)")
    print(f"Rows with phylum: {n_with_phylum} ({100*n_with_phylum/len(merged):.1f}%)")
    
    merged.to_csv(output_csv, index=False)
    print(f"\n✅ Annotated CSV saved to {output_csv}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    input_csv = "./data/interm/training_data.csv"
    output_csv = "./data/tax/training_tax.csv"
    annotate_csv_with_taxonomy(input_csv, output_csv)