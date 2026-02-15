"""Taxonomy retrieval and binary vector generation.

Merges functionality from:
- utils/taxonomy_retriever.py (UniProt ID -> NCBI taxonomy lineage)
- utils/taxonomy_analysis.py (taxonomy CSV -> binary vectors -> HDF5)
"""

from __future__ import annotations

import ast
import logging
import os
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import requests
import taxopy
from tqdm import tqdm

logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------- Constants ----------

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

TAXA = [
    "Porifera",
    "Calcarea",
    "Demospongiae",
    "Hexactinellida",
    "Ctenophora",
    "Nuda",
    "Tentaculata",
    "Cnidaria",
    "Metazoa",
    "Spiralia (Gnathifera)",
    "Rotifera",
    "Annelida",
    "Glyceridae",
    "Bryozoa",
    "Gymnolaemata",
    "Phylactolaemata",
    "Stenolaemata",
    "Mollusca",
    "Bivalvia",
    "Cephalopoda",
    "Octopoda",
    "Neogastropoda",
    "Conoidea",
    "Polyplacophora",
    "Scaphopoda",
    "Echinodermata",
    "Asteroidea",
    "Crinoidea",
    "Echinoidea",
    "Holothuroidea",
    "Ophiuroidea",
    "Panarthropoda",
    "Onychophora",
    "Tardigrada",
    "Myriapoda",
    "Chilopoda",
    "Diplopoda",
    "Arachnida",
    "Araneae",
    "Pseudoscorpiones",
    "Scorpiones",
    "Insecta",
    "Hymenoptera",
    "Chordata",
    "Aves",
    "Chondrichthyes",
    "Dasyatidae",
    "Actinopterygii",
    "Scorpaenoidei",
    "Trachinidae",
    "Reptilia",
    "Squamata",
    "Toxicofera",
    "Mammalia",
    "Solenodontidae",
    "Soricidae",
]

# ---------- Base classes ----------

ProteinFeatures = namedtuple("ProteinFeatures", ["identifier", "features"])


class BaseFeatureRetriever(ABC):
    def __init__(self, headers: list[str] | None = None, features: list | None = None):
        self.headers = headers if headers else []
        self.features = features

    @abstractmethod
    def fetch_features(self) -> list[ProteinFeatures]:
        raise NotImplementedError("Subclasses must implement fetch_features()")


# ---------- Taxonomy retrieval ----------


class UniProtTaxonomyRetriever(BaseFeatureRetriever):
    """Retrieves taxonomy lineages for UniProt protein accessions."""

    def __init__(self, uniprot_ids: list[str], features: list | None = None):
        super().__init__(headers=uniprot_ids, features=features or TAXONOMY_FEATURES)
        self.uniprot_ids = uniprot_ids
        self.tax_retriever = None

    def fetch_features(self) -> dict[str, dict[str, Any]]:
        uniprot_to_taxid = self._get_taxon_ids_from_uniprot(self.uniprot_ids)

        unique_taxids = list(
            {tid for tid in uniprot_to_taxid.values() if tid is not None}
        )
        self.tax_retriever = TaxonomyRetriever(unique_taxids, features=self.features)
        tax_data = self.tax_retriever.fetch_features()

        result = {}
        for uid, taxid in uniprot_to_taxid.items():
            entry = {"taxon_id": taxid}
            if taxid in tax_data:
                entry["features"] = tax_data[taxid]["features"]
            else:
                entry["features"] = dict.fromkeys(self.features, "")
            result[uid] = entry

        return result

    def _get_taxon_ids_from_uniprot(
        self, uniprot_ids: list[str]
    ) -> dict[str, int | None]:
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        batch_size = 100
        result: dict[str, int | None] = {}

        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i : i + batch_size]
            query = " OR ".join(f"accession:{uid}" for uid in batch)
            params = {
                "query": query,
                "fields": "accession,organism_id",
                "format": "tsv",
                "size": batch_size,
            }

            for attempt in range(3):
                try:
                    r = requests.get(base_url, params=params, timeout=30)
                    r.raise_for_status()
                    break
                except requests.RequestException as e:
                    logger.warning(
                        f"Batch {i // batch_size + 1}: {e}, retrying..."
                    )
                    time.sleep(2**attempt)
            else:
                for uid in batch:
                    result[uid] = None
                continue

            lines = r.text.strip().splitlines()
            for line in lines[1:]:
                acc, taxid = line.split("\t")
                result[acc] = int(taxid) if taxid.isdigit() else None

            for uid in batch:
                result.setdefault(uid, None)

            tqdm.write(
                f"Fetched {len(batch)} "
                f"(batch {i // batch_size + 1}/"
                f"{(len(uniprot_ids) - 1) // batch_size + 1})"
            )
            time.sleep(0.3)

        return result


class TaxonomyRetriever(BaseFeatureRetriever):
    """Retrieves taxonomy lineage data from NCBI."""

    def __init__(self, taxon_ids: list[int], features: list | None = None):
        self.taxon_ids = self._validate_taxon_ids(taxon_ids)
        self.features = features
        self.taxdb = self._initialize_taxdb()

    def fetch_features(self) -> dict[int, dict[str, Any]]:
        result = {}

        with tqdm(
            total=len(self.taxon_ids), desc="Fetching taxonomy features", unit="taxon"
        ) as pbar:
            taxonomies_info = self._get_taxonomy_info(self.taxon_ids)

            for taxon_id in self.taxon_ids:
                if taxon_id in taxonomies_info:
                    result[taxon_id] = {"features": taxonomies_info[taxon_id]}
                else:
                    result[taxon_id] = {
                        "features": dict.fromkeys(self.features, "")
                    }
                pbar.update(1)

        return result

    def _validate_taxon_ids(self, taxon_ids: list[int]) -> list[int]:
        for taxon_id in taxon_ids:
            if not isinstance(taxon_id, int):
                raise ValueError(f"Taxon ID {taxon_id} is not an integer")
        return taxon_ids

    def _get_taxonomy_info(self, taxon_ids: list[int]) -> dict[int, dict[str, Any]]:
        result = {}

        for taxon_id in taxon_ids:
            try:
                taxon = taxopy.Taxon(taxon_id, self.taxdb)

                rank_name_dict = taxon.rank_name_dictionary
                ranks = taxon.rank_lineage
                taxids = taxon.taxid_lineage
                rank_id_dict = {r: tid for r, tid in zip(ranks, taxids) if r}

                root_name = rank_name_dict.get(
                    "cellular root", ""
                ) or rank_name_dict.get("acellular root", "")
                root_id = rank_id_dict.get(
                    "cellular root", ""
                ) or rank_id_dict.get("acellular root", "")
                domain_name = rank_name_dict.get(
                    "domain", ""
                ) or rank_name_dict.get("realm", "")
                domain_id = rank_id_dict.get(
                    "domain", ""
                ) or rank_id_dict.get("realm", "")

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

                ranks_of_interest = [
                    "phylum",
                    "class",
                    "order",
                    "family",
                    "genus",
                    "species",
                ]
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

        first_time_setup = not (nodes_file.exists() and names_file.exists())

        needs_refresh = False
        if timestamp_file.exists():
            try:
                with open(timestamp_file) as f:
                    download_time = datetime.fromisoformat(f.read().strip())
                one_week_ago = datetime.now() - timedelta(weeks=1)

                if download_time < one_week_ago:
                    logger.info(
                        "Your taxonomy dataset is more than one week old. "
                        "Refreshing cache..."
                    )
                    needs_refresh = True
            except (ValueError, OSError) as e:
                logger.warning(
                    f"Could not read timestamp file: {e}. Will refresh cache."
                )
                needs_refresh = True
        else:
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

        if existing_db_present:
            if needs_refresh:
                logger.info(
                    "Taxonomy cache is stale. Attempting safe refresh."
                )
                temp_dir_path = None
                try:
                    temp_dir_path = Path(tempfile.mkdtemp(prefix="taxopy_tmp_"))
                    taxopy.TaxDb(taxdb_dir=str(temp_dir_path), keep_files=True)

                    for src_name, dst_path in [
                        ("nodes.dmp", nodes_file),
                        ("names.dmp", names_file),
                        ("merged.dmp", merged_file),
                    ]:
                        src_path = temp_dir_path / src_name
                        if src_path.exists():
                            shutil.move(str(src_path), str(dst_path))

                    with open(timestamp_file, "w") as f:
                        f.write(datetime.now().isoformat())

                except Exception as e:
                    logger.warning(
                        f"Failed to refresh taxonomy database: {e}. "
                        f"Falling back to existing cached database."
                    )
                finally:
                    if temp_dir_path and temp_dir_path.exists():
                        shutil.rmtree(temp_dir_path, ignore_errors=True)

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
            logger.info(f"Downloading taxopy database to {db_dir}")
            try:
                taxdb = taxopy.TaxDb(taxdb_dir=str(db_dir), keep_files=True)
                with open(timestamp_file, "w") as f:
                    f.write(datetime.now().isoformat())
            except Exception as e:
                logger.error(
                    f"Failed to initialize taxopy database (first-time setup): {e}"
                )
                raise

        return taxdb


# ---------- CSV annotation ----------


def annotate_csv_with_taxonomy(input_csv: str | Path, output_csv: str | Path) -> None:
    """Read a CSV with 'identifier' column, fetch taxonomy, write annotated CSV."""
    df = pd.read_csv(input_csv)

    if "identifier" not in df.columns:
        raise ValueError(
            "CSV must contain a column named 'identifier' (UniProt ID)."
        )

    uniprot_ids = df["identifier"].dropna().astype(str).tolist()
    retriever = UniProtTaxonomyRetriever(uniprot_ids)
    taxonomy_results = retriever.fetch_features()

    tax_df = (
        pd.DataFrame.from_dict(
            {
                uid: {**{"taxon_id": data["taxon_id"]}, **data["features"]}
                for uid, data in taxonomy_results.items()
            },
            orient="index",
        )
        .reset_index()
        .rename(columns={"index": "identifier"})
    )

    merged = df.merge(tax_df, on="identifier", how="left")
    merged.to_csv(output_csv, index=False)
    print(f"\nAnnotated CSV saved to {output_csv}")


# ---------- Binary taxonomy vectors ----------


def build_binary_tax_dict(
    detailed_tax_csv_path: str | Path,
    id_col: str = "identifier",
) -> dict[str, np.ndarray]:
    """Build dict: identifier -> np.array of 0/1 (len = len(TAXA))."""
    df = pd.read_csv(detailed_tax_csv_path)

    tax_cols = [
        "domain",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]

    missing = [c for c in tax_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing taxonomy columns in {detailed_tax_csv_path}: {missing}"
        )

    for c in tax_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    taxa_norm = [t.strip().lower() for t in TAXA]

    for original_name, norm_name in zip(TAXA, taxa_norm):
        df[original_name] = (
            (df[tax_cols] == norm_name).any(axis=1).astype(np.float32)
        )

    tax_dict = {}
    for _, row in df.iterrows():
        identifier = row[id_col]
        tax_array = row[TAXA].to_numpy(dtype=np.float32)
        tax_dict[identifier] = tax_array

    print(f"Built binary taxonomy dict for {len(tax_dict)} identifiers")
    print(f"Binary taxonomy vector length: {len(TAXA)}")
    return tax_dict


def run_binary_taxonomy_pipeline(
    tax_csv_path: str | Path,
    input_h5_path: str | Path,
    output_h5_path: str | Path,
    id_col: str = "identifier",
) -> None:
    """Create binary taxonomy vectors in a separate H5 file."""
    tax_dict = build_binary_tax_dict(tax_csv_path, id_col=id_col)
    vec_len = len(TAXA)

    with h5py.File(input_h5_path, "r") as f_in, h5py.File(
        output_h5_path, "w"
    ) as f_out:
        total_entries = len(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids = []

        for i, protein_id in enumerate(f_in.keys()):
            if protein_id in tax_dict:
                vec = tax_dict[protein_id].astype(np.float32)
                matched += 1
            else:
                vec = np.zeros(vec_len, dtype=np.float32)
                unmatched += 1
                unmatched_ids.append(protein_id)

            f_out.create_dataset(protein_id, data=vec)

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{total_entries} entries...")

        print(f"\nProcessing complete! (binary one-hot taxonomy)")
        print(f"Total entries (proteins): {total_entries}")
        print(f"Matched with taxonomy: {matched}")
        print(f"Unmatched (all-zero vector): {unmatched}")

        if unmatched > 0:
            print(f"\nFirst 10 unmatched IDs: {unmatched_ids[:10]}")

    print(f"\nBinary taxonomy pipeline finished.")
    print(f"Output file (only one-hot vectors): {output_h5_path}")


def run_numeric_taxonomy_pipeline(
    tax_csv_path: str | Path,
    input_h5_path: str | Path,
    output_h5_path: str | Path,
    normalize: bool = True,
) -> None:
    """Append numeric taxonomy vectors to embeddings."""
    tax = pd.read_csv(tax_csv_path)

    tax["tax_array"] = tax["tax_array"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    tax_dict = dict(zip(tax["identifier"], tax["tax_array"]))

    if len(tax_dict) > 0:
        example_array = next(iter(tax_dict.values()))
        tax_vec_len = len(example_array)
    else:
        tax_vec_len = 0

    normalize_taxonomy = None
    if normalize and tax_vec_len > 0:
        all_tax_values = []
        for tax_array in tax_dict.values():
            all_tax_values.extend(tax_array)

        tax_min = min(all_tax_values)
        tax_max = max(all_tax_values)

        print(f"Taxonomy value range: [{tax_min}, {tax_max}]")
        print("Will normalize to range: [-2, 2]\n")

        def normalize_taxonomy(tax_array):
            tax_array = np.array(tax_array, dtype=np.float32)
            normalized = (tax_array - tax_min) / (tax_max - tax_min)
            normalized = normalized * 4 - 2
            return normalized

    else:
        print("Taxonomy normalization: DISABLED\n")

    print(f"Loaded {len(tax_dict)} numeric taxonomy entries")

    with h5py.File(input_h5_path, "r") as f_in, h5py.File(
        output_h5_path, "w"
    ) as f_out:
        total_entries = len(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids = []

        for i, protein_id in enumerate(f_in.keys()):
            embedding = f_in[protein_id][:]

            if protein_id in tax_dict:
                tax_array = tax_dict[protein_id]

                if normalize_taxonomy is not None:
                    tax_array = normalize_taxonomy(tax_array)
                else:
                    tax_array = np.array(tax_array, dtype=embedding.dtype)

                combined = np.concatenate([embedding, tax_array])
                matched += 1
            else:
                tax_array = np.zeros(tax_vec_len, dtype=embedding.dtype)
                combined = np.concatenate([embedding, tax_array])
                unmatched += 1
                unmatched_ids.append(protein_id)

            f_out.create_dataset(protein_id, data=combined)

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1}/{total_entries} entries...")

        print(f"\nProcessing complete! (numeric taxonomy)")
        print(f"Total entries: {total_entries}")
        print(f"Matched with taxonomy: {matched}")
        print(f"Unmatched (filled with zeros): {unmatched}")
        print(f"Normalization applied: {normalize and (normalize_taxonomy is not None)}")

        if unmatched > 0:
            print(f"\nFirst 10 unmatched IDs: {unmatched_ids[:10]}")

    print("\nNumeric taxonomy pipeline finished.")
