"""Taxonomy lineage resolution and binary vector generation.

Reads NCBI taxon IDs (already present in the training CSV as ``Organism (ID)``),
resolves full lineage via taxopy, and encodes membership in 56 predefined animal
taxa as binary (one-hot) vectors stored in HDF5.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
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

# ---------- Taxonomy retrieval ----------


class TaxonomyRetriever:
    """Resolves NCBI taxonomy lineage for a list of taxon IDs using taxopy."""

    def __init__(self, taxon_ids: list[int], features: list | None = None):
        self.taxon_ids = self._validate_taxon_ids(taxon_ids)
        self.features = features or TAXONOMY_FEATURES
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
                    result[taxon_id] = {"features": dict.fromkeys(self.features, "")}
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
                root_name = rank_name_dict.get(
                    "cellular root", ""
                ) or rank_name_dict.get("acellular root", "")
                domain_name = rank_name_dict.get("domain", "") or rank_name_dict.get(
                    "realm", ""
                )

                full_taxonomy_info = {
                    "root": root_name,
                    "domain": domain_name,
                    "kingdom": rank_name_dict.get("kingdom", ""),
                    "phylum": rank_name_dict.get("phylum", ""),
                    "class": rank_name_dict.get("class", ""),
                    "order": rank_name_dict.get("order", ""),
                    "family": rank_name_dict.get("family", ""),
                    "genus": rank_name_dict.get("genus", ""),
                    "species": rank_name_dict.get("species", ""),
                }

                result[taxon_id] = full_taxonomy_info

            except Exception as e:
                logger.error(f"Failed to get taxonomy for {taxon_id}: {e}")
                result[taxon_id] = {f: "" for f in TAXONOMY_FEATURES}

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
                logger.info("Taxonomy cache is stale. Attempting safe refresh.")
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


# ---------- Binary taxonomy vectors ----------


def _build_binary_vectors(
    df: pd.DataFrame,
    id_col: str = "identifier",
) -> dict[str, np.ndarray]:
    """Build dict: identifier -> np.array of 0/1 (len = len(TAXA)).

    Expects *df* to contain taxonomy lineage columns (domain, kingdom, …).
    """
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
        raise ValueError(f"Missing taxonomy columns: {missing}")

    for c in tax_cols:
        df[c] = df[c].astype(str).str.strip().str.lower()

    taxa_norm = [t.strip().lower() for t in TAXA]

    for original_name, norm_name in zip(TAXA, taxa_norm):
        df[original_name] = (df[tax_cols] == norm_name).any(axis=1).astype(np.float32)

    tax_dict: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        identifier = row[id_col]
        tax_array = row[TAXA].to_numpy(dtype=np.float32)
        tax_dict[identifier] = tax_array

    print(f"Built binary taxonomy dict for {len(tax_dict)} identifiers")
    print(f"Binary taxonomy vector length: {len(TAXA)}")
    return tax_dict


def run_binary_taxonomy_pipeline(
    input_csv: str | Path,
    input_h5_path: str | Path,
    output_h5_path: str | Path,
    id_col: str = "identifier",
) -> None:
    """Create binary taxonomy vectors from a CSV that contains ``Organism (ID)``.

    1. Reads the CSV (must have *id_col* and ``Organism (ID)`` columns).
    2. Resolves taxonomy lineage for each unique taxon ID via :class:`TaxonomyRetriever`.
    3. Encodes membership in the 56 predefined TAXA as binary vectors.
    4. Writes one vector per protein (keyed by *id_col*) into *output_h5_path*,
       but only for proteins that are also present in *input_h5_path*.
    """
    df = pd.read_csv(input_csv)

    if "Organism (ID)" not in df.columns:
        raise ValueError(
            "CSV must contain an 'Organism (ID)' column with NCBI taxon IDs. "
            "Re-run `toxfam preprocess` to regenerate training_data.csv."
        )

    # Parse taxon IDs, dropping any NaN / non-numeric values
    df["_taxon_id"] = pd.to_numeric(df["Organism (ID)"], errors="coerce")
    valid = df["_taxon_id"].notna()
    unique_taxids = df.loc[valid, "_taxon_id"].astype(int).unique().tolist()

    print(f"Resolving lineage for {len(unique_taxids)} unique taxon IDs ...")
    retriever = TaxonomyRetriever(unique_taxids)
    tax_data = retriever.fetch_features()

    # Map taxon_id -> lineage dict, then join onto df
    lineage_rows = []
    for taxid, info in tax_data.items():
        row = {"_taxon_id": taxid}
        row.update(info["features"])
        lineage_rows.append(row)
    lineage_df = pd.DataFrame(lineage_rows)

    df = df.merge(lineage_df, on="_taxon_id", how="left")

    # Fill missing lineage columns with empty strings
    for col in TAXONOMY_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    tax_dict = _build_binary_vectors(df, id_col=id_col)
    vec_len = len(TAXA)

    with h5py.File(input_h5_path, "r") as f_in, h5py.File(
        output_h5_path, "w"
    ) as f_out:
        total_entries = len(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids: list[str] = []

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

        print("\nProcessing complete! (binary one-hot taxonomy)")
        print(f"Total entries (proteins): {total_entries}")
        print(f"Matched with taxonomy: {matched}")
        print(f"Unmatched (all-zero vector): {unmatched}")

        if unmatched > 0:
            print(f"\nFirst 10 unmatched IDs: {unmatched_ids[:10]}")

    print("\nBinary taxonomy pipeline finished.")
    print(f"Output file (only one-hot vectors): {output_h5_path}")
