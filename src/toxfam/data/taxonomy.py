"""Taxonomy lineage resolution and multi-hot vector generation.

Reads NCBI taxon IDs (already present in the training CSV as ``Organism (ID)``),
resolves full lineage via taxopy, and encodes membership in 50 predefined animal
taxa as multi-hot vectors stored in HDF5.  Multiple taxa can be active per protein
because the predefined taxa span different levels of the taxonomic hierarchy.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import taxopy
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

logger = logging.getLogger(__name__)
console = Console()

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
    # Sponges
    "Porifera",
    # Ctenophora
    "Ctenophora",
    "Tentaculata",
    # Cnidaria
    "Cnidaria",
    "Anthozoa",
    # Spiralia
    "Spiralia",
    "Rotifera",
    "Annelida",
    "Glyceridae",
    "Bryozoa",
    "Gymnolaemata",
    # Mollusca
    "Mollusca",
    "Bivalvia",
    "Cephalopoda",
    "Octopoda",
    "Gastropoda",
    "Neogastropoda",
    "Conidae",
    "Polyplacophora",
    # Echinodermata
    "Echinodermata",
    "Asteroidea",
    "Echinoidea",
    "Holothuroidea",
    # Panarthropoda
    "Panarthropoda",
    "Onychophora",
    "Tardigrada",
    "Myriapoda",
    "Chilopoda",
    # Arachnida
    "Arachnida",
    "Araneae",
    "Theraphosidae",
    "Scorpiones",
    "Buthidae",
    # Insecta
    "Insecta",
    "Hymenoptera",
    # Chordata — fish & cartilaginous
    "Chordata",
    "Chondrichthyes",
    "Dasyatidae",
    "Actinopterygii",
    # Amphibia
    "Amphibia",
    "Anura",
    # Reptiles & birds (Sauropsida = NCBI clade for reptiles + birds)
    "Sauropsida",
    "Aves",
    "Squamata",
    "Serpentes",
    "Viperidae",
    "Crotalinae",
    "Elapidae",
    # Mammals
    "Mammalia",
    "Soricidae",
]

# ---------- TaxDB management ----------


def _resolve_taxdb_dir() -> Path:
    """Return the taxopy database directory, creating it if needed."""
    env_override = os.environ.get("PROTSPACE_TAXDB_DIR")
    db_dir = (
        Path(env_override).expanduser()
        if env_override
        else Path.home() / ".cache" / "taxopy_db"
    )
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir


def _is_cache_stale(timestamp_file: Path) -> bool:
    """Return True if the timestamp file is missing or older than 1 week."""
    if not timestamp_file.exists():
        return True
    try:
        download_time = datetime.fromisoformat(timestamp_file.read_text().strip())
        return download_time < datetime.now() - timedelta(weeks=1)
    except (ValueError, OSError) as e:
        logger.warning(f"Could not read timestamp file: {e}. Will refresh cache.")
        return True


def _download_fresh_taxdb(db_dir: Path, timestamp_file: Path) -> taxopy.TaxDb:
    """Download taxopy database for the first time."""
    console.print(f"Downloading taxopy database to [cyan]{db_dir}[/] ...")
    taxdb = taxopy.TaxDb(taxdb_dir=str(db_dir), keep_files=True)
    timestamp_file.write_text(datetime.now().isoformat())
    return taxdb


def _attempt_refresh(db_dir: Path, timestamp_file: Path) -> None:
    """Refresh taxonomy DB files into a temp dir, then move on success."""
    console.print("Taxonomy cache is stale. Refreshing ...")
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="taxopy_tmp_"))
        taxopy.TaxDb(taxdb_dir=str(temp_dir), keep_files=True)
        for name in ("nodes.dmp", "names.dmp", "merged.dmp"):
            src = temp_dir / name
            if src.exists():
                shutil.move(str(src), str(db_dir / name))
        timestamp_file.write_text(datetime.now().isoformat())
    except Exception as e:
        logger.warning(f"Failed to refresh taxonomy database: {e}. Using cached data.")
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def _load_taxdb(db_dir: Path) -> taxopy.TaxDb:
    """Load a TaxDb from existing dump files in *db_dir*."""
    nodes_file = db_dir / "nodes.dmp"
    names_file = db_dir / "names.dmp"
    merged_file = db_dir / "merged.dmp"
    return taxopy.TaxDb(
        nodes_dmp=str(nodes_file),
        names_dmp=str(names_file),
        merged_dmp=str(merged_file) if merged_file.exists() else None,
    )


def _get_or_refresh_taxdb() -> taxopy.TaxDb:
    """Return a TaxDb, downloading or refreshing as needed."""
    db_dir = _resolve_taxdb_dir()
    nodes_file = db_dir / "nodes.dmp"
    names_file = db_dir / "names.dmp"
    timestamp_file = db_dir / ".download_timestamp"

    if not (nodes_file.exists() and names_file.exists()):
        return _download_fresh_taxdb(db_dir, timestamp_file)

    if _is_cache_stale(timestamp_file):
        _attempt_refresh(db_dir, timestamp_file)

    console.print(f"Loading taxopy database from [cyan]{db_dir}[/]")
    return _load_taxdb(db_dir)


# ---------- Taxonomy lineage resolution ----------


def _resolve_lineages(
    taxon_ids: list[int],
) -> dict[int, tuple[dict[str, str], set[str]]]:
    """Resolve NCBI taxon IDs to lineage dicts and full ancestor names.

    Returns ``{taxon_id: (rank_dict, all_names)}`` where *rank_dict* maps
    standard ranks to names (for the DataFrame columns) and *all_names* is
    the set of **all** ancestor names across every rank (including clades,
    superclasses, infraorders, etc.) for full-lineage matching.
    """
    taxdb = _get_or_refresh_taxdb()
    result: dict[int, tuple[dict[str, str], set[str]]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
    ) as progress:
        task = progress.add_task("Resolving taxonomy lineages", total=len(taxon_ids))

        for taxon_id in taxon_ids:
            try:
                taxon = taxopy.Taxon(taxon_id, taxdb)
                rnd = taxon.rank_name_dictionary
                rank_dict = {
                    "root": rnd.get("cellular root", "") or rnd.get("acellular root", ""),
                    "domain": rnd.get("domain", "") or rnd.get("realm", ""),
                    "kingdom": rnd.get("kingdom", ""),
                    "phylum": rnd.get("phylum", ""),
                    "class": rnd.get("class", ""),
                    "order": rnd.get("order", ""),
                    "family": rnd.get("family", ""),
                    "genus": rnd.get("genus", ""),
                    "species": rnd.get("species", ""),
                }
                # Collect ALL ancestor names (clades, superclasses, etc.)
                all_names = {
                    taxopy.Taxon(tid, taxdb).name
                    for tid in taxon.taxid_lineage
                }
                result[taxon_id] = (rank_dict, all_names)
            except Exception as e:
                logger.error(f"Failed to get taxonomy for {taxon_id}: {e}")
                result[taxon_id] = (dict.fromkeys(TAXONOMY_FEATURES, ""), set())

            progress.advance(task)

    return result


# ---------- Multi-hot taxonomy vectors ----------


def _build_multi_hot_vectors(
    df: pd.DataFrame,
    lineage_names: dict[int, set[str]],
    id_col: str = "identifier",
) -> dict[str, np.ndarray]:
    """Build dict: identifier -> np.array of 0/1 (len = len(TAXA)).

    Each position indicates whether the corresponding taxon from :data:`TAXA`
    appears anywhere in the protein's full taxonomic lineage (including clades,
    superclasses, infraorders, etc.).  Multiple positions can be 1 because
    the predefined taxa span different hierarchy levels (multi-hot encoding).

    *lineage_names* maps taxon IDs to the set of all ancestor names from
    the full NCBI lineage (not just standard ranks).
    """
    taxa_lower = [t.strip().lower() for t in TAXA]
    n_taxa = len(TAXA)

    tax_dict: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        identifier = row[id_col]
        taxon_id = row["_taxon_id"]
        if pd.notna(taxon_id):
            names_lower = {n.lower() for n in lineage_names.get(int(taxon_id), set())}
            vec = np.array(
                [1.0 if t in names_lower else 0.0 for t in taxa_lower],
                dtype=np.float32,
            )
        else:
            vec = np.zeros(n_taxa, dtype=np.float32)
        tax_dict[identifier] = vec

    console.print(
        f"Built multi-hot taxonomy vectors for [green]{len(tax_dict)}[/] identifiers "
        f"(vector length: {n_taxa})"
    )
    return tax_dict


def run_multi_hot_taxonomy_pipeline(
    input_csv: str | Path,
    input_h5_path: str | Path,
    output_h5_path: str | Path,
    id_col: str = "identifier",
) -> None:
    """Create multi-hot taxonomy vectors from a CSV that contains ``Organism (ID)``.

    1. Reads the CSV (must have *id_col* and ``Organism (ID)`` columns).
    2. Resolves taxonomy lineage for each unique taxon ID via :func:`_resolve_lineages`.
    3. Encodes membership in the 50 predefined TAXA as multi-hot vectors.
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

    console.print(
        f"Resolving lineage for [cyan]{len(unique_taxids)}[/] unique taxon IDs ..."
    )
    tax_data = _resolve_lineages(unique_taxids)

    # Split into rank dicts (for DataFrame columns) and full lineage names (for matching)
    rank_dicts = {taxid: rd for taxid, (rd, _) in tax_data.items()}
    lineage_names = {taxid: names for taxid, (_, names) in tax_data.items()}

    # Map taxon_id -> rank dict, then join onto df
    lineage_rows = [{"_taxon_id": taxid, **rd} for taxid, rd in rank_dicts.items()]
    lineage_df = pd.DataFrame(lineage_rows)

    df = df.merge(lineage_df, on="_taxon_id", how="left")

    # Fill missing lineage columns with empty strings
    for col in TAXONOMY_FEATURES:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    tax_dict = _build_multi_hot_vectors(df, lineage_names, id_col=id_col)
    vec_len = len(TAXA)

    with h5py.File(input_h5_path, "r") as f_in, h5py.File(
        output_h5_path, "w"
    ) as f_out:
        # Store TAXA names as metadata so the H5 is self-documenting
        f_out.attrs["taxa"] = TAXA

        protein_ids = list(f_in.keys())
        matched = 0
        unmatched = 0
        unmatched_ids: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task(
                "Writing taxonomy vectors", total=len(protein_ids)
            )

            for protein_id in protein_ids:
                if protein_id in tax_dict:
                    vec = tax_dict[protein_id].astype(np.float32)
                    matched += 1
                else:
                    vec = np.zeros(vec_len, dtype=np.float32)
                    unmatched += 1
                    unmatched_ids.append(protein_id)

                f_out.create_dataset(
                    protein_id, data=vec,
                    compression="gzip", compression_opts=1,
                )
                progress.advance(task)

        console.print("\n[bold green]Multi-hot taxonomy pipeline complete![/]")
        console.print(f"  Total proteins: {len(protein_ids)}")
        console.print(f"  Matched with taxonomy: {matched}")
        console.print(f"  Unmatched (zero vector): {unmatched}")

        if unmatched > 0:
            console.print(f"  First 10 unmatched IDs: {unmatched_ids[:10]}")

    console.print(f"Output: [cyan]{output_h5_path}[/]")
