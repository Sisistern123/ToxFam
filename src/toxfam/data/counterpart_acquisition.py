"""Fetch non-toxic structural counterparts for each toxin family from UniProt.

For each venom protein family, there exist non-venom homologs (same fold, different
function). Adding these as targeted nontox examples helps the model learn venom-specific
signals rather than just fold recognition.

Data sources:
- Swiss-Prot (reviewed): highest quality
- TrEMBL (unreviewed): expand coverage, filtered to existence level 1-3, non-fragment

Signal peptides are trimmed using UniProt annotations to match the training pipeline
(all training sequences are mature, SP-removed).

Counterpart sequences are EXCLUDED if they have UniProt keyword KW-0800 (Toxin).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

from toxfam._paths import get_project_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Toxin family → non-venom counterpart query definitions
# ---------------------------------------------------------------------------

# Each entry: (query_description, uniprot_query_string, priority)
# Query strings use UniProt REST API syntax.
# All queries automatically add: NOT keyword:KW-0800, existence:1-3, fragment:false
COUNTERPART_QUERIES: dict[str, dict] = {
    "Ly6_uPAR": {
        "description": "Ly6/uPAR superfamily (counterpart to Three-finger toxin)",
        "toxic_family": "Three-finger toxin family",
        "query": (
            "(gene:LY6E OR gene:LY6D OR gene:LYNX1 OR gene:SLURP1 OR gene:SLURP2 "
            "OR gene:CD59 OR gene:PSCA OR gene:LYPD1 OR gene:LYPD2 OR gene:LYPD3 "
            "OR gene:LY6G6C OR gene:LY6G6D OR gene:LY6H OR gene:LY6K)"
        ),
        "taxonomy": "40674",  # Mammalia
        "priority": "HIGH",
    },
    "mammalian_PLA2": {
        "description": "Mammalian secreted PLA2 (counterpart to venom Phospholipase)",
        "toxic_family": "Phospholipase family",
        "query": (
            "(gene:PLA2G1B OR gene:PLA2G2A OR gene:PLA2G2D OR gene:PLA2G2E "
            "OR gene:PLA2G2F OR gene:PLA2G5 OR gene:PLA2G10 OR gene:PLA2G12A "
            "OR gene:PLA2G12B)"
        ),
        "taxonomy": "40674",
        "priority": "HIGH",
    },
    "ADAM_metalloproteinases": {
        "description": "ADAM family (counterpart to venom metalloproteinase M12B)",
        "toxic_family": "Venom metalloproteinase (M12B) family",
        "query": (
            "(gene:ADAM7 OR gene:ADAM10 OR gene:ADAM12 OR gene:ADAM15 "
            "OR gene:ADAM17 OR gene:ADAM28 OR gene:ADAM9 OR gene:ADAM19)"
        ),
        "taxonomy": "40674",
        "priority": "HIGH",
    },
    "C_type_lectins": {
        "description": "C-type lectins (counterpart to Snaclec)",
        "toxic_family": "Snaclec family",
        "query": (
            "(gene:CLEC4G OR gene:CLEC4K OR gene:CLEC10A OR gene:ASGR1 "
            "OR gene:OLR1 OR gene:COLEC12 OR gene:CLEC4M OR gene:CLEC4A "
            "OR gene:CLEC1A OR gene:CLEC2D)"
        ),
        "taxonomy": "40674",
        "priority": "HIGH",
    },
    "mammalian_Kunitz": {
        "description": "Mammalian Kunitz inhibitors (counterpart to venom Kunitz-type)",
        "toxic_family": "Venom Kunitz-type family",
        "query": (
            "(gene:SPINT1 OR gene:SPINT2 OR gene:SPINT3 OR gene:SPINT4 "
            "OR gene:TFPI OR gene:TFPI2 OR gene:AMBP OR gene:APP)"
            " AND (protein_name:kunitz)"
        ),
        "taxonomy": "40674",
        "priority": "HIGH",
    },
    "mammalian_CRISP": {
        "description": "Mammalian CRISPs (counterpart to venom CRISP)",
        "toxic_family": "CRISP family",
        "query": (
            "(gene:CRISP1 OR gene:CRISP2 OR gene:CRISP3 "
            "OR gene:GLIPR1 OR gene:GLIPR2)"
        ),
        "taxonomy": "40674",
        "priority": "HIGH",
    },
    "defensins": {
        "description": "Defensins (counterpart to scorpion toxins, CSαβ fold)",
        "toxic_family": "Long (4 C-C) scorpion toxin superfamily",
        "query": "(family:defensin) AND (keyword:KW-0929)",  # antimicrobial
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "natriuretic_peptides": {
        "description": "Mammalian natriuretic peptides (counterpart to Natriuretic/BPP)",
        "toxic_family": "Natriuretic, Bradykinin potentiating peptide family",
        "query": "(gene:NPPA OR gene:NPPB OR gene:NPPC)",
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "mammalian_insulin": {
        "description": "Mammalian insulin/IGF (counterpart to venom Insulin family)",
        "toxic_family": "Insulin family",
        "query": "(gene:INS OR gene:IGF1 OR gene:IGF2 OR gene:RLN1 OR gene:RLN2)",
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "PDGF_VEGF": {
        "description": "Mammalian PDGF/VEGF (counterpart to venom PDGF/VEGF)",
        "toxic_family": "PDGF/VEGF growth factor family",
        "query": (
            "(gene:PDGFA OR gene:PDGFB OR gene:PDGFC OR gene:PDGFD "
            "OR gene:VEGFA OR gene:VEGFB OR gene:VEGFC OR gene:VEGFD)"
        ),
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "vasopressin_oxytocin": {
        "description": "Mammalian vasopressin/oxytocin",
        "toxic_family": "Vasopressin/oxytocin family",
        "query": "(gene:AVP OR gene:OXT)",
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "lipocalins": {
        "description": "Lipocalins (counterpart to Calycin superfamily)",
        "toxic_family": "Calycin superfamily",
        "query": "(gene:LCN1 OR gene:LCN2 OR gene:RBP4 OR gene:OBP2A OR gene:APOD)",
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "perforins": {
        "description": "Perforins (counterpart to Actinoporin)",
        "toxic_family": "Actinoporin family",
        "query": "(gene:PRF1 OR gene:MPEG1)",
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "MAO": {
        "description": "Mammalian MAO (counterpart to Flavin MAO)",
        "toxic_family": "Flavin monoamine oxidase family",
        "query": "(gene:MAOA OR gene:MAOB)",
        "taxonomy": "40674",
        "priority": "LOW",
    },
    # --- New counterpart groups (Phase 2 expansion) ---
    "bradykinin_kininogens": {
        "description": "Mammalian kininogens/bradykinin precursors (counterpart to venom BPPs)",
        "toxic_family": "Bradykinin-related peptide family",
        "query": (
            "(gene:KNG1 OR gene:KNG2 OR gene:BDKRB1 OR gene:BDKRB2 "
            "OR protein_name:kininogen)"
        ),
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "cathelicidins": {
        "description": "Mammalian cathelicidins/AMPs (counterpart to Cationic peptide family)",
        "toxic_family": "Cationic peptide family",
        "query": (
            "(gene:CAMP OR gene:CATHL1 OR gene:CATHL2 OR gene:CATHL3 "
            "OR gene:CATHL4 OR gene:CATHL5 OR gene:CATHL6 "
            "OR family:cathelicidin)"
        ),
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "arthropod_defensins": {
        "description": "Arthropod defensins (counterpart to Short scorpion toxin, CSαβ fold)",
        "toxic_family": "Short scorpion toxin superfamily",
        "query": "(family:defensin) AND (keyword:KW-0929)",
        "taxonomy": "6656",  # Arthropoda
        "priority": "MEDIUM",
    },
    "serine_proteases": {
        "description": "Mammalian serine proteases (counterpart to venom Peptidase S1)",
        "toxic_family": "Peptidase S1 family",
        "query": (
            "(gene:PRSS1 OR gene:PRSS2 OR gene:PRSS3 OR gene:F2 "
            "OR gene:PLAT OR gene:PLG OR gene:KLK1 OR gene:KLK3 "
            "OR gene:TMPRSS2 OR gene:HGFAC)"
        ),
        "taxonomy": "40674",
        "priority": "MEDIUM",
    },
    "integrins_fibrinogen": {
        "description": "Mammalian integrins/fibrinogen (counterpart to Disintegrin family)",
        "toxic_family": "Disintegrin family",
        "query": (
            "(gene:ITGA2B OR gene:ITGB3 OR gene:FGA OR gene:FGB "
            "OR gene:FGG OR gene:VWF)"
        ),
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "mast_cell_peptides": {
        "description": "Mammalian mast cell / antimicrobial peptides (counterpart to MCD family)",
        "toxic_family": "MCD family",
        "query": (
            "(gene:DEFA1 OR gene:DEFA4 OR gene:DEFB1 OR gene:DEFB4A "
            "OR gene:GNLY OR gene:GZMB)"
        ),
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "K_channel_subunits": {
        "description": "Mammalian K+ channel subunits (counterpart to sea anemone K+ channel toxins)",
        "toxic_family": "Sea anemone type 1 potassium channel toxin family",
        "query": (
            "(gene:KCNA1 OR gene:KCNA2 OR gene:KCNA3 OR gene:KCNA4 "
            "OR gene:KCNA5 OR gene:KCNB1 OR gene:KCNB2)"
        ),
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "neuropeptides": {
        "description": "Mammalian neuropeptides (counterpart to NDBP superfamily)",
        "toxic_family": "Non-disulfide-bridged peptide (NDBP) superfamily",
        "query": (
            "(gene:NPY OR gene:PYY OR gene:TAC1 OR gene:PENK "
            "OR gene:PDYN OR gene:GAL OR gene:NTS)"
        ),
        "taxonomy": "40674",
        "priority": "LOW",
    },
    "hymenoptera_amps": {
        "description": "Hymenoptera non-venom AMPs (counterpart to Formicidae venom)",
        "toxic_family": "Formicidae venom family",
        "query": "(keyword:KW-0929) AND (family:defensin OR family:cecropin)",
        "taxonomy": "7399",  # Hymenoptera
        "priority": "LOW",
    },
    "long3cc_defensins": {
        "description": "Invertebrate defensins (counterpart to Long 3 C-C scorpion toxin)",
        "toxic_family": "Long (3 C-C) scorpion toxin superfamily",
        "query": "(family:defensin) AND (keyword:KW-0929)",
        "taxonomy": "6656",  # Arthropoda
        "priority": "LOW",
    },
}

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_FIELDS = (
    "accession,id,protein_name,gene_names,organism_name,organism_id,"
    "length,ft_signal,keyword,sequence"
)


def _build_query(entry: dict, *, include_trembl: bool = False) -> str:
    """Build a full UniProt query string with safety filters."""
    parts = [entry["query"]]

    # Exclude toxin-annotated proteins (KW-0800)
    parts.append("NOT (keyword:KW-0800)")

    # Taxonomy filter
    if "taxonomy" in entry:
        parts.append(f"(taxonomy_id:{entry['taxonomy']})")

    # Existence and fragment filters
    parts.append("(existence:1 OR existence:2 OR existence:3)")
    parts.append("(fragment:false)")

    # Review status
    if not include_trembl:
        parts.append("(reviewed:true)")

    return " AND ".join(parts)


def _fetch_uniprot_batch(
    query: str,
    *,
    max_results: int = 500,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch results from UniProt REST API."""
    params = {
        "query": query,
        "format": "tsv",
        "fields": UNIPROT_FIELDS,
        "size": min(max_results, 500),
    }

    for attempt in range(retries):
        try:
            resp = requests.get(UNIPROT_BASE, params=params, timeout=60)
            resp.raise_for_status()
            if not resp.text.strip():
                return pd.DataFrame()

            from io import StringIO

            df = pd.read_csv(StringIO(resp.text), sep="\t")
            return df
        except Exception as e:
            wait = 10 * (attempt + 1)
            logger.warning(f"UniProt API error (attempt {attempt + 1}): {e}, retrying in {wait}s")
            time.sleep(wait)

    logger.error(f"Failed to fetch from UniProt after {retries} attempts")
    return pd.DataFrame()


def trim_signal_peptide(sequence: str, sp_annotation: str | None) -> str:
    """Trim signal peptide using UniProt ft_signal annotation.

    Annotation format: 'SIGNAL 1..N' or 'SIGNAL 1..21; /evidence=...'
    → remove first N residues to get mature sequence.
    """
    if not sp_annotation or pd.isna(sp_annotation):
        return sequence

    sp_str = str(sp_annotation)
    match = re.search(r"SIGNAL\s+1\.\.(\d+)", sp_str)
    if match:
        sp_len = int(match.group(1))
        if sp_len < len(sequence):
            return sequence[sp_len:]

    return sequence


def fetch_all_counterparts(
    output_dir: Path | None = None,
    *,
    include_trembl: bool = False,
    max_per_family: int = 100,
) -> pd.DataFrame:
    """Fetch non-toxic counterpart sequences from UniProt for all toxin families.

    Returns a DataFrame with columns:
        identifier, Sequence, counterpart_group, toxic_family,
        organism_name, organism_id, length, source
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw" / "nontox_counterparts"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for group_name, entry in COUNTERPART_QUERIES.items():
        query = _build_query(entry, include_trembl=include_trembl)
        logger.info(f"Fetching {group_name}: {entry['description']}")
        print(f"  Fetching {group_name} ({entry['priority']}): ", end="", flush=True)

        df = _fetch_uniprot_batch(query, max_results=max_per_family)

        if df.empty:
            print("0 results")
            continue

        # Rename columns to match our conventions
        col_map = {
            "Entry": "identifier",
            "Entry Name": "entry_name",
            "Protein names": "protein_name",
            "Gene Names": "gene_names",
            "Organism": "organism_name",
            "Organism (ID)": "organism_id",
            "Length": "length",
            "Signal peptide": "signal_peptide",
            "Keywords": "keywords",
            "Sequence": "Sequence",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Double-check: exclude anything with Toxin keyword
        if "keywords" in df.columns:
            toxin_mask = df["keywords"].str.contains("Toxin", case=False, na=False)
            n_excluded = toxin_mask.sum()
            if n_excluded > 0:
                logger.warning(f"  Excluded {n_excluded} entries with Toxin keyword")
                df = df[~toxin_mask]

        # Trim signal peptides
        if "signal_peptide" in df.columns and "Sequence" in df.columns:
            df["Sequence"] = df.apply(
                lambda row: trim_signal_peptide(
                    row["Sequence"], row.get("signal_peptide")
                ),
                axis=1,
            )

        df["counterpart_group"] = group_name
        df["toxic_family"] = entry["toxic_family"]
        df["Protein families"] = "nontox"
        df["source"] = "swissprot" if not include_trembl else "swissprot+trembl"

        print(f"{len(df)} sequences")
        all_dfs.append(df)

        # Rate limiting
        time.sleep(1)

    if not all_dfs:
        print("WARNING: No counterpart sequences found!")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate by identifier
    n_before = len(result)
    result = result.drop_duplicates(subset="identifier").reset_index(drop=True)
    if len(result) < n_before:
        print(f"  Deduplicated: {n_before} → {len(result)}")

    # Save raw data
    result.to_csv(output_dir / "counterparts.csv", index=False)

    # Save FASTA
    fasta_lines = []
    for _, row in result.iterrows():
        if pd.notna(row.get("Sequence")):
            fasta_lines.append(f">{row['identifier']}\n{row['Sequence']}\n")
    (output_dir / "counterparts.fasta").write_text("".join(fasta_lines))

    # Summary
    print(f"\n  Total counterparts: {len(result)}")
    for group, count in result["counterpart_group"].value_counts().items():
        toxic_fam = result[result["counterpart_group"] == group]["toxic_family"].iloc[0]
        print(f"    {group}: {count} (for {toxic_fam})")

    return result


def _clean_sequence(seq: str) -> str:
    """Clean sequence for embedding: uppercase, replace non-standard AAs with X."""
    seq = seq.upper().strip()
    # Replace non-standard amino acids with X (ProtT5 handles X)
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWXY]", "X", seq)


def compute_counterpart_embeddings(
    fasta_dict: dict[str, str],
    output_h5: Path,
    *,
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """Compute ProtT5 embeddings for counterpart sequences via biocentral API.

    Follows the same pattern as SpeciesEmbedding tools
    (generate_embeddings_biocentral in protspace_pipeline.py).
    """
    try:
        from biocentral_api import BiocentralAPI, CommonEmbedder
    except ImportError:
        raise ImportError(
            "biocentral-api required for embedding computation. "
            "Install with: uv add biocentral-api"
        )

    # Clean sequences and filter out empty/very short ones
    cleaned: dict[str, str] = {}
    for sid, seq in fasta_dict.items():
        clean = _clean_sequence(seq)
        if len(clean) >= 5:
            cleaned[sid] = clean
        else:
            logger.warning(f"  Skipping {sid}: too short after cleaning ({len(clean)} aa)")

    if len(cleaned) < len(fasta_dict):
        print(f"  Filtered: {len(fasta_dict)} → {len(cleaned)} valid sequences")

    # Deduplicate by sequence (biocentral API rejects duplicate sequences)
    seq_to_ids: dict[str, list[str]] = {}
    for sid, seq in cleaned.items():
        seq_to_ids.setdefault(seq, []).append(sid)

    unique_seqs = {ids_list[0]: seq for seq, ids_list in seq_to_ids.items()}
    unique_ids = list(unique_seqs.keys())
    n_dupes = len(cleaned) - len(unique_ids)
    if n_dupes > 0:
        print(f"  Deduplicated: {len(cleaned)} → {len(unique_ids)} unique ({n_dupes} duplicates)")

    api = BiocentralAPI()
    unique_embeddings: dict[str, np.ndarray] = {}

    for i in range(0, len(unique_ids), batch_size):
        batch_ids = unique_ids[i : i + batch_size]
        batch = {sid: unique_seqs[sid] for sid in batch_ids}
        print(
            f"  Biocentral batch {i // batch_size + 1} "
            f"({len(batch)} seqs, {i + len(batch)}/{len(unique_ids)})"
        )

        result = None
        for attempt in range(3):
            try:
                result = api.embed(
                    embedder_name=CommonEmbedder.ProtT5,
                    reduce=True,
                    sequence_data=batch,
                    use_half_precision=False,
                ).run()
                break
            except Exception as e:
                wait = 30 * (attempt + 1)
                logger.warning(
                    f"  Biocentral error (attempt {attempt + 1}/3): {e}, "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)

        if result is None:
            logger.error(
                f"  Biocentral failed after 3 attempts for batch starting at {i}"
            )
            continue

        # EmbeddingsResult is NOT a dict — convert to dict first
        result_dict = result.to_dict()
        for sid in batch_ids:
            if sid in result_dict:
                unique_embeddings[sid] = np.array(result_dict[sid], dtype=np.float32)
            else:
                logger.warning(f"  Missing embedding for {sid}")

    # Expand back: copy embeddings to all IDs sharing the same sequence
    all_embeddings: dict[str, np.ndarray] = {}
    for seq, ids_list in seq_to_ids.items():
        rep_id = ids_list[0]
        if rep_id in unique_embeddings:
            emb = unique_embeddings[rep_id]
            for sid in ids_list:
                all_embeddings[sid] = emb

    print(f"  Generated {len(all_embeddings)}/{len(fasta_dict)} embeddings")

    # Save to H5
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_h5), "w") as f:
        for sid, emb in all_embeddings.items():
            f.create_dataset(sid, data=emb.astype(np.float16))

    print(f"  Saved {len(all_embeddings)} embeddings to {output_h5}")
    return all_embeddings


def run_counterpart_pipeline(
    *,
    include_trembl: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """Full pipeline: fetch counterparts, compute embeddings.

    Returns path to the counterpart embeddings H5 file.
    """
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "data" / "raw" / "nontox_counterparts"

    print("=== Fetching non-toxic counterparts from UniProt ===")
    df = fetch_all_counterparts(
        output_dir,
        include_trembl=include_trembl,
    )

    if df.empty:
        print("No counterparts found, skipping embedding computation.")
        return Path()

    # Build FASTA dict for embedding
    fasta_dict = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("Sequence")) and len(str(row["Sequence"])) > 0:
            fasta_dict[row["identifier"]] = str(row["Sequence"])

    print("\n=== Computing ProtT5 embeddings via biocentral API ===")
    print(f"  {len(fasta_dict)} sequences to embed")

    h5_path = root / "data" / "processed" / "counterpart_embeddings.h5"
    compute_counterpart_embeddings(fasta_dict, h5_path)

    # Save a training-ready CSV with just the columns we need
    training_cols = ["identifier", "Sequence", "Protein families"]
    extra_cols = ["organism_id", "counterpart_group", "toxic_family"]
    cols_to_save = [c for c in training_cols + extra_cols if c in df.columns]
    df[cols_to_save].to_csv(
        output_dir / "counterparts_training.csv", index=False
    )

    return h5_path
