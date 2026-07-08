"""Named evaluation-dataset registry + shared resolution helpers.

Lives in the (light) ``data`` layer rather than ``evaluation.runner`` so that
``toxfam predict`` can resolve a dataset name without importing the evaluation +
plotting stack (mmseqs/matplotlib/seaborn). Both ``evaluation.runner`` and
``prediction`` consume these.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

from toxfam._paths import evaluation_data_dir, processed_dir
from toxfam.data.normalization import normalize_protein_families

console = Console()

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "test_set": {
        "source": "training_data",
        "split": "test",
        "task": "multiclass",
    },
    "val_set": {
        "source": "training_data",
        "split": "val",
        "task": "multiclass",
    },
    "non_metazoan": {
        "source": "evaluation",
        "tsv": "non_metazoan.tsv",
        "h5": "non_metazoan.h5",
        "task": "binary",
    },
    "unreviewed": {
        "source": "evaluation",
        "tsv": "unreviewed.tsv",
        "h5": "unreviewed.h5",
        "task": "multiclass",
    },
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def resolve_embeddings_h5(dataset: str) -> Path:
    """Canonical ProtT5 embeddings H5 for a registered dataset.

    Evaluation datasets carry their own H5; the training-split datasets
    (test_set / val_set) share the processed training embeddings.
    """
    cfg = DATASETS[dataset]
    if cfg["source"] == "evaluation":
        return evaluation_data_dir() / dataset / cfg["h5"]
    return processed_dir() / "embeddings.h5"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load ground-truth DataFrame for a dataset.

    Returns DataFrame with at least ``identifier``, ``Sequence``,
    ``Protein families`` columns.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list_datasets()}")

    cfg = DATASETS[dataset]

    if cfg["source"] == "training_data":
        training_csv = processed_dir() / "training_data.csv"
        if not training_csv.exists():
            raise FileNotFoundError(
                f"{training_csv} not found. Run 'toxfam download-data' first."
            )
        df = pd.read_csv(training_csv)
        df = df[df["Split"] == cfg["split"]].reset_index(drop=True)
        console.print(f"   Loaded {len(df)} sequences from {cfg['split']} split")
        return df

    # evaluation datasets
    eval_dir = evaluation_data_dir() / dataset
    tsv_name = cfg.get("tsv")
    if tsv_name is None:
        raise ValueError(f"Dataset '{dataset}' requires --input-tsv")

    tsv_path = eval_dir / tsv_name
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"{tsv_path} not found. Run 'toxfam download-data' first."
        )

    df = pd.read_csv(tsv_path, sep="\t")
    # Normalize column names
    if "Entry" in df.columns and "identifier" not in df.columns:
        df = df.rename(columns={"Entry": "identifier"})

    df = df.dropna(subset=["Protein families"]).copy()
    df = normalize_protein_families(df)

    console.print(f"   Loaded {len(df)} sequences from {tsv_path.name}")
    return df
