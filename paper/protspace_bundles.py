"""Build the ProtSpace bundles behind the embedding-space supplementary figure.

Two ``protspace prepare`` runs over the ProtT5 embeddings of the representative set:

``out_all``    every representative (n=65,179), for the toxin-vs-non-toxin panel.
``out_toxin``  refitted on the toxins alone (n=3,416), for the by-family panel, with
               ``--stats`` so projection quality is scored against the curated family
               labels rather than merely asserted from the picture.

Both runs are fed the pipeline's own ``data/processed/embeddings.h5``. ProtSpace can
embed from FASTA via ``-e prot_t5``, but that delegates to the remote Biocentral API
and would not reproduce the vectors the models were actually trained on.

Each run emits BOTH a ``.parquetbundle`` (drop onto https://protspace.app/explore --
the interactive data-availability artifact) and unbundled parquet parts, which is what
:mod:`paper.figures.figure_embedding_space` reads.

Split assignments come from ``split_manifest.apply_manifest``, never the ``Split``
column of the release CSV, matching every other split-derived artifact.

Run via ``make protspace``. Existing outputs are reused unless ``--force`` is passed;
ProtSpace additionally caches projections under ``{out}/tmp/``, so a re-run without
``--force`` is cheap and returns identical coordinates.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
from rich.console import Console

from paper._paths import protspace_bundle_dir, protspace_dir
from toxfam._paths import processed_dir
from toxfam.data import split_manifest

console = Console()

# ProtSpace 4.8.0 projection identifiers are dimension-suffixed. PCA is kept only as a
# global-variance sanity panel: it scores worse than UMAP on every cluster metric
# (silhouette -0.276 vs -0.057, ARI 0.153 vs 0.372) and is not what the figure plots.
METHODS = "pca2,umap2"

# Pinned so the figure is reproducible -- UMAP is seed dependent. Stated explicitly
# (and in the figure caption) rather than inherited from ProtSpace's defaults.
# n_neighbors/min_dist are raised above those defaults (25 / 0.1): a larger neighbourhood
# weights global structure more, and a larger minimum separation spreads points within a
# cluster, which keeps the dense central mass of panel B legible at print size.
RANDOM_STATE = 42
N_NEIGHBORS = 50
MIN_DIST = 0.3
METRIC = "euclidean"


def _annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Identifier + the three categorical columns the viewer and figure colour by."""
    return pd.DataFrame(
        {
            "identifier": df["identifier"],
            "family": df["Protein families"],
            "split": df["Split"],
            "toxic": (df["Protein families"] != "nontox").map(
                {True: "toxin", False: "non-toxin"}
            ),
        }
    )


def _write_toxin_h5(identifiers: set[str], source: Path, dest: Path) -> int:
    """Copy just the toxin embeddings so UMAP can be refitted on them alone."""
    with h5py.File(source) as src, h5py.File(dest, "w") as dst:
        keys = [k for k in src if k in identifiers]
        for key in keys:
            dst.create_dataset(key, data=src[key][:])
    if len(keys) != len(identifiers):
        # A toxin with no embedding would silently shrink panel B.
        raise RuntimeError(
            f"{len(identifiers) - len(keys)} toxin(s) missing from {source.name}"
        )
    return len(keys)


def _prepare(embeddings: Path, annotations: Path, out: Path, *, stats: bool) -> None:
    """One `protspace prepare` run, emitting both bundled and unbundled outputs."""
    base = [
        "protspace", "prepare",
        "-i", f"{embeddings}:prot_t5",
        "-a", str(annotations),
        "-m", METHODS,
        "-o", str(out),
        "--random-state", str(RANDOM_STATE),
        "--n-neighbors", str(N_NEIGHBORS),
        "--min-dist", str(MIN_DIST),
        "--metric", METRIC,
    ]
    if stats:
        # NOTE: the flag is singular -- `--stats-annotations` does not exist.
        base += ["--stats", "--stats-annotation", "family"]

    # Bundled first (the shareable .parquetbundle), then --no-bundled for the parquet
    # parts the figure reads. The second call reuses cached projections from {out}/tmp,
    # so both outputs are guaranteed to describe identical coordinates.
    for extra in ([], ["--no-bundled"]):
        subprocess.run(base + extra, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="rebuild even if outputs already exist"
    )
    args = parser.parse_args(argv)

    root = protspace_dir()
    root.mkdir(parents=True, exist_ok=True)
    all_dir, toxin_dir = protspace_bundle_dir("all"), protspace_bundle_dir("toxin")

    if not args.force and (all_dir / "projections_data.parquet").exists():
        console.print("[yellow]ProtSpace bundles present; use --force to rebuild.")
        return

    df = split_manifest.apply_manifest(pd.read_csv(processed_dir() / "training_data.csv"))
    ann = _annotations(df)
    ann_all = root / "annotations_all.csv"
    ann_tox = root / "annotations_toxin.csv"
    ann.to_csv(ann_all, index=False)
    ann[ann["toxic"] == "toxin"].to_csv(ann_tox, index=False)
    console.print(f"annotations: {len(ann):,} proteins, {ann['family'].nunique()} families")

    embeddings = processed_dir() / "embeddings.h5"
    toxin_h5 = root / "toxin_embeddings.h5"
    n_tox = _write_toxin_h5(
        set(ann.loc[ann["toxic"] == "toxin", "identifier"]), embeddings, toxin_h5
    )
    console.print(f"toxin-only embeddings: {n_tox:,}")

    console.print("[bold]projecting all representatives[/]")
    _prepare(embeddings, ann_all, all_dir, stats=False)
    console.print("[bold]projecting toxins only (+ stats)[/]")
    _prepare(toxin_h5, ann_tox, toxin_dir, stats=True)
    console.print(f"[green]bundles written to {root}")


if __name__ == "__main__":
    sys.exit(main())
