"""Cheap content invariants that must hold between the split and its artifacts.

Provenance stamps (``split_manifest.write_provenance``) catch "this artifact was
built against a different manifest". These invariants are the defense-in-depth
layer: cheap assertions on artifact *content* that catch a desync even when a
stamp is missing, wrong, or the artifact was hand-edited.

The load-bearing one is ``reference_disjoint_from_holdout``: the HBI reference
must not contain a single val/test protein, or homology self-matches inflate the
baseline. That check is a set intersection on identifiers — milliseconds, no
sequence hashing — and would have caught the 6,785-test-protein contamination
directly, independent of any stamp.

Every check returns an ``InvariantResult`` (never raises for a *failed* invariant)
so ``toxfam verify`` can tabulate all of them; missing inputs are reported as
``skipped`` rather than failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import pandas as pd

from toxfam._paths import processed_dir
from toxfam.data.split_manifest import SplitManifestError, load_manifest

HBI_REFERENCE_CSV = "hbi_train_all.csv"
EMBEDDINGS_H5 = "embeddings.h5"
TAXONOMY_H5 = "taxonomy_vectors.h5"


@dataclass
class InvariantResult:
    """Outcome of one invariant check."""

    name: str
    ok: bool
    detail: str
    skipped: bool = False


def _h5_keys(path: Path) -> set[str]:
    with h5py.File(path, "r") as f:
        return set(f.keys())


def _holdout_ids() -> set[str]:
    man = load_manifest()
    return set(man.loc[man["Split"].isin(("val", "test")), "identifier"])


def reference_disjoint_from_holdout(
    reference_csv: Path | None = None,
) -> InvariantResult:
    """HBI reference must share no identifier with the val/test split.

    This is the check that would have caught the stale ``hbi_train_all`` directly.
    """
    name = "hbi_reference_disjoint"
    path = reference_csv or (processed_dir() / HBI_REFERENCE_CSV)
    if not path.exists():
        return InvariantResult(name, ok=True, detail=f"{path.name} absent", skipped=True)
    try:
        ref_ids = set(pd.read_csv(path, usecols=["identifier"])["identifier"])
        holdout = _holdout_ids()
    except (SplitManifestError, OSError, ValueError) as e:
        return InvariantResult(name, ok=False, detail=f"could not check: {e}")

    leaked = ref_ids & holdout
    if leaked:
        sample = sorted(leaked)[:3]
        return InvariantResult(
            name,
            ok=False,
            detail=(
                f"{len(leaked)} val/test proteins in the HBI reference "
                f"(e.g. {sample}) — homology self-matches inflate the baseline. "
                "Rebuild hbi_train_all from the manifest."
            ),
        )
    return InvariantResult(
        name, ok=True, detail=f"{len(ref_ids):,} reference ids, 0 val/test leakage"
    )


def _h5_covers_manifest(
    name: str, path: Path, *, noun: str, ok_detail: str, fix_hint: str
) -> InvariantResult:
    """Every manifest protein must have a key in the H5 at ``path``."""
    if not path.exists():
        return InvariantResult(name, ok=True, detail=f"{path.name} absent", skipped=True)
    try:
        man_ids = set(load_manifest()["identifier"])
        keys = _h5_keys(path)
    except (SplitManifestError, OSError) as e:
        return InvariantResult(name, ok=False, detail=f"could not check: {e}")

    missing = man_ids - keys
    if missing:
        return InvariantResult(
            name,
            ok=False,
            detail=(
                f"{len(missing):,} manifest proteins have no {noun} "
                f"(e.g. {sorted(missing)[:3]}) — {fix_hint}."
            ),
        )
    return InvariantResult(name, ok=True, detail=f"all {len(man_ids):,} {ok_detail}")


def embeddings_cover_manifest(embeddings_h5: Path | None = None) -> InvariantResult:
    """Every protein in the manifest must have an embedding."""
    return _h5_covers_manifest(
        "embeddings_cover_manifest",
        embeddings_h5 or (processed_dir() / EMBEDDINGS_H5),
        noun="embedding",
        ok_detail="manifest proteins embedded",
        fix_hint="regenerate embeddings",
    )


def taxonomy_matches_embeddings(taxonomy_h5: Path | None = None) -> InvariantResult:
    """Taxonomy vectors must cover every manifest protein.

    Taxonomy is keyed by the same identifiers as the embeddings; a mismatch means
    the two H5s were built against different data.
    """
    return _h5_covers_manifest(
        "taxonomy_matches_embeddings",
        taxonomy_h5 or (processed_dir() / TAXONOMY_H5),
        noun="taxonomy vector",
        ok_detail="manifest proteins have taxonomy",
        fix_hint="regenerate taxonomy",
    )


def all_invariants() -> list[InvariantResult]:
    """Run every content invariant against the artifacts on disk."""
    return [
        reference_disjoint_from_holdout(),
        embeddings_cover_manifest(),
        taxonomy_matches_embeddings(),
    ]
