"""Git-tracked train/val/test assignment, and the guards that enforce it.

The stratified splitter is a function of the *set* of representatives: change the
set (new UniProt data, a different ``--min-seq-id``, a new MMseqs version) and
every protein is reassigned. Nothing used to record which assignment a checkpoint
had trained on, so an April model was once evaluated against a May split and
nothing noticed -- 65% of that "test set" was the model's own training data.

This module turns the assignment into an artifact:

``data/splits/split_manifest.csv``
    ``identifier,Split`` for every representative, tracked in git. If the split
    ever moves, it moves as a reviewable diff in a pull request.
``data/splits/split_manifest.json``
    Provenance: content hash, per-split counts, seed, ``min_seq_id``, commit.
``<run_dir>/models/split_provenance.json``
    The manifest hash a calibrated checkpoint was trained against.

``preprocess`` rewrites the manifest; ``train`` stamps the checkpoint at
*calibration time*; ``eval``/``predict`` refuse a checkpoint whose stamp is
missing or disagrees with the manifest on disk.

Writing a new split is a legal act. Scoring a stale checkpoint against it is not,
and that is the only thing these guards forbid. Because the stamp is written when
the calibrated checkpoint is saved -- not when the run starts -- a run that dies
before calibration leaves no stamp, and the stale checkpoint it failed to replace
is refused rather than silently reused.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from toxfam._git import git_commit_short
from toxfam._paths import splits_dir

SPLIT_VALUES = ("train", "val", "test")

PROVENANCE_FILENAME = "split_provenance.json"


class SplitManifestError(RuntimeError):
    """The split manifest is missing, incomplete, or disagrees with a checkpoint."""


# ---------- Paths ----------


def manifest_csv_path() -> Path:
    return splits_dir() / "split_manifest.csv"


def manifest_json_path() -> Path:
    return splits_dir() / "split_manifest.json"


def manifest_exists() -> bool:
    return manifest_csv_path().exists()


# ---------- Hashing ----------


def _canonical_bytes(pairs: list[tuple[str, str]]) -> bytes:
    """Content hash input: identifier,Split sorted by identifier, one per line.

    Hashing content rather than file bytes keeps the hash stable across CSV
    formatting, line endings, and column order.
    """
    return "".join(f"{ident},{split}\n" for ident, split in sorted(pairs)).encode()


def sha256_of(df: pd.DataFrame) -> str:
    pairs = list(zip(df["identifier"], df["Split"], strict=True))
    return hashlib.sha256(_canonical_bytes(pairs)).hexdigest()


def manifest_sha256() -> str:
    """Content hash of the manifest on disk."""
    return sha256_of(load_manifest())


# ---------- Read / write ----------


def load_manifest() -> pd.DataFrame:
    """Load the manifest as an ``identifier,Split`` frame."""
    path = manifest_csv_path()
    if not path.exists():
        raise SplitManifestError(
            f"Split manifest not found at {path}.\n"
            "It is tracked in git; if it is missing, your checkout is incomplete. "
            "Run 'toxfam preprocess' to regenerate it (this defines a new split and "
            "invalidates every existing checkpoint)."
        )
    df = pd.read_csv(path)
    missing_cols = {"identifier", "Split"} - set(df.columns)
    if missing_cols:
        raise SplitManifestError(f"{path} is missing column(s): {sorted(missing_cols)}")
    bad = set(df["Split"]) - set(SPLIT_VALUES)
    if bad:
        raise SplitManifestError(f"{path} has unexpected Split value(s): {sorted(bad)}")
    if not df["identifier"].is_unique:
        raise SplitManifestError(f"{path} contains duplicate identifiers")
    return df


def _manifest_is_current(new_hash: str, *, seed: int, min_seq_id: float) -> bool:
    """True when the manifest on disk already records exactly this split and params."""
    if not manifest_exists() or not manifest_json_path().exists():
        return False
    try:
        if sha256_of(load_manifest()) != new_hash:
            return False
        meta = json.loads(manifest_json_path().read_text())
    except (SplitManifestError, json.JSONDecodeError, OSError):
        return False
    return meta.get("seed") == seed and meta.get("min_seq_id") == min_seq_id


def write_manifest(
    splits: pd.DataFrame,
    *,
    seed: int,
    min_seq_id: float,
) -> str:
    """Write the manifest + provenance sidecar. Returns the new content hash.

    ``splits`` needs ``identifier`` and ``Split`` columns.

    When nothing moved, the files are left untouched. ``preprocess`` is re-run
    routinely, and a provenance sidecar that rewrites its timestamp every time would
    put a diff in front of a reviewer on every run — exactly the signal this manifest
    exists to preserve for the runs where the split really did change.
    """
    df = splits[["identifier", "Split"]].sort_values("identifier", kind="stable")
    df = df.reset_index(drop=True)
    new_hash = sha256_of(df)

    if _manifest_is_current(new_hash, seed=seed, min_seq_id=min_seq_id):
        return new_hash

    path = manifest_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    counts = df["Split"].value_counts().to_dict()
    manifest_json_path().write_text(
        json.dumps(
            {
                "sha256": new_hash,
                "n_proteins": len(df),
                "counts": {k: int(counts.get(k, 0)) for k in SPLIT_VALUES},
                "seed": seed,
                "min_seq_id": min_seq_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": git_commit_short(),
            },
            indent=2,
        )
        + "\n"
    )
    return new_hash


def diff_against_manifest(splits: pd.DataFrame) -> dict | None:
    """Compare a fresh assignment to the manifest on disk.

    Returns ``None`` when there is no manifest yet, otherwise a summary of how the
    split moved: proteins added, removed, and reassigned between splits.
    """
    if not manifest_exists():
        return None
    old = load_manifest().set_index("identifier")["Split"]
    new = splits.set_index("identifier")["Split"]
    shared = old.index.intersection(new.index)
    return {
        "added": int(len(new.index.difference(old.index))),
        "removed": int(len(old.index.difference(new.index))),
        "reassigned": int((old.loc[shared] != new.loc[shared]).sum()),
        "unchanged": int((old.loc[shared] == new.loc[shared]).sum()),
    }


# ---------- Applying the manifest ----------


def apply_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with its ``Split`` column taken from the manifest.

    Any ``Split`` column already present is discarded. This is what makes a stale
    or re-downloaded ``training_data.csv`` unable to move the split: the CSV
    supplies sequences and labels, the manifest supplies the assignment.
    """
    manifest = load_manifest()
    unknown = set(df["identifier"]) - set(manifest["identifier"])
    if unknown:
        sample = sorted(unknown)[:5]
        raise SplitManifestError(
            f"{len(unknown)} protein(s) are absent from the split manifest "
            f"(e.g. {sample}).\n"
            "The data and the manifest describe different protein sets. Re-run "
            "'toxfam preprocess' to regenerate the manifest for this data -- note "
            "that this defines a new split and invalidates every existing checkpoint."
        )
    out = df.drop(columns=["Split"], errors="ignore")
    return out.merge(manifest, on="identifier", how="left")


# ---------- Generic artifact <-> manifest binding ----------
#
# Every expensive artifact derived from the split (embeddings, taxonomy vectors,
# the HBI reference, benchmark predictions) can silently desync from the manifest
# — the release ships it, a re-pin moves the split, and nothing notices. Each such
# artifact gets a ``<artifact>.provenance.json`` sidecar recording the manifest
# hash it was built against, written at generation time and checked at use.


def _provenance_stamp(**extra: object) -> str:
    """Serialise a provenance payload (manifest hash + timestamp + extras)."""
    return (
        json.dumps(
            {
                "split_manifest_sha256": manifest_sha256(),
                "stamped_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            },
            indent=2,
        )
        + "\n"
    )


def _check_stamped_sha(stamped: str, *, label: str, missing_hint: str) -> None:
    """Raise unless ``stamped`` matches the manifest hash on disk."""
    current = manifest_sha256()
    if stamped != current:
        raise SplitManifestError(
            f"{label} was built against a different train/val/test split than the "
            f"one on disk.\n"
            f"  artifact: {stamped[:12]}\n"
            f"  manifest: {current[:12]}\n" + missing_hint
        )


def sidecar_path(artifact: str | Path) -> Path:
    """Provenance sidecar for a file artifact: ``<artifact>.provenance.json``."""
    return Path(str(artifact) + ".provenance.json")


def write_provenance(artifact: str | Path, **extra: object) -> str:
    """Stamp a file artifact with the current manifest hash. Returns the hash."""
    path = sidecar_path(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_provenance_stamp(**extra))
    return manifest_sha256()


def read_provenance(artifact: str | Path) -> dict | None:
    """Return the sidecar payload for ``artifact``, or None if it has none."""
    path = sidecar_path(artifact)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def verify_provenance(artifact: str | Path, *, label: str | None = None) -> None:
    """Raise unless ``artifact`` was built against the manifest on disk."""
    artifact = Path(artifact)
    name = label or artifact.name
    path = sidecar_path(artifact)
    if not path.exists():
        raise SplitManifestError(
            f"{name} is not pinned to a split manifest ({path.name} is missing).\n"
            "It predates provenance stamping or was produced against a different "
            "split. Regenerate it against the current split before using it."
        )
    try:
        stamped = json.loads(path.read_text())["split_manifest_sha256"]
    except (json.JSONDecodeError, KeyError, OSError) as e:
        raise SplitManifestError(f"Could not read provenance from {path}: {e}")
    _check_stamped_sha(
        stamped,
        label=name,
        missing_hint=(
            "Using it now would mix data across splits. Regenerate it against the "
            "current split (see 'toxfam verify' for the full pipeline state)."
        ),
    )


# ---------- Checkpoint <-> manifest binding ----------


def provenance_path(model_dir: str | Path) -> Path:
    return Path(model_dir) / "models" / PROVENANCE_FILENAME


def write_split_provenance(model_dir: str | Path) -> str:
    """Stamp a run directory with the manifest hash. Call when saving the
    calibrated checkpoint, so a run that dies earlier leaves no stamp."""
    path = provenance_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_provenance_stamp())
    return manifest_sha256()


def verify_split_provenance(model_dir: str | Path) -> None:
    """Raise unless the checkpoint was calibrated against the manifest on disk."""
    path = provenance_path(model_dir)
    name = Path(model_dir).name
    if not path.exists():
        raise SplitManifestError(
            f"{name} is not pinned to a split manifest ({path} is missing).\n"
            "Either it predates split pinning, or its training run died before "
            "calibration and left an older checkpoint in place. Re-run 'toxfam train' "
            "to produce a checkpoint bound to the current split."
        )
    try:
        stamped = json.loads(path.read_text())["split_manifest_sha256"]
    except (json.JSONDecodeError, KeyError, OSError) as e:
        raise SplitManifestError(f"Could not read split provenance from {path}: {e}")
    _check_stamped_sha(
        stamped,
        label=f"{name} (checkpoint)",
        missing_hint=(
            "Evaluating it now would score the model against proteins it may have "
            "trained on. Re-run 'toxfam train' on the current split, or check out the "
            "commit whose data/splits/split_manifest.csv matches this checkpoint."
        ),
    )


def verify_binary_calibrator_provenance(model_dir: str | Path) -> None:
    """Raise unless the deployed binary P(toxic) calibrator matches the split on disk.

    A no-op when ``models/binary_calibrator.json`` is absent (predict then falls
    back to the raw score — nothing to pin). When present it must carry a fresh
    ``.provenance.json`` sidecar, so a calibrator left beside a re-trained
    checkpoint is caught rather than silently applied to it. Verified only where a
    split is scored (predict on test_set/val_set), never on arbitrary proteins.
    """
    calibrator = Path(model_dir) / "models" / "binary_calibrator.json"
    if calibrator.exists():
        verify_provenance(
            calibrator, label=f"{Path(model_dir).name} (binary calibrator)"
        )
