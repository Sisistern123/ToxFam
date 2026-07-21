"""Guards for the git-tracked split manifest and the checkpoint binding.

Two failures these encode, both of which happened:

* ``training_data.csv`` was overwritten by ``download-data --force`` with a CSV
  carrying a different Split column, silently redefining "test_set".
* A training run died before calibration, leaving an older calibrated checkpoint
  in place, which was then evaluated against a split it had never trained on.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from toxfam.data.split_manifest import (
    SplitManifestError,
    apply_manifest,
    diff_against_manifest,
    load_manifest,
    manifest_json_path,
    manifest_sha256,
    provenance_path,
    sha256_of,
    verify_binary_calibrator_provenance,
    verify_split_provenance,
    write_provenance,
    write_split_provenance,
)

SPLITS = {"P1": "train", "P2": "train", "P3": "val", "P4": "test"}


# --------------------------------------------------------------------------- #
# Hashing                                                                      #
# --------------------------------------------------------------------------- #


def test_hash_is_content_based_not_row_order():
    a = pd.DataFrame({"identifier": ["P1", "P2"], "Split": ["train", "val"]})
    b = a.iloc[::-1].reset_index(drop=True)
    assert sha256_of(a) == sha256_of(b)


def test_hash_changes_when_a_single_protein_moves_split():
    a = pd.DataFrame({"identifier": ["P1", "P2"], "Split": ["train", "val"]})
    b = pd.DataFrame({"identifier": ["P1", "P2"], "Split": ["train", "test"]})
    assert sha256_of(a) != sha256_of(b)


# --------------------------------------------------------------------------- #
# Manifest is authoritative over the CSV                                       #
# --------------------------------------------------------------------------- #


def test_apply_manifest_overrides_a_disagreeing_csv_split_column(fake_split_manifest):
    """The download-data --force scenario: a replacement CSV must not move the split."""
    fake_split_manifest(SPLITS)
    csv = pd.DataFrame(
        {
            "identifier": list(SPLITS),
            # Every protein is mislabelled relative to the manifest.
            "Split": ["test", "test", "train", "train"],
            "Sequence": ["M"] * 4,
        }
    )
    out = apply_manifest(csv)
    assert dict(zip(out["identifier"], out["Split"])) == SPLITS
    assert list(out["Sequence"]) == ["M"] * 4  # other columns survive


def test_apply_manifest_works_when_csv_has_no_split_column(fake_split_manifest):
    fake_split_manifest(SPLITS)
    out = apply_manifest(pd.DataFrame({"identifier": list(SPLITS)}))
    assert dict(zip(out["identifier"], out["Split"])) == SPLITS


def test_apply_manifest_refuses_proteins_it_does_not_know(fake_split_manifest):
    fake_split_manifest(SPLITS)
    csv = pd.DataFrame({"identifier": ["P1", "UNKNOWN"]})
    with pytest.raises(SplitManifestError, match="absent from the split manifest"):
        apply_manifest(csv)


def test_load_manifest_rejects_a_bad_split_value(fake_split_manifest):
    path = fake_split_manifest({"P1": "train"})
    path.write_text("identifier,Split\nP1,holdout\n")
    with pytest.raises(SplitManifestError, match="unexpected Split value"):
        load_manifest()


def test_load_manifest_rejects_duplicate_identifiers(fake_split_manifest):
    path = fake_split_manifest({"P1": "train"})
    path.write_text("identifier,Split\nP1,train\nP1,test\n")
    with pytest.raises(SplitManifestError, match="duplicate identifiers"):
        load_manifest()


def test_write_manifest_is_a_noop_when_the_split_did_not_move(fake_split_manifest):
    """`preprocess` is re-run routinely. An unchanged split must leave no git diff,
    or a real split change would be lost in the noise."""
    from toxfam.data.split_manifest import manifest_csv_path, write_manifest

    fake_split_manifest(SPLITS)
    splits = pd.DataFrame({"identifier": list(SPLITS), "Split": list(SPLITS.values())})
    first = write_manifest(splits, seed=42, min_seq_id=0.9)
    before = (manifest_csv_path().read_bytes(), manifest_json_path().read_bytes())

    # Same split, rows in a different order.
    again = write_manifest(splits.iloc[::-1], seed=42, min_seq_id=0.9)
    after = (manifest_csv_path().read_bytes(), manifest_json_path().read_bytes())

    assert first == again
    assert before == after  # timestamp did not churn


def test_write_manifest_rewrites_when_min_seq_id_changes(fake_split_manifest):
    from toxfam.data.split_manifest import write_manifest

    fake_split_manifest(SPLITS)
    splits = pd.DataFrame({"identifier": list(SPLITS), "Split": list(SPLITS.values())})
    write_manifest(splits, seed=42, min_seq_id=0.9)
    write_manifest(splits, seed=42, min_seq_id=0.5)

    assert json.loads(manifest_json_path().read_text())["min_seq_id"] == 0.5


def test_diff_reports_reassignment(fake_split_manifest):
    fake_split_manifest(SPLITS)
    moved = pd.DataFrame(
        {
            "identifier": ["P1", "P2", "P3", "P5"],
            "Split": ["test", "train", "val", "val"],
        }
    )
    d = diff_against_manifest(moved)
    assert d == {"added": 1, "removed": 1, "reassigned": 1, "unchanged": 2}


# --------------------------------------------------------------------------- #
# Checkpoint <-> manifest binding                                              #
# --------------------------------------------------------------------------- #


def test_unpinned_checkpoint_is_refused(tmp_path, fake_split_manifest):
    """A run that died before calibration leaves no stamp — and a stale checkpoint."""
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    with pytest.raises(SplitManifestError, match="not pinned to a split manifest"):
        verify_split_provenance(model_dir)


def test_checkpoint_pinned_to_the_current_split_is_accepted(
    tmp_path, fake_split_manifest
):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    digest = write_split_provenance(model_dir)

    assert digest == manifest_sha256()
    verify_split_provenance(model_dir)  # must not raise


def test_checkpoint_is_refused_after_the_split_moves(tmp_path, fake_split_manifest):
    """The April-model-against-a-May-split failure, in miniature."""
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    write_split_provenance(model_dir)

    # Regenerate the split: same proteins, different assignment.
    fake_split_manifest({"P1": "test", "P2": "train", "P3": "val", "P4": "train"})

    with pytest.raises(SplitManifestError, match="different train/val/test split"):
        verify_split_provenance(model_dir)


def _write_calibrator(model_dir, *, stamp: bool):
    """Create models/binary_calibrator.json, optionally with a fresh sidecar."""
    (model_dir / "models").mkdir(parents=True, exist_ok=True)
    cal = model_dir / "models" / "binary_calibrator.json"
    cal.write_text('{"a": 0.7, "b": -2.3, "eps": 1e-6, "threshold": 0.03}')
    if stamp:
        write_provenance(cal)
    return cal


def test_binary_calibrator_provenance_is_noop_when_absent(
    tmp_path, fake_split_manifest
):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    verify_binary_calibrator_provenance(model_dir)  # no calibrator -> must not raise


def test_binary_calibrator_pinned_to_current_split_is_accepted(
    tmp_path, fake_split_manifest
):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    _write_calibrator(model_dir, stamp=True)
    verify_binary_calibrator_provenance(model_dir)  # fresh sidecar -> must not raise


def test_unstamped_binary_calibrator_is_refused(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    _write_calibrator(model_dir, stamp=False)  # present but no sidecar (pre-feature)
    with pytest.raises(SplitManifestError, match="not pinned to a split manifest"):
        verify_binary_calibrator_provenance(model_dir)


def test_stale_binary_calibrator_is_refused_after_split_moves(
    tmp_path, fake_split_manifest
):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    _write_calibrator(model_dir, stamp=True)
    fake_split_manifest({"P1": "test", "P2": "train", "P3": "val", "P4": "train"})
    with pytest.raises(SplitManifestError, match="different train/val/test split"):
        verify_binary_calibrator_provenance(model_dir)


def test_provenance_is_unreadable_when_corrupt(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    write_split_provenance(model_dir)
    provenance_path(model_dir).write_text("{not json")

    with pytest.raises(SplitManifestError, match="Could not read split provenance"):
        verify_split_provenance(model_dir)


def test_provenance_records_the_hash_it_stamped(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    model_dir = tmp_path / "run"
    (model_dir / "models").mkdir(parents=True)
    digest = write_split_provenance(model_dir)

    stamped = json.loads(provenance_path(model_dir).read_text())
    assert stamped["split_manifest_sha256"] == digest


# --------------------------------------------------------------------------- #
# The manifest checked into this repo                                          #
# --------------------------------------------------------------------------- #


def test_download_verification_accepts_a_csv_whose_only_difference_is_split(
    tmp_path, fake_split_manifest
):
    """The Jun-23 `download-data --force` case: a release CSV carrying a different
    Split column is harmless, because the split is read from the manifest."""
    import typer

    from toxfam.cli import _verify_training_csv_against_manifest

    fake_split_manifest(SPLITS)
    csv = tmp_path / "training_data.csv"
    pd.DataFrame(
        {"identifier": list(SPLITS), "Split": ["test", "test", "train", "train"]}
    ).to_csv(csv, index=False)

    _verify_training_csv_against_manifest(csv)  # must not raise

    # A CSV describing different proteins, however, is a real disagreement.
    pd.DataFrame({"identifier": ["P1", "P9"], "Split": ["train", "test"]}).to_csv(
        csv, index=False
    )
    with pytest.raises(typer.Exit):
        _verify_training_csv_against_manifest(csv)


def test_split_guard_applies_only_to_split_derived_datasets(tmp_path):
    """`predict test_set` reads the split and must verify; `predict my.tsv` must not."""
    from toxfam.prediction import _is_split_dataset

    assert _is_split_dataset("test_set") is True
    assert _is_split_dataset("val_set") is True
    # External datasets carry their own labels; no split involved.
    assert _is_split_dataset("non_metazoan") is False
    assert _is_split_dataset("unreviewed") is False
    # A user-supplied file, even one named like a dataset.
    p = tmp_path / "test_set"
    p.write_text("identifier\nP1\n")
    assert _is_split_dataset(p) is False
    assert _is_split_dataset("whatever.tsv") is False


def test_released_models_carry_the_provenance_stamp():
    """package_models.py must ship models/split_provenance.json, or every released
    checkpoint is refused by `eval` as unpinned."""
    import importlib.util

    from toxfam._paths import get_project_root

    path = get_project_root() / "scripts" / "package_models.py"
    spec = importlib.util.spec_from_file_location("_pkg_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert "models/split_provenance.json" in mod.KEEP_FILES


def test_repo_manifest_is_loadable_and_covers_the_representative_set():
    df = load_manifest()
    assert len(df) == 65_179
    assert df["Split"].value_counts().to_dict() == {
        "train": 45_621,
        "val": 9_779,
        "test": 9_779,
    }
