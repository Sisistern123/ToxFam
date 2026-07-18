"""Guards for the generalized artifact provenance layer, content invariants,
and the `toxfam verify` aggregation.

These encode the failure that motivated them: ``hbi_train_all`` went stale after a
split re-pin and silently contained 6,785 test + 6,857 val proteins, inflating the
HBI baseline. The disjointness invariant catches exactly that, and the provenance
sidecars catch "built against a different manifest" for every derived artifact.
"""

from __future__ import annotations

import json

import h5py
import pandas as pd
import pytest

from toxfam.data.invariants import (
    embeddings_cover_manifest,
    reference_disjoint_from_holdout,
    taxonomy_matches_embeddings,
)
from toxfam.data.split_manifest import (
    SplitManifestError,
    manifest_sha256,
    read_provenance,
    sidecar_path,
    verify_provenance,
    write_provenance,
)

SPLITS = {"P1": "train", "P2": "train", "P3": "val", "P4": "test", "P5": "train"}


# --------------------------------------------------------------------------- #
# Generic artifact provenance round-trip                                       #
# --------------------------------------------------------------------------- #


def test_provenance_round_trip(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    art = tmp_path / "hbi_train_all.csv"
    art.write_text("id\n")

    h = write_provenance(art, min_seq_id=0.9)
    assert sidecar_path(art).exists()
    assert read_provenance(art)["split_manifest_sha256"] == h
    assert read_provenance(art)["min_seq_id"] == 0.9
    verify_provenance(art)  # matches -> no raise


def test_verify_provenance_raises_when_missing(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    art = tmp_path / "unstamped.csv"
    art.write_text("id\n")
    with pytest.raises(SplitManifestError, match="not pinned"):
        verify_provenance(art)


def test_verify_provenance_raises_on_manifest_mismatch(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    art = tmp_path / "hbi_train_all.csv"
    art.write_text("id\n")
    write_provenance(art)

    # Move the split; the stamp now disagrees with the manifest on disk.
    fake_split_manifest({**SPLITS, "P3": "train"})
    with pytest.raises(SplitManifestError, match="different train/val/test split"):
        verify_provenance(art)


def test_read_provenance_none_when_absent(tmp_path):
    assert read_provenance(tmp_path / "nope.csv") is None


# --------------------------------------------------------------------------- #
# Content invariant: HBI reference must not contain val/test proteins          #
# --------------------------------------------------------------------------- #


def _ref_csv(path, ids):
    pd.DataFrame({"identifier": ids, "Protein families": ["fam"] * len(ids)}).to_csv(
        path, index=False
    )
    return path


def test_reference_disjoint_passes_on_clean_reference(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    ref = _ref_csv(tmp_path / "ref.csv", ["P1", "P2", "P5", "M1", "M2"])  # train + members
    r = reference_disjoint_from_holdout(ref)
    assert r.ok and not r.skipped


def test_reference_disjoint_fails_when_holdout_leaks(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    # P3 is val, P4 is test — both must not appear in the reference.
    ref = _ref_csv(tmp_path / "ref.csv", ["P1", "P3", "P4"])
    r = reference_disjoint_from_holdout(ref)
    assert not r.ok
    assert "2 val/test proteins" in r.detail


def test_reference_disjoint_skips_when_absent(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    r = reference_disjoint_from_holdout(tmp_path / "missing.csv")
    assert r.ok and r.skipped


# --------------------------------------------------------------------------- #
# Content invariant: embeddings / taxonomy cover the manifest                  #
# --------------------------------------------------------------------------- #


def _h5(path, ids):
    with h5py.File(path, "w") as f:
        for i in ids:
            f[i] = [0.0]
    return path


def test_embeddings_cover_manifest_passes(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    emb = _h5(tmp_path / "emb.h5", list(SPLITS) + ["M1"])
    r = embeddings_cover_manifest(emb)
    assert r.ok


def test_embeddings_cover_manifest_fails_on_gap(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    emb = _h5(tmp_path / "emb.h5", ["P1", "P2"])  # missing P3,P4,P5
    r = embeddings_cover_manifest(emb)
    assert not r.ok
    assert "no embedding" in r.detail


def test_taxonomy_matches_embeddings(tmp_path, fake_split_manifest):
    fake_split_manifest(SPLITS)
    tax_ok = _h5(tmp_path / "tax.h5", list(SPLITS))
    assert taxonomy_matches_embeddings(tax_ok).ok
    tax_bad = _h5(tmp_path / "tax_bad.h5", ["P1"])
    assert not taxonomy_matches_embeddings(tax_bad).ok


# --------------------------------------------------------------------------- #
# verify aggregation                                                           #
# --------------------------------------------------------------------------- #


def test_verify_or_raise_red_when_reference_contaminated(
    tmp_path, fake_split_manifest, monkeypatch
):
    fake_split_manifest(SPLITS)
    from toxfam.data import invariants
    from toxfam.data import verify as verify_mod

    proc = tmp_path / "processed"
    proc.mkdir()
    _ref_csv(proc / "hbi_train_all.csv", ["P1", "P4"])  # P4 is test -> leak
    write_provenance(proc / "hbi_train_all.csv")
    monkeypatch.setattr(invariants, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "benchmark_dir", lambda: tmp_path / "nobench")

    rows = verify_mod.run_checks()
    assert verify_mod.has_failures(rows)
    with pytest.raises(verify_mod.PipelineNotVerified):
        verify_mod.verify_or_raise()


def test_verify_green_on_clean_setup(tmp_path, fake_split_manifest, monkeypatch):
    fake_split_manifest(SPLITS)
    from toxfam.data import invariants
    from toxfam.data import verify as verify_mod

    proc = tmp_path / "processed"
    proc.mkdir()
    _ref_csv(proc / "hbi_train_all.csv", ["P1", "P2", "P5"])  # train only
    write_provenance(proc / "hbi_train_all.csv")
    _h5(proc / "embeddings.h5", list(SPLITS))
    _h5(proc / "taxonomy_vectors.h5", list(SPLITS))
    monkeypatch.setattr(invariants, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "benchmark_dir", lambda: tmp_path / "nobench")

    rows = verify_mod.run_checks()
    assert not verify_mod.has_failures(rows)
    verify_mod.verify_or_raise()  # no raise


def test_verify_flags_unstamped_benchmark_run(tmp_path, fake_split_manifest, monkeypatch):
    fake_split_manifest(SPLITS)
    from toxfam.data import invariants
    from toxfam.data import verify as verify_mod

    proc = tmp_path / "processed"
    proc.mkdir()
    _ref_csv(proc / "hbi_train_all.csv", ["P1", "P2", "P5"])
    write_provenance(proc / "hbi_train_all.csv")
    bench = tmp_path / "benchmark"
    run = bench / "test_set" / "hbi"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(json.dumps({"method": "hbi"}))  # no stamp
    monkeypatch.setattr(invariants, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "benchmark_dir", lambda: bench)

    rows = verify_mod.run_checks("test_set")
    bench_rows = [r for r in rows if r.name.startswith("benchmark/")]
    assert bench_rows and all(r.status == "fail" for r in bench_rows)


def test_benchmark_stamp_matching_manifest_passes(tmp_path, fake_split_manifest, monkeypatch):
    fake_split_manifest(SPLITS)
    from toxfam.data import invariants
    from toxfam.data import verify as verify_mod

    proc = tmp_path / "processed"
    proc.mkdir()
    _ref_csv(proc / "hbi_train_all.csv", ["P1", "P2", "P5"])
    write_provenance(proc / "hbi_train_all.csv")
    bench = tmp_path / "benchmark"
    run = bench / "test_set" / "hbi"
    run.mkdir(parents=True)
    (run / "run_metadata.json").write_text(
        json.dumps({"method": "hbi", "split_manifest_sha256": manifest_sha256()})
    )
    monkeypatch.setattr(invariants, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "processed_dir", lambda: proc)
    monkeypatch.setattr(verify_mod, "benchmark_dir", lambda: bench)

    rows = verify_mod.run_checks("test_set")
    bench_rows = [r for r in rows if r.name.startswith("benchmark/")]
    assert bench_rows and all(r.status == "ok" for r in bench_rows)
