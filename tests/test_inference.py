"""Tests for toxfam.model.inference.

This module is shared by BOTH `toxfam predict` and `toxfam eval model`, yet was
untested. Covers the CPU-only seams: embedding/taxonomy loading (shape +
identifier ordering + zero-vector fallback), taxonomy-H5 resolution, and the
top-k inference contract (K predictions per row, descending confidences,
P(toxic) derivation) — all on a tiny synthetic ModularMLP checkpoint.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import torch

from toxfam.model.calibration import ModelWithTemperature
from toxfam.model.inference import (
    _load_embeddings,
    _load_tax_vectors,
    _resolve_tax_h5,
    run_topk_inference,
)
from toxfam.model.model_config import ModelConfig

LABELS = {0: "famA", 1: "famB", 2: "nontox"}  # 'nontox' drives the p_toxic path


def _make_model_dir(tmp_path):
    """Build a minimal, loadable standard (ModularMLP) model directory."""
    model_dir = tmp_path / "model"
    (model_dir / "models").mkdir(parents=True)

    cfg = ModelConfig(
        architecture="ModularMLP",
        embedding_dim=1024,
        hidden_dims=[8],
        num_classes=len(LABELS),
        dropout=0.0,
    )
    cfg.save(model_dir / "model_config.json")

    base = cfg.build_model()
    scaled = ModelWithTemperature(base, torch.device("cpu"))
    torch.save(scaled.state_dict(), model_dir / "models" / "best_model_calibrated.pt")

    (model_dir / "class_indices.json").write_text(json.dumps(LABELS))
    return model_dir


# --------------------------------------------------------------------------- #
# _load_embeddings                                                             #
# --------------------------------------------------------------------------- #


def test_load_embeddings_shape_and_ordering(sample_h5):
    idents = ["P002", "P001", "P003"]
    emb = _load_embeddings(sample_h5, idents)

    assert emb.shape == (3, 1024)
    assert emb.dtype == torch.float32
    # Row order must follow the requested identifier order, not H5 key order.
    with h5py.File(sample_h5, "r") as f:
        expected_first = torch.tensor(f["P002"][:], dtype=torch.float32)
    assert torch.allclose(emb[0], expected_first)


# --------------------------------------------------------------------------- #
# _load_tax_vectors                                                            #
# --------------------------------------------------------------------------- #


def test_load_tax_vectors_present_and_missing(tmp_path):
    tax_h5 = tmp_path / "tax.h5"
    tax_dim = 6
    with h5py.File(tax_h5, "w") as f:
        f.create_dataset("P001", data=np.ones(tax_dim, dtype=np.float32))
        f.create_dataset("P002", data=np.ones(tax_dim, dtype=np.float32))

    vecs = _load_tax_vectors(
        tmp_path / "model", ["P001", "MISSING", "P002"], tax_dim, tax_h5_path=tax_h5
    )

    assert vecs.shape == (3, tax_dim)
    assert torch.equal(vecs[0], torch.ones(tax_dim))
    assert torch.equal(vecs[1], torch.zeros(tax_dim))  # absent -> zero vector
    assert torch.equal(vecs[2], torch.ones(tax_dim))


def test_load_tax_vectors_no_source_returns_none(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # No tax_h5_path and no config.yaml -> nothing to resolve.
    assert _load_tax_vectors(model_dir, ["P001"], 6) is None


# --------------------------------------------------------------------------- #
# _resolve_tax_h5                                                              #
# --------------------------------------------------------------------------- #


def test_resolve_tax_h5_no_config_returns_none(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert _resolve_tax_h5(model_dir) is None


def test_resolve_tax_h5_reads_absolute_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tax_h5 = tmp_path / "tax.h5"
    tax_h5.write_bytes(b"")  # only existence matters
    (model_dir / "config.yaml").write_text(f"tax_h5_path: {tax_h5}\n")

    assert _resolve_tax_h5(model_dir) == tax_h5


def test_resolve_tax_h5_null_path_returns_none(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.yaml").write_text("tax_h5_path: null\n")
    assert _resolve_tax_h5(model_dir) is None


# --------------------------------------------------------------------------- #
# run_topk_inference                                                           #
# --------------------------------------------------------------------------- #


def test_run_topk_inference_contract(tmp_path, sample_h5):
    model_dir = _make_model_dir(tmp_path)
    df = pd.DataFrame({"identifier": ["P001", "P002", "P003"]})

    out = run_topk_inference(df, sample_h5, model_dir, top_k=3)

    assert list(out["identifier"]) == ["P001", "P002", "P003"]
    for rank in (1, 2, 3):
        assert f"pred_{rank}" in out.columns
        assert f"conf_{rank}" in out.columns
    # Confidences are the top-k, so must be non-increasing per row.
    for _, row in out.iterrows():
        assert row["conf_1"] >= row["conf_2"] >= row["conf_3"]
    # p_toxic is a probability derived from the 'nontox' class.
    assert ((out["p_toxic"] >= 0) & (out["p_toxic"] <= 1)).all()


def test_load_calibrated_model_does_not_need_a_split_manifest(
    tmp_path, sample_h5, monkeypatch
):
    """Predicting on user proteins involves no split, and must work outside a checkout.

    The Colab notebook pip-installs the package, so `get_project_root()` has nothing
    to find. Loading a checkpoint must not reach for the manifest. The split guard
    lives with the callers that score against a split (eval, predict test_set/val_set).
    """
    from toxfam.data import split_manifest as sm

    model_dir = _make_model_dir(tmp_path)  # no split_provenance.json written
    monkeypatch.setattr(sm, "splits_dir", lambda: tmp_path / "nonexistent")

    out = run_topk_inference(
        pd.DataFrame({"identifier": ["P001"]}), sample_h5, model_dir, top_k=1
    )
    assert list(out["identifier"]) == ["P001"]


def test_run_topk_inference_binary_only(tmp_path, sample_h5):
    model_dir = _make_model_dir(tmp_path)
    df = pd.DataFrame({"identifier": ["P001", "P002"]})

    out = run_topk_inference(df, sample_h5, model_dir, binary_only=True)

    assert list(out.columns) == ["identifier", "p_toxic"]
    assert len(out) == 2
