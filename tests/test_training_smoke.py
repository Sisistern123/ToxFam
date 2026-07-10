"""End-to-end smoke test for the training entrypoint (toxfam.training.orchestrator).

Deselected by default (marked `slow`); run with `pytest -m slow`. Exercises the
full standard-strategy pipeline — config parsing, strategy dispatch, calibration,
evaluation, and both the multiclass + binary-derivation output writers — on a
tiny synthetic dataset. Asserts only structural invariants (files exist + JSON is
well-formed), never numeric metrics, so it is robust to numeric/calibration drift.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from toxfam.config import TrainConfig
from toxfam.training.orchestrator import run_training


@pytest.mark.slow
def test_run_training_standard_smoke(tmp_path, monkeypatch, fake_split_manifest):
    # Keep the test hermetic: skip the optional wandb login/logging path entirely.
    monkeypatch.setattr("toxfam.training.orchestrator.wandb", None)

    rng = np.random.default_rng(0)

    # Two toxin families + a non-toxin class, all present in every split so the
    # binary ROC-AUC / threshold optimization always has both classes.
    classes = ["famA", "famB", "nontox"]
    rows = []
    for split in ("train", "val", "test"):
        per_class = 6 if split == "train" else 3
        for cls in classes:
            for k in range(per_class):
                rows.append((f"{split}_{cls}_{k}", cls, split))
    df = pd.DataFrame(rows, columns=["identifier", "Protein families", "Split"])

    csv_path = tmp_path / "training_data.csv"
    df.to_csv(csv_path, index=False)
    fake_split_manifest(dict(zip(df["identifier"], df["Split"])))

    h5_path = tmp_path / "embeddings.h5"
    with h5py.File(h5_path, "w") as f:
        for ident in df["identifier"]:
            f.create_dataset(ident, data=rng.standard_normal(1024).astype(np.float32))

    out_dir = tmp_path / "run"
    config = TrainConfig(
        input_csv=csv_path,
        h5_path=str(h5_path),
        output_dir=out_dir,
        training_strategy="standard",
        embedding_dim=1024,
        hidden_dims=[8],
        num_epochs=1,
        batch_size=4,
        lr_scheduler="none",
        seed=42,
    )

    run_training(config)

    # Structural invariants only.
    assert (out_dir / "models" / "best_model_calibrated.pt").exists()
    assert (out_dir / "class_indices.json").exists()
    assert (out_dir / "model_config.json").exists()

    test_metrics = json.loads((out_dir / "metrics" / "test_metrics.json").read_text())
    assert "numeric_metrics" in test_metrics

    binary_metrics = json.loads(
        (out_dir / "metrics" / "binary_metrics.json").read_text()
    )
    assert isinstance(binary_metrics, dict) and binary_metrics
