"""Inference-only on the unreviewed TrEMBL set (no labels available).

Writes paper/figures/output/unreviewed_predictions.csv with
identifier, predicted_label, confidence, confidence_uncalibrated.
"""
from __future__ import annotations

import h5py
import pandas as pd
from rich.console import Console

from paper._paths import figures_output_dir
from toxfam._paths import evaluation_data_dir, get_project_root
from toxfam.model.inference import run_inference

console = Console()

MODEL_DIR = get_project_root() / "model" / "model_output" / "combined_run"
OUT = figures_output_dir() / "unreviewed_predictions.csv"


def main() -> None:
    h5 = evaluation_data_dir() / "unreviewed" / "unreviewed.h5"
    with h5py.File(h5, "r") as f:
        ids = list(f.keys())
    df = pd.DataFrame({"identifier": ids})
    out = run_inference(df, h5, MODEL_DIR)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    console.print(f"Wrote {len(out)} unreviewed predictions to {OUT}")
    console.print(out["predicted_label"].value_counts().head(10))


if __name__ == "__main__":
    main()
