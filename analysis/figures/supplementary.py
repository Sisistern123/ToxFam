"""Supplementary figures: reliability/ECE + macro-F1 convention table.

(Identity-binned null and confusion matrices reuse existing analysis/plots and
model_output confusion matrices; this script adds the calibration + convention
artifacts that did not previously exist.)
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures._common import apply_style, load_preds, save_fig, test_class_list
from toxfam.evaluation.manuscript import binary_reliability, macro_f1_conventions
from toxfam.evaluation.metrics import to_binary_class


def reliability_panel() -> None:
    """Binary reliability diagram + ECE from non-metazoan/test p_toxic.

    Uses the test-set NN predictions' calibrated 'confidence' as a multiclass
    reliability proxy; for the binary head, recompute p_toxic via eval binary
    (model_output/.../metrics/binary_metrics.json holds AUROC/AUPRC).
    """
    apply_style()
    nn = load_preds("test_set", "nn_combined_run")
    # multiclass reliability from calibrated vs uncalibrated max-confidence
    correct = (nn["predicted_label"] == nn["actual_label"]).astype(float).values
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for col, name, color in (("confidence", "calibrated", "#c0504d"),
                             ("confidence_uncalibrated", "uncalibrated", "#7f7f7f")):
        if col not in nn.columns:
            continue
        conf = nn[col].values
        edges = np.linspace(0, 1, 16)
        xs, ys, ece = [], [], 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.mean() > 0:
                xs.append(conf[m].mean()); ys.append(correct[m].mean())
                ece += abs(conf[m].mean() - correct[m].mean()) * m.mean()
        ax.plot(xs, ys, "o-", color=color, label=f"{name} (ECE={ece:.3f})", ms=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy"); ax.legend()
    ax.set_title("Reliability (multiclass top-class)")
    save_fig(fig, "supp_reliability")


def convention_table() -> None:
    """Write the macro-F1 no-hit convention values to a CSV for the manuscript."""
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi")
    conv = macro_f1_conventions(hbi, class_list=classes)
    out = pd.DataFrame([
        {"method": "HBI", **conv},
    ])
    from analysis.figures._common import FIG_DIR
    out.to_csv(FIG_DIR / "supp_macro_f1_conventions.csv", index=False)
    print(out.to_string(index=False))


def main() -> None:
    reliability_panel()
    convention_table()


if __name__ == "__main__":
    main()
