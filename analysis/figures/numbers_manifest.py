"""Emit analysis/manuscript_figures/results_numbers.json — every cited number."""
from __future__ import annotations

import json

import pandas as pd

from analysis.figures._common import FIG_DIR, load_preds, test_class_list
from toxfam.evaluation.manuscript import (
    adjudication_summary, bootstrap_label_metric_ci, correctness, macro_mcc_by_support,
    mcnemar_test, micro_mcc, overall_mcc, paired_bootstrap_accuracy_diff, subset_accuracy,
    toxin_mask,
)
from toxfam._paths import benchmark_dir, get_project_root
from toxfam.evaluation.hbi import NO_HIT_LABEL

ADJ_CSV = get_project_root() / "analysis" / "model_test_wrong_conf_annotated.csv"


def main() -> None:
    classes = test_class_list()
    hbi = load_preds("test_set", "hbi"); nn = load_preds("test_set", "nn_combined_run")
    std = load_preds("test_set", "nn_standard_run")
    nohit = hbi["predicted_label"] == NO_HIT_LABEL
    nn_nh = nn[nn["identifier"].isin(hbi.loc[nohit, "identifier"])]
    out = {
        "n_test": int(len(nn)),
        "non_toxin_prior": round(float((nn["actual_label"].str.lower() == "nontox").mean()), 4),
        "toxin_only_n": int(toxin_mask(nn).sum()),
        "toxin_only_acc": {"nn_combined": subset_accuracy(nn, toxin_mask(nn)),
                            "hbi": subset_accuracy(hbi, toxin_mask(hbi))},
        "all_class_acc": {"nn_combined": subset_accuracy(nn), "hbi": subset_accuracy(hbi)},
        "mcnemar": mcnemar_test(correctness(nn), correctness(hbi)),
        "paired_bootstrap": paired_bootstrap_accuracy_diff(correctness(nn), correctness(hbi)),
        "no_hit": {"n": int(nohit.sum()), "n_toxin": int(toxin_mask(nn_nh).sum()),
                   "nn_acc": subset_accuracy(nn_nh)},
        "mcc": {
            m: {"overall": overall_mcc(d["actual_label"], d["predicted_label"]),
                "micro": micro_mcc(d["actual_label"], d["predicted_label"], class_list=classes)}
            for m, d in [("hbi", hbi), ("nn_standard", std), ("nn_combined", nn)]
        },
        "mcc_ci_nn_combined": bootstrap_label_metric_ci(
            nn["actual_label"].values, nn["predicted_label"].values, overall_mcc, n_boot=2000),
        "macro_mcc_by_support": macro_mcc_by_support(nn, hbi, class_list=classes).to_dict("records"),
        "adjudication": adjudication_summary(ADJ_CSV),
    }

    # Non-metazoan scope boundary — a NEGATIVE OOD result (ToxFam is metazoan-specific).
    # benchmark/ is gitignored; include only if the regenerated predictions are present.
    nm_path = benchmark_dir() / "non_metazoan" / "nn_combined_run" / "predictions.csv"
    if nm_path.exists():
        nm = pd.read_csv(nm_path)
        recognized = int(toxin_mask(nm, "predicted_label").sum())
        out["non_metazoan_scope_boundary"] = {
            "n": int(len(nm)),
            "recognized_as_toxic": recognized,
            "recognition_rate": round(recognized / len(nm), 4) if len(nm) else 0.0,
            "note": "all entries are non-metazoan toxins; low rate = ToxFam is metazoan-specific.",
        }

    (FIG_DIR / "results_numbers.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
