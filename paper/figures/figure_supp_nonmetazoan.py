"""Supplementary: does the combined model flag toxicity outside Metazoa?

Every entry in the non-metazoan set is a reviewed KW-0800 toxin, so every row is a true
positive and **recall is the only measurable quantity** — specificity would need a
non-metazoan *non-toxin* set, which does not exist. The figure is therefore a P(toxic)
distribution with the decision threshold marked, not an ROC curve.

Two independent reasons to expect the model to struggle here, both worth stating because
they point at different fixes:

* the taxonomy branch is trained on 50 *metazoan* taxa, and every non-metazoan organism
  falls outside them, so the branch contributes a zero vector for all 812 entries;
* the sequences themselves are out-of-distribution — the model has never seen a
  non-metazoan toxin.

This is a negative result that motivates future work (non-metazoan training data, domain
adaptation, a taxonomy vocabulary beyond Metazoa), not an application claim.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from paper.figures._common import (
    METHODS,
    NEUTRAL,
    SINGLE_COL,
    apply_style,
    deployed_binary_threshold,
    load_predict,
    save_fig,
)
from paper.stats import nonmetazoan_toxicity_recall


def main() -> None:
    apply_style()
    # The deployed operating point, read from the checkpoint rather than hardcoded:
    # t* is in calibrated (Platt) score space and `toxfam predict` writes a calibrated
    # p_toxic, so 0.5 is NOT the decision threshold -- reporting recall at 0.5
    # understated it by an order of magnitude (13/812 against 218/812).
    threshold = deployed_binary_threshold()
    preds = load_predict("non_metazoan")
    # The figure re-thresholds p_toxic, so it can only be right while the calibrator it
    # reads is the one `toxfam predict` used to write predicted_toxic. Re-deploying the
    # calibrator without re-running predict would silently move the line; check rather
    # than trust.
    n_flagged = int((preds["p_toxic"] >= threshold).sum())
    if n_flagged != int(preds["predicted_toxic"].sum()):
        raise SystemExit(
            f"threshold {threshold:.6f} flags {n_flagged} proteins, but the predict "
            f"output records {int(preds['predicted_toxic'].sum())}. The deployed "
            "calibrator has moved since predictions.tsv was written -- re-run "
            "'uv run toxfam predict non_metazoan ...' before rendering this figure."
        )
    s = nonmetazoan_toxicity_recall(preds, threshold=threshold)

    fig, ax = plt.subplots(
        figsize=(SINGLE_COL, SINGLE_COL * 0.72), layout="constrained"
    )
    ax.hist(
        preds["p_toxic"],
        bins=30,
        color=METHODS["nn_combined_run"][1],
        edgecolor="white",
    )
    ax.axvline(threshold, color=NEUTRAL["ink"], ls=":", lw=1.0)
    ax.annotate(
        f"deployed threshold {threshold:.3f}",
        xy=(threshold, ax.get_ylim()[1]),
        xytext=(2, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color=NEUTRAL["ink"],
    )
    ax.set_xlabel("Predicted P(toxic)")
    ax.set_ylabel("Proteins")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Known non-metazoan toxins (n={s['n']:,})\n"
        f"recall {s['recall']:.0%} · median P(toxic) {s['median_p_toxic']:.3f}"
    )
    save_fig(fig, "figure_supp_nonmetazoan")

    print(
        f"non-metazoan: n={s['n']}  recall@{threshold:.3f}={s['recall']:.1%}  "
        f"median P(toxic)={s['median_p_toxic']:.3f}  flagged={s['n_flagged']}"
    )


if __name__ == "__main__":
    main()
