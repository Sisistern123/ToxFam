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
    SINGLE_COL,
    apply_style,
    load_predict,
    save_fig,
)
from paper.stats import nonmetazoan_toxicity_recall

THRESHOLD = 0.5


def main() -> None:
    apply_style()
    preds = load_predict("non_metazoan")
    s = nonmetazoan_toxicity_recall(preds, threshold=THRESHOLD)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.72), layout="constrained")
    ax.hist(preds["p_toxic"], bins=30, color=METHODS["nn_combined_run"][1], edgecolor="white")
    ax.axvline(THRESHOLD, color="#333333", ls=":", lw=1.0)
    ax.annotate(
        f"threshold {THRESHOLD:.2f}",
        xy=(THRESHOLD, ax.get_ylim()[1]),
        xytext=(2, -2),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=7,
        color="#333333",
    )
    ax.set_xlabel("Predicted P(toxic)")
    ax.set_ylabel("Proteins")
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Known non-metazoan toxins (n={s['n']:,})\n"
        f"recall {s['recall']:.0%} · median P(toxic) {s['median_p_toxic']:.2f}"
    )
    save_fig(fig, "figure_supp_nonmetazoan")

    print(
        f"non-metazoan: n={s['n']}  recall@{THRESHOLD}={s['recall']:.1%}  "
        f"median P(toxic)={s['median_p_toxic']:.3f}  flagged={s['n_flagged']}"
    )


if __name__ == "__main__":
    main()
