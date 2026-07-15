"""Supplementary: the combined model as a family-annotation suggestion tool (TrEMBL).

The unreviewed set is TrEMBL, so its family labels are auto-assigned rather than curated.
It is **not** an evaluation set and nothing here is an accuracy claim — the point is that
a large real-world pool splits into two useful cases, one per panel:

* **A — gap filling.** A large fraction of entries carry no UniProt family at all. The
  model emits three ranked candidates for every protein, so it proposes an annotation
  where none exists.
* **B — corroboration.** Where UniProt *does* carry a family the model's 38-class
  vocabulary can express, how often is that family the model's top pick, and how often
  does it appear anywhere in the top-3? A label that is the model's second suggestion is
  still a useful, reviewable agreement.

Panel B is restricted to entries whose collapsed family is a *specific* family: agreeing
with the "other" catch-all or with "nontox" would say nothing about naming the right
family, and including them would inflate the agreement rate.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from paper.figures._common import (
    ADJUDICATION,
    DOUBLE_COL,
    METHODS,
    apply_style,
    load_predict,
    model_vocab,
    panel_label,
    save_fig,
    unreviewed_families,
)
from paper.stats import unreviewed_annotation_summary

TOP_K = 3
GREY = "#BBBBBB"


def main() -> None:
    apply_style()
    preds = load_predict("unreviewed")
    families = preds["identifier"].map(unreviewed_families())
    s = unreviewed_annotation_summary(preds, families, vocab=model_vocab(), top_k=TOP_K)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.34), layout="constrained"
    )

    # --- A: annotation coverage -------------------------------------------------
    counts = [s["n_annotated"], s["n_unannotated"]]
    bars = axa.bar(
        ["Has UniProt family", "No family"],
        counts,
        color=[METHODS["nn_combined_run"][1], GREY],
        edgecolor="white",
    )
    axa.bar_label(
        bars,
        labels=[f"{v:,}\n({v / s['n']:.0%})" for v in counts],
        padding=2,
        fontsize=7,
    )
    axa.set_ylim(0, max(counts) * 1.25)
    axa.set_ylabel("Proteins")
    axa.set_title(f"Annotation coverage (n={s['n']:,})")
    panel_label(axa, "A")

    # --- B: rank of the UniProt family among the model's top-3 -------------------
    labels = [f"top-{i}" for i in range(1, TOP_K + 1)] + [f"not in top-{TOP_K}"]
    vals = [s["rank_counts"][f"top_{i}"] for i in range(1, TOP_K + 1)]
    vals.append(s["rank_counts"]["not_in_top_k"])
    # Luminance-ordered good->bad (never green/red: the deuteranopia failure case).
    colors = [ADJUDICATION["correct"], "#3C7DBF", "#8FB8DC", ADJUDICATION["incorrect"]]
    bars = axb.bar(labels, vals, color=colors, edgecolor="white")
    axb.bar_label(bars, labels=[f"{v:,}" for v in vals], padding=2, fontsize=7)
    axb.set_ylim(0, max(vals) * 1.18)
    axb.set_ylabel("Annotated proteins")
    axb.set_title(
        f"Rank of the UniProt family (n={s['n_comparable']:,})\n"
        f"top-1 {s['top_1']:.0%} · in top-{TOP_K} {s['top_k']:.0%}"
    )
    panel_label(axb, "B")

    save_fig(fig, "figure_supp_unreviewed")

    print(
        f"unreviewed: n={s['n']:,}  unannotated={s['n_unannotated']:,} "
        f"({s['frac_unannotated']:.0%})  comparable={s['n_comparable']:,}  "
        f"top-1={s['top_1']:.1%}  top-{TOP_K}={s['top_k']:.1%}  "
        f"out-of-vocab={s['n_out_of_vocab']:,} "
        f"({s['n_out_of_vocab_families']} distinct families)"
    )


if __name__ == "__main__":
    main()
