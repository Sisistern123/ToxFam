"""Supplementary figure: sub-family structure inside two curated toxin families.

Zooms into the toxin-only UMAP (see :mod:`paper.figures.figure_embedding_space`) and
re-colours two families by a sub-classification the model never saw. Both sub-labels are
recovered from the RAW UniProt family string, which is hierarchical -- "Venom
metalloproteinase (M12B) family, P-III subfamily, P-IIIa sub-subfamily" -- and which the
training pipeline collapses to the top-level family. So these labels exist nowhere in the
training data: any structure here is unsupervised.

  A  Snake venom metalloproteinases by domain-architecture class. P-I is the
     metalloproteinase domain alone, P-II adds a disintegrin domain, P-III adds
     disintegrin-like + cysteine-rich domains; the classes arose by successive domain
     loss from an ancestral P-III (Casewell et al. 2011). The three classes are almost
     perfectly separated in the projection.

  B  Conotoxins by gene superfamily. Note that superfamilies are DEFINED by the
     hyper-conserved signal peptide, which this pipeline removes with SignalP6 before
     embedding -- so any separation here is driven by mature-region and cysteine-framework
     divergence, not by the defining region itself.

Numbers quoted in the caption are printed by this script; regenerate before citing.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from paper._paths import protspace_bundle_dir
from paper.figures._common import DOUBLE_COL, apply_style, console, save_fig
from paper.figures.figure_embedding_space import (
    GREY,
    LEGEND_KW,
    axis_glyph,
    legend_handle,
    load,
    panel_header,
)
from toxfam._paths import raw_dir

# Validated six-slot palette: PASSES all-pairs CVD (worst deutan deltaE 11.0) and the
# normal-vision floor, so plain circles need no shape encoding. Six is the ceiling --
# see figure_embedding_space for the search that establishes it.
SUB_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#882255", "#D55E00", "#7570B3"]
PAD = 0.12  # zoom window padding, as a fraction of each family's extent


def _svmp_class(families: str) -> str | None:
    """P-I / P-II / P-III from the raw hierarchical UniProt family string."""
    match = re.search(r"(P-I{1,3})\s+subfamily", str(families))
    return match.group(1) if match else None


def _cono_superfamily(families: str) -> str | None:
    """Gene superfamily letter code, with I1/I2 merged.

    I1 and I2 have near-identical centroids in the projection and comparable size, so
    splitting them would present an arbitrary division as a finding.
    """
    match = re.search(r"Conotoxin\s+([A-Z][0-9]?)\s+superfamily", str(families))
    if not match:
        return None
    code = match.group(1)
    return "I" if code in {"I1", "I2"} else code


def load_with_subclass() -> pd.DataFrame:
    """Toxin projection joined to the raw (un-collapsed) UniProt family string."""
    raw = pd.read_csv(raw_dir() / "0800.tsv", sep="\t").rename(columns={"Entry": "identifier"})
    tox = load(protspace_bundle_dir("toxin"))
    return tox.merge(raw[["identifier", "Protein families"]], on="identifier", how="left")


def separability(sub: pd.DataFrame) -> tuple[float, float]:
    """5-NN cross-validated accuracy on the 2D coordinates, and the majority baseline.

    Reported instead of silhouette: silhouette assumes one convex cluster per label and
    goes negative for multi-lobed classes even when they are perfectly distinguishable.
    Neighbour accuracy asks the question that matters -- can you tell the class from where
    the point sits?
    """
    X, y = sub[["x", "y"]].values, sub["cls"].values
    acc = cross_val_score(KNeighborsClassifier(5), X, y, cv=5).mean()
    return acc, y.tolist().count(max(set(y), key=y.tolist().count)) / len(y)


def draw_panel(ax, tox, sub, letter: str, title: str, min_n: int) -> tuple[float, float]:
    """One zoomed panel: the family's classes coloured over a grey local backdrop."""
    keep = sub["cls"].value_counts()
    sub = sub[sub["cls"].isin(keep[keep >= min_n].index)]

    # Zoom window = the family's own extent, padded.
    (x0, x1), (y0, y1) = (sub["x"].min(), sub["x"].max()), (sub["y"].min(), sub["y"].max())
    dx, dy = (x1 - x0) * PAD, (y1 - y0) * PAD
    x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy

    # Every other toxin inside the window, greyed -- shows the family against its
    # neighbourhood rather than floating in a void.
    near = tox[tox["x"].between(x0, x1) & tox["y"].between(y0, y1)]
    ax.scatter(near["x"], near["y"], s=2, c=GREY, linewidths=0, rasterized=True)

    handles = []
    order = keep[keep >= min_n].index  # size-descending
    for cls, color in zip(order, SUB_COLORS):
        g = sub[sub["cls"] == cls]
        ax.scatter(g["x"], g["y"], s=7, c=color, linewidths=0, rasterized=True)
        handles.append(legend_handle(color, f"{cls} ({len(g)})"))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(handles=handles, loc="upper right", ncol=1, **LEGEND_KW)
    axis_glyph(ax)
    panel_header(ax, letter, title)
    return separability(sub)


def main() -> None:
    apply_style()
    tox = load_with_subclass()

    svmp = tox[tox["family"].str.contains("metalloproteinase", case=False, na=False)].copy()
    svmp["cls"] = svmp["Protein families"].map(_svmp_class)
    svmp = svmp.dropna(subset=["cls"])

    cono = tox[tox["family"].str.contains("Conotoxin", case=False, na=False)].copy()
    cono["cls"] = cono["Protein families"].map(_cono_superfamily)
    cono = cono.dropna(subset=["cls"])

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2))
    acc_s, base_s = draw_panel(ax_a, tox, svmp, "A",
                               "venom metalloproteinase, by class", min_n=5)
    acc_c, base_c = draw_panel(ax_b, tox, cono, "B",
                               "conotoxin, by gene superfamily", min_n=15)
    fig.tight_layout()
    save_fig(fig, "figure_embedding_subclasses")

    console.print(f"SVMP     5-NN acc {acc_s:.3f} (baseline {base_s:.3f}), n={len(svmp)}")
    console.print(f"Conotoxin 5-NN acc {acc_c:.3f} (baseline {base_c:.3f}), n={len(cono)}")


if __name__ == "__main__":
    main()
