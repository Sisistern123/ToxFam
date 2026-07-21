"""Supplementary figure: ProtT5 embedding space of the ToxFam representative set.

Draws the ProtSpace-computed UMAP projection twice:

  A  all 65,179 representatives, non-toxins greyed out, toxins highlighted --
     shows that toxicity is largely a *global* property of the ProtT5 space.
  B  toxins only, coloured by curated family -- shows that the family signal the
     MLP head exploits is already present as local cluster structure.

Reads the unbundled parquet parts written by ``paper.protspace_bundles``
(``make protspace``); run that first or this raises.

Projection hyperparameters are pinned in :mod:`paper.protspace_bundles`
(n_neighbors=25, min_dist=0.1, metric=euclidean, random_state=42) and must be
restated in the caption -- UMAP is seed dependent.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from paper._paths import protspace_bundle_dir
from paper.figures._common import DOUBLE_COL, apply_style, console, save_fig

PROJECTION = "ProtT5 — UMAP 2"

# Colours are bound to families BY NAME, not by rank, because which pairs may safely be
# similar depends on where their clusters land. Measured centroid separations (fraction
# of plot width) put five families in one crowded core -- Conotoxin, Neurotoxin,
# Scoloptoxin, Long- and Short-scorpion toxin, all within 0.11-0.24 of each other -- and
# four on isolated islands. The core takes the most separable hues; the islands absorb
# the weaker ones, where similarity cannot cause confusion. Binding by rank instead once
# put two near-identical dark reds 0.13 apart, inside the same cluster.
# Re-check this binding by eye after any reprojection: a new seed moves the clusters.
FAMILY_ORDER = [
    "Conotoxin", "Neurotoxin", "Scoloptoxin",            # core
    "Long scorpion toxin", "Short scorpion toxin",       # core
    "Three-finger toxin", "Phospholipase",               # isolates
    "Venom metalloproteinase", "Snaclec",                # isolates
]

# Kelly's first nine "maximum contrast" colours. Chosen over an Okabe-Ito-based
# alternative that scored better on paper (normal deltaE 13.5 / deutan 7.6 vs Kelly's
# 12.4 / protan 3.1) -- Kelly reads better at this mark size against the grey backdrop.
# Neither clears the floors: nine categories exceed what categorical colour can separate,
# so the ordering below matters. The five crowded-core families take the most separable
# hues; Kelly's protan-confusable pair (#0067A5 vs #875692) is split across core and
# isolate so the two never sit adjacent on the page.
FAMILY_COLORS = ["#F38400", "#0067A5", "#BE0032", "#008856", "#875692",
                 "#F3C300", "#E68FAC", "#C2B280", "#A1CAF1"]
FALLBACK_COLOR = "#999999"
N_FAMILIES = len(FAMILY_ORDER)

GREY = "#D9D9D9"      # backdrop / residual category
TOXIN_DARK = "#333333"   # not blue: blue is a FAMILY colour in panel B, in both palettes
INK = "#333333"       # axis glyph + panel furniture

# Legends sit INSIDE the axes to buy plot width at fixed journal column width, so the
# curated family names have to be shortened to fit. Counts stay: they carry the class
# imbalance the per-family results depend on.
SHORT_NAMES = {
    "Long (4 C-C) scorpion toxin superfamily": "Long scorpion toxin",
    "Short scorpion toxin superfamily": "Short scorpion toxin",
    "Venom metalloproteinase (M12B) family": "Venom metalloproteinase",
}
# 7 pt is the TOP of Bioinformatics' 5-7 pt body range, and it only fits panel B because
# the per-family counts moved OUT of the legend and into the caption: dropping "(753)"
# from every entry makes each label single-line and much narrower, which is what buys the
# type size without the legend growing leftward over the point cloud. Panel A keeps its
# counts inline -- two short entries, and the space is there.
LEGEND_FONTSIZE = 7
LEGEND_MARKERSIZE = 4.5  # points; identical for every entry, see legend_handle()
LEGEND_KW = {
    "frameon": False,  # the panels have no box; a legend frame would be the only rule
    "fontsize": LEGEND_FONTSIZE,
    "handletextpad": 0.4,
    "borderpad": 0.3,
    "labelspacing": 0.4,
}


def legend_handle(color: str, name: str, count: int | None = None) -> Line2D:
    """A legend entry whose dot size is independent of the plotted mark size.

    The plotted marks differ in size deliberately (the non-toxin backdrop is smaller than
    the toxins so 61,763 points cannot out-shout 3,416), but that must not leak into the
    legend, where a smaller dot reads as a different KIND of thing rather than as
    background. Every legend dot is therefore drawn at one fixed size via a proxy handle,
    instead of scaling the real series with `markerscale`.

    ``count=None`` omits the count; the per-family numbers live in the caption instead.
    """
    label = name if count is None else f"{name} ({count:,})"
    return Line2D([], [], linestyle="", marker="o", markersize=LEGEND_MARKERSIZE,
                  color=color, label=label)


def split_columns(handles: list[Line2D]) -> list[Line2D]:
    """Reorder legend entries so short labels fill one column and long labels the other.

    A two-column legend is as wide as (widest label in column 1) + (widest in column 2),
    so pairing short-with-short and long-with-long is the narrowest possible arrangement
    -- mixing them makes BOTH columns as wide as their own longest entry. Two columns are
    needed here for height, not width: a single column runs down into a grey "other"
    cluster on the right of the panel.

    matplotlib fills column-major, so the first half of the returned list becomes the left
    column. Size ordering is preserved WITHIN each column, and the "other" residual stays
    last overall, at the foot of the right column.
    """
    ordered = sorted(handles, key=lambda h: len(h.get_label()))
    half = len(ordered) // 2
    short, long = ordered[:half], ordered[half:]
    # restore the original (size-descending, "other"-last) order inside each column
    rank = {h.get_label(): i for i, h in enumerate(handles)}
    key = lambda h: rank[h.get_label()]
    return sorted(short, key=key) + sorted(long, key=key)


def family_color(family: str) -> str:
    """Colour for a curated family name, matched on the short (display) form.

    Falls back to grey rather than raising: if re-curation renames a family, the figure
    still renders and the grey mark makes the omission obvious instead of silent.
    """
    name = short_name(family)
    if name not in FAMILY_ORDER:
        return FALLBACK_COLOR
    return FAMILY_COLORS[FAMILY_ORDER.index(name)]


def short_name(family: str) -> str:
    """Legend label for a curated family name, trimmed to fit inside the axes."""
    if family in SHORT_NAMES:
        return SHORT_NAMES[family]
    return family.replace(" family", "")


def panel_header(ax, letter: str, subtitle: str) -> None:
    """Panel letter and subtitle sharing one TRUE baseline at the axes' left edge.

    Uses va="baseline", not va="bottom". Bottom alignment matches bounding boxes, and the
    subtitle has descenders ("p", "y", parentheses) while a capital letter has none -- so
    its box hangs lower and its visible baseline rides up. The letter stays bold and the
    subtitle regular: the bold letter is the panel's index, the subtitle is prose, and
    bolding both would leave nothing to distinguish them.
    """
    ax.text(0, 1.03, letter, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="baseline", ha="left")
    ax.text(0.035, 1.03, subtitle, transform=ax.transAxes,
            fontsize=7, va="baseline", ha="left")


def axis_glyph(ax) -> None:
    """A small corner L instead of a full axis frame.

    UMAP coordinates have no units and no meaningful origin or scale, so ticks, a box,
    and full-length spines all assert precision the projection does not have. A corner
    glyph states the two directions and nothing more.
    """
    kw = {"transform": ax.transAxes, "color": INK, "lw": 0.6, "clip_on": False}
    ax.plot([0.0, 0.0], [0.0, 0.055], **kw)
    ax.plot([0.0, 0.055], [0.0, 0.0], **kw)
    ax.text(0.068, -0.004, "UMAP 1", transform=ax.transAxes, fontsize=5,
            color=INK, va="bottom", ha="left")
    ax.text(-0.006, 0.068, "UMAP 2", transform=ax.transAxes, fontsize=5,
            color=INK, va="bottom", ha="left", rotation=90)


def load(out_dir: Path) -> pd.DataFrame:
    data = out_dir / "projections_data.parquet"
    if not data.exists():
        raise FileNotFoundError(f"{data} missing -- run `make protspace` first")
    proj = pd.read_parquet(data)
    proj = proj[proj["projection_name"] == PROJECTION]
    ann = pd.read_parquet(out_dir / "selected_annotations.parquet")
    return proj.merge(ann, left_on="identifier", right_on="protein_id")


def cluster_stats(toxin_dir: Path) -> dict[str, float]:
    """UMAP-vs-``family`` agreement scores, for the caption.

    Reports adjusted Rand and NMI rather than silhouette: silhouette assumes one
    convex cluster per label and so penalises genuinely multi-modal families (e.g.
    Conotoxin, which occupies several distinct lobes), scoring -0.057 despite the
    clear per-lobe separation the panel shows.
    """
    stats = pd.read_parquet(toxin_dir / "statistics.parquet")
    umap = stats[
        (stats["space_name"] == PROJECTION) & (stats["stat_family"] == "cluster_agreement")
    ]
    return dict(zip(umap["metric"], umap["value"]))


def main(
    out_dir: Path | None = None,
    toxin_dir: Path | None = None,
) -> None:
    out_dir = out_dir or protspace_bundle_dir("all")
    toxin_dir = toxin_dir or protspace_bundle_dir("toxin")
    apply_style()
    df = load(out_dir)
    # Panel B uses a UMAP fitted on the toxins alone -- in the global projection the
    # 3,416 toxins occupy one small region and no family structure is resolvable.
    tox = load(toxin_dir)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2))

    # --- A: whole space, toxins on top of a grey non-toxin backdrop ---
    non = df[df["toxic"] == "non-toxin"]
    tox_global = df[df["toxic"] == "toxin"]
    # The backdrop is context, not a series: smaller and lighter than the focal points
    # so 61,763 grey markers cannot out-shout 3,416 blue ones.
    ax_a.scatter(non["x"], non["y"], s=0.8, c=GREY, linewidths=0, rasterized=True)
    ax_a.scatter(tox_global["x"], tox_global["y"], s=2.0, c=TOXIN_DARK,
                 linewidths=0, rasterized=True)
    ax_a.legend(handles=[legend_handle(GREY, "non-toxin", len(non)),
                         legend_handle(TOXIN_DARK, "toxin", len(tox_global))],
                loc="upper right", **LEGEND_KW)
    panel_header(ax_a, "A", f"all representatives (n={len(df):,})")

    # --- B: toxins only, coloured by the six largest families ---
    counts = tox["family"].value_counts().drop(labels=["other"], errors="ignore")
    top = list(counts.head(N_FAMILIES).index)
    rest = tox[~tox["family"].isin(top)]
    n_rest_families = rest["family"].nunique()
    # Greyed remainder first so the coloured families draw on top of it.
    ax_b.scatter(rest["x"], rest["y"], s=1.5, c=GREY, linewidths=0, rasterized=True)
    handles = []
    for family in top:
        sub = tox[tox["family"] == family]
        color = family_color(family)
        ax_b.scatter(sub["x"], sub["y"], s=3.5, c=color, linewidths=0, rasterized=True)
        handles.append(legend_handle(color, short_name(family)))
    # "Other" is a residual, not a peer of the named families -- it names how many
    # families it absorbs rather than only how many proteins, so the reader sees a long
    # tail instead of one big unnamed group.
    handles.append(legend_handle(GREY, f"{n_rest_families} other families"))
    ax_b.legend(handles=split_columns(handles), loc="upper right", ncol=2,
                columnspacing=1.0, **LEGEND_KW)
    panel_header(ax_b, "B", f"toxins only, refitted UMAP (n={len(tox):,})")

    # Headroom for the in-axes legends, added at the TOP only -- ax.margins() is
    # symmetric and would leave as much dead space below the cloud as it opens above.
    # Panel A's legend overlays empty space and needs no headroom. Panel B's does not fit
    # at 7 pt -- its left column reaches the three-finger cluster -- so a little top
    # headroom drops the cloud clear of it. This clearance is MEASURED, not structural:
    # re-check it after any reprojection, because a new seed moves the clusters.
    for ax, headroom in ((ax_a, 0.0), (ax_b, 0.0)):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.margins(0.03)
        if headroom:
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + (hi - lo) * headroom)
        axis_glyph(ax)

    fig.tight_layout()
    save_fig(fig, "figure_embedding_space")

    stats = cluster_stats(toxin_dir)
    console.print(
        "toxin UMAP vs family: "
        f"ARI={stats['adjusted_rand']:.3f}, NMI={stats['normalized_mutual_info']:.3f}"
    )


if __name__ == "__main__":
    main()
