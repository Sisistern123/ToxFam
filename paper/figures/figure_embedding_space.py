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
import pandas as pd

from paper._paths import protspace_bundle_dir
from paper.figures._common import DOUBLE_COL, apply_style, console, save_fig

PROJECTION = "ProtT5 — UMAP 2"
# Paul Tol 'bright' + 'muted' -- 10 hues that stay separable under deuteranopia.
FAMILY_COLORS = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#EE8866", "#44BB99", "#BBCC33", "#994455",
]
N_FAMILIES = 10


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
    ax_a.scatter(non["x"], non["y"], s=1, c="#DDDDDD", linewidths=0, rasterized=True)
    ax_a.scatter(tox_global["x"], tox_global["y"], s=1.5, c="#0072B2",
                 linewidths=0, rasterized=True)
    ax_a.set_title(f"all representatives (n={len(df):,})", loc="left")

    # --- B: toxins only, coloured by the N largest families ---
    top = tox["family"].value_counts().drop(labels=["other"], errors="ignore")
    top = list(top.head(N_FAMILIES).index)
    rest = tox[~tox["family"].isin(top)]
    ax_b.scatter(rest["x"], rest["y"], s=2, c="#DDDDDD", linewidths=0,
                 rasterized=True, label="other families")
    for family, color in zip(top, FAMILY_COLORS):
        sub = tox[tox["family"] == family]
        ax_b.scatter(sub["x"], sub["y"], s=3, c=color, linewidths=0, rasterized=True,
                     label=f"{family.replace(' family', '')} ({len(sub)})")
    ax_b.set_title(f"toxins only, refitted UMAP (n={len(tox):,})", loc="left")
    ax_b.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                markerscale=4, handletextpad=0.3, borderaxespad=0)

    for ax in (ax_a, ax_b):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    fig.tight_layout()
    save_fig(fig, "figure_embedding_space")

    stats = cluster_stats(toxin_dir)
    console.print(
        "toxin UMAP vs family: "
        f"ARI={stats['adjusted_rand']:.3f}, NMI={stats['normalized_mutual_info']:.3f}"
    )


if __name__ == "__main__":
    main()
