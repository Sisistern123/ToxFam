"""Generate taxonomy sunburst plots for toxin and non-toxin proteins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from rich.console import Console

from toxfam._paths import get_project_root
from toxfam.data.taxonomy import _resolve_lineages

console = Console()

RANKS = ["kingdom", "phylum", "class", "order", "family"]


def _build_sunburst_df(
    df: pd.DataFrame,
    lineages: dict[int, dict[str, str]],
) -> pd.DataFrame:
    """Map each protein to its lineage and count occurrences at each rank."""
    taxid_col = pd.to_numeric(df["Organism (ID)"], errors="coerce")

    rows: list[dict[str, str]] = []
    for taxid in taxid_col:
        if pd.isna(taxid):
            continue
        lin = lineages.get(int(taxid), {})
        rows.append({r: lin.get(r, "") for r in RANKS})

    lin_df = pd.DataFrame(rows)

    # Build parent-child pairs for sunburst
    sunburst_rows: list[dict[str, str | int]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in lin_df.iterrows():
        parent = ""
        for rank in RANKS:
            child = row[rank]
            if not child:
                break
            key = (child, parent)
            if key not in seen:
                seen.add(key)
                sunburst_rows.append(
                    {"id": child, "parent": parent, "rank": rank, "count": 0}
                )
            # Increment count
            for sr in sunburst_rows:
                if sr["id"] == child and sr["parent"] == parent:
                    sr["count"] += 1  # type: ignore[operator]
                    break
            parent = child

    return pd.DataFrame(sunburst_rows)


def _build_sunburst_data(
    df: pd.DataFrame,
    lineages: dict[int, dict[str, str]],
) -> tuple[list[str], list[str], list[int], list[str]]:
    """Build parallel lists for plotly sunburst: ids, parents, values, labels."""
    taxid_col = pd.to_numeric(df["Organism (ID)"], errors="coerce")

    # Collect lineage path per protein
    paths: list[list[str]] = []
    for taxid in taxid_col:
        if pd.isna(taxid):
            continue
        lin = lineages.get(int(taxid), {})
        path = []
        for r in RANKS:
            val = lin.get(r, "")
            if not val:
                break
            path.append(val)
        paths.append(path)

    # Count occurrences per (node, parent_path) to handle duplicate names
    # Use full path as unique id to avoid collisions (e.g., same family name
    # in different orders)
    from collections import Counter

    node_counts: Counter[tuple[str, str]] = Counter()  # (full_id, parent_id) -> count

    for path in paths:
        for depth in range(len(path)):
            full_id = " - ".join(path[: depth + 1])
            parent_id = " - ".join(path[:depth]) if depth > 0 else ""
            node_counts[(full_id, parent_id)] += 1

    ids: list[str] = []
    parents: list[str] = []
    values: list[int] = []
    labels: list[str] = []

    for (full_id, parent_id), count in sorted(node_counts.items()):
        ids.append(full_id)
        parents.append(parent_id)
        values.append(count)
        # Show only the leaf name, not the full path
        labels.append(full_id.split(" - ")[-1])

    return ids, parents, values, labels


def create_sunburst(
    df: pd.DataFrame,
    lineages: dict[int, dict[str, str]],
    title: str,
    output_path: Path,
) -> None:
    """Create and save a sunburst plot."""
    ids, parents, values, labels = _build_sunburst_data(df, lineages)

    fig = go.Figure(
        go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percentRoot:.1%} of total<extra></extra>",
            maxdepth=4,
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        width=1000,
        height=1000,
        margin=dict(t=60, l=10, r=10, b=10),
    )

    fig.write_html(output_path.with_suffix(".html"))
    fig.write_image(output_path.with_suffix(".png"), scale=2)
    console.print(f"  Saved: [cyan]{output_path.with_suffix('.html')}[/]")
    console.print(f"  Saved: [cyan]{output_path.with_suffix('.png')}[/]")


def main() -> None:
    root = get_project_root()
    csv_path = root / "data" / "processed" / "training_data.csv"
    figures_dir = root / "figures" / "taxonomy"
    figures_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"Reading [cyan]{csv_path}[/] ...")
    df = pd.read_csv(csv_path)

    tox_df = df[df["Protein families"] != "nontox"]
    nontox_df = df[df["Protein families"] == "nontox"]
    console.print(f"  Toxin proteins: {len(tox_df)}")
    console.print(f"  Non-toxin proteins: {len(nontox_df)}")

    # Resolve lineages for all unique taxon IDs
    all_taxids = (
        pd.to_numeric(df["Organism (ID)"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    console.print(f"Resolving lineages for {len(all_taxids)} unique taxon IDs ...")
    raw = _resolve_lineages(all_taxids)
    lineages = {taxid: rank_dict for taxid, (rank_dict, _) in raw.items()}

    console.print("\nCreating toxin sunburst ...")
    create_sunburst(
        tox_df,
        lineages,
        title=f"Taxonomy of Toxin Proteins (n={len(tox_df):,})",
        output_path=figures_dir / "sunburst_toxin",
    )

    console.print("\nCreating non-toxin sunburst ...")
    create_sunburst(
        nontox_df,
        lineages,
        title=f"Taxonomy of Non-Toxin Proteins (n={len(nontox_df):,})",
        output_path=figures_dir / "sunburst_nontoxin",
    )

    console.print("\n[bold green]Done![/]")


if __name__ == "__main__":
    main()
