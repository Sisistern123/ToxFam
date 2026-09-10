"""Measure how much of our test split ToxinPred 3.0 was trained on.

The manuscript used to assert that ToxinPred 3.0 "is not ToxProt-trained and carries no
such overlap", in contrast to ToxDL 2.0. That is false. ToxinPred 3.0's positives were
curated in part from *reviewed UniProtKB entries matching the keywords "toxin" or "toxic"*
-- which is our KW-0800 query with extra noise -- and the only redundancy reduction applied
was removal of exact duplicates. No sequence-identity clustering. So its training set
contains our test positives outright, wherever they fall inside its <=35-residue window.

This script quantifies that, from the tool's own published dataset. Run it before quoting
any ToxinPred 3.0 overlap number in the manuscript:

    uv run python scripts/external_tools/toxinpred3_overlap.py

Two criteria are reported, and the manuscript should quote the first:

  exact  -- the test sequence is byte-identical to a ToxinPred 3.0 positive. Unassailable:
            no interpretation, no threshold. This is the number to cite.
  seg15  -- the test sequence shares an exact 15-residue segment with one. Looser, and it
            catches near-duplicates the exact test misses, but a 15-mer can also be shared
            through ordinary family homology. The script prints the non-toxin rate as the
            background so the figure can be read honestly.

It also prints what the overlap does to the tool's scores, using the committed score
snapshot. The direction matters for the paper: correcting this WIDENS ToxFam's margin,
so it is not a claim we have any incentive to soften.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from toxfam._paths import get_project_root

# The published training data: one bare sequence per line, no header, no accessions, so
# the intersection has to be by sequence string. Byte-identical to the copy on the
# webserver (webs.iiitd.edu.in/raghava/toxinpred3/download/) and in the Zenodo bundle
# (10.5281/zenodo.19877839); GitHub is used here because it needs no scraping.
TP3_BASE = "https://raw.githubusercontent.com/raghavagps/toxinpred3/main/dataset/"
POSITIVE_FILES = ("train_pos.csv", "test_pos.csv")

SEG = 15  # segment length for the loose criterion
SCORES = "scripts/external_tools/results/scores/toxinpred3/test_scores.csv"


def positives() -> set[str]:
    seqs: set[str] = set()
    for name in POSITIVE_FILES:
        with urllib.request.urlopen(TP3_BASE + name) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8").read()
        seqs |= {line.strip().upper() for line in text.split() if line.strip()}
    return seqs


def test_split(root) -> pd.DataFrame:
    manifest = pd.read_csv(root / "data" / "splits" / "split_manifest.csv")
    ids = set(manifest.loc[manifest.Split == "test", "identifier"])
    df = pd.read_csv(root / "data" / "processed" / "training_data.csv")
    df = df[df.identifier.isin(ids)][
        ["identifier", "Protein families", "Sequence"]
    ].copy()
    df["y"] = (df["Protein families"] != "nontox").astype(int)
    df["seq"] = df.Sequence.str.upper()
    return df


def main() -> None:
    root = get_project_root()
    pos = positives()
    df = test_split(root)
    n_tox = int(df.y.sum())
    print(f"ToxinPred 3.0 positives: {len(pos)}")
    print(f"test split: {len(df)} proteins, {n_tox} toxins\n")

    df["exact"] = df.seq.isin(pos)
    kmers = {p[i : i + SEG] for p in pos for i in range(len(p) - SEG + 1)}
    df["seg"] = [
        (s in pos) if len(s) < SEG else any(s[i : i + SEG] in kmers for i in range(len(s) - SEG + 1))
        for s in df.seq
    ]

    for crit in ("exact", "seg"):
        tox = int((df[crit] & df.y.astype(bool)).sum())
        non = int((df[crit] & ~df.y.astype(bool)).sum())
        label = "byte-identical" if crit == "exact" else f">={SEG}-aa segment"
        print(
            f"{label:22s} toxins {tox:4d}/{n_tox} ({100 * tox / n_tox:5.1f}%)"
            f"   non-toxins {non:4d}/{len(df) - n_tox} "
            f"({100 * non / (len(df) - n_tox):.1f}%  <- background)"
        )

    short = df.seq.str.len() <= 35  # ToxinPred 3.0's training window
    print(
        f"\ntest toxins <=35 aa: {int((short & df.y.astype(bool)).sum())}/{n_tox}; "
        f"every byte-identical toxin is one of them: "
        f"{bool((df.exact & df.y.astype(bool) & ~short).sum() == 0)}"
    )

    fam = (
        df[df.y == 1]
        .groupby("Protein families")
        .agg(n=("seg", "size"), exposed=("seg", "sum"))
    )
    fam = fam[fam.n >= 15].sort_values("exposed", ascending=False)
    print(f"\nexposure by family ({SEG}-aa criterion, families with n>=15):")
    print((fam.assign(pct=(100 * fam.exposed / fam.n).round(1))).to_string())

    scores = root / SCORES
    if not scores.exists():
        print(f"\n[skip] no score snapshot at {SCORES}")
        return
    d = df.merge(pd.read_csv(scores), on="identifier")
    print(f"\nscore impact (snapshot: {SCORES}, n={len(d)}):")
    for name, sub in (
        ("full test set", d),
        ("minus byte-identical", d[~d.exact]),
        (f"minus >={SEG}-aa segment", d[~d.seg]),
    ):
        print(
            f"  {name:24s} n={len(sub):5d} toxins={int(sub.y.sum()):4d} "
            f"prior={100 * sub.y.mean():5.2f}%  "
            f"ROC-AUC={roc_auc_score(sub.y, sub.score):.3f}  "
            f"PR-AUC={average_precision_score(sub.y, sub.score):.3f}  "
            f"MCC@0.5={matthews_corrcoef(sub.y, (sub.score >= 0.5).astype(int)):.3f}"
        )
    tox = d[d.y == 1]
    print(
        f"  sensitivity at t=0.5: exposed {(tox[tox.exact].score >= 0.5).mean():.3f} "
        f"(n={int(tox.exact.sum())})  vs unexposed "
        f"{(tox[~tox.exact].score >= 0.5).mean():.3f} (n={int((~tox.exact).sum())})"
    )


if __name__ == "__main__":
    main()
