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
    write_subsets(root, df)


def write_subsets(root, df: pd.DataFrame) -> None:
    """Emit the exposed-id list, and the labels for a both-tools-excluded subset.

    Mirrors toxdl2/build_clean_subset.py, which writes _shared/toxdl2_seen_in_train.txt
    and _shared_clean/. The manuscript's contamination-excluded subset used to remove
    ToxDL 2.0's overlap alone, which left ToxinPred 3.0 scored on proteins it had
    memorised; _shared_clean_both/ removes both, so the two decontaminated rows of
    Table 3 sit on one subset and are comparable to each other.

    EXPOSURE CRITERION IS `exact`, NOT `seg`. Byte-identical means the sequence is
    literally in the published training file. A shared 15-residue segment can also come
    from ordinary family homology, and excluding on it would quietly drop genuine
    conotoxins that ToxinPred 3.0 never saw -- a harder subset, but not a cleaner one.
    """
    # The COMMITTED snapshot, not benchmark/test_set/_shared. benchmark/ is gitignored and
    # regenerated, and a regenerated copy no longer lines up with the committed
    # toxdl2_seen_in_train.txt (on this tree only 118 of its 828 ids still matched). The
    # snapshot under results/ is what compare.py's --labels-dir is pointed at and what the
    # manuscript numbers were computed from, so it is the one to extend.
    shared = root / "scripts" / "external_tools" / "results" / "ground_truth"
    if not (shared / "test_labels.csv").exists():
        print(f"\n[skip] no {shared}/test_labels.csv")
        return

    exposed = set(df.loc[df.exact, "identifier"])
    (shared / "toxinpred3_seen_in_train.txt").write_text("\n".join(sorted(exposed)))

    toxdl2_file = shared / "toxdl2_seen_in_train.txt"
    if not toxdl2_file.exists():
        print(f"\n[skip] no {toxdl2_file} -- run toxdl2/build_clean_subset.py first")
        return
    toxdl2 = {ln.strip() for ln in toxdl2_file.read_text().split() if ln.strip()}

    labels = pd.read_csv(shared / "test_labels.csv")
    # A stale id list is the failure mode this whole function is exposed to, and it is
    # silent: the subset just comes out too big. Refuse instead.
    hit = int(labels.identifier.isin(toxdl2).sum())
    if hit != len(toxdl2):
        raise SystemExit(
            f"{toxdl2_file.name} lists {len(toxdl2)} ids but only {hit} are in "
            f"{shared.name}/test_labels.csv -- the two are from different splits"
        )

    seen = exposed | toxdl2
    keep = labels[~labels.identifier.isin(seen)]
    out = shared.parent / "ground_truth_clean_both"
    out.mkdir(parents=True, exist_ok=True)
    keep.to_csv(out / "test_labels.csv", index=False)
    # Full val, unchanged: it is only used to pick each method's Youden threshold, and
    # thresholding on a decontaminated val would change the operating point as well as
    # the scored subset, confounding the two.
    pd.read_csv(shared / "val_labels.csv").to_csv(out / "val_labels.csv", index=False)
    print(
        f"\nwrote {shared.name}/toxinpred3_seen_in_train.txt ({len(exposed)} ids) and "
        f"{out.name}/ ({len(keep)} proteins, {int(keep.is_toxic.sum())} toxins; "
        f"removed {len(labels) - len(keep)} = {len(toxdl2)} ToxDL 2.0 "
        f"+ {len(exposed)} ToxinPred 3.0, overlapping in "
        f"{len(exposed & toxdl2)})"
    )


if __name__ == "__main__":
    main()
