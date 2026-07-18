"""Regression guards for the stratified split in toxfam.data.preprocessing.

These pin the reported train/val/test partition against accidental seed or
proportion edits — not a leakage proof (disjointness is inherent to the splitter).
"""

from __future__ import annotations

import pandas as pd

from toxfam.data.preprocessing import _cluster_cache_key, multilabel_stratified_splits


def _sample_reps() -> pd.DataFrame:
    # Each family appears several times (required by stratified splitting); a few
    # rows are multi-label to exercise the comma split/rejoin roundtrip.
    fams = (
        ["famA"] * 8
        + ["famB"] * 8
        + ["famC"] * 8
        + ["famA,famB", "famB,famC", "famA,famC"] * 2
    )
    return pd.DataFrame(
        {
            "identifier": [f"P{i:03d}" for i in range(len(fams))],
            "Protein families": fams,
        }
    )


def test_splits_are_deterministic():
    df = _sample_reps()
    a = multilabel_stratified_splits(df.copy())
    b = multilabel_stratified_splits(df.copy())
    for sa, sb in zip(a, b):  # train, val, test
        assert sorted(sa["identifier"]) == sorted(sb["identifier"])


def test_splits_are_invariant_under_row_permutation():
    """The split must be a function of protein identity, not row order.

    The splitter selects rows positionally (``df.iloc[idx]``), so ``random_state``
    pins which *positions* land in each split, never which proteins. Callers
    assemble the representative frame from ``os.listdir``, whose order is a
    filesystem detail. Feeding the same proteins in a different order must not
    move a single one between splits.
    """
    df = _sample_reps()
    expected = [
        set(part["identifier"]) for part in multilabel_stratified_splits(df.copy())
    ]

    for seed in range(5):
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        got = [
            set(part["identifier"]) for part in multilabel_stratified_splits(shuffled)
        ]
        assert got == expected, f"split moved under row permutation (seed={seed})"


def test_splits_partition_all_rows_disjointly():
    df = _sample_reps()
    train_df, val_df, test_df = multilabel_stratified_splits(df.copy())
    train, val, test = (
        set(train_df["identifier"]),
        set(val_df["identifier"]),
        set(test_df["identifier"]),
    )
    # Pairwise disjoint...
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    # ...and collectively cover exactly the input.
    assert train | val | test == set(df["identifier"])


def test_cluster_cache_key_depends_on_min_seq_id(tmp_path):
    """`preprocess --min-seq-id 0.5` must not reuse clusters built at 0.9.

    The cache used to be keyed on md5(input.fasta) plus "does cluster_rep_seq.fasta
    exist", so a changed cutoff silently reused the old clusters while the console
    printed the new one.
    """
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">P001\nMKTA\n")

    assert _cluster_cache_key(fasta, 0.9) != _cluster_cache_key(fasta, 0.5)
    assert _cluster_cache_key(fasta, 0.9) == _cluster_cache_key(fasta, 0.9)


def test_cluster_cache_key_depends_on_fasta_content(tmp_path):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">P001\nMKTA\n")
    before = _cluster_cache_key(fasta, 0.9)
    fasta.write_text(">P001\nMKTAYIAK\n")

    assert _cluster_cache_key(fasta, 0.9) != before


def test_multilabel_family_string_roundtrips():
    df = _sample_reps()
    parts = multilabel_stratified_splits(df.copy())
    original = dict(zip(df["identifier"], df["Protein families"]))
    for subset in parts:
        for ident, fams in zip(subset["identifier"], subset["Protein families"]):
            assert fams == original[ident]
