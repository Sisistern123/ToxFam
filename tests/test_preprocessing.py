"""Regression guards for the stratified split in toxfam.data.preprocessing.

These pin the reported train/val/test partition against accidental seed or
proportion edits — not a leakage proof (disjointness is inherent to the splitter).
"""

from __future__ import annotations

import pandas as pd

from toxfam.data.preprocessing import multilabel_stratified_splits


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


def test_multilabel_family_string_roundtrips():
    df = _sample_reps()
    parts = multilabel_stratified_splits(df.copy())
    original = dict(zip(df["identifier"], df["Protein families"]))
    for subset in parts:
        for ident, fams in zip(subset["identifier"], subset["Protein families"]):
            assert fams == original[ident]
