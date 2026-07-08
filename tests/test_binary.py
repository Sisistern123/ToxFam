"""Tests for the pure P(toxic) column definition in toxfam.evaluation.binary."""

from __future__ import annotations

from toxfam.evaluation.binary import _nontox_indices


class _FakeLE:
    """Minimal stand-in for a fitted LabelEncoder (only .classes_ is read)."""

    def __init__(self, classes):
        self.classes_ = classes


def test_nontox_indices_case_insensitive():
    le = _FakeLE(
        ["Conotoxin family", "NonTox", "Three-finger toxin family", "nontoxin"]
    )
    # 'NonTox' and 'nontoxin' are the non-toxin classes (matched case-insensitively).
    assert _nontox_indices(le) == [1, 3]


def test_nontox_indices_none_when_no_nontoxin_class():
    le = _FakeLE(["Conotoxin family", "Three-finger toxin family"])
    assert _nontox_indices(le) == []
