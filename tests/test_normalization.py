"""Tests for toxfam.data.normalization."""

from __future__ import annotations

import pandas as pd

from toxfam.data.normalization import normalize_protein_families


def test_conotoxin_prefix_fix():
    df = pd.DataFrame({"Protein families": ["I1 superfamily", "O1 superfamily"]})
    result = normalize_protein_families(df, min_count=1)
    assert all(v.startswith("Conotoxin") for v in result["Protein families"])


def test_regex_consolidation():
    df = pd.DataFrame(
        {
            "Protein families": [
                "Conotoxin A superfamily",
                "Conotoxin B superfamily",
                "Neurotoxin 3 family",
                "Neurotoxin long chain",
            ]
        }
    )
    result = normalize_protein_families(df, min_count=1)
    assert set(result["Protein families"]) == {"Conotoxin family", "Neurotoxin family"}


def test_min_count_threshold():
    families = ["famA"] * 15 + ["famB"] * 3
    df = pd.DataFrame({"Protein families": families})
    result = normalize_protein_families(df, min_count=10)
    assert "famA" in result["Protein families"].values
    assert "famB" not in result["Protein families"].values
    assert "other" in result["Protein families"].values


def test_does_not_mutate_input():
    df = pd.DataFrame({"Protein families": ["I1 superfamily"]})
    original_val = df["Protein families"].iloc[0]
    normalize_protein_families(df, min_count=1)
    assert df["Protein families"].iloc[0] == original_val


def test_semicolon_and_comma_split():
    df = pd.DataFrame(
        {"Protein families": ["FamilyA;FamilyB", "FamilyC,FamilyD"]}
    )
    result = normalize_protein_families(df, min_count=1)
    # Should take first part only
    assert result["Protein families"].iloc[0] == "FamilyA"
    assert result["Protein families"].iloc[1] == "FamilyC"


def test_phospholipase_consolidation():
    df = pd.DataFrame(
        {
            "Protein families": [
                "Snake phospholipase A2",
                "Phospholipase D family",
            ]
        }
    )
    result = normalize_protein_families(df, min_count=1)
    assert all(v == "Phospholipase family" for v in result["Protein families"])
