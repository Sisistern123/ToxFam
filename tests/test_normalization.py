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
    # First letter is capitalized: "famA" → "FamA"
    assert "FamA" in result["Protein families"].values
    assert "FamB" not in result["Protein families"].values
    assert "other" in result["Protein families"].values


def test_does_not_mutate_input():
    df = pd.DataFrame({"Protein families": ["I1 superfamily"]})
    original_val = df["Protein families"].iloc[0]
    normalize_protein_families(df, min_count=1)
    assert df["Protein families"].iloc[0] == original_val


def test_semicolon_and_comma_split():
    df = pd.DataFrame({"Protein families": ["FamilyA;FamilyB", "FamilyC,FamilyD"]})
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


def test_xml_lowercase_normalized_to_match_tsv():
    """XML extracts lowercase names; TSV has title-case. Both should normalize the same."""
    df = pd.DataFrame(
        {
            "Protein families": [
                # XML-style (lowercase)
                "three-finger toxin family",
                "conotoxin A superfamily",
                # TSV-style (capitalized)
                "Three-finger toxin family",
                "Conotoxin A superfamily",
            ]
        }
    )
    result = normalize_protein_families(df, min_count=1)
    vals = result["Protein families"].tolist()
    # Both conotoxin variants → "Conotoxin family"
    assert vals[1] == "Conotoxin family"
    assert vals[3] == "Conotoxin family"
    # Both three-finger variants → same capitalized form
    assert vals[0] == vals[2] == "Three-finger toxin family"


def test_case_insensitive_conotoxin_prefix():
    """Lowercase conotoxin prefix corrections from XML data."""
    df = pd.DataFrame({"Protein families": ["i1 superfamily", "o1 superfamily"]})
    result = normalize_protein_families(df, min_count=1)
    # Should still become "Conotoxin family" via prefix fix → regex consolidation
    assert all(v == "Conotoxin family" for v in result["Protein families"])
