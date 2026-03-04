"""Protein family label normalization."""

from __future__ import annotations

import pandas as pd


# Conotoxin superfamily prefix corrections
_CONOTOXIN_REPLACEMENTS = {
    "I1 superfamily": "Conotoxin I1 superfamily",
    "O1 superfamily": "Conotoxin O1 superfamily",
    "O2 superfamily": "Conotoxin O2 superfamily",
    "E superfamily": "Conotoxin E superfamily",
    "F superfamily": "Conotoxin F superfamily",
}

# Regex-based family consolidation
_FAMILY_MAPPING = {
    r"Conotoxin.*": "Conotoxin family",
    r"Neurotoxin.*": "Neurotoxin family",
    r"Scoloptoxin.*|Scolopendra.*": "Scoloptoxin family",
    r"Caterpillar.*": "Caterpillar family",
    r"Teretoxin.*": "Teretoxin family",
    r"Limacoditoxin.*": "Limacoditoxin family",
    r"Scutigerotoxin.*": "Scutigerotoxin family",
    r"Cationic peptide.*": "Cationic peptide family",
    r"Formicidae venom.*": "Formicidae venom family",
    r"Bradykinin-potentiating peptide family|Natriuretic peptide family|Natriuretic": "Natriuretic, Bradykinin potentiating peptide family",
    r".*phospholipase.*|.*Phospholipase.*": "Phospholipase family",
}


def normalize_protein_families(
    df: pd.DataFrame,
    column: str = "Protein families",
    *,
    min_count: int = 10,
) -> pd.DataFrame:
    """Normalize protein family labels: conotoxin fixes, regex mapping, min-count threshold.

    Returns a copy — the input DataFrame is not modified.
    """
    df = df.copy()

    df[column] = df[column].str.split(";").str[0]
    df[column] = df[column].str.split(",").str[0]

    df[column] = df[column].replace(_CONOTOXIN_REPLACEMENTS)

    for pattern, replacement in _FAMILY_MAPPING.items():
        df[column] = df[column].str.replace(pattern, replacement, regex=True)

    df[column] = df[column].where(
        df[column].map(df[column].value_counts()) >= min_count,
        "other",
    )

    return df
