"""Protein family label normalization."""

from __future__ import annotations

import pandas as pd

# Conotoxin superfamily prefix corrections (matched case-insensitively)
_CONOTOXIN_REPLACEMENTS = {
    "i1 superfamily": "Conotoxin I1 superfamily",
    "o1 superfamily": "Conotoxin O1 superfamily",
    "o2 superfamily": "Conotoxin O2 superfamily",
    "e superfamily": "Conotoxin E superfamily",
    "f superfamily": "Conotoxin F superfamily",
}

# Regex-based family consolidation (applied case-insensitively)
_FAMILY_MAPPING = {
    r"conotoxin.*": "Conotoxin family",
    r"neurotoxin.*": "Neurotoxin family",
    r"scoloptoxin.*|scolopendra.*": "Scoloptoxin family",
    r"caterpillar.*": "Caterpillar family",
    r"teretoxin.*": "Teretoxin family",
    r"limacoditoxin.*": "Limacoditoxin family",
    r"scutigerotoxin.*": "Scutigerotoxin family",
    r"cationic peptide.*": "Cationic peptide family",
    r"formicidae venom.*": "Formicidae venom family",
    r"bradykinin-potentiating peptide family|natriuretic peptide family|natriuretic": "Natriuretic, Bradykinin potentiating peptide family",
    r".*phospholipase.*": "Phospholipase family",
}


def normalize_protein_families(
    df: pd.DataFrame,
    column: str = "Protein families",
    *,
    min_count: int = 10,
) -> pd.DataFrame:
    """Normalize protein family labels: conotoxin fixes, regex mapping, min-count threshold.

    Handles both TSV-style (Title Case) and XML-style (lowercase) family names
    via case-insensitive matching and first-letter capitalization.

    Returns a copy — the input DataFrame is not modified.
    """
    df = df.copy()

    df[column] = df[column].str.strip()
    df[column] = df[column].str.split(";").str[0]
    df[column] = df[column].str.split(",").str[0]
    df[column] = df[column].str.strip()

    # Capitalize first letter to normalize XML (lowercase) vs TSV (title case).
    # "three-finger toxin family" → "Three-finger toxin family"
    mask = df[column].str.len() > 0
    df.loc[mask, column] = (
        df.loc[mask, column].str[0].str.upper() + df.loc[mask, column].str[1:]
    )

    # Conotoxin replacements — case-insensitive via lowered lookup
    lowered = df[column].str.lower()
    for pattern_lower, replacement in _CONOTOXIN_REPLACEMENTS.items():
        df.loc[lowered == pattern_lower, column] = replacement

    # Regex family consolidation — case-insensitive
    for pattern, replacement in _FAMILY_MAPPING.items():
        df[column] = df[column].str.replace(
            pattern, replacement, regex=True, case=False
        )

    df[column] = df[column].where(
        df[column].map(df[column].value_counts()) >= min_count,
        "other",
    )

    return df
