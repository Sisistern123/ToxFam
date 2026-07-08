"""Tests for the pure multi-hot taxonomy encoder in toxfam.data.taxonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from toxfam.data.taxonomy import TAXA, _build_multi_hot_vectors


def test_multi_hot_matches_lineage_case_insensitive():
    # Two real taxa appear in the lineage, in deliberately mismatched casing.
    t0, t1 = TAXA[0], TAXA[5]
    df = pd.DataFrame({"identifier": ["P1"], "_taxon_id": [9606]})
    lineage_names = {9606: {t0.upper(), t1.lower(), "SomeOtherClade"}}

    vec = _build_multi_hot_vectors(df, lineage_names)["P1"]

    assert vec.shape == (len(TAXA),) == (50,)
    assert vec[0] == 1.0
    assert vec[5] == 1.0
    assert vec.sum() == 2.0  # only the two matched taxa are set


def test_multi_hot_zero_when_taxon_unresolved():
    """Taxon absent from lineage_names (unresolvable) -> all-zero vector."""
    df = pd.DataFrame({"identifier": ["P1"], "_taxon_id": [12345]})
    vec = _build_multi_hot_vectors(df, lineage_names={})["P1"]
    assert vec.shape == (len(TAXA),)
    assert vec.sum() == 0.0


def test_multi_hot_zero_when_taxon_id_nan():
    """Missing organism id (NaN) -> all-zero vector."""
    df = pd.DataFrame({"identifier": ["P1"], "_taxon_id": [np.nan]})
    vec = _build_multi_hot_vectors(df, lineage_names={9606: {TAXA[0]}})["P1"]
    assert vec.sum() == 0.0
