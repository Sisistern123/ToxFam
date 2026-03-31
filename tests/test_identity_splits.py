"""Tests for identity-aware splitting.

NOTE: Full identity_aware_splits() requires MMseqs2 installed.
These tests focus on the rebalancing logic which is pure Python.
"""

import pandas as pd
import pytest

from toxfam.data.preprocessing import _rebalance_splits


class TestRebalanceSplits:
    @pytest.fixture
    def sample_data(self):
        """Create sample data with clusters and families."""
        df = pd.DataFrame(
            {
                "identifier": [f"P{i}" for i in range(20)],
                "Protein families": (
                    ["famA"] * 8 + ["famB"] * 6 + ["nontox"] * 6
                ),
                "_cluster_id": [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5,
            }
        )
        cluster_df = pd.DataFrame(
            {
                "_cluster_id": [0, 1, 2, 3],
                "families": [
                    {"famA"},
                    {"famA", "famB"},
                    {"famB", "nontox"},
                    {"nontox"},
                ],
                "size": [5, 5, 5, 5],
            }
        )
        return df, cluster_df

    def test_returns_three_sets(self, sample_data):
        df, cluster_df = sample_data
        train, val, test = _rebalance_splits(
            df, cluster_df, {0}, {1, 2}, {3}
        )
        assert isinstance(train, set)
        assert isinstance(val, set)
        assert isinstance(test, set)

    def test_no_overlap(self, sample_data):
        df, cluster_df = sample_data
        train, val, test = _rebalance_splits(
            df, cluster_df, {0}, {1, 2}, {3}
        )
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_preserves_all_clusters(self, sample_data):
        df, cluster_df = sample_data
        train, val, test = _rebalance_splits(
            df, cluster_df, {0}, {1, 2}, {3}
        )
        assert train | val | test == {0, 1, 2, 3}

    def test_never_empties_split(self, sample_data):
        df, cluster_df = sample_data
        # Start with only 1 cluster in val — it should never be moved out
        train, val, test = _rebalance_splits(
            df, cluster_df, {0, 3}, {1}, {2}
        )
        assert len(val) >= 1
        assert len(test) >= 1
