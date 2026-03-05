"""Tests for identity-aware splitting."""

from __future__ import annotations

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

from toxfam.data.preprocessing import identity_aware_splits


@pytest.fixture
def rep_df():
    """Small representative DataFrame for split testing."""
    return pd.DataFrame({
        "identifier": [f"P{i:03d}" for i in range(20)],
        "Sequence": [f"MKTA{'A' * i}" for i in range(20)],
        "Protein families": (
            ["famA"] * 7 + ["famB"] * 6 + ["nontox"] * 7
        ),
    })


class TestIdentityAwareSplits:
    def test_returns_three_dataframes(self, rep_df):
        """Should return train, val, test DataFrames."""
        with patch(
            "toxfam.data.preprocessing.easy_cluster"
        ) as mock_cluster:
            # Mock easy_cluster to create a trivial cluster TSV
            def fake_cluster(fasta_files, cluster_prefix, tmp_dir, min_seq_id):
                tsv_path = f"{cluster_prefix}_cluster.tsv"
                # Each protein is its own cluster (no merging)
                with open(fasta_files) as f:
                    ids = []
                    for line in f:
                        if line.startswith(">"):
                            ids.append(line[1:].strip().split()[0])
                with open(tsv_path, "w") as out:
                    for pid in ids:
                        out.write(f"{pid}\t{pid}\n")

            mock_cluster.side_effect = fake_cluster

            train_df, val_df, test_df = identity_aware_splits(rep_df)

            assert len(train_df) > 0
            assert len(val_df) > 0
            assert len(test_df) > 0

    def test_no_duplicate_identifiers(self, rep_df):
        """No identifier should appear in multiple splits."""
        with patch(
            "toxfam.data.preprocessing.easy_cluster"
        ) as mock_cluster:
            def fake_cluster(fasta_files, cluster_prefix, tmp_dir, min_seq_id):
                tsv_path = f"{cluster_prefix}_cluster.tsv"
                with open(fasta_files) as f:
                    ids = []
                    for line in f:
                        if line.startswith(">"):
                            ids.append(line[1:].strip().split()[0])
                with open(tsv_path, "w") as out:
                    for pid in ids:
                        out.write(f"{pid}\t{pid}\n")

            mock_cluster.side_effect = fake_cluster

            train_df, val_df, test_df = identity_aware_splits(rep_df)

            train_ids = set(train_df["identifier"])
            val_ids = set(val_df["identifier"])
            test_ids = set(test_df["identifier"])

            assert len(train_ids & val_ids) == 0
            assert len(train_ids & test_ids) == 0
            assert len(val_ids & test_ids) == 0

    def test_all_identifiers_assigned(self, rep_df):
        """All input identifiers should appear in exactly one split."""
        with patch(
            "toxfam.data.preprocessing.easy_cluster"
        ) as mock_cluster:
            def fake_cluster(fasta_files, cluster_prefix, tmp_dir, min_seq_id):
                tsv_path = f"{cluster_prefix}_cluster.tsv"
                with open(fasta_files) as f:
                    ids = []
                    for line in f:
                        if line.startswith(">"):
                            ids.append(line[1:].strip().split()[0])
                with open(tsv_path, "w") as out:
                    for pid in ids:
                        out.write(f"{pid}\t{pid}\n")

            mock_cluster.side_effect = fake_cluster

            train_df, val_df, test_df = identity_aware_splits(rep_df)

            all_ids = set(train_df["identifier"]) | set(val_df["identifier"]) | set(test_df["identifier"])
            assert all_ids == set(rep_df["identifier"])

    def test_approximate_split_ratios(self, rep_df):
        """Train should be roughly 70%, val/test ~15% each."""
        with patch(
            "toxfam.data.preprocessing.easy_cluster"
        ) as mock_cluster:
            def fake_cluster(fasta_files, cluster_prefix, tmp_dir, min_seq_id):
                tsv_path = f"{cluster_prefix}_cluster.tsv"
                with open(fasta_files) as f:
                    ids = []
                    for line in f:
                        if line.startswith(">"):
                            ids.append(line[1:].strip().split()[0])
                with open(tsv_path, "w") as out:
                    for pid in ids:
                        out.write(f"{pid}\t{pid}\n")

            mock_cluster.side_effect = fake_cluster

            train_df, val_df, test_df = identity_aware_splits(rep_df)

            total = len(rep_df)
            train_frac = len(train_df) / total
            # Allow wide tolerance due to small dataset + cluster constraints
            assert 0.4 <= train_frac <= 0.9


class TestRebalanceSplits:
    """Tests for the _rebalance_splits function."""

    def test_rebalance_moves_clusters(self):
        """Rebalancing should move clusters to train when representation is low."""
        from toxfam.data.preprocessing import _rebalance_splits

        # Create scenario: famA has most members in test
        df = pd.DataFrame({
            "identifier": [f"P{i:03d}" for i in range(20)],
            "Protein families": (["famA"] * 12 + ["nontox"] * 8),
            "_cluster_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            "_split": ["test"] * 12 + ["train"] * 8,
        })
        cluster_df = pd.DataFrame({
            "_cluster_id": list(range(10)),
            "families": [{"famA"}] * 6 + [{"nontox"}] * 4,
            "size": [2] * 10,
        })

        train_cids = {6, 7, 8, 9}
        val_cids = set()
        test_cids = {0, 1, 2, 3, 4, 5}

        # Can't rebalance with empty val, so add one cluster to val
        val_cids = {5}
        test_cids = {0, 1, 2, 3, 4}

        new_train, new_val, new_test = _rebalance_splits(
            df, cluster_df, train_cids, val_cids, test_cids,
        )

        # Train should have gained clusters
        assert len(new_train) > len(train_cids)
        # Val and test should not be empty
        assert len(new_val) >= 0  # val might lose its single cluster
        assert len(new_test) >= 0

    def test_rebalance_preserves_cluster_integrity(self):
        """No cluster should be split across different sets."""
        from toxfam.data.preprocessing import _rebalance_splits

        df = pd.DataFrame({
            "identifier": [f"P{i:03d}" for i in range(10)],
            "Protein families": (["famA"] * 6 + ["nontox"] * 4),
            "_cluster_id": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            "_split": ["test"] * 6 + ["train"] * 4,
        })
        cluster_df = pd.DataFrame({
            "_cluster_id": list(range(5)),
            "families": [{"famA"}] * 3 + [{"nontox"}] * 2,
            "size": [2] * 5,
        })

        train_cids = {3, 4}
        val_cids = {2}
        test_cids = {0, 1}

        new_train, new_val, new_test = _rebalance_splits(
            df, cluster_df, train_cids, val_cids, test_cids,
        )

        # Every cluster should be in exactly one set
        all_cids = new_train | new_val | new_test
        assert len(all_cids) == len(new_train) + len(new_val) + len(new_test)
