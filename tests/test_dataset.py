"""Tests for toxfam.data.dataset."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch

from toxfam.data.dataset import ToxDataset, analyze_data_splits


class TestToxDataset:
    def test_len(self, sample_h5):
        df = pd.DataFrame(
            {
                "identifier": ["P001", "P002", "P003"],
                "Protein families": ["famA", "famA", "famB"],
            }
        )
        ds = ToxDataset(df, [str(sample_h5)], is_train=True)
        assert len(ds) == 3
        ds.close()

    def test_getitem_returns_tensor_and_label(self, sample_h5):
        df = pd.DataFrame(
            {
                "identifier": ["P001", "P002"],
                "Protein families": ["famA", "famB"],
            }
        )
        ds = ToxDataset(df, [str(sample_h5)], is_train=True)
        features, label = ds[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (1024,)
        assert isinstance(label, (int, np.integer))
        ds.close()

    def test_num_classes(self, sample_h5):
        df = pd.DataFrame(
            {
                "identifier": ["P001", "P002", "P003"],
                "Protein families": ["famA", "famB", "famC"],
            }
        )
        ds = ToxDataset(df, [str(sample_h5)], is_train=True)
        assert ds.num_classes == 3
        ds.close()

    def test_with_taxonomy(self, sample_h5, tmp_path):
        tax_h5 = tmp_path / "tax.h5"
        with h5py.File(tax_h5, "w") as f:
            for pid in ["P001", "P002"]:
                f.create_dataset(pid, data=np.zeros(56, dtype=np.float32))

        df = pd.DataFrame(
            {
                "identifier": ["P001", "P002"],
                "Protein families": ["famA", "famB"],
            }
        )
        ds = ToxDataset(
            df, [str(sample_h5)], is_train=True, tax_h5_path=str(tax_h5)
        )
        features, label = ds[0]
        assert isinstance(features, (tuple, list))
        emb, tax = features
        assert emb.shape == (1024,)
        assert tax.shape == (56,)
        ds.close()


    def test_with_cpp_features(self, sample_h5, tmp_path):
        cpp_h5 = tmp_path / "cpp.h5"
        with h5py.File(cpp_h5, "w") as f:
            for pid in ["P001", "P002"]:
                f.create_dataset(pid, data=np.ones(100, dtype=np.float32))

        df = pd.DataFrame(
            {
                "identifier": ["P001", "P002"],
                "Protein families": ["famA", "famB"],
            }
        )
        ds = ToxDataset(
            df, [str(sample_h5)], is_train=True, cpp_h5_path=str(cpp_h5)
        )
        features, label = ds[0]
        assert isinstance(features, torch.Tensor)
        # 1024 (ProtT5) + 100 (CPP) = 1124
        assert features.shape == (1124,)
        ds.close()


class TestAnalyzeDataSplits:
    def test_splits_correctly(self):
        df = pd.DataFrame(
            {
                "identifier": ["A", "B", "C", "D"],
                "Split": ["train", "train", "val", "test"],
            }
        )
        train, val, test = analyze_data_splits(df)
        assert len(train) == 2
        assert len(val) == 1
        assert len(test) == 1

    def test_invalid_split_raises(self):
        df = pd.DataFrame({"identifier": ["A"], "Split": ["unknown"]})
        import pytest

        with pytest.raises(ValueError):
            analyze_data_splits(df)
