"""Tests for toxfam.training.strategies.DataSelector.

DataSelector is the routing layer of the "central design axis" (which inputs
each training strategy receives). It decides whether a batch is fed as
embeddings-only or as an (embeddings, taxonomy) pair, and raises if a strategy
asks for taxonomy the dataset can't supply. These are the routing invariants,
exercised on tiny tensors with no optimizer loop.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from toxfam.data.dataset import ToxDataset
from toxfam.training.strategies import DataSelector

TAX_DIM = 6


@pytest.fixture
def emb_loader(sample_h5):
    """DataLoader over an embeddings-only dataset."""
    df = pd.DataFrame(
        {"identifier": ["P001", "P002"], "Protein families": ["famA", "famB"]}
    )
    ds = ToxDataset(df, [str(sample_h5)], is_train=True)
    yield DataLoader(ds, batch_size=2)
    ds.close()


@pytest.fixture
def tax_loader(sample_h5, tmp_path):
    """DataLoader over a dataset that also carries taxonomy vectors."""
    tax_h5 = tmp_path / "tax.h5"
    with h5py.File(tax_h5, "w") as f:
        for pid in ["P001", "P002"]:
            f.create_dataset(pid, data=np.ones(TAX_DIM, dtype=np.float32))

    df = pd.DataFrame(
        {"identifier": ["P001", "P002"], "Protein families": ["famA", "famB"]}
    )
    ds = ToxDataset(df, [str(sample_h5)], is_train=True, tax_h5_path=str(tax_h5))
    yield DataLoader(ds, batch_size=2)
    ds.close()


def test_emb_only_yields_embedding_tensor(emb_loader):
    batches = list(DataSelector(emb_loader, "emb_only"))
    assert len(batches) == 1
    features, label = batches[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (2, 1024)
    assert label.shape == (2,)


def test_emb_dataset_with_both_mode_raises(emb_loader):
    """A 'both' strategy over an embeddings-only dataset is a config error."""
    with pytest.raises(RuntimeError, match="tax_h5_path"):
        list(DataSelector(emb_loader, "both"))


def test_both_mode_yields_embedding_and_taxonomy(tax_loader):
    features, label = next(iter(DataSelector(tax_loader, "both")))
    assert isinstance(features, (list, tuple))
    emb, tax = features
    assert emb.shape == (2, 1024)
    assert tax.shape == (2, TAX_DIM)


def test_emb_only_mode_drops_taxonomy(tax_loader):
    """emb_only over a taxonomy dataset routes just the embeddings through."""
    features, label = next(iter(DataSelector(tax_loader, "emb_only")))
    assert isinstance(features, torch.Tensor)
    assert features.shape == (2, 1024)


def test_len_passes_through(emb_loader):
    assert len(DataSelector(emb_loader, "emb_only")) == len(emb_loader)
