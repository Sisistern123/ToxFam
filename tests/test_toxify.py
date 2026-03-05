"""Tests for TOXIFY reimplementation."""

import numpy as np
import pytest
import torch

from toxfam.evaluation.toxify_benchmark import ToxifyGRU, encode_atchley


def test_encode_atchley_shape():
    seqs = ["ACDEFG", "MK", "WWWWWWWWWW"]
    X, lengths = encode_atchley(seqs, max_len=20)
    assert X.shape == (3, 20, 5)
    assert lengths.tolist() == [6, 2, 10]


def test_encode_atchley_padding():
    seqs = ["AC"]
    X, lengths = encode_atchley(seqs, max_len=10)
    # Positions beyond length should be zero
    assert torch.all(X[0, 2:] == 0)
    # First position (A) should have non-zero Atchley factors
    assert not torch.all(X[0, 0] == 0)


def test_encode_atchley_truncation():
    long_seq = "A" * 600
    X, lengths = encode_atchley([long_seq], max_len=500)
    assert X.shape == (1, 500, 5)
    assert lengths[0] == 500


def test_toxify_gru_forward():
    model = ToxifyGRU(input_dim=5, hidden_dim=32, num_classes=2)
    X = torch.randn(4, 20, 5)
    lengths = torch.tensor([20, 15, 10, 5])
    out = model(X, lengths)
    assert out.shape == (4, 2)


def test_toxify_gru_train_tiny():
    """Train on tiny synthetic data to verify the training loop works."""
    model = ToxifyGRU(input_dim=5, hidden_dim=16, num_classes=2)
    # Tiny dataset: 10 sequences
    seqs = ["ACDEF", "GHIKL", "MNPQR", "STVWY", "AAAA", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"]
    labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

    X, lengths = encode_atchley(seqs, max_len=10)
    y = torch.tensor(labels, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for _ in range(3):
        logits = model(X, lengths)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Just verify it doesn't crash and produces valid output
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X, lengths), dim=1)
    assert probs.shape == (10, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(10), atol=1e-5)
