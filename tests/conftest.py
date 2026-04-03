"""Shared fixtures for ToxFam tests."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_fasta(tmp_path) -> Path:
    """Write a small FASTA file and return its path."""
    fasta = tmp_path / "sample.fasta"
    fasta.write_text(
        ">P001\nMKTAYIAKQR\n>P002\nMLLPVLLLALL\n>P003\nACDEFGHIKLM\n"
    )
    return fasta


@pytest.fixture
def sample_h5(tmp_path) -> Path:
    """HDF5 file with fake 1024-d embeddings keyed by protein ID."""
    h5_path = tmp_path / "embeddings.h5"
    rng = np.random.default_rng(42)
    with h5py.File(h5_path, "w") as f:
        for pid in ["P001", "P002", "P003", "P004", "P005"]:
            f.create_dataset(pid, data=rng.standard_normal(1024).astype(np.float32))
    return h5_path


