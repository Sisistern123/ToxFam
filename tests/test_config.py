"""Tests for toxfam.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from toxfam.config import TrainConfig


def test_from_yaml(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()

    yaml_content = f"""\
input_csv: "dummy.csv"
h5_path: "{h5}"
output_dir: "{tmp_path / 'out'}"
training_strategy: "standard"
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)

    cfg = TrainConfig.from_yaml(cfg_file)
    assert cfg.training_strategy == "standard"
    assert cfg.h5_paths == [h5]
    assert cfg.embedding_dim == 1024
    assert cfg.tax_dim == 56


def test_combined_strategy(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()

    yaml_content = f"""\
input_csv: "dummy.csv"
h5_path: "{h5}"
tax_h5_path: "{tmp_path / 'tax.h5'}"
output_dir: "{tmp_path / 'out'}"
training_strategy: "combined"
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)

    cfg = TrainConfig.from_yaml(cfg_file)
    assert cfg.training_strategy == "combined"
    assert cfg.tax_h5_path is not None


def test_extra_fields_ignored(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()

    yaml_content = f"""\
input_csv: "dummy.csv"
h5_path: "{h5}"
output_dir: "{tmp_path / 'out'}"
training_strategy: "standard"
use_focal_loss: true
focal_loss_gamma: 2.0
"""
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml_content)

    # Should not raise even with extra fields (model_config extra=ignore)
    cfg = TrainConfig.from_yaml(cfg_file)
    assert cfg.training_strategy == "standard"


def test_no_h5_raises():
    with pytest.raises(Exception):
        TrainConfig(
            input_csv=Path("dummy.csv"),
            output_dir=Path("out"),
            training_strategy="standard",
        )


def test_effective_embedding_dim_no_cpp(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()
    cfg = TrainConfig(
        input_csv=Path("dummy.csv"),
        h5_path=str(h5),
        output_dir=Path("out"),
        training_strategy="standard",
    )
    assert cfg.effective_embedding_dim == 1024


def test_effective_embedding_dim_with_cpp(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()
    cfg = TrainConfig(
        input_csv=Path("dummy.csv"),
        h5_path=str(h5),
        output_dir=Path("out"),
        training_strategy="binary",
        cpp_h5_path=tmp_path / "cpp.h5",
        cpp_dim=100,
    )
    assert cfg.effective_embedding_dim == 1124


def test_n_folds_default(tmp_path):
    h5 = tmp_path / "emb.h5"
    h5.touch()
    cfg = TrainConfig(
        input_csv=Path("dummy.csv"),
        h5_path=str(h5),
        output_dir=Path("out"),
        training_strategy="standard",
    )
    assert cfg.n_folds == 1
