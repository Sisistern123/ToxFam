"""Tests for ModelConfig — save/load/build round-trip and checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from toxfam.model.model_config import ModelConfig


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arch,kwargs",
    [
        (
            "ModularMLP",
            {"embedding_dim": 1024, "hidden_dims": [256, 256], "num_classes": 38},
        ),
        (
            "MultiInputMLP",
            {
                "embedding_dim": 1024,
                "hidden_dims": [256, 256],
                "num_classes": 38,
                "tax_dim": 50,
                "tax_hidden_dim": 8,
            },
        ),
    ],
)
def test_config_roundtrip(tmp_path: Path, arch: str, kwargs: dict):
    """Config survives JSON save → load with identical values."""
    cfg = ModelConfig(architecture=arch, **kwargs)
    path = tmp_path / "model_config.json"
    cfg.save(path)

    loaded = ModelConfig.load(path)
    assert loaded == cfg
    assert loaded.architecture == arch
    assert loaded.hidden_dims == kwargs["hidden_dims"]


# ---------------------------------------------------------------------------
# build_model produces correct architecture
# ---------------------------------------------------------------------------


def test_build_modular_mlp():
    cfg = ModelConfig(
        architecture="ModularMLP",
        embedding_dim=64,
        hidden_dims=[32, 16],
        num_classes=5,
        dropout=0.1,
    )
    model = cfg.build_model()
    assert model.__class__.__name__ == "ModularMLP"

    # Smoke test forward pass
    x = torch.randn(2, 64)
    out = model(x)
    assert out.shape == (2, 5)


def test_build_multi_input_mlp():
    cfg = ModelConfig(
        architecture="MultiInputMLP",
        embedding_dim=64,
        hidden_dims=[32],
        num_classes=5,
        tax_dim=10,
        tax_hidden_dim=4,
        dropout=0.1,
    )
    model = cfg.build_model()
    assert model.__class__.__name__ == "MultiInputMLP"

    # Smoke test forward pass
    emb = torch.randn(2, 64)
    tax = torch.randn(2, 10)
    out = model(emb, tax)
    assert out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Loading a real checkpoint (integration test)
# ---------------------------------------------------------------------------

COMBINED_DIR = Path("model/model_output/combined_run")
STANDARD_DIR = Path("model/model_output/standard_run")


@pytest.mark.skipif(
    not (COMBINED_DIR / "model_config.json").exists(),
    reason="combined_run model not available",
)
def test_load_combined_checkpoint():
    """ModelConfig can load the real combined_run checkpoint."""
    cfg = ModelConfig.load(COMBINED_DIR / "model_config.json")
    assert cfg.architecture == "MultiInputMLP"
    assert cfg.tax_dim is not None

    model = cfg.build_model()

    from toxfam.model.calibration import ModelWithTemperature

    scaled = ModelWithTemperature(model, torch.device("cpu"))
    state_dict = torch.load(
        COMBINED_DIR / "models" / "best_model_calibrated.pt",
        map_location="cpu",
    )
    # This is the critical test — no size mismatch, no unexpected keys
    scaled.load_state_dict(state_dict)


@pytest.mark.skipif(
    not (STANDARD_DIR / "model_config.json").exists(),
    reason="standard_run model not available",
)
def test_load_standard_checkpoint():
    """ModelConfig can load the real standard_run checkpoint."""
    cfg = ModelConfig.load(STANDARD_DIR / "model_config.json")
    assert cfg.architecture == "ModularMLP"
    assert cfg.tax_dim is None

    model = cfg.build_model()

    from toxfam.model.calibration import ModelWithTemperature

    scaled = ModelWithTemperature(model, torch.device("cpu"))
    state_dict = torch.load(
        STANDARD_DIR / "models" / "best_model_calibrated.pt",
        map_location="cpu",
    )
    scaled.load_state_dict(state_dict)
