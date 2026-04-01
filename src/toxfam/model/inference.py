"""Model loading and inference for evaluation.

Loads a calibrated (temperature-scaled) model from a training output directory
using the saved ``model_config.json`` for deterministic architecture
reconstruction. No fragile state_dict key parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
import torch
import yaml
from rich.console import Console

from toxfam.model.model_config import ModelConfig

console = Console()


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_calibrated_model(
    model_dir: str | Path,
    device: str | None = None,
):
    """Load a calibrated model from a training output directory.

    Reads ``model_config.json`` to reconstruct the architecture, then loads
    weights from ``models/best_model_calibrated.pt``.

    Returns (model, model_config, idx_to_label).
    """
    from toxfam.model.calibration import ModelWithTemperature

    model_dir = Path(model_dir)
    if device is None:
        device = get_device()

    # Load architecture config
    config_path = model_dir / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"model_config.json not found in {model_dir}. "
            "Re-run training or generate it from config.yaml."
        )
    model_config = ModelConfig.load(config_path)

    # Load class mapping
    with open(model_dir / "class_indices.json") as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    # Build model from config and load weights
    base_model = model_config.build_model()
    scaled_model = ModelWithTemperature(base_model, torch.device(device))

    checkpoint = model_dir / "models" / "best_model_calibrated.pt"
    state_dict = torch.load(checkpoint, map_location=torch.device(device))
    scaled_model.load_state_dict(state_dict)
    scaled_model.to(device)
    scaled_model.eval()

    console.print(
        f"   Loaded {model_config.architecture} "
        f"(T={scaled_model.temperature.item():.3f})"
    )

    return scaled_model, model_config, idx_to_label


def _resolve_tax_h5(model_dir: Path) -> Path | None:
    """Read tax_h5_path from the saved config.yaml in the model directory."""
    config_yaml = model_dir / "config.yaml"
    if not config_yaml.exists():
        return None
    cfg = yaml.safe_load(config_yaml.read_text())
    tax_path = cfg.get("tax_h5_path")
    if tax_path is None:
        return None
    p = Path(tax_path)
    if not p.is_absolute():
        # Resolve relative to project root (config paths are relative to project root)
        from toxfam._paths import get_project_root

        p = get_project_root() / p
    return p if p.exists() else None


def run_inference(
    df: pd.DataFrame,
    h5_path: str | Path,
    model_dir: str | Path,
) -> pd.DataFrame:
    """Run model inference on a DataFrame of proteins.

    For MultiInputMLP models, loads real taxonomy vectors from the taxonomy H5
    specified in the model's ``config.yaml``. Falls back to zero vectors if the
    taxonomy H5 is unavailable.

    Returns DataFrame with columns: identifier, predicted_label, confidence.
    """
    device = get_device()
    model_dir = Path(model_dir)
    model, model_config, idx_to_label = load_calibrated_model(model_dir, device=device)

    is_multi_input = model_config.architecture == "MultiInputMLP"

    # Load taxonomy vectors for combined models
    tax_h5_file = None
    if is_multi_input:
        tax_h5_path = _resolve_tax_h5(model_dir)
        if tax_h5_path is not None:
            tax_h5_file = h5py.File(tax_h5_path, "r")
            console.print(f"   Loading taxonomy vectors from {tax_h5_path.name}")
        else:
            console.print(
                "   [yellow]Warning: taxonomy H5 not found, "
                "using zero vectors (predictions may differ from training)[/]"
            )

    preds = []
    confs = []
    try:
        with h5py.File(h5_path, "r") as f:
            for ident in df["identifier"]:
                emb = torch.tensor(f[ident][:]).unsqueeze(0).to(device)

                with torch.no_grad():
                    if is_multi_input:
                        if tax_h5_file is not None and ident in tax_h5_file:
                            tax = (
                                torch.tensor(tax_h5_file[ident][:])
                                .unsqueeze(0)
                                .to(device)
                            )
                        else:
                            tax = torch.zeros(1, model_config.tax_dim).to(device)
                        logits = model(emb, tax)
                    else:
                        logits = model(emb)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred_idx = probs.max(dim=1)
                preds.append(idx_to_label.get(pred_idx.item(), "other"))
                confs.append(conf.item())
    finally:
        if tax_h5_file is not None:
            tax_h5_file.close()

    return pd.DataFrame(
        {
            "identifier": df["identifier"].values,
            "predicted_label": preds,
            "confidence": confs,
        }
    )
