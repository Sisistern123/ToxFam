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


def run_inference(
    df: pd.DataFrame,
    h5_path: str | Path,
    model_dir: str | Path,
) -> pd.DataFrame:
    """Run model inference on a DataFrame of proteins.

    Returns DataFrame with columns: identifier, predicted_label, confidence.
    """
    device = get_device()
    model, model_config, idx_to_label = load_calibrated_model(model_dir, device=device)

    is_multi_input = model_config.architecture == "MultiInputMLP"
    tax_dim = model_config.tax_dim if is_multi_input else None

    preds = []
    confs = []
    with h5py.File(h5_path, "r") as f:
        for ident in df["identifier"]:
            emb = torch.tensor(f[ident][:]).unsqueeze(0).to(device)

            with torch.no_grad():
                if is_multi_input:
                    dummy_tax = torch.zeros(1, tax_dim).to(device)
                    logits = model(emb, dummy_tax)
                else:
                    logits = model(emb)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = probs.max(dim=1)
            preds.append(idx_to_label.get(pred_idx.item(), "other"))
            confs.append(conf.item())

    return pd.DataFrame(
        {
            "identifier": df["identifier"].values,
            "predicted_label": preds,
            "confidence": confs,
        }
    )
