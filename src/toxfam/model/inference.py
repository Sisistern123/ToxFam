"""Model loading and inference for evaluation.

Loads a calibrated (temperature-scaled) model from a training output directory
using the saved ``model_config.json`` for deterministic architecture
reconstruction. No fragile state_dict key parsing.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from rich.console import Console

from toxfam.device import get_device
from toxfam.model.model_config import ModelConfig

console = Console()


def load_calibrated_model(
    model_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple:
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
        from toxfam._paths import get_project_root

        p = get_project_root() / p
    return p if p.exists() else None


def run_inference(
    df: pd.DataFrame,
    h5_path: str | Path,
    model_dir: str | Path,
) -> pd.DataFrame:
    """Run batched model inference on a DataFrame of proteins.

    For MultiInputMLP models, loads real taxonomy vectors from the taxonomy H5
    specified in the model's ``config.yaml``. Falls back to zero vectors if the
    taxonomy H5 is unavailable.

    Returns DataFrame with columns: identifier, predicted_label, confidence.
    """
    device = get_device()
    model_dir = Path(model_dir)
    model, model_config, idx_to_label = load_calibrated_model(model_dir, device=device)

    is_multi_input = model_config.architecture == "MultiInputMLP"
    tax_dim = model_config.tax_dim if is_multi_input else None

    identifiers = df["identifier"].tolist()

    # Read all embeddings into a single tensor
    with h5py.File(h5_path, "r") as f:
        embeddings = torch.stack(
            [torch.tensor(f[ident][:], dtype=torch.float32) for ident in identifiers]
        )

    # Load taxonomy vectors for combined models
    tax_vectors = None
    if is_multi_input:
        tax_h5_path = _resolve_tax_h5(model_dir)
        if tax_h5_path is not None:
            console.print(f"   Loading taxonomy vectors from {tax_h5_path.name}")
            with h5py.File(tax_h5_path, "r") as tf:
                tax_list = []
                for ident in identifiers:
                    if ident in tf:
                        tax_list.append(torch.tensor(tf[ident][:], dtype=torch.float32))
                    else:
                        tax_list.append(torch.zeros(tax_dim, dtype=torch.float32))
                tax_vectors = torch.stack(tax_list)
        else:
            console.print(
                "   [yellow]Warning: taxonomy H5 not found, "
                "using zero vectors (predictions may differ from training)[/]"
            )

    # Batched forward pass
    all_preds: list[str] = []
    all_confs: list[float] = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size].to(device)
            if is_multi_input:
                if tax_vectors is not None:
                    tax_batch = tax_vectors[i : i + batch_size].to(device)
                else:
                    tax_batch = torch.zeros(batch.shape[0], tax_dim, device=device)
                logits = model(batch, tax_batch)
            else:
                logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            confs, pred_idxs = probs.max(dim=1)
            all_preds.extend(
                idx_to_label.get(idx.item(), "other") for idx in pred_idxs
            )
            all_confs.extend(confs.cpu().tolist())

    return pd.DataFrame(
        {
            "identifier": identifiers,
            "predicted_label": all_preds,
            "confidence": all_confs,
        }
    )
