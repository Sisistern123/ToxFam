"""Model loading and inference for evaluation.

Loads a calibrated (temperature-scaled) model from a training output directory,
auto-detecting the architecture (ModularMLP vs MultiInputMLP) from the state
dict keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pandas as pd
import torch
from rich.console import Console

console = Console()


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_calibrated_model(
    model_path: Path,
    class_map_path: Path,
    h5_path: Path,
    device: str | None = None,
):
    """Load a calibrated model, auto-detecting architecture from state dict.

    Returns (model, is_multi_input, idx_to_label).
    """
    from toxfam.model.architectures import ModularMLP, MultiInputMLP
    from toxfam.model.calibration import ModelWithTemperature

    if device is None:
        device = get_device()

    with open(class_map_path, "r") as f:
        class_indices = json.load(f)
    num_classes = len(class_indices)
    idx_to_label = {int(k): v for k, v in class_indices.items()}

    with h5py.File(h5_path, "r") as f:
        first_key = list(f.keys())[0]
        embedding_dim = f[first_key][:].shape[0]

    state_dict = torch.load(model_path, map_location=torch.device(device))
    is_multi_input = any(k.startswith("model.tax_net.") for k in state_dict)

    if is_multi_input:
        tax_dim = state_dict["model.tax_net.0.weight"].shape[1]
        tax_hidden_dim = state_dict["model.tax_net.0.weight"].shape[0]
        hidden_dims = []
        i = 0
        while f"model.joint.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.joint.{i}.weight"].shape[0])
            i += 3
        if hidden_dims:
            hidden_dims.pop()

        base_model = MultiInputMLP(
            embed_dim=embedding_dim,
            tax_dim=tax_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            tax_hidden_dim=tax_hidden_dim,
        )
    else:
        hidden_dims = [state_dict["model.projector.0.weight"].shape[0]]
        i = 0
        while f"model.backbone.{i}.weight" in state_dict:
            hidden_dims.append(state_dict[f"model.backbone.{i}.weight"].shape[0])
            i += 3
        if hidden_dims and len(hidden_dims) > 1:
            hidden_dims.pop()

        base_model = ModularMLP(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
        )

    scaled_model = ModelWithTemperature(base_model, torch.device(device))
    scaled_model.load_state_dict(state_dict)
    scaled_model.eval()

    console.print(
        f"   Loaded calibrated model (T={scaled_model.temperature.item():.3f}, "
        f"{'MultiInputMLP' if is_multi_input else 'ModularMLP'})"
    )

    return scaled_model, is_multi_input, idx_to_label


def run_inference(
    df: pd.DataFrame,
    h5_path: Path,
    model_path: Path,
    class_map_path: Path,
) -> pd.DataFrame:
    """Run model inference on a DataFrame of proteins.

    Returns DataFrame with columns: identifier, predicted_label, confidence.
    """
    device = get_device()
    model, is_multi_input, idx_to_label = load_calibrated_model(
        model_path, class_map_path, h5_path, device=device
    )

    if is_multi_input:
        tax_dim = model.model.tax_net[0].in_features
    else:
        tax_dim = None

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
