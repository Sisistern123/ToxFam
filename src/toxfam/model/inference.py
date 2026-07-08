"""Model loading and inference for evaluation.

Loads a calibrated (temperature-scaled) model from a training output directory
using the saved ``model_config.json`` for deterministic architecture
reconstruction. No fragile state_dict key parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import pandas as pd
import torch
import yaml
from rich.console import Console

from toxfam.device import get_device
from toxfam.model.model_config import ModelConfig

if TYPE_CHECKING:
    from toxfam.model.calibration import ModelWithTemperature

console = Console()


def load_calibrated_model(
    model_dir: str | Path,
    device: str | torch.device | None = None,
) -> tuple[ModelWithTemperature, ModelConfig, dict[int, str]]:
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
    class_indices_path = model_dir / "class_indices.json"
    if not class_indices_path.exists():
        raise FileNotFoundError(
            f"class_indices.json not found in {model_dir}. "
            "Re-run training to generate class index mapping."
        )
    with open(class_indices_path) as f:
        idx_to_label = {int(k): v for k, v in json.load(f).items()}

    # Build model from config and load weights
    base_model = model_config.build_model()
    scaled_model = ModelWithTemperature(base_model, torch.device(device))

    checkpoint = model_dir / "models" / "best_model_calibrated.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"best_model_calibrated.pt not found at {checkpoint}. "
            "Re-run training to generate the calibrated model checkpoint."
        )
    state_dict = torch.load(
        checkpoint, map_location=torch.device(device), weights_only=True
    )
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


def _load_embeddings(h5_path: str | Path, identifiers: list[str]) -> torch.Tensor:
    """Stack per-protein embeddings (keyed by identifier) into one tensor."""
    with h5py.File(h5_path, "r") as f:
        return torch.stack(
            [torch.tensor(f[ident][:], dtype=torch.float32) for ident in identifiers]
        )


def _load_tax_vectors(
    model_dir: Path,
    identifiers: list[str],
    tax_dim: int,
    tax_h5_path: str | Path | None = None,
) -> torch.Tensor | None:
    """Load taxonomy vectors for combined models, keyed by identifier.

    Resolves the taxonomy H5 from ``tax_h5_path`` if given, otherwise from the
    model's saved ``config.yaml`` (training taxonomy H5). Proteins absent from
    the H5 fall back to a zero vector. Returns ``None`` if no H5 is available.
    """
    resolved = Path(tax_h5_path) if tax_h5_path is not None else _resolve_tax_h5(model_dir)
    if resolved is None or not resolved.exists():
        console.print(
            "   [yellow]Warning: taxonomy H5 not found, "
            "using zero vectors (predictions may differ from training)[/]"
        )
        return None

    console.print(f"   Loading taxonomy vectors from {resolved.name}")
    with h5py.File(resolved, "r") as tf:
        tax_list = []
        for ident in identifiers:
            if ident in tf:
                tax_list.append(torch.tensor(tf[ident][:], dtype=torch.float32))
            else:
                tax_list.append(torch.zeros(tax_dim, dtype=torch.float32))
    return torch.stack(tax_list)


def _calibrated_probs_in_batches(
    model: ModelWithTemperature,
    embeddings: torch.Tensor,
    tax_vectors: torch.Tensor | None,
    *,
    is_multi_input: bool,
    tax_dim: int | None,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass over all proteins; return (calibrated_probs, uncalibrated_probs)."""
    cal_chunks: list[torch.Tensor] = []
    uncal_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i : i + batch_size].to(device)
            if is_multi_input:
                if tax_vectors is not None:
                    tax_batch = tax_vectors[i : i + batch_size].to(device)
                else:
                    tax_batch = torch.zeros(batch.shape[0], tax_dim, device=device)
                raw_logits = model.model(batch, tax_batch)
            else:
                raw_logits = model.model(batch)
            scaled_logits = model.temperature_scale(raw_logits)
            cal_chunks.append(torch.softmax(scaled_logits, dim=1).cpu())
            uncal_chunks.append(torch.softmax(raw_logits, dim=1).cpu())
    return torch.cat(cal_chunks), torch.cat(uncal_chunks)


def run_inference(
    df: pd.DataFrame,
    h5_path: str | Path,
    model_dir: str | Path,
    *,
    tax_h5_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run batched model inference on a DataFrame of proteins.

    For MultiInputMLP models, loads real taxonomy vectors from ``tax_h5_path``
    if provided, otherwise from the taxonomy H5 specified in the model's
    ``config.yaml``. Falls back to zero vectors if the taxonomy H5 is
    unavailable.

    Returns DataFrame with columns: identifier, predicted_label, confidence,
    confidence_uncalibrated.
    """
    device = get_device()
    model_dir = Path(model_dir)
    model, model_config, idx_to_label = load_calibrated_model(model_dir, device=device)

    is_multi_input = model_config.architecture == "MultiInputMLP"
    tax_dim = model_config.tax_dim if is_multi_input else None

    identifiers = df["identifier"].tolist()
    embeddings = _load_embeddings(h5_path, identifiers)

    tax_vectors = None
    if is_multi_input:
        tax_vectors = _load_tax_vectors(model_dir, identifiers, tax_dim, tax_h5_path)

    cal_probs, uncal_probs = _calibrated_probs_in_batches(
        model, embeddings, tax_vectors,
        is_multi_input=is_multi_input, tax_dim=tax_dim, device=device,
    )

    confs, pred_idxs = cal_probs.max(dim=1)
    uncal_confs, _ = uncal_probs.max(dim=1)

    return pd.DataFrame(
        {
            "identifier": identifiers,
            "predicted_label": [
                idx_to_label.get(idx.item(), "other") for idx in pred_idxs
            ],
            "confidence": confs.tolist(),
            "confidence_uncalibrated": uncal_confs.tolist(),
        }
    )


def run_topk_inference(
    df: pd.DataFrame,
    h5_path: str | Path,
    model_dir: str | Path,
    *,
    tax_h5_path: str | Path | None = None,
    top_k: int = 3,
    binary_only: bool = False,
) -> pd.DataFrame:
    """Run inference returning the top-``k`` family predictions plus P(toxic).

    Unlike :func:`run_inference` (argmax + metrics), this returns the highest
    ``k`` calibrated family probabilities per protein and a score-based binary
    toxicity probability ``p_toxic`` = 1 - sum(P(nontoxin classes)).

    Returns DataFrame with columns: identifier, pred_1..k, conf_1..k, p_toxic.
    When ``binary_only`` is set, the per-family columns are skipped and only
    identifier + p_toxic are returned.
    """
    from toxfam.evaluation.metrics import nontoxin_indices

    device = get_device()
    model_dir = Path(model_dir)
    model, model_config, idx_to_label = load_calibrated_model(model_dir, device=device)

    is_multi_input = model_config.architecture == "MultiInputMLP"
    tax_dim = model_config.tax_dim if is_multi_input else None

    identifiers = df["identifier"].tolist()
    embeddings = _load_embeddings(h5_path, identifiers)

    tax_vectors = None
    if is_multi_input:
        tax_vectors = _load_tax_vectors(model_dir, identifiers, tax_dim, tax_h5_path)

    cal_probs, _ = _calibrated_probs_in_batches(
        model, embeddings, tax_vectors,
        is_multi_input=is_multi_input, tax_dim=tax_dim, device=device,
    )

    # P(toxic) = 1 - sum over nontoxin class probabilities
    ordered_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    nontox_indices = nontoxin_indices(ordered_labels)
    if nontox_indices:
        p_toxic = 1.0 - cal_probs[:, nontox_indices].sum(dim=1)
    else:
        p_toxic = torch.ones(len(identifiers))

    if binary_only:
        return pd.DataFrame({"identifier": identifiers, "p_toxic": p_toxic.tolist()})

    # Top-k families (clamped to available classes)
    k = min(top_k, cal_probs.shape[1])
    top_confs, top_idxs = cal_probs.topk(k, dim=1)

    out: dict[str, object] = {"identifier": identifiers}
    for rank in range(top_k):
        if rank < k:
            out[f"pred_{rank + 1}"] = [
                idx_to_label.get(top_idxs[row, rank].item(), "other")
                for row in range(len(identifiers))
            ]
            out[f"conf_{rank + 1}"] = top_confs[:, rank].tolist()
        else:
            out[f"pred_{rank + 1}"] = [None] * len(identifiers)
            out[f"conf_{rank + 1}"] = [None] * len(identifiers)
    out["p_toxic"] = p_toxic.tolist()

    return pd.DataFrame(out)
