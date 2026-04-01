"""Ensemble model evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.training.trainer import get_device
from toxfam.evaluation.metrics import calculate_binary_metrics_with_scores
from toxfam.model.calibration import ModelWithTemperature
from toxfam.training.strategies import DataSelector
from toxfam.visualization.analysis import plot_binary_pr, plot_binary_roc


def evaluate_ensemble(
    model_dirs: list[Path],
    *,
    output_dir: Path | None = None,
    method: str = "mean",
) -> dict:
    """Evaluate an ensemble of trained models.

    Supports heterogeneous models (different strategies/class counts).
    Each model produces a p_toxic score, which are ensembled at the binary level.

    1. Load each model's calibrated weights + class_indices.json + config.yaml
    2. For each model, derive p_toxic on the test set
    3. Average p_toxic across models (or majority vote at threshold 0.5)
    4. Compute binary metrics from ensembled p_toxic
    """
    if output_dir is None:
        output_dir = Path("model/model_output/ensemble")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    # Load configs and class maps for each model
    configs = []
    class_maps = []
    for md in model_dirs:
        cfg_path = md / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        configs.append(TrainConfig.from_yaml(cfg_path))

        cj_path = md / "class_indices.json"
        if not cj_path.exists():
            raise FileNotFoundError(f"class_indices.json not found in {md}")
        with open(cj_path) as f:
            class_maps.append(json.load(f))

    # Use first model's config for loading test data
    ref_config = configs[0]

    # Load test data
    df = pd.read_csv(ref_config.input_csv)
    _, _, test_df = analyze_data_splits(df)

    # Prepare binary ground truth from family labels
    if "is_toxic" not in test_df.columns:
        from toxfam.evaluation.metrics import to_binary_class

        test_df["is_toxic"] = test_df["Protein families"].apply(
            lambda x: to_binary_class(x) == "toxin"
        )
    y_true_binary = test_df["is_toxic"].astype(int).to_numpy()

    from sklearn.preprocessing import LabelEncoder

    # Collect p_toxic from each model
    all_p_toxic = []

    for i, md in enumerate(model_dirs):
        cfg = configs[i]
        cm = class_maps[i]
        classes_i = [cm[str(j)] for j in range(len(cm))]
        strategy = cfg.training_strategy

        # Determine label column for this model's strategy
        label_col = "Protein families"
        df_model = test_df.copy()
        if strategy in ("binary", "hierarchical"):
            label_col = "is_toxic_label"
            df_model[label_col] = df_model["is_toxic"].map({True: "toxic", False: "nontoxic"})

        le = LabelEncoder()
        le.classes_ = np.array(classes_i)

        print(f"Loading model {i + 1}/{len(model_dirs)} from {md} (strategy={strategy}, classes={len(classes_i)})")

        model = _load_calibrated_model(md, cfg, len(classes_i), device)
        probs = _get_model_probs(model, df_model, label_col, le, cfg, device)

        # Derive p_toxic from this model's softmax output
        if strategy in ("binary", "hierarchical"):
            toxic_idx = list(le.classes_).index("toxic") if "toxic" in le.classes_ else 1
            p_toxic_i = probs[:, toxic_idx]
        else:
            nontox_idx = list(le.classes_).index("nontox") if "nontox" in le.classes_ else 0
            p_toxic_i = 1.0 - probs[:, nontox_idx]

        all_p_toxic.append(p_toxic_i)
        print(f"  p_toxic range: [{p_toxic_i.min():.4f}, {p_toxic_i.max():.4f}]")

    # Ensemble at binary probability level
    stacked = np.stack(all_p_toxic, axis=0)  # (n_models, n_samples)
    if method == "mean":
        p_toxic = stacked.mean(axis=0)
    elif method == "vote":
        # Majority vote at threshold 0.5
        votes = (stacked > 0.5).astype(int)  # (n_models, n_samples)
        p_toxic = votes.mean(axis=0)  # fraction of models voting toxic
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    binary_metrics = calculate_binary_metrics_with_scores(y_true_binary, p_toxic)

    # Save results
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    serializable = {
        k: v for k, v in binary_metrics.items()
        if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                      "roc_thresholds", "pr_thresholds")
    }
    serializable["method"] = method
    serializable["n_models"] = len(model_dirs)
    serializable["model_dirs"] = [str(md) for md in model_dirs]
    (metrics_dir / "ensemble_binary_metrics.json").write_text(
        json.dumps(serializable, indent=4)
    )

    # Plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_binary_roc(
        binary_metrics["fpr"], binary_metrics["tpr"],
        binary_metrics["roc_auc"],
        plots_dir / "ensemble_binary_roc.png",
    )
    plot_binary_pr(
        binary_metrics["precision_curve"], binary_metrics["recall_curve"],
        binary_metrics["pr_auc"],
        plots_dir / "ensemble_binary_pr.png",
    )

    print(
        f"Ensemble ({method}, {len(model_dirs)} models): "
        f"ROC-AUC={binary_metrics['roc_auc']:.4f}, "
        f"PR-AUC={binary_metrics['pr_auc']:.4f}, "
        f"MCC={binary_metrics['mcc']:.4f}"
    )

    return binary_metrics


def _load_calibrated_model(
    model_dir: Path, config: TrainConfig, num_classes: int, device: torch.device,
) -> torch.nn.Module:
    """Load a calibrated model from a model directory."""
    strategy = config.training_strategy
    calibrated_path = model_dir / "best_model_calibrated.pt"

    if strategy == "multitask":
        from toxfam.model.architectures import MultiTaskMLP

        base = MultiTaskMLP(
            input_dim=config.effective_embedding_dim,
            hidden_dims=config.hidden_dims,
            num_family_classes=num_classes,
            dropout=config.dropout,
        )
    elif strategy == "hierarchical":
        from toxfam.model.architectures import HierarchicalMLP, ModularMLP

        # Build a dummy projector to get the state dict structure
        dummy = ModularMLP(
            input_dim=config.effective_embedding_dim,
            hidden_dims=config.hidden_dims,
            num_classes=2,
            dropout=config.dropout,
        )
        base = HierarchicalMLP(
            projector_state=dummy.projector.state_dict(),
            projector_out_dim=config.hidden_dims[0],
            hidden_dim=config.stage2_hidden_dim,
            num_binary_classes=num_classes,
            freeze_backbone=config.stage2_freeze_backbone,
        )
    elif strategy == "combined":
        from toxfam.model.architectures import MultiInputMLP

        base = MultiInputMLP(
            embed_dim=config.effective_embedding_dim,
            tax_dim=config.tax_dim,
            hidden_dims=config.hidden_dims,
            num_classes=num_classes,
            dropout=config.dropout,
        )
    else:
        from toxfam.model.architectures import ModularMLP

        base = ModularMLP(
            input_dim=config.effective_embedding_dim,
            hidden_dims=config.hidden_dims,
            num_classes=num_classes,
            dropout=config.dropout,
        )

    state_dict = torch.load(calibrated_path, map_location=device, weights_only=True)

    if strategy == "multitask":
        # MultiTaskMLP returns a tuple, which ModelWithTemperature can't handle.
        # Load the model directly and store temperature separately.
        model_sd = {k.removeprefix("model."): v for k, v in state_dict.items() if k.startswith("model.")}
        base.load_state_dict(model_sd)
        base.to(device)
        base.eval()
        temperature = state_dict.get("temperature", torch.ones(1))
        base._ensemble_temperature = temperature.to(device)
        return base
    else:
        model = ModelWithTemperature(base, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model


def _get_model_probs(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    label_col: str,
    label_encoder,
    config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    """Get softmax probabilities from a model on test data."""
    h5_paths = [str(p) for p in config.h5_paths]
    cpp_h5 = str(config.cpp_h5_path) if config.cpp_h5_path else None
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None

    ds = ToxDataset(
        test_df, h5_paths, label_encoder=label_encoder, is_train=False,
        label_col=label_col, tax_h5_path=tax_h5, cpp_h5_path=cpp_h5,
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)
    selector = DataSelector(loader, "both" if config.training_strategy == "combined" else "emb_only")

    all_probs = []
    temperature = getattr(model, "_ensemble_temperature", None)
    model.eval()
    with torch.no_grad():
        for features, _ in selector:
            if isinstance(features, (tuple, list)):
                features = [f.to(device) for f in features]
                outputs = model(*features)
            else:
                outputs = model(features.to(device))
            # MultiTaskMLP returns (family_logits, binary_logits) tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # use family logits
            # Apply temperature scaling for manually-loaded multitask models
            if temperature is not None:
                outputs = outputs / temperature
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    ds.close()
    return np.concatenate(all_probs, axis=0)
