from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from toxfam.config import TrainConfig
from toxfam.device import get_device

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.model.calibration import ModelWithTemperature
from toxfam.evaluation.metrics import (
    calculate_binary_metrics_with_scores,
    find_optimal_threshold,
    to_binary_class,
)
from toxfam.training.strategies import (
    DataSelector,
    _MultiTaskBinaryWrapper,
    evaluate_label_on_dataset,
    run_binary_strategy,
    run_combined_strategy,
    run_standard_strategy,
)
from toxfam.training.trainer import get_class_weights
from toxfam.visualization.analysis import (
    analyze_label_distribution_for_split,
    plot_binary_roc_curve,
    plot_precision_recall_curve,
)


def _extra_dataset_kwargs(config: TrainConfig) -> dict:
    """Build extra kwargs for ToxDataset from config (HBI, length, venom)."""
    kw: dict = {}
    if config.hbi_h5_path:
        kw["hbi_h5_path"] = str(config.hbi_h5_path)
    if config.handcrafted_h5_path:
        kw["handcrafted_h5_path"] = str(config.handcrafted_h5_path)
    if config.include_length:
        kw["include_length"] = True
    if config.include_venom_indicator:
        kw["include_venom_indicator"] = True
    return kw


def run_training(config: TrainConfig) -> None:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config for later re-evaluation
    import yaml

    config_copy_path = out_root / "config.yaml"
    with open(config_copy_path, "w") as f:
        yaml.dump(config.model_dump(mode="json"), f, default_flow_style=False)

    device = get_device()
    print(f"Using device: {device}", flush=True)

    # ---- Weights & Biases (wandb) setup (optional) ----
    if wandb is not None:
        os.environ.setdefault("WANDB_PROJECT", "toxfam")
        os.environ.setdefault("WANDB_LOG_MODEL", "true")
        wandb.login()
        wandb_config = {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "training_strategy": config.training_strategy,
        }
        wandb.init(project=os.environ["WANDB_PROJECT"], config=wandb_config)

    # Create organized subdirectories
    plots_dir = out_root / "plots"
    metrics_dir = out_root / "metrics"
    predictions_dir = out_root / "predictions"
    for d in (plots_dir, metrics_dir, predictions_dir):
        d.mkdir(exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    # Derive is_toxic if missing (Phase 2 compat)
    for split_df in (train_df, val_df, test_df):
        if "is_toxic" not in split_df.columns:
            split_df["is_toxic"] = split_df["Protein families"] != "nontox"

    label_col = "Protein families"
    strategy = config.training_strategy

    # For binary strategy, derive binary labels upfront
    if strategy == "binary":
        label_col = "is_toxic_label"
        for split_df in (train_df, val_df, test_df):
            split_df[label_col] = split_df["is_toxic"].map(
                {True: "toxic", False: "nontoxic"}
            )

    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_root)

    # 2. Init Datasets
    h5_paths = [str(p) for p in config.h5_paths]
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None
    cpp_h5 = str(config.cpp_h5_path) if config.cpp_h5_path else None
    extra_ds_kwargs = _extra_dataset_kwargs(config)

    train_ds = ToxDataset(
        train_df, h5_paths, is_train=True, label_col=label_col,
        tax_h5_path=tax_h5, cpp_h5_path=cpp_h5, **extra_ds_kwargs,
    )

    class_indices = {int(i): label for i, label in enumerate(train_ds.le.classes_)}
    class_json_path = out_root / "class_indices.json"
    with open(class_json_path, "w") as f:
        json.dump(class_indices, f, indent=4)
    print(f"Saved class mapping to {class_json_path}")

    val_ds = ToxDataset(
        val_df,
        h5_paths,
        label_encoder=train_ds.le,
        is_train=False,
        label_col=label_col,
        tax_h5_path=tax_h5,
        cpp_h5_path=cpp_h5,
        **extra_ds_kwargs,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    _, w_tensor, _ = get_class_weights(train_ds)

    # 3. Dispatch Strategy
    final_model = None

    if strategy == "hierarchical":
        from toxfam.training.hierarchical import run_hierarchical_strategy

        train_ds.close()
        val_ds.close()

        final_model = run_hierarchical_strategy(
            train_df, val_df, h5_paths, out_root, config,
        )

        # For hierarchical, Stage 2 produces a binary model — adjust label_col and encoder
        label_col = "is_toxic_label"
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        train_df[label_col] = train_df["is_toxic"].map({True: "toxic", False: "nontoxic"})
        val_df[label_col] = val_df["is_toxic"].map({True: "toxic", False: "nontoxic"})
        test_df[label_col] = test_df["is_toxic"].map({True: "toxic", False: "nontoxic"})

        # Rebuild datasets for evaluation with binary labels
        train_ds = ToxDataset(
            train_df, h5_paths, is_train=True, label_col=label_col,
            cpp_h5_path=cpp_h5, **extra_ds_kwargs,
        )
        val_ds = ToxDataset(
            val_df, h5_paths, label_encoder=train_ds.le, is_train=False,
            label_col=label_col, cpp_h5_path=cpp_h5, **extra_ds_kwargs,
        )
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    elif strategy == "binary":
        final_model = run_binary_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "standard":
        final_model = run_standard_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "combined":
        final_model = run_combined_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "multitask":
        from toxfam.training.strategies import run_multitask_strategy

        final_model = run_multitask_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Evaluation: Uncalibrated
    print("\nRunning Final Evaluation (Uncalibrated)...")
    evaluate_label_on_dataset(
        final_model,
        val_df,
        label_col,
        train_ds.le,
        loss_fn,
        "Validation",
        out_root,
        config,
    )
    evaluate_label_on_dataset(
        final_model,
        test_df,
        label_col,
        train_ds.le,
        loss_fn,
        "Test",
        out_root,
        config,
    )

    # 5. Calibration (Temperature Scaling)
    print("\nRunning Calibration (Temperature Scaling)...")

    if strategy == "combined":
        val_selector = DataSelector(val_loader, "both")
    else:
        val_selector = DataSelector(val_loader, "emb_only")

    device = get_device()
    final_model = final_model.to(device)

    scaled_model = ModelWithTemperature(final_model, device)
    scaled_model.set_temperature(val_selector)
    calibrated_path = out_root / "best_model_calibrated.pt"
    torch.save(scaled_model.state_dict(), calibrated_path)
    print(f"Saved calibrated model to {calibrated_path}")

    # Log calibrated model as a wandb artifact (model generation tracking).
    if wandb is not None and wandb.run is not None:
        calibrated_artifact = wandb.Artifact(
            name="toxfam-best-model-calibrated",
            type="model",
            metadata={"strategy": strategy},
        )
        calibrated_artifact.add_file(str(calibrated_path))
        wandb.log_artifact(calibrated_artifact)

    # 6. Evaluation: Calibrated
    print("\nRunning Final Evaluation (Calibrated)...")
    evaluate_label_on_dataset(
        scaled_model,
        val_df,
        label_col,
        train_ds.le,
        loss_fn,
        "Validation_Calibrated",
        out_root,
        config,
    )
    evaluate_label_on_dataset(
        scaled_model,
        test_df,
        label_col,
        train_ds.le,
        loss_fn,
        "Test_Calibrated",
        out_root,
        config,
    )

    # 7. Binary metrics for ALL strategies
    # 7a. Compute on validation set first (for threshold optimization)
    print("\nComputing binary toxic/nontoxic metrics on validation set...")
    val_binary = _compute_and_save_binary_metrics(
        scaled_model,
        val_df,
        label_col,
        train_ds.le,
        config,
        out_root,
        tag="Validation_Calibrated",
    )

    # 7b. Threshold optimization on validation set
    optimal_threshold = 0.5
    if val_binary is not None:
        print("\nOptimizing classification threshold on validation set...")
        thresh_result = find_optimal_threshold(
            val_binary["y_true"],
            val_binary["p_toxic"],
            method="youden",
        )
        optimal_threshold = thresh_result["optimal_threshold"]
        print(
            f"  Optimal threshold (Youden's J): {optimal_threshold:.4f}"
        )
        (out_root / "metrics" / "threshold_optimization.json").write_text(
            json.dumps(thresh_result, indent=4)
        )

    # 7c. Compute on test set with default threshold (0.5)
    print("\nComputing binary toxic/nontoxic metrics on test set...")
    _compute_and_save_binary_metrics(
        scaled_model,
        test_df,
        label_col,
        train_ds.le,
        config,
        out_root,
        tag="Test_Calibrated",
    )

    # 7d. Compute on test set with optimized threshold
    if optimal_threshold != 0.5:
        print(f"\nComputing test metrics with optimized threshold ({optimal_threshold:.4f})...")
        _compute_and_save_binary_metrics(
            scaled_model,
            test_df,
            label_col,
            train_ds.le,
            config,
            out_root,
            tag="Test_Calibrated_Optimized",
            threshold=optimal_threshold,
        )

    # 7e. Multitask: also evaluate binary head directly
    if strategy == "multitask":
        print("\nEvaluating multitask binary head directly...")
        inner_model = scaled_model.model  # _MultiTaskFamilyWrapper
        if hasattr(inner_model, "model"):
            mt_model = inner_model.model  # MultiTaskMLP
            binary_wrapper = _MultiTaskBinaryWrapper(mt_model)
            binary_wrapper = binary_wrapper.to(device)

            # Build binary labels and encoder for the 2-class binary head
            from sklearn.preprocessing import LabelEncoder

            test_df_bin = test_df.copy()
            test_df_bin["_binary_label"] = test_df_bin["is_toxic"].map(
                {True: "toxic", False: "nontoxic"}
            )
            binary_le = LabelEncoder()
            binary_le.classes_ = __import__("numpy").array(["nontoxic", "toxic"])

            # Create a temporary config that looks like a binary strategy
            binary_config = config.model_copy(
                update={"training_strategy": "binary"}
            )

            # Build binary val loader for calibration
            bin_val_ds = ToxDataset(
                val_df.assign(
                    _binary_label=val_df["is_toxic"].map(
                        {True: "toxic", False: "nontoxic"}
                    )
                ),
                h5_paths,
                label_encoder=binary_le,
                is_train=False,
                label_col="_binary_label",
                cpp_h5_path=cpp_h5,
                **extra_ds_kwargs,
            )
            bin_val_loader = DataLoader(
                bin_val_ds, batch_size=config.batch_size, shuffle=False
            )
            bin_val_selector = DataSelector(bin_val_loader, "emb_only")

            binary_scaled = ModelWithTemperature(binary_wrapper, device)
            binary_scaled.set_temperature(bin_val_selector)
            _compute_and_save_binary_metrics(
                binary_scaled,
                test_df_bin,
                "_binary_label",
                binary_le,
                binary_config,
                out_root,
                tag="Test_Calibrated_BinaryHead",
            )
            bin_val_ds.close()

    train_ds.close()
    val_ds.close()

    if wandb is not None and wandb.run is not None:
        wandb.finish()


def _compute_and_save_binary_metrics(
    model,
    test_df: pd.DataFrame,
    label_col: str,
    label_encoder,
    config: TrainConfig,
    out_root: Path,
    *,
    tag: str = "Test_Calibrated",
    threshold: float = 0.5,
) -> dict | None:
    """Compute binary toxic/nontoxic metrics and save results + plots.

    Works for all strategies:
    - Binary/hierarchical: toxic class probability directly.
    - Standard/combined multiclass: p_toxic = 1 - softmax[nontox_idx].

    Returns dict with y_true, p_toxic arrays for downstream threshold optimization,
    or None if binary metrics could not be computed.
    """
    import numpy as np
    import torch.nn.functional as F

    device = get_device()
    model = model.to(device)

    h5_paths = [str(p) for p in config.h5_paths]
    cpp_h5 = str(config.cpp_h5_path) if config.cpp_h5_path else None
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None
    extra_kw = _extra_dataset_kwargs(config)

    ds = ToxDataset(
        test_df,
        h5_paths,
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=tax_h5,
        cpp_h5_path=cpp_h5,
        **extra_kw,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    strategy = config.training_strategy
    if strategy == "combined":
        selector = DataSelector(loader, "both")
    else:
        selector = DataSelector(loader, "emb_only")

    all_scores = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for features, labels in selector:
            labels_dev = labels.to(device)
            if isinstance(features, (tuple, list)):
                features = [f.to(device) for f in features]
                outputs = model(*features)
            else:
                outputs = model(features.to(device))
            probs = F.softmax(outputs, dim=1)
            all_scores.append(probs.cpu().numpy())
            all_labels.extend(labels_dev.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    classes = list(label_encoder.classes_)

    # Derive binary y_true and p_toxic
    toxic_idx = None
    nontox_idx = None
    for i, cls_name in enumerate(classes):
        bc = to_binary_class(cls_name)
        if bc == "toxin" and toxic_idx is None:
            toxic_idx = i
        elif bc == "nontoxin" and nontox_idx is None:
            nontox_idx = i

    if config.training_strategy in ("binary", "hierarchical"):
        if toxic_idx is None:
            toxic_idx = 1
        y_true_binary = np.array(all_labels) == toxic_idx
        p_toxic = all_scores[:, toxic_idx]
    else:
        if nontox_idx is None:
            print("  Warning: could not find nontox class for binary metrics")
            ds.close()
            return None

        p_toxic = 1.0 - all_scores[:, nontox_idx]
        y_true_binary = np.array(
            [to_binary_class(classes[lbl]) == "toxin" for lbl in all_labels]
        )

    y_true_int = y_true_binary.astype(int)
    binary_metrics = calculate_binary_metrics_with_scores(
        y_true_int, p_toxic, threshold=threshold,
    )

    # Save metrics
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    serializable = {
        k: v for k, v in binary_metrics.items()
        if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                      "roc_thresholds", "pr_thresholds")
    }
    (metrics_dir / f"binary_{tag.lower()}_metrics.json").write_text(
        json.dumps(serializable, indent=4)
    )
    print(
        f"  Binary ROC-AUC: {binary_metrics['roc_auc']:.4f}, "
        f"PR-AUC: {binary_metrics['pr_auc']:.4f}, "
        f"MCC: {binary_metrics['mcc']:.4f}"
    )

    # Save plots
    plots_dir = out_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_binary_roc_curve(
        binary_metrics["fpr"],
        binary_metrics["tpr"],
        binary_metrics["roc_auc"],
        plots_dir / f"binary_{tag.lower()}_roc.png",
    )
    plot_precision_recall_curve(
        binary_metrics["precision_curve"],
        binary_metrics["recall_curve"],
        binary_metrics["pr_auc"],
        plots_dir / f"binary_{tag.lower()}_pr.png",
    )

    ds.close()
    return {"y_true": y_true_int, "p_toxic": p_toxic}
