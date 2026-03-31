from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.model.calibration import ModelWithTemperature
from toxfam.training.strategies import (
    DataSelector,
    _MultiTaskFamilyWrapper,
    evaluate_label_on_dataset,
    run_binary_strategy,
    run_combined_strategy,
    run_multitask_strategy,
    run_standard_strategy,
)
from toxfam.training.trainer import _forward_model, get_class_weights, get_device, set_seed
from toxfam.visualization.analysis import (
    analyze_label_distribution_for_split,
    plot_binary_pr,
    plot_binary_roc,
)

console = Console()


def _compute_binary_labels(df: pd.DataFrame, label_col: str = "Protein families") -> np.ndarray:
    """Convert family labels to binary: 1 = toxic, 0 = nontoxin."""
    from toxfam.evaluation.metrics import to_binary_class

    return (df[label_col].apply(to_binary_class) == "toxin").astype(int).values


def _compute_p_toxic(
    model, dataset_df, config, label_encoder, label_col="Protein families"
) -> np.ndarray:
    """Compute P(toxic) for each sample by summing toxic-class probabilities."""
    from toxfam.evaluation.metrics import NONTOXIN_LABELS

    ds = ToxDataset(
        dataset_df,
        [str(p) for p in config.h5_paths],
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=str(config.tax_h5_path) if config.tax_h5_path else None,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    strategy = config.training_strategy
    if strategy == "combined":
        selector = DataSelector(loader, "both")
    else:
        selector = DataSelector(loader, "emb_only")

    device = get_device()
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for features, _ in selector:
            outputs = _forward_model(model, features, device)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    ds.close()

    # Sum probabilities of all nontoxin classes
    nontox_indices = [
        i for i, cls in enumerate(label_encoder.classes_)
        if cls.lower() in NONTOXIN_LABELS
    ]
    p_nontox = all_probs[:, nontox_indices].sum(axis=1)
    return 1.0 - p_nontox


def _run_binary_metrics_pipeline(
    model, train_ds, val_df, test_df, config, out_root, label_col="Protein families"
):
    """Full binary metrics pipeline: threshold optimization on val, evaluate on test."""
    from toxfam.evaluation.metrics import (
        calculate_binary_metrics_with_scores,
        find_optimal_threshold,
    )

    console.print("\n[bold]Running Binary Metrics Pipeline...[/bold]")

    # Val set
    val_y_true = _compute_binary_labels(val_df, label_col)
    val_p_toxic = _compute_p_toxic(model, val_df, config, train_ds.le, label_col)

    # Threshold optimization on val
    opt_threshold = find_optimal_threshold(val_y_true, val_p_toxic, method="youden")
    console.print(f"  Optimized threshold (Youden's J): {opt_threshold:.4f}")

    # Test set
    test_y_true = _compute_binary_labels(test_df, label_col)
    test_p_toxic = _compute_p_toxic(model, test_df, config, train_ds.le, label_col)

    # Test with default threshold
    test_default = calculate_binary_metrics_with_scores(test_y_true, test_p_toxic, threshold=0.5)
    console.print(
        f"  Test (t=0.5): ROC-AUC={test_default['roc_auc']:.4f}, "
        f"PR-AUC={test_default['pr_auc']:.4f}, MCC={test_default['mcc']:.4f}"
    )

    # Test with optimized threshold
    test_opt = calculate_binary_metrics_with_scores(test_y_true, test_p_toxic, threshold=opt_threshold)
    console.print(
        f"  Test (t={opt_threshold:.3f}): ROC-AUC={test_opt['roc_auc']:.4f}, "
        f"PR-AUC={test_opt['pr_auc']:.4f}, MCC={test_opt['mcc']:.4f}"
    )

    # Save metrics
    metrics_dir = out_root / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    binary_results = {
        "optimized_threshold": opt_threshold,
        "test_default": {k: v for k, v in test_default.items() if not isinstance(v, np.ndarray)},
        "test_optimized": {k: v for k, v in test_opt.items() if not isinstance(v, np.ndarray)},
    }
    (metrics_dir / "binary_metrics.json").write_text(json.dumps(binary_results, indent=4))

    # Plots
    plots_dir = out_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_binary_roc(
        test_default["fpr"], test_default["tpr"], test_default["roc_auc"],
        plots_dir / "binary_roc.png",
    )
    plot_binary_pr(
        test_default["precision_curve"], test_default["recall_curve"],
        test_default["pr_auc"], plots_dir / "binary_pr.png",
    )


def _extra_dataset_kwargs(config: TrainConfig) -> dict:
    """Build auxiliary feature kwargs for ToxDataset from config."""
    kwargs: dict = {}
    if config.cpp_h5_path:
        kwargs["cpp_h5_path"] = str(config.cpp_h5_path)
    if config.hbi_h5_path:
        kwargs["hbi_h5_path"] = str(config.hbi_h5_path)
    if config.include_length:
        kwargs["include_length"] = True
    if config.include_venom_indicator:
        kwargs["include_venom_indicator"] = True
    return kwargs


def run_training(config: TrainConfig) -> None:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Set seed early for full reproducibility
    set_seed(config.seed)

    # Save resolved config to output dir
    config_copy = out_root / "config.yaml"
    config_copy.write_text(yaml.dump(config.model_dump(mode="json"), sort_keys=False))

    # wandb setup (optional — gracefully degrade if not logged in)
    _use_wandb = False
    if wandb is not None:
        try:
            wandb.login()
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=config.model_dump(mode="json"),
            )
            _use_wandb = True
        except Exception:
            console.print("[yellow]wandb login failed — continuing without wandb[/yellow]")

    device = get_device()
    console.print(f"Using device: [bold]{device}[/bold]")

    # Create organized subdirectories
    models_dir = out_root / "models"
    plots_dir = out_root / "plots"
    metrics_dir = out_root / "metrics"
    predictions_dir = out_root / "predictions"
    for d in (models_dir, plots_dir, metrics_dir, predictions_dir):
        d.mkdir(exist_ok=True)

    # 1. Load Data
    console.print("Loading data...")
    strategy = config.training_strategy
    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    label_col = "Protein families"
    is_binary = strategy == "binary"

    if is_binary:
        from toxfam.evaluation.metrics import to_binary_class

        for split_df in (train_df, val_df, test_df):
            split_df["binary_label"] = split_df[label_col].apply(to_binary_class)
        effective_label_col = "binary_label"
    else:
        effective_label_col = label_col

    analyze_label_distribution_for_split(
        train_df, val_df, test_df, effective_label_col, out_root
    )

    # 2. Init Datasets
    h5_paths = [str(p) for p in config.h5_paths]
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None
    extra_ds_kwargs = _extra_dataset_kwargs(config)

    train_ds = ToxDataset(
        train_df, h5_paths, is_train=True, label_col=effective_label_col,
        tax_h5_path=tax_h5, **extra_ds_kwargs,
    )

    # Validate taxonomy vector dimension matches config
    if tax_h5 is not None and config.training_strategy == "combined":
        import h5py as _h5py

        with _h5py.File(tax_h5, "r") as _f:
            _first = next(iter(_f))
            actual_dim = _f[_first][:].shape[0]
            if actual_dim != config.tax_dim:
                raise ValueError(
                    f"Taxonomy H5 has vectors of dim {actual_dim} but config.tax_dim={config.tax_dim}. "
                    f"Regenerate with `toxfam taxonomy` or update tax_dim in your config."
                )

    class_indices = {int(i): label for i, label in enumerate(train_ds.le.classes_)}
    class_json_path = out_root / "class_indices.json"
    with open(class_json_path, "w") as f:
        json.dump(class_indices, f, indent=4)
    console.print(f"Saved class mapping to {class_json_path}")

    val_ds = ToxDataset(
        val_df,
        h5_paths,
        label_encoder=train_ds.le,
        is_train=False,
        label_col=effective_label_col,
        tax_h5_path=tax_h5,
        **extra_ds_kwargs,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    _, w_tensor, _ = get_class_weights(train_ds)

    # 3. Dispatch Strategy
    final_model = None

    if strategy == "standard":
        final_model = run_standard_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "binary":
        final_model = run_binary_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "hierarchical":
        from toxfam.training.hierarchical import run_hierarchical_strategy

        final_model, _binary_le = run_hierarchical_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes,
            train_df, val_df, h5_paths, out_root, config,
        )
        # Hierarchical produces a binary model — override label col for eval
        effective_label_col = "binary_label"
        if "binary_label" not in val_df.columns:
            from toxfam.evaluation.metrics import to_binary_class

            for split_df in (train_df, val_df, test_df):
                split_df["binary_label"] = split_df[label_col].apply(to_binary_class)
        # Rebuild datasets with binary labels for evaluation
        train_ds.close()
        val_ds.close()
        train_ds = ToxDataset(
            train_df, h5_paths, is_train=True,
            label_col="binary_label", tax_h5_path=tax_h5, **extra_ds_kwargs,
        )
        val_ds = ToxDataset(
            val_df, h5_paths, label_encoder=train_ds.le, is_train=False,
            label_col="binary_label", tax_h5_path=tax_h5, **extra_ds_kwargs,
        )
    elif strategy == "multitask":
        final_model = run_multitask_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes,
            train_df, out_root, config,
        )
        # Wrap for family evaluation
        final_model = _MultiTaskFamilyWrapper(final_model)
    elif strategy == "combined":
        final_model = run_combined_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    # Watch model gradients in wandb
    if _use_wandb:
        wandb.watch(final_model, log="gradients", log_freq=50)

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Evaluation: Uncalibrated
    console.print("\n[bold]Running Final Evaluation (Uncalibrated)...[/bold]")
    val_metrics = evaluate_label_on_dataset(
        final_model,
        val_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "val",
        out_root,
        config,
    )
    test_metrics = evaluate_label_on_dataset(
        final_model,
        test_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "test",
        out_root,
        config,
    )

    # 5. Calibration (Temperature Scaling)
    console.print("\n[bold]Running Calibration (Temperature Scaling)...[/bold]")

    if strategy == "combined":
        val_selector = DataSelector(val_loader, "both")
    else:
        val_selector = DataSelector(val_loader, "emb_only")

    device = get_device()
    final_model = final_model.to(device)

    scaled_model = ModelWithTemperature(final_model, device)
    scaled_model.set_temperature(val_selector)
    calibrated_path = models_dir / "best_model_calibrated.pt"
    torch.save(scaled_model.state_dict(), calibrated_path)
    console.print(f"Saved calibrated model to {calibrated_path}")

    # Log calibrated model as a wandb artifact
    if _use_wandb:
        calibrated_artifact = wandb.Artifact(
            name="toxfam-best-model-calibrated",
            type="model",
            metadata={"strategy": strategy},
        )
        calibrated_artifact.add_file(str(calibrated_path))
        wandb.log_artifact(calibrated_artifact)

    # 6. Evaluation: Calibrated
    console.print("\n[bold]Running Final Evaluation (Calibrated)...[/bold]")
    val_cal_metrics = evaluate_label_on_dataset(
        scaled_model,
        val_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "val_calibrated",
        out_root,
        config,
    )
    test_cal_metrics = evaluate_label_on_dataset(
        scaled_model,
        test_df,
        effective_label_col,
        train_ds.le,
        loss_fn,
        "test_calibrated",
        out_root,
        config,
    )

    # wandb summary
    if _use_wandb:
        for tag, m in [
            ("val", val_metrics),
            ("test", test_metrics),
            ("val_calibrated", val_cal_metrics),
            ("test_calibrated", test_cal_metrics),
        ]:
            for k, v in m.items():
                wandb.run.summary[k] = v

    # 7. Binary Metrics Pipeline
    # For binary strategy, the model directly outputs binary classes,
    # so we pass the effective label col. For other strategies, we derive
    # binary from the original family labels.
    _run_binary_metrics_pipeline(
        scaled_model, train_ds, val_df, test_df, config, out_root, effective_label_col
    )

    train_ds.close()
    val_ds.close()

    if _use_wandb:
        wandb.finish()
