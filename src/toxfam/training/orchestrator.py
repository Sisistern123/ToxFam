from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from rich.console import Console
from torch.utils.data import DataLoader
import wandb

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.model.calibration import ModelWithTemperature
from toxfam.training.strategies import (
    DataSelector,
    evaluate_label_on_dataset,
    run_combined_strategy,
    run_standard_strategy,
)
from toxfam.training.trainer import get_class_weights, get_device, set_seed
from toxfam.visualization.analysis import analyze_label_distribution_for_split

console = Console()


def run_training(config: TrainConfig) -> None:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Set seed early for full reproducibility
    set_seed(config.seed)

    # Save resolved config to output dir
    config_copy = out_root / "config.yaml"
    config_copy.write_text(yaml.dump(config.model_dump(mode="json"), sort_keys=False))

    # wandb setup (required)
    wandb.login()
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=config.model_dump(mode="json"),
    )

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
    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    label_col = "Protein families"
    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_root)

    # 2. Init Datasets
    h5_paths = [str(p) for p in config.h5_paths]
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None

    train_ds = ToxDataset(train_df, h5_paths, is_train=True, tax_h5_path=tax_h5)

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

    # Save model config for deterministic architecture reconstruction at inference
    from toxfam.model.model_config import ModelConfig

    num_classes = len(class_indices)
    model_config = ModelConfig(
        architecture="MultiInputMLP"
        if config.training_strategy == "combined"
        else "ModularMLP",
        embedding_dim=config.embedding_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
        tax_dim=config.tax_dim if config.training_strategy == "combined" else None,
    )
    model_config.save(out_root / "model_config.json")

    val_ds = ToxDataset(
        val_df,
        h5_paths,
        label_encoder=train_ds.le,
        is_train=False,
        tax_h5_path=tax_h5,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    _, w_tensor, _ = get_class_weights(train_ds)

    # 3. Dispatch Strategy
    strategy = config.training_strategy
    final_model = None

    if strategy == "standard":
        final_model = run_standard_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    elif strategy == "combined":
        final_model = run_combined_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    # Watch model gradients in wandb
    if wandb.run is not None:
        wandb.watch(final_model, log="gradients", log_freq=50)

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Evaluation: Uncalibrated
    console.print("\n[bold]Running Final Evaluation (Uncalibrated)...[/bold]")
    val_metrics = evaluate_label_on_dataset(
        final_model,
        val_df,
        label_col,
        train_ds.le,
        loss_fn,
        "val",
        out_root,
        config,
    )
    test_metrics = evaluate_label_on_dataset(
        final_model,
        test_df,
        label_col,
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
    if wandb.run is not None:
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
        label_col,
        train_ds.le,
        loss_fn,
        "val_calibrated",
        out_root,
        config,
    )
    test_cal_metrics = evaluate_label_on_dataset(
        scaled_model,
        test_df,
        label_col,
        train_ds.le,
        loss_fn,
        "test_calibrated",
        out_root,
        config,
    )

    # wandb summary
    if wandb.run is not None:
        for tag, m in [
            ("val", val_metrics),
            ("test", test_metrics),
            ("val_calibrated", val_cal_metrics),
            ("test_calibrated", test_cal_metrics),
        ]:
            for k, v in m.items():
                wandb.run.summary[k] = v

    train_ds.close()
    val_ds.close()

    wandb.finish()
