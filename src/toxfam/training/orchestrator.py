from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import wandb

from toxfam.config import TrainConfig
from toxfam.data.dataset import ToxDataset, analyze_data_splits
from toxfam.model.calibration import ModelWithTemperature
from toxfam.training.strategies import (
    DataSelector,
    evaluate_label_on_dataset,
    run_combined_strategy,
    run_pretrain_finetune_strategy,
    run_standard_strategy,
)
from toxfam.training.trainer import get_class_weights
from toxfam.visualization.analysis import analyze_label_distribution_for_split


def run_training(config: TrainConfig) -> None:
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Weights & Biases (wandb) setup ----
    # Can be overridden from the shell; these are just sensible defaults.
    os.environ.setdefault("WANDB_PROJECT", "toxfam")
    os.environ.setdefault("WANDB_LOG_MODEL", "true")

    # Initial login (expects WANDB_API_KEY to be configured externally if needed).
    wandb.login()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print("Using device: ", device, flush=True)

    # Initialize a run and log key hyperparameters.
    wandb_config = {
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "training_strategy": config.training_strategy,
    }
    wandb.init(project=os.environ["WANDB_PROJECT"], config=wandb_config)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(config.input_csv)
    train_df, val_df, test_df = analyze_data_splits(df)

    label_col = "Protein families"
    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, out_root)

    # 2. Init Datasets
    h5_paths = [str(p) for p in config.h5_paths]
    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None

    train_ds = ToxDataset(
        train_df, h5_paths, is_train=True, tax_h5_path=tax_h5
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
    elif strategy == "pretrain_finetune":
        final_model = run_pretrain_finetune_strategy(
            train_loader, val_loader, w_tensor, train_ds.num_classes, out_root, config
        )
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

    loss_fn = torch.nn.CrossEntropyLoss()

    # 4. Evaluation: Uncalibrated
    print("\nRunning Final Evaluation (Uncalibrated)...")
    evaluate_label_on_dataset(
        final_model, val_df, label_col, train_ds.le, loss_fn, "Validation", out_root,
        config,
    )
    evaluate_label_on_dataset(
        final_model, test_df, label_col, train_ds.le, loss_fn, "Test", out_root,
        config,
    )

    # 5. Calibration (Temperature Scaling)
    print("\nRunning Calibration (Temperature Scaling)...")

    if strategy == "combined":
        val_selector = DataSelector(val_loader, "both")
    else:
        val_selector = DataSelector(val_loader, "emb_only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)

    scaled_model = ModelWithTemperature(final_model, device)
    scaled_model.set_temperature(val_selector)
    calibrated_path = out_root / "best_model_calibrated.pt"
    torch.save(scaled_model.state_dict(), calibrated_path)
    print(f"Saved calibrated model to {calibrated_path}")

    # Log calibrated model as a wandb artifact (model generation tracking).
    if wandb.run is not None:
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

    train_ds.close()
    val_ds.close()
