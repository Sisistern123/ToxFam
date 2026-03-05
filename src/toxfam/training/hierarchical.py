"""Hierarchical two-stage training strategy.

Stage 1: Family classification — trains ModularMLP to classify proteins into families
         using BOTH toxic and non-toxic members. This teaches the model family-level
         structural/functional features.

Stage 2: Tox/nontox binary classification — loads Stage 1's projector as a frozen (or
         fine-tunable) backbone into HierarchicalMLP with a binary head.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from toxfam.data.dataset import ToxDataset
from toxfam.device import get_device
from toxfam.model.architectures import HierarchicalMLP, ModularMLP
from toxfam.training.strategies import DataSelector
from toxfam.training.trainer import get_class_weights, train_model
from toxfam.visualization.plots import plot_loss_curve

if TYPE_CHECKING:
    from toxfam.config import TrainConfig


def _train_stage1(
    train_loader: DataLoader,
    val_loader: DataLoader,
    w_tensor: torch.Tensor,
    num_classes: int,
    out_dir: Path,
    config: TrainConfig,
) -> ModularMLP:
    """Stage 1: Train family classifier."""
    print("\n" + "=" * 60)
    print("STAGE 1: Family Classification")
    print("=" * 60)

    model = ModularMLP(
        input_dim=config.effective_embedding_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
    )

    model, hist = train_model(
        model,
        DataSelector(train_loader, "emb_only"),
        DataSelector(val_loader, "emb_only"),
        w_tensor,
        config,
    )

    plot_loss_curve(hist, out_dir / "plots" / "loss_stage1_family.png")

    # Save Stage 1 model and projector separately
    stage1_path = out_dir / "stage1_family_model.pt"
    torch.save(model.state_dict(), stage1_path)
    print(f"Saved Stage 1 model to {stage1_path}")

    projector_path = out_dir / "stage1_projector.pt"
    torch.save(model.projector.state_dict(), projector_path)
    print(f"Saved Stage 1 projector to {projector_path}")

    return model


def _train_stage2(
    stage1_model: ModularMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    w_tensor: torch.Tensor,
    out_dir: Path,
    config: TrainConfig,
) -> HierarchicalMLP:
    """Stage 2: Train binary tox/nontox classifier using Stage 1 backbone."""
    print("\n" + "=" * 60)
    print("STAGE 2: Tox/Nontox Binary Classification")
    print("=" * 60)

    # Extract the projector output dim from the first linear layer
    projector = stage1_model.projector
    backbone_out_dim = config.hidden_dims[0]

    model = HierarchicalMLP(
        backbone=projector,
        backbone_out_dim=backbone_out_dim,
        num_classes=2,
        dropout=config.dropout,
        freeze_backbone=config.stage2_freeze_backbone,
        head_hidden_dim=config.stage2_hidden_dim,
    )

    freeze_label = "frozen" if config.stage2_freeze_backbone else "fine-tuned"
    print(f"  Backbone: {freeze_label}")
    print(f"  Head hidden dim: {config.stage2_hidden_dim}")

    # Use a separate learning rate for Stage 2 if configured
    stage2_lr = config.stage2_learning_rate or config.learning_rate / 10
    print(f"  Learning rate: {stage2_lr}")

    # Create a temporary config with Stage 2 learning rate
    stage2_config = config.model_copy(update={"learning_rate": stage2_lr})

    model, hist = train_model(
        model,
        DataSelector(train_loader, "emb_only"),
        DataSelector(val_loader, "emb_only"),
        w_tensor,
        stage2_config,
    )

    plot_loss_curve(hist, out_dir / "plots" / "loss_stage2_binary.png")

    return model


def run_hierarchical_strategy(
    train_df,
    val_df,
    h5_paths: list[str],
    out_dir: Path,
    config: TrainConfig,
) -> HierarchicalMLP:
    """Run the full hierarchical two-stage training pipeline.

    This function manages both stages including dataset creation with
    different label columns for each stage.

    Args:
        train_df: Training DataFrame with 'Protein families' and 'is_toxic' columns.
        val_df: Validation DataFrame.
        h5_paths: List of HDF5 embedding file paths.
        out_dir: Output directory.
        config: Training configuration.

    Returns:
        Trained HierarchicalMLP model.
    """
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ──────── Stage 1: Family classification ────────
    stage1_model = None

    if config.stage1_model_path and config.stage1_model_path.exists():
        # Load pretrained Stage 1 model
        print(f"\nLoading pretrained Stage 1 model from {config.stage1_model_path}")
        device = get_device()

        # We need to know num_classes — try loading class_indices from same dir
        stage1_dir = config.stage1_model_path.parent
        class_json = stage1_dir / "stage1_class_indices.json"
        if class_json.exists():
            with open(class_json) as f:
                class_map = json.load(f)
            num_family_classes = len(class_map)
        else:
            # Infer from the checkpoint
            state_dict = torch.load(
                config.stage1_model_path, map_location=device, weights_only=True
            )
            # Last layer weight shape tells us num_classes
            last_key = [k for k in state_dict if "backbone" in k and "weight" in k][-1]
            num_family_classes = state_dict[last_key].shape[0]

        stage1_model = ModularMLP(
            input_dim=config.effective_embedding_dim,
            hidden_dims=config.hidden_dims,
            num_classes=num_family_classes,
            dropout=config.dropout,
        )
        stage1_model.load_state_dict(
            torch.load(config.stage1_model_path, map_location=device, weights_only=True)
        )
        print(f"  Loaded Stage 1 model with {num_family_classes} family classes")
    else:
        # Train Stage 1 from scratch
        family_col = "Protein families"
        cpp_h5 = str(config.cpp_h5_path) if config.cpp_h5_path else None

        train_ds_s1 = ToxDataset(
            train_df, h5_paths, is_train=True, label_col=family_col, cpp_h5_path=cpp_h5,
        )
        val_ds_s1 = ToxDataset(
            val_df, h5_paths, label_encoder=train_ds_s1.le, is_train=False,
            label_col=family_col, cpp_h5_path=cpp_h5,
        )

        # Save Stage 1 class mapping
        s1_class_indices = {int(i): label for i, label in enumerate(train_ds_s1.le.classes_)}
        s1_json_path = out_dir / "stage1_class_indices.json"
        with open(s1_json_path, "w") as f:
            json.dump(s1_class_indices, f, indent=4)
        print(f"Stage 1 classes: {train_ds_s1.num_classes}")

        train_loader_s1 = DataLoader(train_ds_s1, batch_size=config.batch_size, shuffle=True)
        val_loader_s1 = DataLoader(val_ds_s1, batch_size=config.batch_size, shuffle=False)
        _, w_tensor_s1, _ = get_class_weights(train_ds_s1)

        stage1_model = _train_stage1(
            train_loader_s1, val_loader_s1, w_tensor_s1,
            train_ds_s1.num_classes, out_dir, config,
        )

        train_ds_s1.close()
        val_ds_s1.close()

    # ──────── Stage 2: Binary tox/nontox ────────
    binary_col = "is_toxic_label"

    # Create binary label column
    train_df_s2 = train_df.copy()
    val_df_s2 = val_df.copy()
    train_df_s2[binary_col] = train_df_s2["is_toxic"].map({True: "toxic", False: "nontoxic"})
    val_df_s2[binary_col] = val_df_s2["is_toxic"].map({True: "toxic", False: "nontoxic"})

    cpp_h5_s2 = str(config.cpp_h5_path) if config.cpp_h5_path else None
    extra_kw: dict = {}
    if config.hbi_h5_path:
        extra_kw["hbi_h5_path"] = str(config.hbi_h5_path)
    if config.handcrafted_h5_path:
        extra_kw["handcrafted_h5_path"] = str(config.handcrafted_h5_path)
    if config.include_length:
        extra_kw["include_length"] = True
    if config.include_venom_indicator:
        extra_kw["include_venom_indicator"] = True

    train_ds_s2 = ToxDataset(
        train_df_s2, h5_paths, is_train=True, label_col=binary_col,
        cpp_h5_path=cpp_h5_s2, **extra_kw,
    )
    val_ds_s2 = ToxDataset(
        val_df_s2, h5_paths, label_encoder=train_ds_s2.le, is_train=False,
        label_col=binary_col, cpp_h5_path=cpp_h5_s2, **extra_kw,
    )

    # Save Stage 2 class mapping
    s2_class_indices = {int(i): label for i, label in enumerate(train_ds_s2.le.classes_)}
    s2_json_path = out_dir / "class_indices.json"
    with open(s2_json_path, "w") as f:
        json.dump(s2_class_indices, f, indent=4)
    print(f"Stage 2 classes: {s2_class_indices}")

    train_loader_s2 = DataLoader(train_ds_s2, batch_size=config.batch_size, shuffle=True)
    val_loader_s2 = DataLoader(val_ds_s2, batch_size=config.batch_size, shuffle=False)
    _, w_tensor_s2, _ = get_class_weights(train_ds_s2)

    final_model = _train_stage2(
        stage1_model, train_loader_s2, val_loader_s2, w_tensor_s2, out_dir, config,
    )

    train_ds_s2.close()
    val_ds_s2.close()

    return final_model
