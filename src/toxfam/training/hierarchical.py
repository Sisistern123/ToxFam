"""Two-stage hierarchical training: family classification → binary toxic/nontoxin.

Stage 1 trains a ModularMLP on family classification (or loads a pre-trained one).
Stage 2 extracts the projector weights and trains a HierarchicalMLP with a binary
head on top of the (optionally frozen) projector backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from rich.console import Console
from torch.utils.data import DataLoader

from toxfam.data.dataset import ToxDataset
from toxfam.model.architectures import HierarchicalMLP, ModularMLP
from toxfam.training.strategies import DataSelector
from toxfam.training.trainer import get_class_weights, get_device, train_model
from toxfam.visualization.plots import plot_loss_curve

if TYPE_CHECKING:
    from toxfam.config import TrainConfig

console = Console()


def _train_stage1(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    """Stage 1: Train a family classifier and return its projector weights."""
    console.print("[bold]>>> Hierarchical Stage 1: Family Classification[/bold]")
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

    stage1_dir = Path(out_dir) / "stage1"
    stage1_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(hist, stage1_dir / "loss_curve.png")

    # Save projector weights for reuse
    projector_path = stage1_dir / "stage1_projector.pt"
    torch.save(model.projector.state_dict(), projector_path)
    console.print(f"  Saved Stage 1 projector to {projector_path}")

    # Save full Stage 1 model too
    torch.save(model.state_dict(), stage1_dir / "stage1_family_model.pt")

    return model


def _train_stage2(
    train_df, val_df, projector_state, h5_paths, out_dir, config: TrainConfig
):
    """Stage 2: Train a binary head on top of the Stage 1 projector."""
    from toxfam.evaluation.metrics import to_binary_class

    console.print("[bold]>>> Hierarchical Stage 2: Binary Classification[/bold]")

    # Create binary labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["binary_label"] = train_df["Protein families"].apply(to_binary_class)
    val_df["binary_label"] = val_df["Protein families"].apply(to_binary_class)

    from toxfam.training.orchestrator import _extra_dataset_kwargs

    tax_h5 = str(config.tax_h5_path) if config.tax_h5_path else None
    extra_kwargs = _extra_dataset_kwargs(config)

    train_ds = ToxDataset(
        train_df, h5_paths, is_train=True, label_col="binary_label",
        tax_h5_path=tax_h5, **extra_kwargs,
    )
    val_ds = ToxDataset(
        val_df, h5_paths, label_encoder=train_ds.le, is_train=False,
        label_col="binary_label", tax_h5_path=tax_h5, **extra_kwargs,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    _, w_tensor, _ = get_class_weights(train_ds)

    projector_out_dim = config.hidden_dims[0]
    model = HierarchicalMLP(
        projector_state=projector_state,
        projector_out_dim=projector_out_dim,
        hidden_dim=config.stage2_hidden_dim,
        num_binary_classes=2,
        freeze_backbone=config.stage2_freeze_backbone,
    )

    # Stage 2 uses its own learning rate
    stage2_config = config.model_copy(
        update={"learning_rate": config.stage2_learning_rate}
    )

    model, hist = train_model(
        model,
        DataSelector(train_loader, "emb_only"),
        DataSelector(val_loader, "emb_only"),
        w_tensor,
        stage2_config,
    )

    stage2_dir = Path(out_dir) / "stage2"
    stage2_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(hist, stage2_dir / "loss_curve.png")

    train_ds.close()
    val_ds.close()

    return model, train_ds.le


def run_hierarchical_strategy(
    train_loader,
    val_loader,
    w_tensor,
    num_classes,
    train_df,
    val_df,
    h5_paths,
    out_dir,
    config: TrainConfig,
):
    """Full two-stage hierarchical training.

    Returns the Stage 2 binary model and its label encoder.
    """
    # Stage 1: family classification (or load pre-trained)
    if config.stage1_model_path:
        console.print(
            f"  Loading Stage 1 projector from {config.stage1_model_path}"
        )
        device = get_device()
        projector_state = torch.load(
            config.stage1_model_path, map_location=device, weights_only=True
        )
    else:
        stage1_model = _train_stage1(
            train_loader, val_loader, w_tensor, num_classes, out_dir, config
        )
        projector_state = stage1_model.projector.state_dict()

    # Stage 2: binary classification
    model, binary_le = _train_stage2(
        train_df, val_df, projector_state, h5_paths, out_dir, config
    )

    return model, binary_le
