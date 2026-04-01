from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from rich.console import Console
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

import torch.nn as nn

from toxfam.data.dataset import ToxDataset
from toxfam.model.architectures import ModularMLP, MultiInputMLP, MultiTaskMLP
from toxfam.training.trainer import (
    _forward_model,
    evaluate_model,
    get_device,
    train_model,
)
from toxfam.visualization.analysis import plot_multiclass_roc_from_scores
from toxfam.visualization.plots import plot_confusion_matrix, plot_loss_curve

if TYPE_CHECKING:
    from toxfam.config import TrainConfig

console = Console()


class DataSelector:
    """Wraps the loader to yield only the specific input needed by the strategy."""

    def __init__(self, loader, mode):
        self.loader = loader
        self.mode = mode  # 'emb_only', 'both'

    def __iter__(self):
        for batch in self.loader:
            features, label = batch

            if isinstance(features, torch.Tensor):
                if self.mode == "emb_only":
                    yield features, label
                else:
                    raise RuntimeError(
                        f"Strategy requested '{self.mode}' but Dataset provided "
                        f"only Embeddings. Check if tax_h5_path is set in config."
                    )
            elif isinstance(features, (list, tuple)):
                emb, tax = features
                if self.mode == "emb_only":
                    yield emb, label
                else:
                    yield (emb, tax), label

    def __len__(self):
        return len(self.loader)


def run_standard_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    console.print("[bold]>>> Running Strategy: STANDARD (Embeddings Only)[/bold]")
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
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_curve.png")
    return model


def run_combined_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    console.print("[bold]>>> Running Strategy: COMBINED (Branched Architecture)[/bold]")
    model = MultiInputMLP(
        embed_dim=config.effective_embedding_dim,
        tax_dim=config.tax_dim,
        hidden_dims=config.hidden_dims,
        num_classes=num_classes,
        dropout=config.dropout,
    )
    model, hist = train_model(
        model,
        DataSelector(train_loader, "both"),
        DataSelector(val_loader, "both"),
        w_tensor,
        config,
    )
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_curve.png")
    return model


def run_binary_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    """Train a direct binary toxic/nontoxin classifier.

    The dataset must already have binary labels ("toxin"/"nontoxin").
    num_classes should be 2.
    """
    console.print("[bold]>>> Running Strategy: BINARY (Toxic vs Non-toxic)[/bold]")
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
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_curve.png")
    return model


class _MultiTaskFamilyWrapper(nn.Module):
    """Wraps MultiTaskMLP to return only family logits for evaluation."""

    def __init__(self, model: MultiTaskMLP):
        super().__init__()
        self.model = model

    def forward(self, x):
        fam_out, _ = self.model(x)
        return fam_out


class _MultiTaskBinaryWrapper(nn.Module):
    """Wraps MultiTaskMLP to return only binary logits for evaluation."""

    def __init__(self, model: MultiTaskMLP):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, bin_out = self.model(x)
        return bin_out


def run_multitask_strategy(
    train_loader,
    val_loader,
    w_tensor,
    num_classes,
    train_df,
    out_dir,
    config: TrainConfig,
):
    """Train a multitask model with shared backbone, joint family + binary heads.

    Returns the trained MultiTaskMLP (not wrapped — caller wraps as needed).
    """
    from toxfam.evaluation.metrics import to_binary_class

    console.print("[bold]>>> Running Strategy: MULTITASK (Family + Binary)[/bold]")

    model = MultiTaskMLP(
        input_dim=config.effective_embedding_dim,
        hidden_dims=config.hidden_dims,
        num_family_classes=num_classes,
        num_binary_classes=2,
        dropout=config.dropout,
    )

    device = get_device()
    model.to(device)
    w_tensor = w_tensor.to(device)

    # Build binary label mapping: family_class_index → binary {0=nontoxin, 1=toxin}
    le = train_loader.dataset.le
    family_to_binary = {}
    for idx, cls in enumerate(le.classes_):
        family_to_binary[idx] = 0 if to_binary_class(cls) == "nontoxin" else 1
    binary_mapping = torch.tensor(
        [family_to_binary[i] for i in range(len(le.classes_))], device=device
    )

    # Binary class weights from sample counts
    train_labels = train_df["Protein families"].apply(to_binary_class)
    n_tox = (train_labels == "toxin").sum()
    n_nontox = (train_labels == "nontoxin").sum()
    total = n_tox + n_nontox
    binary_weights = torch.tensor(
        [total / (2 * max(n_nontox, 1)), total / (2 * max(n_tox, 1))],
        dtype=torch.float32,
        device=device,
    )

    family_loss_fn = torch.nn.CrossEntropyLoss(weight=w_tensor)
    binary_loss_fn = torch.nn.CrossEntropyLoss(weight=binary_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    alpha = config.multitask_family_weight
    beta = config.multitask_binary_weight

    best_mcc = float("-inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    models_dir = Path(out_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model.pt"

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for features, labels in DataSelector(train_loader, "emb_only"):
            labels = labels.to(device)
            binary_labels = binary_mapping[labels]

            optimizer.zero_grad()
            outputs = _forward_model(model, features, device)
            fam_out, bin_out = outputs

            loss = alpha * family_loss_fn(fam_out, labels) + beta * binary_loss_fn(
                bin_out, binary_labels
            )
            loss.backward()
            if config.max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation using family wrapper
        family_wrapper = _MultiTaskFamilyWrapper(model)
        val_metrics, _, _, _ = evaluate_model(
            family_wrapper,
            DataSelector(val_loader, "emb_only"),
            family_loss_fn,
            device,
        )
        val_mcc = val_metrics["Validation_mcc"]
        val_loss = val_metrics["Validation_loss"]
        val_losses.append(val_loss)

        console.print(
            f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Val MCC={val_mcc:.4f}"
        )

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            console.print("[yellow]Early stopping triggered.[/yellow]")
            break

    # Load best
    if best_model_path.exists():
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

    plot_loss_curve(
        {"train_losses": train_losses, "val_losses": val_losses},
        Path(out_dir) / "plots" / "loss_curve.png",
    )

    return model


def evaluate_label_on_dataset(
    model,
    dataset_df,
    label_col,
    label_encoder,
    loss_fn,
    tag,
    out_dir,
    config: TrainConfig,
) -> dict:
    """Evaluate the model on a dataframe. Returns the metrics dict."""
    strategy = config.training_strategy

    from toxfam.training.orchestrator import _extra_dataset_kwargs

    extra_kwargs = _extra_dataset_kwargs(config)

    ds = ToxDataset(
        dataset_df,
        [str(p) for p in config.h5_paths],
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=str(config.tax_h5_path) if config.tax_h5_path else None,
        **extra_kwargs,
    )
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    if strategy == "combined":
        selector = DataSelector(loader, "both")
    else:
        selector = DataSelector(loader, "emb_only")

    device = get_device()
    model = model.to(device)

    metrics, y_true, y_pred, y_scores = evaluate_model(
        model, selector, loss_fn, device, dataset_type=tag
    )

    confidences = y_scores.max(axis=1)
    conf_df = pd.DataFrame(
        {
            "identifier": dataset_df["identifier"].reset_index(drop=True),
            "actual_label": label_encoder.inverse_transform(y_true),
            "predicted_label": label_encoder.inverse_transform(y_pred),
            "confidence": confidences,
        }
    )

    out_path = Path(out_dir)
    conf_df.to_csv(out_path / "predictions" / f"{tag}_predictions.csv", index=False)

    plot_confusion_matrix(
        y_true,
        y_pred,
        ds.le,
        out_path / "plots" / f"{tag}_confusion_matrix.png",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ds.le.classes_))),
        target_names=ds.le.classes_,
        output_dict=True,
        zero_division=0,
    )
    (out_path / "metrics" / f"{tag}_metrics.json").write_text(
        json.dumps(
            {"numeric_metrics": metrics, "classification_report": report}, indent=4
        )
    )

    plot_multiclass_roc_from_scores(
        y_true, y_scores, ds.le.classes_, out_path / "plots" / f"{tag}_roc.png"
    )

    # Log to wandb
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics)
        class_names = list(label_encoder.classes_)
        wandb.log(
            {
                f"{tag}_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names,
                )
            }
        )

    ds.close()
    return metrics
