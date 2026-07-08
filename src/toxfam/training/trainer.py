from __future__ import annotations

import math
import random
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize

from toxfam.device import get_device
from toxfam.model.forward import forward_model

try:
    import wandb
except ImportError:
    wandb = None

if TYPE_CHECKING:
    from toxfam.config import TrainConfig
    from toxfam.data.dataset import ToxDataset

console = Console()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility across all frameworks."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    data_loader,
    loss_fn: nn.Module,
    device: torch.device | str,
    dataset_type: str = "Validation",
) -> tuple[dict[str, float], list, list, np.ndarray]:
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    total_loss = 0.0
    n_classes = None

    with torch.no_grad():
        for features, labels in data_loader:
            labels = labels.to(device)
            outputs = forward_model(model, features, device)
            if n_classes is None:
                n_classes = outputs.size(1)
            probs = F.softmax(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = probs.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_scores.append(probs.cpu().numpy())

    all_scores = np.concatenate(all_scores, axis=0)
    avg_loss = total_loss / len(data_loader)

    # Standard multi-class MCC
    mcc = matthews_corrcoef(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    # Macro MCC: average of per-class one-vs-rest MCCs
    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    y_pred_bin = label_binarize(all_preds, classes=list(range(n_classes)))
    # sklearn returns (N, 1) for binary case — expand to (N, 2)
    if n_classes == 2 and y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        y_pred_bin = np.hstack([1 - y_pred_bin, y_pred_bin])
    per_class_mcc = []
    for c in range(n_classes):
        mcc_c = matthews_corrcoef(y_true_bin[:, c], y_pred_bin[:, c])
        per_class_mcc.append(mcc_c)
    macro_mcc = float(np.mean(per_class_mcc))

    # Micro MCC: MCC on flattened one-hot arrays
    micro_mcc = float(matthews_corrcoef(y_true_bin.ravel(), y_pred_bin.ravel()))

    metrics = {
        f"{dataset_type}_accuracy": accuracy,
        f"{dataset_type}_mcc": mcc,
        f"{dataset_type}_macro_mcc": macro_mcc,
        f"{dataset_type}_micro_mcc": micro_mcc,
        f"{dataset_type}_loss": avg_loss,
    }

    return metrics, all_labels, all_preds, all_scores


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------


def get_class_weights(
    train_dataset: ToxDataset,
) -> tuple[dict[str, float], torch.Tensor, dict[int, str]]:
    encoded_col = train_dataset.label_col + "_encoded"
    class_counts = Counter(train_dataset.df[encoded_col])
    num_classes = train_dataset.num_classes

    encoded_to_label: dict[int, str] = {
        enc: train_dataset.le.inverse_transform([enc])[0] for enc in range(num_classes)
    }

    total_samples = sum(class_counts.values())
    weights_dict = {
        encoded_to_label[c]: total_samples / class_counts[c] for c in range(num_classes)
    }

    max_weight = max(weights_dict.values())
    weights_dict = {label: w / max_weight for label, w in weights_dict.items()}

    weights_tensor = torch.tensor(
        [weights_dict[encoded_to_label[i]] for i in range(num_classes)],
        dtype=torch.float32,
    )

    return weights_dict, weights_tensor, encoded_to_label


# ---------------------------------------------------------------------------
# LR Scheduler helpers
# ---------------------------------------------------------------------------


class _LinearWarmupCosineScheduler(optim.lr_scheduler.LRScheduler):
    """Linear warmup for `warmup_epochs`, then cosine annealing to 0."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            return [
                base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(model, train_loader, val_loader, weights_tensor, config: TrainConfig):
    device = get_device()
    console.print(f"Using device: [bold]{device}[/bold]")

    set_seed(config.seed)

    model.to(device)
    weights_tensor = weights_tensor.to(device)

    # Optimizer
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # LR Scheduler
    scheduler = None
    if config.lr_scheduler == "cosine":
        scheduler = _LinearWarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            total_epochs=config.num_epochs,
        )

    # Loss function
    if config.use_focal_loss:
        loss_fn = FocalLoss(
            gamma=config.focal_loss_gamma,
            weight=weights_tensor,
            label_smoothing=config.label_smoothing,
        )
        console.print(
            f"Loss function: [bold]Focal Loss[/bold] (gamma={config.focal_loss_gamma})"
        )
    else:
        loss_fn = nn.CrossEntropyLoss(
            weight=weights_tensor, label_smoothing=config.label_smoothing
        )
        console.print("Loss function: [bold]CrossEntropyLoss[/bold]")

    # Early stopping setup
    use_mcc = config.early_stopping_metric == "mcc"
    best_score = float("-inf") if use_mcc else float("inf")

    epochs_no_improve = 0
    train_losses, val_losses = [], []

    models_dir = config.output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = models_dir / "best_model.pt"

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = forward_model(model, features, device)

            loss = loss_fn(outputs, labels)

            if torch.isnan(loss):
                console.print("[bold red]Stopping: Loss became NaN.[/bold red]")
                return model, {"train_losses": train_losses, "val_losses": val_losses}

            loss.backward()

            if config.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        val_metrics, _, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        val_loss = val_metrics["Validation_loss"]
        val_mcc = val_metrics["Validation_mcc"]
        val_macro_mcc = val_metrics["Validation_macro_mcc"]
        val_accuracy = val_metrics["Validation_accuracy"]
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        console.print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}, LR: {current_lr:.2e}"
        )

        if wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mcc": val_mcc,
                    "val_macro_mcc": val_macro_mcc,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr,
                }
            )

        # Early stopping check
        if use_mcc:
            improved = val_mcc > best_score
        else:
            improved = val_loss < best_score

        if improved:
            best_score = val_mcc if use_mcc else val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)

            if wandb is not None and wandb.run is not None:
                artifact = wandb.Artifact(
                    name="toxfam-best-model",
                    type="model",
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_mcc": val_mcc,
                    },
                )
                artifact.add_file(str(best_model_path))
                wandb.log_artifact(artifact)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            metric_name = "MCC" if use_mcc else "loss"
            console.print(
                f"[bold yellow]Early stopping triggered. "
                f"Val {metric_name} did not improve for "
                f"{config.early_stopping_patience} epochs.[/bold yellow]"
            )
            break

    if best_model_path.exists():
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

    history = {"train_losses": train_losses, "val_losses": val_losses}
    return model, history
