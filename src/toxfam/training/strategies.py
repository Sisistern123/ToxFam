from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from rich.console import Console
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import wandb

from toxfam.data.dataset import ToxDataset
from toxfam.model.architectures import ModularMLP, MultiInputMLP
from toxfam.training.trainer import evaluate_model, get_device, train_model
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
        input_dim=config.embedding_dim,
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
        embed_dim=config.embedding_dim,
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

    ds = ToxDataset(
        dataset_df,
        [str(p) for p in config.h5_paths],
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=str(config.tax_h5_path) if config.tax_h5_path else None,
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
    conf_df.to_csv(
        out_path / "predictions" / f"{tag}_predictions.csv", index=False
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        ds.le,
        out_path / "plots" / f"{tag}_confusion_matrix.png",
    )

    report = classification_report(
        y_true,
        y_pred,
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
    if wandb.run is not None:
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
