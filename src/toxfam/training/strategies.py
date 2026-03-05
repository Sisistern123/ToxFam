from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from toxfam.data.dataset import ToxDataset
from toxfam.device import get_device
from toxfam.model.architectures import ModularMLP, MultiInputMLP, MultiTaskMLP
from toxfam.training.trainer import _build_loss_fn, train_model, evaluate_model
from toxfam.visualization.analysis import plot_multiclass_roc_from_scores
from toxfam.visualization.plots import plot_confusion_matrix, plot_loss_curve

if TYPE_CHECKING:
    from toxfam.config import TrainConfig


class _MultiTaskFamilyWrapper(torch.nn.Module):
    """Wraps a MultiTaskMLP so forward() returns only family logits.

    This is needed for downstream evaluation and calibration, which expect
    a model that returns a single logits tensor.
    """

    def __init__(self, model: MultiTaskMLP):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_both=False)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


class _MultiTaskBinaryWrapper(torch.nn.Module):
    """Wraps a MultiTaskMLP to return only binary logits."""

    def __init__(self, model: MultiTaskMLP):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_both=True)[1]

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


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
    print(">>> Running Strategy: STANDARD (Embeddings Only)")
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
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_standard.png")
    return model


def run_binary_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    """Binary toxic/non-toxic strategy — same architecture as standard but 2 classes."""
    print(">>> Running Strategy: BINARY (Toxic vs Non-toxic)")
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
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_binary.png")
    return model


def run_combined_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    print(">>> Running Strategy: COMBINED (Branched Architecture)")
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
    plot_loss_curve(hist, Path(out_dir) / "plots" / "loss_combined.png")
    return model


def run_multitask_strategy(
    train_loader, val_loader, w_tensor, num_classes, out_dir, config: TrainConfig
):
    """Multi-task strategy: joint family + binary classification.

    Uses MultiTaskMLP with shared backbone. Loss = alpha*family + beta*binary.
    Returns the trained model (which produces family logits by default when
    return_both=False, but the binary head is also available).
    """
    import os

    import torch.optim as optim

    from toxfam.evaluation.metrics import to_binary_class

    try:
        import wandb as _wandb
    except ModuleNotFoundError:
        _wandb = None

    print(">>> Running Strategy: MULTITASK (Family + Binary)")
    device = get_device()

    model = MultiTaskMLP(
        input_dim=config.effective_embedding_dim,
        hidden_dims=config.hidden_dims,
        num_family_classes=num_classes,
        dropout=config.dropout,
    )
    model.to(device)

    alpha = config.multitask_family_weight
    beta = config.multitask_binary_weight
    print(f"  Family weight (alpha): {alpha}, Binary weight (beta): {beta}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    w_tensor_dev = w_tensor.to(device)
    family_loss_fn = _build_loss_fn(config, w_tensor_dev)

    # We need the label encoder from the dataset to map family labels to binary
    # Access via the loader's dataset
    ds = train_loader.dataset
    classes = list(ds.le.classes_)
    # Build a mapping: family_idx -> binary_idx (0=nontoxic, 1=toxic)
    family_to_binary = torch.tensor(
        [0 if to_binary_class(cls) == "nontoxin" else 1 for cls in classes],
        dtype=torch.long,
        device=device,
    )

    # Compute binary class weights from training data
    encoded_col = ds.label_col + "_encoded"
    binary_counts = torch.zeros(2)
    for enc_label, count in ds.df[encoded_col].value_counts().items():
        bin_idx = family_to_binary[enc_label].item()
        binary_counts[bin_idx] += count
    total = binary_counts.sum()
    binary_weights = total / (2.0 * binary_counts)
    binary_weights = binary_weights / binary_weights.max()
    binary_weights = binary_weights.to(device)

    binary_loss_fn = _build_loss_fn(
        config, binary_weights,
    )

    train_sel = DataSelector(train_loader, "emb_only")
    val_sel = DataSelector(val_loader, "emb_only")

    best_score = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0.0

        for features, labels in train_sel:
            labels = labels.to(device)
            if isinstance(features, (tuple, list)):
                features = [f.to(device) for f in features]
                family_logits, binary_logits = model(*features)
            else:
                family_logits, binary_logits = model(features.to(device))

            binary_labels = family_to_binary[labels]

            loss_family = family_loss_fn(family_logits, labels)
            loss_binary = binary_loss_fn(binary_logits, binary_labels)
            loss = alpha * loss_family + beta * loss_binary

            if torch.isnan(loss):
                print("Stopping: Loss became NaN.", flush=True)
                return model

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_sel)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for features, labels in val_sel:
                labels = labels.to(device)
                if isinstance(features, (tuple, list)):
                    features = [f.to(device) for f in features]
                    fam_out, bin_out = model(*features)
                else:
                    fam_out, bin_out = model(features.to(device))

                binary_labels = family_to_binary[labels]
                val_total += (
                    alpha * family_loss_fn(fam_out, labels)
                    + beta * binary_loss_fn(bin_out, binary_labels)
                ).item()

        val_loss = val_total / len(val_sel)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}",
            flush=True,
        )

        if _wandb is not None and _wandb.run is not None:
            _wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_score:
            best_score = val_loss
            epochs_no_improve = 0
            os.makedirs(str(config.output_dir), exist_ok=True)
            best_path = os.path.join(str(config.output_dir), "best_model.pt")
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print("Early stopping triggered.", flush=True)
            break

    best_path = os.path.join(str(config.output_dir), "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )

    plot_loss_curve(
        {"train_losses": train_losses, "val_losses": val_losses},
        Path(out_dir) / "plots" / "loss_multitask.png",
    )

    # Wrap the model so forward() returns only family logits for downstream evaluation
    return _MultiTaskFamilyWrapper(model)


def evaluate_label_on_dataset(
    model,
    dataset_df,
    label_col,
    label_encoder,
    loss_fn,
    tag,
    out_dir,
    config: TrainConfig,
):
    """Evaluate the model on a dataframe, using the correct DataSelector per strategy."""
    strategy = config.training_strategy

    extra_kw: dict = {}
    if config.hbi_h5_path:
        extra_kw["hbi_h5_path"] = str(config.hbi_h5_path)
    if config.handcrafted_h5_path:
        extra_kw["handcrafted_h5_path"] = str(config.handcrafted_h5_path)
    if config.include_length:
        extra_kw["include_length"] = True
    if config.include_venom_indicator:
        extra_kw["include_venom_indicator"] = True

    ds = ToxDataset(
        dataset_df,
        [str(p) for p in config.h5_paths],
        label_encoder=label_encoder,
        is_train=False,
        label_col=label_col,
        tax_h5_path=str(config.tax_h5_path) if config.tax_h5_path else None,
        cpp_h5_path=str(config.cpp_h5_path) if config.cpp_h5_path else None,
        **extra_kw,
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
        out_path / "predictions" / f"{tag.lower()}_predictions.csv", index=False
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        ds.le,
        out_path / "plots" / f"{tag.lower()}_confusion_matrix.png",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(ds.le.classes_))),
        target_names=ds.le.classes_,
        output_dict=True,
        zero_division=0,
    )
    (out_path / "metrics" / f"{tag.lower()}_metrics.json").write_text(
        json.dumps(
            {"numeric_metrics": metrics, "classification_report": report}, indent=4
        )
    )

    plot_multiclass_roc_from_scores(
        y_true, y_scores, ds.le.classes_, out_path / "plots" / f"{tag.lower()}_roc.png"
    )
    ds.close()
