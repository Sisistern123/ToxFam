from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import wandb

if TYPE_CHECKING:
    from toxfam.config import TrainConfig


def _forward_model(model, features, device):
    """Handle single-input (Tensor) or multi-input ((emb, tax)) forwarding."""
    if isinstance(features, (tuple, list)):
        features = [f.to(device) for f in features]
        return model(*features)
    else:
        return model(features.to(device))


def evaluate_model(model, data_loader, loss_fn, device, dataset_type="Validation"):
    model.eval()
    all_labels, all_preds, all_scores = [], [], []
    total_loss = 0.0
    n_classes = None

    with torch.no_grad():
        for features, labels in data_loader:
            labels = labels.to(device)
            outputs = _forward_model(model, features, device)
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

    metrics = {
        f"{dataset_type}_Accuracy": accuracy_score(all_labels, all_preds),
        f"{dataset_type}_MCC": matthews_corrcoef(all_labels, all_preds),
        f"{dataset_type}_Avg_Loss": avg_loss,
    }

    y_true_bin = label_binarize(all_labels, classes=list(range(n_classes)))
    y_pred_bin = label_binarize(all_preds, classes=list(range(n_classes)))
    metrics[f"{dataset_type}_Micro_MCC"] = matthews_corrcoef(
        y_true_bin.ravel(), y_pred_bin.ravel()
    )

    return metrics, all_labels, all_preds, all_scores


def get_class_weights(train_dataset):
    encoded_col = train_dataset.label_col + "_encoded"
    class_counts = Counter(train_dataset.df[encoded_col])
    num_classes = train_dataset.num_classes

    encoded_to_label = {
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


def train_model(model, train_loader, val_loader, weights_tensor, config: TrainConfig):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using Device: CUDA", flush=True)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Device: MPS (Apple Silicon)", flush=True)
    else:
        device = torch.device("cpu")
        print("Using Device: CPU", flush=True)

    model.to(device)

    best_score = float("inf")

    weights_tensor = weights_tensor.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("Loss Function: Cross Entropy", flush=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    epochs_no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = _forward_model(model, features, device)

            loss = loss_fn(outputs, labels)

            if torch.isnan(loss):
                print("Stopping: Loss became NaN.", flush=True)
                return model, {"train_losses": train_losses, "val_losses": val_losses}

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        val_metrics, _, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        val_loss = val_metrics["Validation_Avg_Loss"]
        val_mcc = val_metrics["Validation_MCC"]
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}",
            flush=True,
        )

        # Log epoch metrics to wandb if a run is active.
        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_mcc": val_mcc,
                }
            )

        improvement = False
        if val_loss < best_score:
            best_score = val_loss
            improvement = True

        if improvement:
            epochs_no_improve = 0
            output_dir = str(config.output_dir)
            os.makedirs(output_dir, exist_ok=True)
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)

            # Log best model checkpoint as a wandb artifact for model tracking.
            if wandb.run is not None:
                artifact = wandb.Artifact(
                    name="toxfam-best-model",
                    type="model",
                    metadata={
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_mcc": val_mcc,
                    },
                )
                artifact.add_file(best_model_path)
                wandb.log_artifact(artifact)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.early_stopping_patience:
            print("Early stopping triggered. Loss did not improve)", flush=True)
            break

    best_model_path = os.path.join(str(config.output_dir), "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    history = {"train_losses": train_losses, "val_losses": val_losses}
    return model, history
