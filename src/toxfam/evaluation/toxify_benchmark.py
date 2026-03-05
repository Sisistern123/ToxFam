"""TOXIFY reimplementation in PyTorch and benchmark on our test set.

The original TOXIFY (Cole & Bhatt 2019, PeerJ 7:e7200) uses TensorFlow 1.8,
which cannot run on macOS ARM64 (AVX instructions not supported by Rosetta 2).
We reimplement the architecture from scratch:

    Atchley factors (5-dim per AA) → GRU(270) → Dense(2)

Training data is downloaded from the original repo (4,808 venom + 32,391
non-venom proteins). The reimplementation trains in ~2 min on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from toxfam._paths import get_project_root
from toxfam.data.handcrafted_features import ATCHLEY_FACTORS
from toxfam.evaluation.metrics import calculate_binary_metrics_with_scores

# Default zero vector for unknown amino acids
_ATCHLEY_ZERO = [0.0, 0.0, 0.0, 0.0, 0.0]

MAX_SEQ_LEN = 500
GRU_HIDDEN = 270


# ── Model ────────────────────────────────────────────────────────────────────

class ToxifyGRU(nn.Module):
    """Single-layer GRU on Atchley factor encodings.

    Matches the original TOXIFY architecture: GRU(input=5, hidden=270)
    followed by a dense layer mapping to 2 classes.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = GRU_HIDDEN, num_classes: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        return self.fc(h_n.squeeze(0))


# ── Atchley encoding ────────────────────────────────────────────────────────

def encode_atchley(
    sequences: list[str],
    max_len: int = MAX_SEQ_LEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode sequences as Atchley factor tensors with padding.

    Returns
    -------
    X : float tensor (N, max_len, 5) — padded Atchley factors
    lengths : long tensor (N,) — actual sequence lengths (capped at max_len)
    """
    n = len(sequences)
    X = torch.zeros(n, max_len, 5, dtype=torch.float32)
    lengths = torch.zeros(n, dtype=torch.long)

    for i, seq in enumerate(sequences):
        seq_upper = seq.upper()[:max_len]
        seq_len = len(seq_upper)
        lengths[i] = seq_len
        for j, aa in enumerate(seq_upper):
            factors = ATCHLEY_FACTORS.get(aa, _ATCHLEY_ZERO)
            X[i, j] = torch.tensor(factors, dtype=torch.float32)

    return X, lengths


# ── Dataset ──────────────────────────────────────────────────────────────────

class _AtchleyDataset(Dataset):
    def __init__(self, X: torch.Tensor, lengths: torch.Tensor, labels: torch.Tensor):
        self.X = X
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.lengths[idx], self.labels[idx]


# ── Data download ────────────────────────────────────────────────────────────

def _parse_fasta_text(text: str) -> list[str]:
    """Parse FASTA from text content, return list of sequences."""
    sequences = []
    current = []
    for line in text.splitlines():
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    return sequences


def download_toxify_data(cache_dir: Path) -> tuple[list[str], list[str]]:
    """Download TOXIFY training data from GitHub, return (venom, non_venom) seqs.

    Caches FASTA files locally to avoid re-downloading.
    """
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/tijeco/toxify/master/sequence_data/training_data"
    files = {
        "venom": "pre.venom.fasta",
        "nonvenom": "pre.NOT.venom.fasta",
    }

    result = {}
    for key, fname in files.items():
        local = cache_dir / fname
        if not local.exists():
            url = f"{base_url}/{fname}"
            print(f"  Downloading {url}...")
            urllib.request.urlretrieve(url, local)
        result[key] = _parse_fasta_text(local.read_text())

    return result["venom"], result["nonvenom"]


# ── Training ─────────────────────────────────────────────────────────────────

def _train_toxify(
    model: ToxifyGRU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.01,
    patience: int = 5,
) -> ToxifyGRU:
    """Train the TOXIFY GRU model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, lens, labels in train_loader:
            X_batch, lens, labels = X_batch.to(device), lens.to(device), labels.to(device)
            logits = model(X_batch, lens)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, lens, labels in val_loader:
                X_batch, lens, labels = X_batch.to(device), lens.to(device), labels.to(device)
                logits = model(X_batch, lens)
                val_loss += loss_fn(logits, labels).item()
        val_loss /= len(val_loader)

        print(f"  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Benchmark ────────────────────────────────────────────────────────────────

def run_toxify_benchmark(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    output_dir: Path,
) -> dict:
    """Train TOXIFY from scratch on original data and evaluate on our test set.

    Parameters
    ----------
    test_df : DataFrame with 'Sequence' column.
    y_true : binary ground truth (1=toxic, 0=nontoxic).
    output_dir : where to save model and results.

    Returns dict with binary metrics, or empty dict on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached trained model
    model_path = output_dir / "toxify_model.pt"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if model_path.exists():
        print("Loading cached TOXIFY model...")
        model = ToxifyGRU()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
    else:
        # Download training data
        cache_dir = get_project_root() / "data" / "intermediate" / "toxify"
        print("Downloading TOXIFY training data...")
        venom_seqs, nonvenom_seqs = download_toxify_data(cache_dir)
        print(f"  Venom: {len(venom_seqs)}, Non-venom: {len(nonvenom_seqs)}")

        # Prepare training data
        all_seqs = venom_seqs + nonvenom_seqs
        all_labels = [1] * len(venom_seqs) + [0] * len(nonvenom_seqs)

        # Shuffle and split 80/20
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_seqs))
        split = int(0.8 * len(idx))
        train_idx, val_idx = idx[:split], idx[split:]

        train_seqs = [all_seqs[i] for i in train_idx]
        val_seqs = [all_seqs[i] for i in val_idx]
        train_labels = torch.tensor([all_labels[i] for i in train_idx], dtype=torch.long)
        val_labels = torch.tensor([all_labels[i] for i in val_idx], dtype=torch.long)

        print(f"  Train: {len(train_seqs)}, Val: {len(val_seqs)}")

        # Encode
        print("  Encoding Atchley factors...")
        X_train, lens_train = encode_atchley(train_seqs)
        X_val, lens_val = encode_atchley(val_seqs)

        train_ds = _AtchleyDataset(X_train, lens_train, train_labels)
        val_ds = _AtchleyDataset(X_val, lens_val, val_labels)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        # Train
        print(f"  Training TOXIFY GRU on {device}...")
        model = ToxifyGRU().to(device)
        model = _train_toxify(model, train_loader, val_loader, device)

        # Save
        torch.save(model.state_dict(), model_path)
        print(f"  Saved trained model to {model_path}")

    # ── Predict on our test set ──
    test_seqs = test_df["Sequence"].tolist()
    print(f"  Predicting on {len(test_seqs)} test sequences...")

    X_test, lens_test = encode_atchley(test_seqs)
    test_ds = _AtchleyDataset(X_test, lens_test, torch.zeros(len(test_seqs), dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for X_batch, lens, _ in test_loader:
            X_batch, lens = X_batch.to(device), lens.to(device)
            logits = model(X_batch, lens)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 1].cpu().numpy())

    p_toxic = np.concatenate(all_probs).astype(np.float64)
    print(f"  Predictions: {(p_toxic >= 0.5).sum()} toxic, {(p_toxic < 0.5).sum()} non-toxic")

    metrics = calculate_binary_metrics_with_scores(y_true, p_toxic)

    # Save metrics
    serializable = {
        k: v for k, v in metrics.items()
        if k not in ("fpr", "tpr", "precision_curve", "recall_curve",
                      "roc_thresholds", "pr_thresholds")
    }
    serializable["model"] = "TOXIFY reimpl. (Atchley GRU)"
    serializable["n_predictions"] = len(test_seqs)

    (output_dir / "toxify_metrics.json").write_text(
        json.dumps(serializable, indent=4)
    )

    print(
        f"  TOXIFY (reimpl.): "
        f"ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}, "
        f"MCC={metrics['mcc']:.4f}"
    )

    return metrics
