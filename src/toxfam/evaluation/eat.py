"""EAT — embedding-based annotation transfer (the embedding-space analog of HBI).

For each query protein, find its nearest neighbour among a labelled reference
set in ProtT5 embedding space and transfer that neighbour's family label — the
k=1 EAT of Heinzinger et al., "Contrastive learning on protein embeddings
enriches the function" (NAR Genomics & Bioinformatics 4(2):lqac043, 2022).
Both ``cosine`` (default; selected on val_set — beat Euclidean on every metric)
and ``euclidean`` (the original EAT distance) are supported.

Where HBI transfers the label of the nearest *sequence* homolog (MMseqs2), EAT
transfers the label of the nearest *embedding* neighbour. A continuous P(toxic)
score is derived from the distance margin between the nearest toxic and nearest
non-toxic reference, so EAT can also be ranked on the binary toxicity task
(threshold-free ROC-AUC / PR-AUC are rank-invariant to the sigmoid transform).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from toxfam.evaluation.metrics import is_nontoxin

DEFAULT_BATCH_SIZE = 512
METRICS = ("euclidean", "cosine")


@dataclass
class EATResult:
    """Results from an EAT search."""

    predictions: pd.DataFrame  # identifier, eat_prediction, eat_confidence, p_toxic
    n_queries: int
    n_reference: int


def _load_matrix(h5_path: str | Path, identifiers: list[str]) -> torch.Tensor:
    """Stack per-protein embeddings (keyed by identifier) into one (N, D) tensor.

    Mirrors ``toxfam.model.inference._load_embeddings``; kept local so the
    evaluation layer does not import a private model-layer helper.
    """
    with h5py.File(h5_path, "r") as f:
        return torch.stack(
            [torch.tensor(f[ident][:], dtype=torch.float32) for ident in identifiers]
        )


def run_eat_search(
    query_h5: str | Path,
    ref_h5: str | Path,
    reference_df: pd.DataFrame,
    query_ids: list[str],
    *,
    id_column: str = "identifier",
    label_column: str = "Protein families",
    metric: str = "cosine",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EATResult:
    """k=1 annotation transfer from a reference set to query proteins.

    Parameters
    ----------
    query_h5, ref_h5 : HDF5 files keyed by identifier (1-D float embeddings).
        ``ref_h5`` must contain every ``reference_df[id_column]``; ``query_h5``
        every ``query_ids``.
    reference_df : labelled reference (e.g. the training split) with
        ``id_column`` + ``label_column``.
    query_ids : identifiers to score, in output order.
    metric : ``"euclidean"`` (exact L2; canonical EAT) or ``"cosine"``
        (1 − cosine similarity on L2-normalised embeddings).

    Returns
    -------
    EATResult whose ``predictions`` DataFrame has columns
    ``identifier``, ``eat_prediction`` (nearest-neighbour family label),
    ``eat_confidence`` (``1 / (1 + d_nearest)`` ∈ (0, 1]), and ``p_toxic``
    (``sigmoid(d_nearest_nontoxic - d_nearest_toxic)`` ∈ [0, 1]). Distances are
    in the chosen metric; the toxic-vs-nontoxic ranking (ROC-AUC/PR-AUC) is
    invariant to the sigmoid transform.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; use one of {METRICS}.")
    ref_ids = reference_df[id_column].tolist()
    if not ref_ids:
        raise ValueError("reference_df is empty; cannot run EAT search.")
    if not query_ids:
        raise ValueError("query_ids is empty; nothing to score.")

    ref_labels = reference_df[label_column].to_numpy()
    tox_mask = torch.tensor(
        [not is_nontoxin(lbl) for lbl in ref_labels], dtype=torch.bool
    )  # (R,)

    ref = _load_matrix(ref_h5, ref_ids)  # (R, D)
    if metric == "cosine":
        ref = F.normalize(ref, p=2, dim=1)

    preds: list[str] = []
    confidences: list[float] = []
    p_toxic: list[float] = []
    for start in range(0, len(query_ids), batch_size):
        batch = query_ids[start : start + batch_size]
        q = _load_matrix(query_h5, batch)  # (B, D)
        if metric == "cosine":
            # Cosine distance = 1 − cosine similarity ∈ [0, 2] on unit vectors.
            d = 1.0 - F.normalize(q, p=2, dim=1) @ ref.T  # (B, R)
        else:
            # Exact Euclidean: the matmul expansion (default at D>25) carries ~1e-2
            # float32 error that can re-order near-tie neighbours of different families.
            d = torch.cdist(q, ref, compute_mode="donot_use_mm_for_euclid_dist")  # (B, R)

        nn_idx = d.argmin(dim=1)  # (B,) index of nearest reference
        d_nn = d.gather(1, nn_idx.unsqueeze(1)).squeeze(1)  # (B,) nearest distance

        # Distance margin: nearest toxic vs nearest non-toxic reference.
        d_tox = d.masked_fill(~tox_mask.unsqueeze(0), float("inf")).min(dim=1).values
        d_non = d.masked_fill(tox_mask.unsqueeze(0), float("inf")).min(dim=1).values
        p_batch = torch.sigmoid(d_non - d_tox)  # high when query is closer to toxic

        preds.extend(ref_labels[nn_idx.numpy()].tolist())
        confidences.extend((1.0 / (1.0 + d_nn)).tolist())
        p_toxic.extend(p_batch.tolist())

    predictions = pd.DataFrame(
        {
            "identifier": query_ids,
            "eat_prediction": preds,
            "eat_confidence": confidences,
            "p_toxic": p_toxic,
        }
    )
    return EATResult(
        predictions=predictions, n_queries=len(query_ids), n_reference=len(ref_ids)
    )
