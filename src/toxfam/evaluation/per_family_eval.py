"""Per-family evaluation: compute metrics broken down by protein family.

Allows comparing methods on contested families (Phospholipase, Kunitz, etc.)
and identifying which families drive overall performance differences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


def evaluate_per_family(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    output_dir: Path | None = None,
    min_family_size: int = 5,
) -> pd.DataFrame:
    """Compute per-family binary metrics for each prediction method.

    Parameters
    ----------
    test_df : DataFrame with 'Protein families' column.
    y_true : binary ground truth (1=toxic, 0=nontoxic).
    predictions : {method_name: p_toxic array} for each method.
    threshold : classification threshold for p_toxic.
    output_dir : if provided, save results CSV and JSON here.
    min_family_size : minimum test samples for per-family reporting.

    Returns DataFrame with per-family metrics for each method.
    """
    families = test_df["Protein families"].values
    unique_families = sorted(set(families))

    rows = []
    for fam in unique_families:
        mask = families == fam
        n_total = int(mask.sum())
        if n_total < min_family_size:
            continue

        y_fam = y_true[mask]
        n_toxic = int(y_fam.sum())
        n_nontox = n_total - n_toxic

        row = {
            "family": fam,
            "n_total": n_total,
            "n_toxic": n_toxic,
            "n_nontox": n_nontox,
        }

        for method_name, p_toxic in predictions.items():
            p_fam = p_toxic[mask]
            y_pred = (p_fam >= threshold).astype(int)

            # Only compute MCC if both classes are present
            if n_toxic > 0 and n_nontox > 0:
                mcc = matthews_corrcoef(y_fam, y_pred)
            elif n_toxic == 0:
                # All nontox: MCC=1 if all predicted nontox, else penalize
                mcc = 1.0 if y_pred.sum() == 0 else -1.0
            else:
                # All toxic: MCC=1 if all predicted toxic, else penalize
                mcc = 1.0 if y_pred.sum() == n_total else -1.0

            acc = accuracy_score(y_fam, y_pred)
            f1 = f1_score(y_fam, y_pred, zero_division=0)
            fp = int(((y_pred == 1) & (y_fam == 0)).sum())
            fn = int(((y_pred == 0) & (y_fam == 1)).sum())

            row[f"{method_name}_acc"] = round(acc, 4)
            row[f"{method_name}_f1"] = round(f1, 4)
            row[f"{method_name}_mcc"] = round(mcc, 4)
            row[f"{method_name}_fp"] = fp
            row[f"{method_name}_fn"] = fn

        rows.append(row)

    result = pd.DataFrame(rows)

    # Sort by family size (largest first)
    result = result.sort_values("n_total", ascending=False).reset_index(drop=True)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "per_family_metrics.csv", index=False)

        # Also save summary for top contested families
        summary = result[result["n_toxic"] > 0].head(20)
        print("\nPer-family metrics (top 20 families with toxic members):")
        for _, row in summary.iterrows():
            fam = row["family"]
            n = row["n_total"]
            line = f"  {fam:<40} (n={n:>4}, tox={row['n_toxic']:>3})"
            for method_name in predictions:
                mcc = row.get(f"{method_name}_mcc", "—")
                line += f"  {method_name}={mcc:.3f}" if isinstance(mcc, float) else f"  {method_name}={mcc}"
            print(line)

    return result
