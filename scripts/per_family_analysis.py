"""Per-family binary classification analysis for the best model."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from pathlib import Path

MODEL_DIR = Path("model/model_output/binary_augmented_counterparts_run")
TRAINING_CSV = Path("data/processed/training_data_with_counterparts.csv")
OUT_DIR = MODEL_DIR / "per_family_analysis"
OUT_DIR.mkdir(exist_ok=True)

# Load threshold
with open(MODEL_DIR / "metrics" / "threshold_optimization.json") as f:
    opt = json.load(f)
threshold = opt["optimal_threshold"]
print(f"Using optimized threshold: {threshold:.4f}")

# Load predictions (calibrated)
preds = pd.read_csv(MODEL_DIR / "predictions" / "test_calibrated_predictions.csv")
print(f"Predictions loaded: {len(preds)} samples")

# Load training data to get family info
train = pd.read_csv(TRAINING_CSV, usecols=["identifier", "Protein families", "Split"])
test_families = train[train["Split"] == "test"][["identifier", "Protein families"]]

# Merge
merged = preds.merge(test_families, on="identifier", how="left")
assert merged["Protein families"].notna().all(), "Some predictions have no family info"

# Binary ground truth: nontox → 0, all toxin families → 1
merged["true_binary"] = (merged["Protein families"] != "nontox").astype(int)

# Predicted binary using optimized threshold
merged["pred_binary_opt"] = (merged["confidence"].values > threshold).astype(int)
# For the "toxic" class, confidence is P(toxic). For "nontoxic" predicted, confidence is P(nontoxic).
# Let's check: if predicted_label == "toxic", confidence = P(toxic), else confidence = P(nontoxic)
# So actual P(toxic) = confidence when predicted_label=="toxic", else 1-confidence
merged["p_toxic"] = np.where(
    merged["predicted_label"] == "toxic",
    merged["confidence"],
    1 - merged["confidence"],
)
merged["pred_binary_opt"] = (merged["p_toxic"] > threshold).astype(int)
merged["pred_binary_default"] = (merged["p_toxic"] > 0.5).astype(int)

# ----- Overall metrics -----
print("\n=== Overall Test Metrics (optimized threshold) ===")
y_true = merged["true_binary"]
y_pred = merged["pred_binary_opt"]
print(f"  Accuracy:  {(y_true == y_pred).mean():.4f}")
print(f"  F1:        {f1_score(y_true, y_pred):.4f}")
print(f"  MCC:       {matthews_corrcoef(y_true, y_pred):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_true, y_pred):.4f}")

# ----- Per-family analysis -----
families = sorted(merged["Protein families"].unique())
toxin_families = [f for f in families if f != "nontox"]

results = []
for fam in toxin_families:
    # Subset: this family + all nontox
    mask = (merged["Protein families"] == fam) | (merged["Protein families"] == "nontox")
    sub = merged[mask]

    y_t = sub["true_binary"].values
    y_p_opt = sub["pred_binary_opt"].values
    y_p_def = sub["pred_binary_default"].values

    n_toxic = (y_t == 1).sum()
    n_nontox = (y_t == 0).sum()

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm_opt = confusion_matrix(y_t, y_p_opt, labels=[0, 1])
    tn, fp, fn, tp = cm_opt.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    results.append({
        "Family": fam,
        "N_toxic": int(n_toxic),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Recall": recall,
        "Precision": precision,
        "F1": f1,
        "Avg_P_toxic": float(sub[sub["true_binary"] == 1]["p_toxic"].mean()),
    })

df = pd.DataFrame(results).sort_values("N_toxic", ascending=False)
df.to_csv(OUT_DIR / "per_family_metrics.csv", index=False)

# Print table
print("\n=== Per-Family Binary Classification (Optimized Threshold) ===")
print(f"{'Family':<60} {'N':>4} {'TP':>4} {'FN':>4} {'Recall':>7} {'Prec':>7} {'F1':>7} {'Avg P(toxic)':>12}")
print("-" * 115)
for _, r in df.iterrows():
    print(f"{r['Family']:<60} {r['N_toxic']:>4} {r['TP']:>4} {r['FN']:>4} {r['Recall']:>7.3f} {r['Precision']:>7.3f} {r['F1']:>7.3f} {r['Avg_P_toxic']:>12.4f}")

# ---- MISSED TOXINS (False Negatives) ----
print("\n=== Missed Toxins (False Negatives) by Family ===")
fn_mask = (merged["true_binary"] == 1) & (merged["pred_binary_opt"] == 0)
fn_df = merged[fn_mask].copy()
print(f"Total missed toxins: {len(fn_df)}")
print(fn_df.groupby("Protein families").size().sort_values(ascending=False).to_string())

# ===== FIGURE 1: Per-family recall heatmap =====
fig, ax = plt.subplots(figsize=(14, 10))
df_plot = df.copy()
# Create matrix: rows = families, columns = [TP, FN]
fam_names = df_plot["Family"].tolist()
tp_vals = df_plot["TP"].values
fn_vals = df_plot["FN"].values
recall_vals = df_plot["Recall"].values

# Horizontal bar chart of recall
colors = ["#2ecc71" if r >= 0.9 else "#f39c12" if r >= 0.5 else "#e74c3c" for r in recall_vals]
bars = ax.barh(range(len(fam_names)), recall_vals, color=colors, edgecolor="white", height=0.7)

# Add count annotations
for i, (tp, fn, rec) in enumerate(zip(tp_vals, fn_vals, recall_vals)):
    total = tp + fn
    ax.text(rec + 0.01, i, f"  {tp}/{total} detected", va="center", fontsize=9, fontweight="bold")

ax.set_yticks(range(len(fam_names)))
ax.set_yticklabels(fam_names, fontsize=9)
ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax.set_title("Per-Family Toxin Detection Recall\n(Best Model: Binary Augmented + Counterparts, Optimized Threshold)", fontsize=13)
ax.set_xlim(0, 1.25)
ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.3)
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#2ecc71", label="Recall ≥ 0.90"),
    Patch(facecolor="#f39c12", label="Recall 0.50–0.89"),
    Patch(facecolor="#e74c3c", label="Recall < 0.50"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_recall_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR / 'per_family_recall_bar.png'}")

# ===== FIGURE 2: Confusion matrix heatmap (per-family TP/FN stacked) =====
fig, ax = plt.subplots(figsize=(14, 10))
# Stacked bar: TP (green) and FN (red)
ax.barh(range(len(fam_names)), tp_vals, color="#2ecc71", label="TP (correctly detected)", height=0.7)
ax.barh(range(len(fam_names)), fn_vals, left=tp_vals, color="#e74c3c", label="FN (missed)", height=0.7)
for i, (tp, fn) in enumerate(zip(tp_vals, fn_vals)):
    if tp > 0:
        ax.text(tp / 2, i, str(tp), ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    if fn > 0:
        ax.text(tp + fn / 2, i, str(fn), ha="center", va="center", fontsize=9, fontweight="bold", color="white")

ax.set_yticks(range(len(fam_names)))
ax.set_yticklabels(fam_names, fontsize=9)
ax.set_xlabel("Number of Test Samples", fontsize=12)
ax.set_title("Per-Family: True Positives vs False Negatives\n(Best Model: Binary Augmented + Counterparts, Optimized Threshold)", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_tp_fn_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_tp_fn_bar.png'}")

# ===== FIGURE 3: Full confusion matrix heatmap grid =====
# Build a matrix: rows = families (sorted by size), cols = [TP, FP, FN, TN]
# But since TN is always ~9010, normalize to make it interpretable.
# Better: show per-family mini confusion matrices as a heatmap of recall + FP rate

fig, axes = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1, 1]})

# Left panel: Recall (TP / (TP+FN))
ax1 = axes[0]
# Heatmap data: each row is a family, cell value is recall
recall_matrix = np.array(recall_vals).reshape(-1, 1)
im1 = ax1.imshow(recall_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax1.set_yticks(range(len(fam_names)))
ax1.set_yticklabels(fam_names, fontsize=8)
ax1.set_xticks([0])
ax1.set_xticklabels(["Recall"], fontsize=11)
ax1.set_title("Recall\n(TP / [TP+FN])", fontsize=12)
for i in range(len(fam_names)):
    ax1.text(0, i, f"{recall_vals[i]:.2f}", ha="center", va="center", fontsize=9,
             fontweight="bold", color="black" if recall_vals[i] > 0.4 else "white")
plt.colorbar(im1, ax=ax1, shrink=0.6)

# Right panel: Average P(toxic) for toxic samples
avg_p = df_plot["Avg_P_toxic"].values
avg_matrix = np.array(avg_p).reshape(-1, 1)
im2 = axes[1].imshow(avg_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
axes[1].set_yticks(range(len(fam_names)))
axes[1].set_yticklabels(fam_names, fontsize=8)
axes[1].set_xticks([0])
axes[1].set_xticklabels(["Avg P(toxic)"], fontsize=11)
axes[1].set_title("Model Confidence\n(Avg P(toxic) for toxic samples)", fontsize=12)
for i in range(len(fam_names)):
    axes[1].text(0, i, f"{avg_p[i]:.3f}", ha="center", va="center", fontsize=9,
                 fontweight="bold", color="black" if avg_p[i] > 0.4 else "white")
plt.colorbar(im2, ax=axes[1], shrink=0.6)

fig.suptitle("Per-Family Toxin Classification Performance\n(Best Model: Binary Augmented + Counterparts)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_heatmap.png'}")

# ===== FIGURE 4: Full 2x2 confusion matrices for top families =====
top_families = df_plot[df_plot["N_toxic"] >= 5]["Family"].tolist()
n_fams = len(top_families)
ncols = 4
nrows = (n_fams + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
axes_flat = axes.flatten() if n_fams > ncols else (axes if n_fams > 1 else [axes])

for idx, fam in enumerate(top_families):
    ax = axes_flat[idx]
    row = df_plot[df_plot["Family"] == fam].iloc[0]
    cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]]).astype(int)

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Non-toxic", "Pred Toxic"], fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-toxic", "Toxic"], fontsize=7)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    short_name = fam[:35] + "..." if len(fam) > 35 else fam
    ax.set_title(f"{short_name}\n(N={int(row['N_toxic'])}, Recall={row['Recall']:.2f})", fontsize=8, fontweight="bold")

# Hide unused subplots
for idx in range(n_fams, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Per-Family Confusion Matrices (families with ≥5 test samples)\n(Best Model: Binary Augmented + Counterparts, Optimized Threshold)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_confusion_matrices.png'}")

# ===== FIGURE 5: P(toxic) distribution per family (violin/box) =====
fig, ax = plt.subplots(figsize=(14, 10))
toxin_data = merged[merged["true_binary"] == 1].copy()

# Order by median P(toxic)
fam_order = toxin_data.groupby("Protein families")["p_toxic"].median().sort_values(ascending=False).index.tolist()

positions = []
labels = []
data_list = []
for i, fam in enumerate(fam_order):
    vals = toxin_data[toxin_data["Protein families"] == fam]["p_toxic"].values
    data_list.append(vals)
    positions.append(i)
    labels.append(fam)

bp = ax.boxplot(data_list, positions=positions, vert=False, widths=0.6,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.5))
for patch in bp["boxes"]:
    patch.set_facecolor("#3498db")
    patch.set_alpha(0.7)

ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold:.3f})")
ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, label="Default (0.5)")

ax.set_yticks(positions)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("P(toxic)", fontsize=12)
ax.set_title("Distribution of P(toxic) per Toxin Family\n(Test Set, True Positives + False Negatives)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_ptoxic_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_ptoxic_distribution.png'}")

print(f"\nAll outputs saved to: {OUT_DIR}")
