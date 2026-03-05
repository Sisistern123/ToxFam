"""Per-family analysis for the multitask augmented counterparts model.
Analyzes both family classification AND binary toxic/non-toxic performance."""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from pathlib import Path

MODEL_DIR = Path("model/model_output/multitask_augmented_counterparts_run")
BINARY_MODEL_DIR = Path("model/model_output/binary_augmented_counterparts_run")
TRAINING_CSV = Path("data/processed/training_data_with_counterparts.csv")
OUT_DIR = MODEL_DIR / "per_family_analysis"
OUT_DIR.mkdir(exist_ok=True)

# ---- Load data ----
train = pd.read_csv(TRAINING_CSV, usecols=["identifier", "Protein families", "Split"])
test_families = train[train["Split"] == "test"][["identifier", "Protein families"]]

# Load multitask family predictions
preds = pd.read_csv(MODEL_DIR / "predictions" / "test_calibrated_predictions.csv")
merged = preds.merge(test_families, on="identifier", how="left")
assert merged["Protein families"].notna().all()

# True binary
merged["true_binary"] = (merged["Protein families"] != "nontox").astype(int)
# Family prediction correct?
merged["family_correct"] = merged["actual_label"] == merged["predicted_label"]

# Load class indices for this model
with open(MODEL_DIR / "class_indices.json") as f:
    class_map = json.load(f)

# ---- Derive binary from family softmax (how orchestrator does it) ----
# For multitask: the family head predictions are in test_calibrated_predictions.csv
# Binary: predicted as toxic if predicted family != nontox
merged["pred_binary_from_family"] = (merged["predicted_label"] != "nontox").astype(int)

# ---- Binary head metrics (from saved metrics) ----
with open(MODEL_DIR / "metrics" / "binary_test_calibrated_binaryhead_metrics.json") as f:
    bh_metrics = json.load(f)

with open(MODEL_DIR / "metrics" / "threshold_optimization.json") as f:
    threshold_info = json.load(f)

print("=" * 80)
print("MULTITASK AUGMENTED COUNTERPARTS MODEL — TEST SET ANALYSIS")
print("=" * 80)

print(f"\nTest set: {len(merged)} samples ({merged['true_binary'].sum()} toxic, {(~merged['true_binary'].astype(bool)).sum()} non-toxic)")

# ---- Overall family classification ----
with open(MODEL_DIR / "metrics" / "test_calibrated_metrics.json") as f:
    fam_metrics = json.load(f)

print(f"\n--- Family Classification (38-class) ---")
print(f"  Accuracy:  {fam_metrics['numeric_metrics']['Test_Calibrated_Accuracy']:.4f}")
print(f"  MCC:       {fam_metrics['numeric_metrics']['Test_Calibrated_MCC']:.4f}")
print(f"  Micro MCC: {fam_metrics['numeric_metrics']['Test_Calibrated_Micro_MCC']:.4f}")

print(f"\n--- Binary Head (dedicated toxic/nontox head) ---")
print(f"  ROC-AUC:   {bh_metrics['roc_auc']:.4f}")
print(f"  PR-AUC:    {bh_metrics['pr_auc']:.4f}")
print(f"  MCC:       {bh_metrics['mcc']:.4f}")
print(f"  F1:        {bh_metrics['f1']:.4f}")

print(f"\n--- Binary Derived from Family Head (predicted != nontox → toxic) ---")
y_true = merged["true_binary"].values
y_pred_fam = merged["pred_binary_from_family"].values
print(f"  Accuracy:  {(y_true == y_pred_fam).mean():.4f}")
print(f"  F1:        {f1_score(y_true, y_pred_fam):.4f}")
print(f"  MCC:       {matthews_corrcoef(y_true, y_pred_fam):.4f}")
print(f"  Precision: {precision_score(y_true, y_pred_fam):.4f}")
print(f"  Recall:    {recall_score(y_true, y_pred_fam):.4f}")

# ---- Per-family analysis ----
families = sorted(merged["Protein families"].unique())
toxin_families = [f for f in families if f != "nontox"]

results = []
for fam in toxin_families:
    sub = merged[merged["Protein families"] == fam]
    n = len(sub)

    # Family classification accuracy for this family
    fam_correct = sub["family_correct"].sum()
    fam_acc = fam_correct / n if n > 0 else 0

    # What did the model predict for these samples?
    pred_dist = sub["predicted_label"].value_counts().to_dict()

    # Binary from family: correctly identified as toxic (predicted != nontox)
    tp_from_fam = (sub["pred_binary_from_family"] == 1).sum()
    fn_from_fam = (sub["pred_binary_from_family"] == 0).sum()
    recall_from_fam = tp_from_fam / n if n > 0 else 0

    # Most common misprediction
    wrong = sub[~sub["family_correct"]]
    top_mispredict = wrong["predicted_label"].value_counts().head(1)
    top_mis = f"{top_mispredict.index[0]} ({top_mispredict.values[0]})" if len(top_mispredict) > 0 else "-"

    results.append({
        "Family": fam,
        "N": n,
        "Family_Correct": int(fam_correct),
        "Family_Acc": fam_acc,
        "Toxic_Detected": int(tp_from_fam),
        "Missed_as_Nontox": int(fn_from_fam),
        "Binary_Recall": recall_from_fam,
        "Top_Misprediction": top_mis,
    })

df = pd.DataFrame(results).sort_values("N", ascending=False)
df.to_csv(OUT_DIR / "per_family_metrics.csv", index=False)

print(f"\n{'='*120}")
print(f"{'Family':<55} {'N':>4} {'FamAcc':>7} {'Correct':>7} {'ToxDet':>7} {'Missed':>7} {'BinRec':>7}  Top Misprediction")
print(f"{'='*120}")
for _, r in df.iterrows():
    print(f"{r['Family']:<55} {r['N']:>4} {r['Family_Acc']:>7.3f} {r['Family_Correct']:>7} {r['Toxic_Detected']:>7} {r['Missed_as_Nontox']:>7} {r['Binary_Recall']:>7.3f}  {r['Top_Misprediction']}")

# ---- Nontox analysis ----
nontox = merged[merged["Protein families"] == "nontox"]
nontox_correct = nontox["family_correct"].sum()
nontox_fp = (~nontox["family_correct"]).sum()
print(f"\n--- Nontox (N={len(nontox)}) ---")
print(f"  Correctly identified as nontox: {nontox_correct} ({nontox_correct/len(nontox)*100:.1f}%)")
print(f"  Misclassified as a toxin family: {nontox_fp} ({nontox_fp/len(nontox)*100:.1f}%)")
fp_families = nontox[~nontox["family_correct"]]["predicted_label"].value_counts()
print(f"  False positive families:")
for fam, cnt in fp_families.head(10).items():
    print(f"    {fam}: {cnt}")

# ===== FIGURE 1: Per-family classification accuracy + binary recall =====
fig, ax = plt.subplots(figsize=(16, 10))
df_plot = df[df["N"] >= 2].copy()
fam_names = df_plot["Family"].tolist()
fam_acc = df_plot["Family_Acc"].values
bin_rec = df_plot["Binary_Recall"].values
n_vals = df_plot["N"].values

x = np.arange(len(fam_names))
width = 0.35

bars1 = ax.barh(x - width/2, fam_acc, width, label="Family Classification Accuracy", color="#3498db", alpha=0.85)
bars2 = ax.barh(x + width/2, bin_rec, width, label="Binary Recall (detected as toxic)", color="#2ecc71", alpha=0.85)

for i, (fa, br, n) in enumerate(zip(fam_acc, bin_rec, n_vals)):
    ax.text(max(fa, br) + 0.02, i, f"N={n}", va="center", fontsize=8, color="gray")

ax.set_yticks(x)
ax.set_yticklabels(fam_names, fontsize=8)
ax.set_xlabel("Score", fontsize=12)
ax.set_title("Multitask Model: Per-Family Accuracy vs Binary Recall\n(Family = exact family match, Binary = detected as any toxin)", fontsize=13)
ax.set_xlim(0, 1.15)
ax.legend(loc="lower right", fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "family_acc_vs_binary_recall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR / 'family_acc_vs_binary_recall.png'}")

# ===== FIGURE 2: Confusion matrix heatmap (family level) =====
# Build per-family confusion: what gets predicted as what
all_labels = sorted(merged["actual_label"].unique())
toxin_labels = [l for l in all_labels if l != "nontox"]
# Only show families with test samples
active = [f for f in toxin_labels if (merged["actual_label"] == f).sum() > 0]
active_with_nontox = active + ["nontox"]

# Build confusion matrix subset
from sklearn.metrics import confusion_matrix as cm_fn
cm_full = cm_fn(merged["actual_label"], merged["predicted_label"], labels=active_with_nontox)

# Normalize by row (recall)
cm_norm = cm_full.astype(float)
row_sums = cm_norm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
cm_norm = cm_norm / row_sums

# Plot only the toxin families (rows), showing where they get classified
fig, ax = plt.subplots(figsize=(20, 12))
n_rows = len(active)
n_cols = len(active_with_nontox)
cm_toxin = cm_norm[:n_rows, :]

im = ax.imshow(cm_toxin, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(n_cols))
ax.set_xticklabels(active_with_nontox, rotation=90, fontsize=7)
ax.set_yticks(range(n_rows))
ax.set_yticklabels([f"{f} (N={cm_full[i].sum()})" for i, f in enumerate(active)], fontsize=7)
ax.set_xlabel("Predicted Family", fontsize=11)
ax.set_ylabel("True Family", fontsize=11)
ax.set_title("Family Classification Confusion Matrix (Row-Normalized)\nMultitask Augmented + Counterparts Model", fontsize=13)

# Annotate cells
for i in range(n_rows):
    for j in range(n_cols):
        val = cm_toxin[i, j]
        count = cm_full[i, j]
        if val > 0.01:
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}\n({count})", ha="center", va="center", fontsize=6, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, shrink=0.6, label="Row-Normalized Proportion")
plt.tight_layout()
plt.savefig(OUT_DIR / "family_confusion_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'family_confusion_heatmap.png'}")

# ===== FIGURE 3: Comparison with pure binary model =====
# Load binary model per-family metrics if available
binary_pf = BINARY_MODEL_DIR / "per_family_analysis" / "per_family_metrics.csv"
if binary_pf.exists():
    binary_df = pd.read_csv(binary_pf)
    comp = df.merge(binary_df[["Family", "Recall"]], on="Family", how="left", suffixes=("", "_binary_model"))
    comp = comp.rename(columns={"Recall": "Binary_Model_Recall"})
    comp = comp[comp["N"] >= 2].sort_values("N", ascending=False)

    fig, ax = plt.subplots(figsize=(16, 10))
    fam_names_c = comp["Family"].tolist()
    x = np.arange(len(fam_names_c))
    width = 0.25

    ax.barh(x - width, comp["Family_Acc"].values, width, label="Multitask: Family Accuracy", color="#3498db", alpha=0.85)
    ax.barh(x, comp["Binary_Recall"].values, width, label="Multitask: Binary Recall", color="#2ecc71", alpha=0.85)
    ax.barh(x + width, comp["Binary_Model_Recall"].values, width, label="Pure Binary Model: Recall", color="#e74c3c", alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(fam_names_c, fontsize=8)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Multitask vs Pure Binary Model: Per-Family Performance Comparison", fontsize=13)
    ax.set_xlim(0, 1.15)
    ax.legend(loc="lower right", fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "multitask_vs_binary_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_DIR / 'multitask_vs_binary_comparison.png'}")

# ===== FIGURE 4: Per-family mini confusion matrices (top families) =====
top_families = df_plot[df_plot["N"] >= 5]["Family"].tolist()
n_fams = len(top_families)
ncols = 4
nrows = (n_fams + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
axes_flat = axes.flatten()

for idx, fam in enumerate(top_families):
    ax = axes_flat[idx]
    sub = merged[(merged["Protein families"] == fam) | (merged["Protein families"] == "nontox")]
    y_t = (sub["Protein families"] != "nontox").astype(int).values
    y_p = (sub["predicted_label"] != "nontox").astype(int).values
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred Non-toxic", "Pred Toxic"], fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-toxic", "Toxic"], fontsize=7)

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    n_fam = (sub["Protein families"] == fam).sum()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    short = fam[:35] + "..." if len(fam) > 35 else fam
    ax.set_title(f"{short}\n(N={n_fam}, Recall={rec:.2f})", fontsize=8, fontweight="bold")

for idx in range(n_fams, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Per-Family Binary Confusion Matrices (families with ≥5 test samples)\nMultitask Augmented + Counterparts Model", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_confusion_matrices.png'}")

# ===== Summary comparison table =====
print(f"\n{'='*80}")
print("MODEL COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"{'Metric':<35} {'Pure Binary':>15} {'Multitask (BinHead)':>20} {'Multitask (FamDerived)':>22}")
print(f"{'-'*92}")

# Load pure binary metrics
with open(BINARY_MODEL_DIR / "metrics" / "binary_test_calibrated_optimized_metrics.json") as f:
    pure_bin = json.load(f)

mt_fam_derived_recall = recall_score(y_true, y_pred_fam)
mt_fam_derived_mcc = matthews_corrcoef(y_true, y_pred_fam)
mt_fam_derived_f1 = f1_score(y_true, y_pred_fam)

print(f"{'ROC-AUC':<35} {pure_bin['roc_auc']:>15.4f} {bh_metrics['roc_auc']:>20.4f} {'N/A':>22}")
print(f"{'PR-AUC':<35} {pure_bin['pr_auc']:>15.4f} {bh_metrics['pr_auc']:>20.4f} {'N/A':>22}")
print(f"{'MCC':<35} {pure_bin['mcc']:>15.4f} {bh_metrics['mcc']:>20.4f} {mt_fam_derived_mcc:>22.4f}")
print(f"{'F1':<35} {pure_bin['f1']:>15.4f} {bh_metrics['f1']:>20.4f} {mt_fam_derived_f1:>22.4f}")
print(f"{'Recall (toxic)':<35} {'':>15} {'':>20} {mt_fam_derived_recall:>22.4f}")

print(f"\nAll outputs saved to: {OUT_DIR}")
