"""Full comparison: Standard (main branch baseline) vs Multitask Augmented (Exploration branch).
Per-family metrics, binary metrics, confusion matrices, and heatmaps."""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score
from pathlib import Path

STANDARD_DIR = Path("model/model_output/standard_run")
MULTITASK_DIR = Path("model/model_output/multitask_augmented_counterparts_run")
TRAINING_CSV = Path("data/processed/training_data.csv")
OUT_DIR = Path("model/model_output/comparison_standard_vs_multitask")
OUT_DIR.mkdir(exist_ok=True)

# ---- Load training data ----
train = pd.read_csv(TRAINING_CSV, usecols=["identifier", "Protein families", "Split"])
test_families = train[train["Split"] == "test"][["identifier", "Protein families"]]
test_families["true_binary"] = (test_families["Protein families"] != "nontox").astype(int)

print(f"Test set: {len(test_families)} samples")
print(f"  Toxic: {test_families['true_binary'].sum()}, Non-toxic: {(~test_families['true_binary'].astype(bool)).sum()}")
print(f"  Families: {test_families['Protein families'].nunique()}")

# ---- Load metrics ----
def load_metrics(model_dir):
    m = {}
    # Family classification
    fam_path = model_dir / "metrics" / "test_calibrated_metrics.json"
    if fam_path.exists():
        with open(fam_path) as f:
            fam = json.load(f)
        m["family_accuracy"] = fam["numeric_metrics"].get("Test_Calibrated_Accuracy", None)
        m["family_mcc"] = fam["numeric_metrics"].get("Test_Calibrated_MCC", None)
        m["classification_report"] = fam.get("classification_report", {})

    # Binary (derived from family softmax)
    for tag in ["binary_test_calibrated_metrics", "binary_test_calibrated_optimized_metrics"]:
        p = model_dir / "metrics" / f"{tag}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            prefix = "binary_opt" if "optimized" in tag else "binary_default"
            m[f"{prefix}_roc_auc"] = d.get("roc_auc")
            m[f"{prefix}_pr_auc"] = d.get("pr_auc")
            m[f"{prefix}_mcc"] = d.get("mcc")
            m[f"{prefix}_f1"] = d.get("f1")
            m[f"{prefix}_threshold"] = d.get("threshold")

    # Binary head (multitask only)
    bh_path = model_dir / "metrics" / "binary_test_calibrated_binaryhead_metrics.json"
    if bh_path.exists():
        with open(bh_path) as f:
            d = json.load(f)
        m["binaryhead_roc_auc"] = d.get("roc_auc")
        m["binaryhead_pr_auc"] = d.get("pr_auc")
        m["binaryhead_mcc"] = d.get("mcc")
        m["binaryhead_f1"] = d.get("f1")

    return m

std_m = load_metrics(STANDARD_DIR)
mt_m = load_metrics(MULTITASK_DIR)

# ---- Load predictions ----
def load_predictions(model_dir):
    p = model_dir / "predictions" / "test_calibrated_predictions.csv"
    return pd.read_csv(p) if p.exists() else None

std_preds = load_predictions(STANDARD_DIR)
mt_preds = load_predictions(MULTITASK_DIR)

# ---- Per-family analysis function ----
def per_family_analysis(preds, test_fam):
    merged = preds.merge(test_fam, on="identifier", how="left")
    merged["pred_binary"] = (merged["predicted_label"] != "nontox").astype(int)
    merged["family_correct"] = merged["actual_label"] == merged["predicted_label"]

    families = sorted(merged["Protein families"].unique())
    toxin_families = [f for f in families if f != "nontox"]

    results = []
    for fam in toxin_families:
        sub = merged[merged["Protein families"] == fam]
        n = len(sub)
        if n == 0:
            continue
        fam_correct = sub["family_correct"].sum()
        tp = (sub["pred_binary"] == 1).sum()
        fn = (sub["pred_binary"] == 0).sum()
        results.append({
            "Family": fam, "N": n,
            "Family_Acc": fam_correct / n,
            "Binary_Recall": tp / n if n > 0 else 0,
            "TP": int(tp), "FN": int(fn),
        })

    # Nontox
    nontox = merged[merged["Protein families"] == "nontox"]
    nontox_correct = (nontox["predicted_label"] == "nontox").sum()
    fp = len(nontox) - nontox_correct

    return pd.DataFrame(results).sort_values("N", ascending=False), fp, len(nontox)

std_pf, std_fp, std_n_nontox = per_family_analysis(std_preds, test_families)
mt_pf, mt_fp, mt_n_nontox = per_family_analysis(mt_preds, test_families)

# ---- Print comparison table ----
print("\n" + "=" * 100)
print("OVERALL METRICS COMPARISON")
print("=" * 100)
print(f"{'Metric':<40} {'Standard (Main)':>18} {'Multitask Augmented':>20}")
print("-" * 80)
print(f"{'Family Classification Accuracy':<40} {std_m.get('family_accuracy',0):>18.4f} {mt_m.get('family_accuracy',0):>20.4f}")
print(f"{'Family MCC':<40} {std_m.get('family_mcc',0):>18.4f} {mt_m.get('family_mcc',0):>20.4f}")
print(f"{'Binary ROC-AUC (from family softmax)':<40} {std_m.get('binary_default_roc_auc',0):>18.4f} {mt_m.get('binary_default_roc_auc',0):>20.4f}")
print(f"{'Binary PR-AUC (from family softmax)':<40} {std_m.get('binary_default_pr_auc',0):>18.4f} {mt_m.get('binary_default_pr_auc',0):>20.4f}")
print(f"{'Binary MCC (from family softmax)':<40} {std_m.get('binary_default_mcc',0):>18.4f} {mt_m.get('binary_default_mcc',0):>20.4f}")
print(f"{'Binary MCC (optimized threshold)':<40} {std_m.get('binary_opt_mcc',0):>18.4f} {mt_m.get('binary_opt_mcc',0):>20.4f}")
if mt_m.get("binaryhead_roc_auc"):
    print(f"{'Binary Head ROC-AUC (multitask only)':<40} {'N/A':>18} {mt_m['binaryhead_roc_auc']:>20.4f}")
    print(f"{'Binary Head PR-AUC (multitask only)':<40} {'N/A':>18} {mt_m['binaryhead_pr_auc']:>20.4f}")
    print(f"{'Binary Head MCC (multitask only)':<40} {'N/A':>18} {mt_m['binaryhead_mcc']:>20.4f}")
    print(f"{'Binary Head F1 (multitask only)':<40} {'N/A':>18} {mt_m['binaryhead_f1']:>20.4f}")
print(f"{'Nontox False Positives':<40} {std_fp:>18} {mt_fp:>20}")
print(f"{'Nontox FP Rate':<40} {std_fp/std_n_nontox:>18.4f} {mt_fp/mt_n_nontox:>20.4f}")

# ---- Per-family comparison table ----
comp = std_pf.merge(mt_pf, on="Family", how="outer", suffixes=("_std", "_mt"))
comp = comp.sort_values("N_std", ascending=False, na_position="last")
comp.to_csv(OUT_DIR / "per_family_comparison.csv", index=False)

print("\n" + "=" * 130)
print("PER-FAMILY COMPARISON")
print("=" * 130)
print(f"{'Family':<50} {'N':>4} {'FamAcc_Std':>10} {'FamAcc_MT':>10} {'Diff':>7} {'BinRec_Std':>10} {'BinRec_MT':>10} {'Diff':>7}")
print("-" * 130)
for _, r in comp.iterrows():
    n = int(r.get("N_std", r.get("N_mt", 0)))
    fa_s = r.get("Family_Acc_std", 0) or 0
    fa_m = r.get("Family_Acc_mt", 0) or 0
    br_s = r.get("Binary_Recall_std", 0) or 0
    br_m = r.get("Binary_Recall_mt", 0) or 0
    fa_d = fa_m - fa_s
    br_d = br_m - br_s
    fa_arrow = "+" if fa_d > 0.01 else ("-" if fa_d < -0.01 else "=")
    br_arrow = "+" if br_d > 0.01 else ("-" if br_d < -0.01 else "=")
    print(f"{r['Family']:<50} {n:>4} {fa_s:>10.3f} {fa_m:>10.3f} {fa_arrow}{abs(fa_d):>5.3f} {br_s:>10.3f} {br_m:>10.3f} {br_arrow}{abs(br_d):>5.3f}")

# ===== FIGURE 1: Side-by-side per-family comparison =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

families = comp["Family"].tolist()
x = np.arange(len(families))
width = 0.35

# Family accuracy
fa_std = comp["Family_Acc_std"].fillna(0).values
fa_mt = comp["Family_Acc_mt"].fillna(0).values
ax1.barh(x - width/2, fa_std, width, label="Standard (Main Branch)", color="#e74c3c", alpha=0.8)
ax1.barh(x + width/2, fa_mt, width, label="Multitask Augmented", color="#3498db", alpha=0.8)
ax1.set_yticks(x)
ax1.set_yticklabels(families, fontsize=7)
ax1.set_xlabel("Family Classification Accuracy", fontsize=11)
ax1.set_title("Family Classification Accuracy", fontsize=13)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_xlim(0, 1.1)
ax1.invert_yaxis()

# Binary recall
br_std = comp["Binary_Recall_std"].fillna(0).values
br_mt = comp["Binary_Recall_mt"].fillna(0).values
ax2.barh(x - width/2, br_std, width, label="Standard (Main Branch)", color="#e74c3c", alpha=0.8)
ax2.barh(x + width/2, br_mt, width, label="Multitask Augmented", color="#3498db", alpha=0.8)
ax2.set_yticks(x)
ax2.set_yticklabels(families, fontsize=7)
ax2.set_xlabel("Binary Recall (Detected as Toxic)", fontsize=11)
ax2.set_title("Binary Toxin Detection Recall", fontsize=13)
ax2.legend(loc="lower right", fontsize=9)
ax2.set_xlim(0, 1.1)
ax2.invert_yaxis()

fig.suptitle("Standard (Main Branch) vs Multitask Augmented (Exploration)\nPer-Family Performance", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_DIR / 'per_family_comparison.png'}")

# ===== FIGURE 2: Delta heatmap (improvement) =====
fig, ax = plt.subplots(figsize=(12, 10))
delta_fam = (fa_mt - fa_std).reshape(-1, 1)
delta_bin = (br_mt - br_std).reshape(-1, 1)
delta_matrix = np.hstack([delta_fam, delta_bin])

im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
ax.set_yticks(range(len(families)))
ax.set_yticklabels(families, fontsize=7)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Family Accuracy Δ", "Binary Recall Δ"], fontsize=11)
ax.set_title("Improvement: Multitask Augmented over Standard\n(Green = better, Red = worse)", fontsize=13)

for i in range(len(families)):
    for j in range(2):
        val = delta_matrix[i, j]
        sign = "+" if val > 0 else ""
        color = "black" if abs(val) < 0.3 else "white"
        ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center", fontsize=8, fontweight="bold", color=color)

plt.colorbar(im, ax=ax, shrink=0.6, label="Δ (Multitask - Standard)")
plt.tight_layout()
plt.savefig(OUT_DIR / "improvement_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'improvement_heatmap.png'}")

# ===== FIGURE 3: Overall metrics comparison bar chart =====
fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ["Family\nAccuracy", "Family\nMCC", "Binary\nROC-AUC", "Binary\nPR-AUC", "Binary\nMCC (opt)", "Binary Head\nMCC"]
std_vals = [
    std_m.get("family_accuracy", 0),
    std_m.get("family_mcc", 0),
    std_m.get("binary_default_roc_auc", 0),
    std_m.get("binary_default_pr_auc", 0),
    std_m.get("binary_opt_mcc", 0),
    0,  # no binary head
]
mt_vals = [
    mt_m.get("family_accuracy", 0),
    mt_m.get("family_mcc", 0),
    mt_m.get("binary_default_roc_auc", 0),
    mt_m.get("binary_default_pr_auc", 0),
    mt_m.get("binary_opt_mcc", 0),
    mt_m.get("binaryhead_mcc", 0),
]

x = np.arange(len(metrics_names))
width = 0.3
bars1 = ax.bar(x - width/2, std_vals, width, label="Standard (Main Branch)", color="#e74c3c", alpha=0.8)
bars2 = ax.bar(x + width/2, mt_vals, width, label="Multitask Augmented", color="#3498db", alpha=0.8)

for bar_group in [bars1, bars2]:
    for bar in bar_group:
        if bar.get_height() > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=10)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Overall Metrics: Standard vs Multitask Augmented", fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(0, 1.15)
plt.tight_layout()
plt.savefig(OUT_DIR / "overall_metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'overall_metrics_comparison.png'}")

# ===== FIGURE 4: Per-family confusion matrices side by side (top families) =====
top_fams = comp[comp["N_std"] >= 5]["Family"].tolist()
n_fams = len(top_fams)
ncols = 2
nrows = n_fams

fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 2.2))

for idx, fam in enumerate(top_fams):
    for col, (preds, label) in enumerate([(std_preds, "Standard"), (mt_preds, "Multitask")]):
        ax = axes[idx, col]
        merged = preds.merge(test_families, on="identifier", how="left")
        sub = merged[(merged["Protein families"] == fam) | (merged["Protein families"] == "nontox")]
        y_t = (sub["Protein families"] != "nontox").astype(int).values
        y_p = (sub["predicted_label"] != "nontox").astype(int).values
        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-toxic", "Toxic"], fontsize=6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Non-toxic", "Toxic"], fontsize=6)
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.6 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10, fontweight="bold", color=color)

        n_fam = (sub["Protein families"] == fam).sum()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        short = fam[:30] + "..." if len(fam) > 30 else fam
        ax.set_title(f"{label}: {short}\n(N={n_fam}, Recall={rec:.2f})", fontsize=7, fontweight="bold")

fig.suptitle("Per-Family Confusion Matrices: Standard (Left) vs Multitask (Right)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / "per_family_confusion_side_by_side.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_DIR / 'per_family_confusion_side_by_side.png'}")

print(f"\nAll outputs in: {OUT_DIR}")
