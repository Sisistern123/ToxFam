import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


def plot_loss_curve(history, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_losses"], label="Train Loss")
    plt.plot(history["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, label_encoder_or_names, output_path):
    if hasattr(label_encoder_or_names, "classes_"):
        class_names = list(label_encoder_or_names.classes_)
    else:
        class_names = list(label_encoder_or_names)

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_percentage = cm.astype(float) / row_sums * 100.0

    colors = [
        (1, 1, 1),
        (0.9, 0.9, 1),
        (0.8, 0.8, 1),
        (0.6, 0.6, 1),
        (0.4, 0.4, 0.8),
        (0.2, 0.2, 0.6),
    ]
    cmap = LinearSegmentedColormap.from_list("custom_lilac", colors)
    boundaries = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    norm = BoundaryNorm(boundaries, cmap.N)
    cm_percent_display = np.where(cm == 0, np.nan, cm_percentage)

    plt.figure(figsize=(15, 10), dpi=200)
    ax = sns.heatmap(
        cm_percent_display,
        cmap=cmap,
        norm=norm,
        annot=False,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Percentage (%)", "ticks": boundaries},
        linecolor="lightgrey",
        linewidths=0.4,
    )
    plt.yticks(rotation=0)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count_val = cm[i, j]
            if count_val > 0:
                pct_val = cm_percentage[i, j]
                if pct_val % 1 == 0:
                    pct_str = f"{pct_val:.0f}%"
                elif (round(pct_val, 2) * 10) % 1 == 0:
                    pct_str = f"{pct_val:.1f}%"
                else:
                    pct_str = f"{math.ceil(pct_val * 10) / 10:.1f}%"

                color_bg = cmap(norm(pct_val))
                brightness = (
                    color_bg[0] * 0.299 + color_bg[1] * 0.587 + color_bg[2] * 0.114
                )
                font_color = "white" if brightness < 0.5 else "black"

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{pct_str}\n({int(count_val)})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=font_color,
                )

    plt.title("Confusion Matrix – % per class w/ absolute counts")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
