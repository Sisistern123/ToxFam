import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import json
import numpy as np

# For the custom lilac confusion matrix
import seaborn as sns
import math
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

class ToxDataset(Dataset):
    def __init__(self, df, h5_path, label_encoder=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.h5f = h5py.File(h5_path, 'r')

        if is_train:
            self.le = LabelEncoder()
            self.df['family_encoded'] = self.le.fit_transform(self.df['protein_category'])
        else:
            self.le = label_encoder
            self.df['family_encoded'] = self.le.transform(self.df['protein_category'])

        self.num_family_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_id = row['Entry']
        embedding = self.h5f[protein_id][:]
        family_label = row['family_encoded']
        return torch.tensor(embedding, dtype=torch.float32), family_label

    def __del__(self):
        # h5py handles close automatically, but for cleanliness:
        if hasattr(self, "h5f") and self.h5f is not None:
            self.h5f.close()


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_family_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_family = nn.Linear(hidden_dim, num_family_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc_family(x)


def evaluate_model(model, data_loader, loss_fn, device, dataset_type="Validation"):
    """
    Returns dict with accuracy, MCC, and average loss.
    """
    model.eval()
    all_family_labels, all_family_preds = [], []
    total_loss = 0

    with torch.no_grad():
        for features, family_labels in data_loader:
            features, family_labels = features.to(device), family_labels.to(device)
            family_pred = model(features)
            loss = loss_fn(family_pred, family_labels)
            total_loss += loss.item()

            family_preds = torch.argmax(family_pred, dim=1).cpu().numpy()
            all_family_labels.extend(family_labels.cpu().numpy())
            all_family_preds.extend(family_preds)

    avg_loss = total_loss / len(data_loader)
    metrics = {
        f"{dataset_type}_Family_Accuracy": accuracy_score(all_family_labels, all_family_preds),
        f"{dataset_type}_Family_MCC": matthews_corrcoef(all_family_labels, all_family_preds),
        f"{dataset_type}_Avg_Loss": avg_loss
    }
    return metrics


def train_model(model, train_loader, val_loader, weights_tensor, num_epochs=500, lr=0.0001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_family = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        total_loss_family = 0

        for features, family_labels in train_loader:
            features, family_labels = features.to(device), family_labels.to(device)
            optimizer.zero_grad()
            family_pred = model(features)
            loss_family = loss_fn_family(family_pred, family_labels)
            loss_family.backward()
            optimizer.step()
            total_loss_family += loss_family.item()

        train_loss = total_loss_family / len(train_loader)

        val_metrics = evaluate_model(model, val_loader, loss_fn_family, device)
        val_loss = val_metrics["Validation_Avg_Loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Save Loss Plot
    os.makedirs("model_output", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.legend()
    plt.savefig("model_output/loss_plot.png")
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv("data/model_input.csv")
    h5_path = "data/toxins_no_na.h5"

    # --- 1) Convert your cluster columns to boolean if they aren’t already ---
    # If your CSV uses "True"/"False", "1"/"0", or something else,
    # adjust this part as needed to ensure they become True/False booleans.
    df["Train_Cluster_Rep"] = df["Train_Cluster_Rep"].astype(bool)
    df["Val_Cluster_Rep"]   = df["Val_Cluster_Rep"].astype(bool)
    if "Test_Cluster_Rep" in df.columns:
        df["Test_Cluster_Rep"] = df["Test_Cluster_Rep"].astype(bool)

    # --- 2) Create filtered dataframes for each split ---
    train_df = df[df["Train_Cluster_Rep"] == True]
    val_df   = df[df["Val_Cluster_Rep"]   == True]
    test_df  = df[df["Test_Cluster_Rep"]  == True] if "Test_Cluster_Rep" in df.columns else pd.DataFrame()

    # --- 3) Identify families in each split *from these filtered dataframes* ---
    train_families = set(train_df["protein_category"])
    val_families   = set(val_df["protein_category"])
    test_families  = set(test_df["protein_category"]) if not test_df.empty else set()

    all_families   = train_families.union(val_families, test_families)

    missing_from_train = all_families - train_families
    missing_from_val   = all_families - val_families
    missing_from_test  = all_families - test_families if not test_df.empty else set()

    print("Missing from Train:", sorted(missing_from_train) if missing_from_train else "None")
    print("Missing from Val:",   sorted(missing_from_val)   if missing_from_val   else "None")
    if not test_df.empty:
        print("Missing from Test:", sorted(missing_from_test) if missing_from_test else "None")

    # --- 4) Build the ToxDataset for train & val from the same filtered DataFrames ---
    train_dataset = ToxDataset(
        train_df.copy(),  # is_train=True => new LabelEncoder
        h5_path,
        is_train=True
    )
    val_dataset = ToxDataset(
        val_df.copy(),
        h5_path,
        label_encoder=train_dataset.le,  # re-use the label encoder from train
        is_train=False
    )

    # Optional: check what's *actually* in val_dataset
    print("\nFamilies actually in val_dataset:")
    print(val_dataset.df["protein_category"].value_counts())

    # --- 5) Prepare Dataloaders ---
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False)

    # --- 6) Build model & train ---
    weights_tensor = torch.ones(train_dataset.num_family_classes, dtype=torch.float32)
    model = MLP(
        input_dim=1024,
        hidden_dim=128,
        num_family_classes=train_dataset.num_family_classes
    )

    train_model(model, train_loader, val_loader, weights_tensor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn_family = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    # Evaluate final metrics on validation
    val_metrics = evaluate_model(model, val_loader, loss_fn_family, device)
    print("\nFinal Validation metrics:", val_metrics)

    # --- 7) Gather full predictions for classification_report & confusion_matrix ---
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for features, family_labels in val_loader:
            features = features.to(device)
            family_labels = family_labels.to(device)
            output = model(features)
            preds = torch.argmax(output, dim=1).cpu().numpy()
            all_labels.extend(family_labels.cpu().numpy())
            all_preds.extend(preds)

    # Classification report
    val_class_report = classification_report(
        all_labels,
        all_preds,
        labels=range(val_dataset.num_family_classes),
        target_names=val_dataset.le.classes_,
        output_dict=True,
        zero_division=0
    )
    print("\nClassification Report (Validation):")
    for class_name, metrics_dict in val_class_report.items():
        if isinstance(metrics_dict, dict):
            print(f"{class_name}\n  Precision: {metrics_dict['precision']:.4f}"
                  f"\n  Recall: {metrics_dict['recall']:.4f}"
                  f"\n  F1-score: {metrics_dict['f1-score']:.4f}")
        else:
            # keys like 'accuracy'
            print(f"{class_name}: {metrics_dict:.4f}")

    # --- 8) Confusion Matrix ---
    true_labels_str = val_dataset.le.inverse_transform(all_labels)
    pred_labels_str = val_dataset.le.inverse_transform(all_preds)
    cm = confusion_matrix(true_labels_str, pred_labels_str)

    # Convert to % per row
    cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    # Lilac colormap
    colors = [
        (1, 1, 1),       # White
        (0.9, 0.9, 1),   # Light lilac
        (0.8, 0.8, 1),   # Lilac
        (0.6, 0.6, 1),   # Medium lilac
        (0.4, 0.4, 0.8), # Deep lilac
        (0.2, 0.2, 0.6), # Dark lilac
    ]
    cmap = LinearSegmentedColormap.from_list("custom_lilac", colors)
    boundaries = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    norm = BoundaryNorm(boundaries, cmap.N)

    class_names = list(val_dataset.le.classes_)
    cm_percent_display = np.where(cm == 0, np.nan, cm_percentage)

    plt.figure(figsize=(15, 10), dpi=200)
    ax = sns.heatmap(
        cm_percent_display,
        cmap=cmap,
        norm=norm,
        annot=False,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'},
        linecolor='lightgrey',
        linewidths=0.4,
    )
    plt.yticks(rotation=0)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count_val = cm[i, j]
            if count_val > 0:
                pct_val = cm_percentage[i, j]
                # Rounding logic
                if pct_val % 1 == 0:
                    pct_str = f"{pct_val:.0f}%"
                elif (round(pct_val, 2) * 10) % 1 == 0:
                    pct_str = f"{pct_val:.1f}%"
                else:
                    pct_str = f"{math.ceil(pct_val * 10) / 10:.1f}%"

                # Annotation color (white or black depending on background brightness)
                color_bg = cmap(norm(pct_val))
                brightness = (color_bg[0]*0.299 + color_bg[1]*0.587 + color_bg[2]*0.114)
                font_color = 'white' if brightness < 0.5 else 'black'

                ax.text(
                    j+0.5, i+0.5,
                    f"{pct_str}\n({int(count_val)})",
                    ha='center', va='center',
                    fontsize=7, color=font_color
                )

    plt.title("Confusion Matrix (Validation) – % per class w/ absolute counts")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = "model_output/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")

    # --- 9) Save numeric metrics + classification report to JSON ---
    save_dict = dict(
        numeric_metrics=val_metrics,
        classification_report=val_class_report
    )
    with open("model_output/metrics.json", "w") as f:
        json.dump(save_dict, f, indent=4)

    print("\nMetrics + classification report saved to model_output/metrics.json")
    print("Done!")
