import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from collections import Counter
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
import contextlib
import seaborn as sns
import math
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Configuration parameters
CONFIG = {
    'input_csv': "data/model_input.csv",
    'h5_path': "data/toxins_no_na.h5",
    'output_dir': "model_output",
    'embedding_dim': 1024,
    'hidden_dim': 128,
    'batch_size': 64,
    'num_epochs': 500,
    'learning_rate': 0.0001,
    'early_stopping_patience': 5,
}


class ToxDataset(Dataset):
    def __init__(self, df, h5_path, label_encoder=None, is_train=True):
        """
        Dataset for toxin protein data

        Args:
            df: DataFrame with protein data
            h5_path: Path to h5 file with embeddings
            label_encoder: Optional pre-fitted LabelEncoder
            is_train: Whether this is a training dataset
        """
        self.df = df.reset_index(drop=True)
        self.h5f = h5py.File(h5_path, 'r')

        if is_train:
            self.le = LabelEncoder()
            self.df['family_encoded'] = self.le.fit_transform(self.df['Protein families'])
        else:
            self.le = label_encoder
            self.df['family_encoded'] = self.le.transform(self.df['Protein families'])

        self.num_family_classes = len(self.le.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_id = row['Entry']
        embedding = self.h5f[protein_id][:]
        family_label = row['family_encoded']
        return torch.tensor(embedding, dtype=torch.float32), family_label

    def close(self):
        """Explicitly close the h5 file"""
        if hasattr(self, "h5f") and self.h5f is not None:
            try:
                self.h5f.close()
            except Exception:
                pass
            self.h5f = None

    def __del__(self):
        """Safely clean up resources on object destruction"""
        try:
            self.close()
        except Exception:
            pass


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_family_classes):
        """
        Simple MLP model for toxin family classification

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            num_family_classes: Number of output classes
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_family = nn.Linear(hidden_dim, num_family_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc_family(x)


def evaluate_model(model, data_loader, loss_fn, device, dataset_type="Validation"):
    """
    Evaluate model performance

    Args:
        model: The model to evaluate
        data_loader: DataLoader with evaluation data
        loss_fn: Loss function
        device: Device to run evaluation on
        dataset_type: String indicating dataset type for metrics naming

    Returns:
        Dict with accuracy, MCC, and average loss.
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
    return metrics, all_family_labels, all_family_preds


def get_class_weights(train_dataset):
    """
    Calculate class weights for imbalanced dataset

    Args:
        train_dataset: Training dataset

    Returns:
        Tuple of (weights_dict, weights_tensor, encoded_to_label_dict)
    """
    # Get class counts
    class_counts = Counter(train_dataset.df["family_encoded"])
    num_classes = train_dataset.num_family_classes

    # Map encoded labels to decoded labels
    encoded_to_label = {
        enc: train_dataset.le.inverse_transform([enc])[0]
        for enc in range(num_classes)
    }

    # Calculate inverse frequency weights
    total_samples = sum(class_counts.values())
    weights_dict = {
        encoded_to_label[c]: total_samples / class_counts[c]
        for c in range(num_classes)
    }

    # Normalize weights to max 1.0
    max_weight = max(weights_dict.values())
    weights_dict = {label: w / max_weight for label, w in weights_dict.items()}

    # Convert to tensor for loss function
    weights_tensor = torch.tensor(
        [weights_dict[encoded_to_label[i]] for i in range(num_classes)],
        dtype=torch.float32
    )

    return weights_dict, weights_tensor, encoded_to_label


def train_model(model, train_loader, val_loader, weights_tensor, label_encoder, config):
    """
    Train the model and save validation metrics, confusion matrix, and classification report.

    Args:
        model: The model to train
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        weights_tensor: Tensor of class weights
        label_encoder: LabelEncoder used for class names
        config: Dictionary of configuration parameters

    Returns:
        Trained model and training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []  # Track losses for training curve
    val_metrics_list = []  # Store metrics for each epoch (optional)

    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # Validation phase (only calculate validation loss for training curve)
        val_metrics, _, _ = evaluate_model(model, val_loader, loss_fn, device)
        val_loss = val_metrics["Validation_Avg_Loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics_list.append(val_metrics)  # Optional: Store metrics for each epoch

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(config['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['early_stopping_patience']:
            print("Early stopping triggered.")
            break

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f"{config['output_dir']}/best_model.pt"))

    # Calculate overall validation metrics AFTER training
    final_val_metrics, val_preds, val_labels = evaluate_model(model, val_loader, loss_fn, device, dataset_type="Validation")

    # Generate and save confusion matrix plot
    plot_path = os.path.join(config['output_dir'], "confusion_matrix.png")
    plot_confusion_matrix(val_labels, val_preds, label_encoder, plot_path)

    # Generate classification report as a dictionary
    val_class_report = classification_report(
        val_labels,
        val_preds,
        labels=range(len(label_encoder.classes_)),  # Use the number of classes from the label encoder
        target_names=label_encoder.classes_,  # Use class names from the label encoder
        output_dict=True,  # Return the report as a dictionary
        zero_division=0  # Handle division by zero in metrics
    )

    # Combine numeric metrics and classification report into a single dictionary
    val_save_dict = {
        'numeric_metrics': final_val_metrics,
        'classification_report': val_class_report
    }

    # Save combined metrics and classification report to JSON
    metrics_path = os.path.join(config['output_dir'], "validation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(val_save_dict, f, indent=4)

    print("Validation Metrics and Classification Report saved to:", metrics_path)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    return model, history


def plot_loss_curve(history, output_path):
    """
    Plot and save training/validation loss curves

    Args:
        history: Dictionary with training history
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_losses'], label="Train Loss")
    plt.plot(history['val_losses'], label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, label_encoder, output_path):
    """
    Create and save confusion matrix visualization

    Args:
        all_labels: True labels
        all_preds: Predicted labels
        label_encoder: LabelEncoder used for class names
        output_path: Path to save the plot
    """
    # Confusion Matrix
    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=range(len(label_encoder.classes_))
    )

    # Convert to % per row
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid 0-division
    cm_percentage = cm.astype(float) / row_sums * 100.0

    # Lilac colormap
    colors = [
        (1, 1, 1),  # White
        (0.9, 0.9, 1),  # Light lilac
        (0.8, 0.8, 1),  # Lilac
        (0.6, 0.6, 1),  # Medium lilac
        (0.4, 0.4, 0.8),  # Deep lilac
        (0.2, 0.2, 0.6),  # Dark lilac
    ]
    cmap = LinearSegmentedColormap.from_list("custom_lilac", colors)
    boundaries = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    norm = BoundaryNorm(boundaries, cmap.N)

    class_names = list(label_encoder.classes_)
    cm_percent_display = np.where(cm == 0, np.nan, cm_percentage)

    plt.figure(figsize=(15, 10), dpi=200)
    ax = sns.heatmap(
        cm_percent_display,
        cmap=cmap,
        norm=norm,
        annot=False,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)', 'ticks': boundaries},
        linecolor='lightgrey',
        linewidths=0.4,
    )
    plt.yticks(rotation=0)

    # Add text annotations
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
                brightness = (color_bg[0] * 0.299 + color_bg[1] * 0.587 + color_bg[2] * 0.114)
                font_color = 'white' if brightness < 0.5 else 'black'

                ax.text(
                    j + 0.5, i + 0.5,
                    f"{pct_str}\n({int(count_val)})",
                    ha='center', va='center',
                    fontsize=7, color=font_color
                )

    plt.title("Confusion Matrix (Validation) – % per class w/ absolute counts")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def analyze_data_splits(df):
    """
    Analyze data splits for train/val/test and return split DataFrames

    Args:
        df: Main DataFrame with split indicators

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Convert cluster columns to bool and filter for each split
    df["Train_Cluster_Rep"] = df["Train_Cluster_Rep"].astype(bool)
    df["Val_Cluster_Rep"] = df["Val_Cluster_Rep"].astype(bool)
    df["Test_Cluster_Rep"] = df["Test_Cluster_Rep"].astype(bool)

    train_df = df[df["Train_Cluster_Rep"] == True]
    val_df = df[df["Val_Cluster_Rep"] == True]
    test_df = df[df["Test_Cluster_Rep"] == True]

    # Identify families in each split
    train_families = set(train_df["Protein families"])
    val_families = set(val_df["Protein families"])
    test_families = set(test_df["Protein families"]) if not test_df.empty else set()

    all_families = train_families.union(val_families, test_families)

    missing_from_train = all_families - train_families
    missing_from_val = all_families - val_families
    missing_from_test = all_families - test_families if not test_df.empty else set()

    print("Missing from Train:", sorted(missing_from_train) if missing_from_train else "None")
    print("Missing from Val:", sorted(missing_from_val) if missing_from_val else "None")
    if not test_df.empty:
        print("Missing from Test:", sorted(missing_from_test) if missing_from_test else "None")

    return train_df, val_df, test_df


# custom logging
class CustomLogger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        if message == "\n":
            return
        if "Epoch" in message:
            message += "\n"
            self.terminal.write(message)  # Print to CLI
        else:
            self.file.write(message)  # Write to file

    def flush(self):
        self.terminal.flush()
        self.file.flush()


@contextlib.contextmanager
def custom_logging(output_dir):
    """Context manager for custom logging"""
    os.makedirs(output_dir, exist_ok=True)
    original_stdout = sys.stdout
    log_file = open(f"{output_dir}/model_output.txt", "w")
    try:
        sys.stdout = CustomLogger(log_file)
        yield
    finally:
        sys.stdout = original_stdout
        log_file.close()


def main():
    """Main execution function"""
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(CONFIG['input_csv'])
    except FileNotFoundError:
        print(f"Error: Input file {CONFIG['input_csv']} not found.")
        return

    with custom_logging(CONFIG['output_dir']):
        # Analyze data splits
        train_df, val_df, test_df = analyze_data_splits(df)

        # Build datasets
        try:
            train_dataset = ToxDataset(
                train_df.copy(),  # is_train=True => new LabelEncoder
                CONFIG['h5_path'],
                is_train=True
            )

            val_dataset = ToxDataset(
                val_df.copy(),
                CONFIG['h5_path'],
                label_encoder=train_dataset.le,  # re-use the label encoder from train
                is_train=False
            )

            test_dataset = ToxDataset(
                test_df.copy(),
                CONFIG['h5_path'],
                label_encoder=train_dataset.le,  # re-use the label encoder from train
                is_train=False
            )
        except Exception as e:
            print(f"Error creating datasets: {e}")
            return

        # Prepare Dataloaders
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # Calculate class weights
        weights_dict, weights_tensor, encoded_to_label = get_class_weights(train_dataset)

        # Print class labels, counts, and weights
        print("\nClass Label | Count | Normalized Weight")
        class_counts = Counter(train_dataset.df["family_encoded"])
        for encoded, count in class_counts.items():
            label = encoded_to_label[encoded]
            weight = weights_dict[label]
            print(f"{label}:\n {count} samples \t Weight: {weight:.4f}\n")

        # Build model
        model = MLP(
            input_dim=CONFIG['embedding_dim'],
            hidden_dim=CONFIG['hidden_dim'],
            num_family_classes=train_dataset.num_family_classes
        )

        # Train model
        model, history = train_model(model, train_loader, val_loader, weights_tensor, val_dataset.le, CONFIG)

        # Plot loss curves
        plot_loss_curve(history, f"{CONFIG['output_dir']}/loss_plot.png")

        # Evaluate final model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn_family = nn.CrossEntropyLoss(weight=weights_tensor.to(device))

        # Evaluate on test set
        test_metrics, test_labels, test_preds = evaluate_model(model, test_loader, loss_fn_family, device, dataset_type="Test")
        print("\nFinal Test metrics:", test_metrics)

        # Generate test classification report
        test_class_report = classification_report(
            test_labels,
            test_preds,
            labels=range(test_dataset.num_family_classes),
            target_names=test_dataset.le.classes_,
            output_dict=True,
            zero_division=0
        )

        # Plot test confusion matrix
        plot_confusion_matrix(
            test_labels,
            test_preds,
            test_dataset.le,
            f"{CONFIG['output_dir']}/test_confusion_matrix.png"
        )

        # Save test metrics to JSON
        test_save_dict = {
            'numeric_metrics': test_metrics,
            'classification_report': test_class_report
        }
        with open(f"{CONFIG['output_dir']}/test_metrics.json", "w") as f:
            json.dump(test_save_dict, f, indent=4)

        print(f"\nValidation confusion matrix saved to: {CONFIG['output_dir']}/val_confusion_matrix.png")
        print(f"Validation metrics + classification report saved to {CONFIG['output_dir']}/val_metrics.json")
        print(f"\nTest confusion matrix saved to: {CONFIG['output_dir']}/test_confusion_matrix.png")
        print(f"Test metrics + classification report saved to {CONFIG['output_dir']}/test_metrics.json")
        print("Done!")

        # Clean up datasets
        train_dataset.close()
        val_dataset.close()
        test_dataset.close()


if __name__ == "__main__":
    main()