import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import torch
from torch.utils.data import DataLoader

from config import CONFIG
from dataset import ToxDataset, analyze_data_splits
from model import MLP
from training import train_model, evaluate_model, get_class_weights
from visualization import plot_loss_curve, plot_confusion_matrix
from utils import custom_logging


def analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, output_dir):
    """
    Analyze the distribution of the given label across train, validation, and test splits.
    Saves:
      - A JSON file with label counts.
      - A grouped bar chart (PNG) showing counts.
      - A JSON file with chi-square test results.
    """
    # Compute label counts for each split
    train_counts = train_df[label_col].value_counts().sort_index()
    val_counts = val_df[label_col].value_counts().sort_index()
    test_counts = test_df[label_col].value_counts().sort_index()

    # Combine counts into a DataFrame
    dist_df = pd.DataFrame({
        'Train': train_counts,
        'Validation': val_counts,
        'Test': test_counts
    }).fillna(0)

    # Save the distribution counts as JSON
    dist_json_path = os.path.join(output_dir, f"{label_col.replace(' ', '_')}_distribution.json")
    dist_df.to_json(dist_json_path, orient='index')
    print(f"Label distribution saved to: {dist_json_path}")

    # Plot grouped bar chart and save it
    plt.figure(figsize=(10, 6))
    dist_df.plot(kind='bar')
    plt.title(f'Distribution of {label_col} Across Splits')
    plt.xlabel(label_col)
    plt.ylabel('Count')
    plt.tight_layout()
    dist_plot_path = os.path.join(output_dir, f"{label_col.replace(' ', '_')}_distribution.png")
    plt.savefig(dist_plot_path)
    plt.close()
    print(f"Label distribution plot saved to: {dist_plot_path}")

    # Create a contingency table for chi-square test
    # Rows: splits; Columns: label counts
    contingency_table = dist_df.T
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi_square_results = {
        "chi2_statistic": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "expected": expected.tolist()  # convert numpy array to list
    }
    chi_json_path = os.path.join(output_dir, f"{label_col.replace(' ', '_')}_chi_square.json")
    with open(chi_json_path, "w") as f:
        json.dump(chi_square_results, f, indent=4)
    print(f"Chi-square test results saved to: {chi_json_path}")

    # Print the chi-square results
    print(f"Chi-square test for {label_col}: Chi2={chi2:.2f}, p={p:.4f}, dof={dof}")


def run_training_for_label(label_col, output_subdir, train_df, val_df, test_df):
    print(f"\n=== Running training for label: {label_col} ===")
    # Create a separate output directory for this model
    output_dir = os.path.join(CONFIG['output_dir'], output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Compute missing info for the current label column
    train_labels = set(train_df[label_col])
    val_labels = set(val_df[label_col])
    test_labels = set(test_df[label_col]) if not test_df.empty else set()
    all_labels = train_labels.union(val_labels, test_labels)

    missing_from_train = sorted(list(all_labels - train_labels))
    missing_from_val = sorted(list(all_labels - val_labels))
    missing_from_test = sorted(list(all_labels - test_labels)) if not test_df.empty else []

    missing_info = {
        "missing_from_train": missing_from_train,
        "missing_from_val": missing_from_val,
        "missing_from_test": missing_from_test
    }

    # Save split stats (absolute counts, relative percentages, and missing info)
    train_count = len(train_df)
    val_count = len(val_df)
    test_count = len(test_df)
    total = train_count + val_count + test_count
    split_stats = {
        "absolute": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
            "total": total
        },
        "relative": {
            "train": train_count / total if total > 0 else None,
            "val": val_count / total if total > 0 else None,
            "test": test_count / total if total > 0 else None
        },
        "missing_info": missing_info
    }
    with open(os.path.join(output_dir, "split_stats.json"), "w") as f:
        json.dump(split_stats, f, indent=4)

    # Analyze label distribution for this label column
    analyze_label_distribution_for_split(train_df, val_df, test_df, label_col, output_dir)

    # Build datasets with the specified label column
    try:
        train_dataset = ToxDataset(train_df.copy(), CONFIG['h5_path'], is_train=True, label_col=label_col)
        val_dataset = ToxDataset(val_df.copy(), CONFIG['h5_path'], label_encoder=train_dataset.le, is_train=False,
                                 label_col=label_col)
        test_dataset = ToxDataset(test_df.copy(), CONFIG['h5_path'], label_encoder=train_dataset.le, is_train=False,
                                  label_col=label_col)
    except Exception as e:
        print(f"Error creating datasets for {label_col}: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Calculate class weights
    weights_dict, weights_tensor, encoded_to_label = get_class_weights(train_dataset)

    # Print class counts and normalized weights
    from collections import Counter
    encoded_col = label_col + '_encoded'
    total_counts = Counter(train_dataset.df[encoded_col]) + \
                   Counter(val_dataset.df[encoded_col]) + \
                   Counter(test_dataset.df[encoded_col])
    print(f"\nClass counts for {label_col}:")
    for encoded, count in total_counts.items():
        lbl = encoded_to_label[encoded]
        weight = weights_dict[lbl]
        print(f"{lbl}: {count} samples, weight: {weight:.4f}")

    # Build and train model
    model = MLP(
        input_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_family_classes=train_dataset.num_classes  # now generic "num_classes"
    )
    # Create a temporary run configuration with the new output_dir
    run_config = CONFIG.copy()
    run_config['output_dir'] = output_dir

    model, history = train_model(model, train_loader, val_loader, weights_tensor, val_dataset.le, run_config)
    plot_loss_curve(history, os.path.join(output_dir, "loss_plot.png"))

    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    test_metrics, test_labels, test_preds = evaluate_model(model, test_loader, loss_fn, device, dataset_type="Test")

    from sklearn.metrics import classification_report
    test_class_report = classification_report(
        test_labels,
        test_preds,
        labels=range(test_dataset.num_classes),
        target_names=test_dataset.le.classes_,
        output_dict=True,
        zero_division=0
    )
    plot_confusion_matrix(test_labels, test_preds, test_dataset.le,
                          os.path.join(output_dir, "test_confusion_matrix.png"))

    test_save_dict = {
        'numeric_metrics': test_metrics,
        'classification_report': test_class_report
    }
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_save_dict, f, indent=4)

    # Clean up datasets
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()


def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    try:
        df = pd.read_csv(CONFIG['input_csv'])
    except FileNotFoundError:
        print(f"Error: Input file {CONFIG['input_csv']} not found.")
        return

    # Get train, validation, and test splits
    train_df, val_df, test_df = analyze_data_splits(df)

    # Save overall split stats (absolute and relative numbers) in the main output directory
    overall_train = len(train_df)
    overall_val = len(val_df)
    overall_test = len(test_df)
    overall_total = overall_train + overall_val + overall_test
    overall_stats = {
        "absolute": {
            "train": overall_train,
            "val": overall_val,
            "test": overall_test,
            "total": overall_total
        },
        "relative": {
            "train": overall_train / overall_total if overall_total > 0 else None,
            "val": overall_val / overall_total if overall_total > 0 else None,
            "test": overall_test / overall_total if overall_total > 0 else None
        }
    }
    with open(os.path.join(CONFIG['output_dir'], "overall_split_stats.json"), "w") as f:
        json.dump(overall_stats, f, indent=4)

    # List of label columns to train on, along with output subdirectories
    models_to_run = [
        ("Protein families", "protein_families"),
        ("protein_category", "protein_category"),
        ("Cluster ID", "cluster_id")
    ]

    # Loop over each target and run a separate training process
    for label_col, subdir in models_to_run:
        with custom_logging(os.path.join(CONFIG['output_dir'], subdir)):
            run_training_for_label(label_col, subdir, train_df, val_df, test_df)


if __name__ == "__main__":
    main()
