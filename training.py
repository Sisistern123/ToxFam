import torch
import torch.optim as optim
import os
import json
from collections import Counter
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
from visualization import plot_confusion_matrix

def evaluate_model(model, data_loader, loss_fn, device, dataset_type="Validation"):
    """
    Evaluate model performance
    """
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)

    avg_loss = total_loss / len(data_loader)
    metrics = {
        f"{dataset_type}_Accuracy": accuracy_score(all_labels, all_preds),
        f"{dataset_type}_MCC": matthews_corrcoef(all_labels, all_preds),
        f"{dataset_type}_Avg_Loss": avg_loss
    }
    return metrics, all_labels, all_preds

def get_class_weights(train_dataset):
    """
    Calculate class weights for imbalanced dataset
    """
    encoded_col = train_dataset.label_col + '_encoded'
    class_counts = Counter(train_dataset.df[encoded_col])
    num_classes = train_dataset.num_classes

    encoded_to_label = {
        enc: train_dataset.le.inverse_transform([enc])[0]
        for enc in range(num_classes)
    }

    total_samples = sum(class_counts.values())
    weights_dict = {
        encoded_to_label[c]: total_samples / class_counts[c]
        for c in range(num_classes)
    }

    max_weight = max(weights_dict.values())
    weights_dict = {label: w / max_weight for label, w in weights_dict.items()}

    weights_tensor = torch.tensor(
        [weights_dict[encoded_to_label[i]] for i in range(num_classes)],
        dtype=torch.float32
    )

    return weights_dict, weights_tensor, encoded_to_label

def train_model(model, train_loader, val_loader, weights_tensor, label_encoder, config):
    """
    Train the model and save validation metrics, confusion matrix, and classification report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))

    best_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []

    for epoch in range(config['num_epochs']):
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
        train_losses.append(train_loss)

        val_metrics, val_labels, val_preds = evaluate_model(model, val_loader, loss_fn, device)
        val_loss = val_metrics["Validation_Avg_Loss"]
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(config['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['output_dir'], "best_model.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config['early_stopping_patience']:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(os.path.join(config['output_dir'], "best_model.pt")))

    final_val_metrics, val_preds, val_labels = evaluate_model(model, val_loader, loss_fn, device, dataset_type="Validation")

    # Save validation confusion matrix plot
    plot_confusion_matrix(val_labels, val_preds, label_encoder, os.path.join(config['output_dir'], "val_confusion_matrix.png"))

    # Generate and save classification report for validation
    val_class_report = classification_report(
        val_labels,
        val_preds,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    val_save_dict = {
        'numeric_metrics': final_val_metrics,
        'classification_report': val_class_report
    }
    metrics_path = os.path.join(config['output_dir'], "validation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(val_save_dict, f, indent=4)
    print("Validation Metrics and Classification Report saved to:", metrics_path)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

    return model, history
