import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console

from toxfam.training.trainer import _forward_model

console = Console()


class ModelWithTemperature(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *inputs):
        logits = self.model(*inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(
            logits.size(0), logits.size(1)
        )
        return logits / temperature

    def set_temperature(self, valid_loader):
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        logits_list = []
        labels_list = []

        console.print("Collecting validation logits for calibration...")
        with torch.no_grad():
            for features, label in valid_loader:
                label = label.to(self.device)
                logits = _forward_model(self.model, features, self.device)
                logits_list.append(logits)
                labels_list.append(label)

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        console.print(
            f"Before Calibration - NLL: {before_temperature_nll:.3f}, "
            f"ECE: {before_temperature_ece:.3f}"
        )

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        after_temperature_nll = nll_criterion(
            self.temperature_scale(logits), labels
        ).item()
        after_temperature_ece = ece_criterion(
            self.temperature_scale(logits), labels
        ).item()
        console.print(
            f"Optimal Temperature: [bold]{self.temperature.item():.3f}[/bold]"
        )
        console.print(
            f"After Calibration  - NLL: {after_temperature_nll:.3f}, "
            f"ECE: {after_temperature_ece:.3f}"
        )

        return self


class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
