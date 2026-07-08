from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from torch.utils.data import DataLoader

from toxfam.model.forward import forward_model

console = Console()


class ModelWithTemperature(nn.Module):
    """Wraps a trained classifier with a learned temperature parameter for
    post-hoc probability calibration via Platt-style temperature scaling."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(*inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def set_temperature(self, valid_loader: DataLoader) -> ModelWithTemperature:
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        logits_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []

        console.print("Collecting validation logits for calibration...")
        with torch.no_grad():
            for features, label in valid_loader:
                label = label.to(self.device)
                logits = forward_model(self.model, features, self.device)
                logits_list.append(logits)
                labels_list.append(label)

        if not logits_list:
            raise ValueError(
                "Validation loader was empty; cannot calibrate temperature."
            )

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        console.print(
            f"Before Calibration - NLL: {before_temperature_nll:.3f}, "
            f"ECE: {before_temperature_ece:.3f}"
        )

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

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
    """Expected Calibration Error loss, binned into equal-width confidence intervals."""

    def __init__(self, n_bins: int = 15) -> None:
        super().__init__()
        self.n_bins = n_bins

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=logits.device)

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
