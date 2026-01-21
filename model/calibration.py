import torch
import torch.nn as nn
import torch.optim as optim
from training import _forward_model

class ModelWithTemperature(nn.Module):
    def __init__(self, model, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.device = device
        # Initialize temperature to 1.5 (default starting point)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *inputs):
        """
        Modified to accept variable arguments (*inputs).
        This handles:
          1. Standard Strategy: inputs = (features_tensor,)
          2. Combined Strategy: inputs = (emb_tensor, tax_tensor)
        """
        # The inputs are already moved to device by the external _forward_model caller
        logits = self.model(*inputs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the temperature of the model using the validation set.
        """
        self.to(self.device)
        nll_criterion = nn.CrossEntropyLoss().to(self.device)
        ece_criterion = _ECELoss().to(self.device)

        # 1. Collect all logits and labels for the validation set
        logits_list = []
        labels_list = []

        print("Collecting validation logits for calibration...")
        with torch.no_grad():
            for features, label in valid_loader:
                label = label.to(self.device)
                # We use _forward_model here to handle device movement and unpacking correctly
                # for the INTERNAL model (self.model), not the wrapper.
                logits = _forward_model(self.model, features, self.device)

                logits_list.append(logits)
                labels_list.append(label)

        logits = torch.cat(logits_list).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        # 2. Calculate NLL and ECE before scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before Calibration - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # 3. Optimize the temperature parameter
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        # 4. Calculate metrics after scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f'Optimal Temperature: {self.temperature.item():.3f}')
        print(f'After Calibration  - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        super(_ECELoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = torch.nn.functional.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)

        for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece