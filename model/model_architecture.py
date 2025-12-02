# model_architecture.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_family_classes, dropout=0.3):
        """
        Flexible MLP with optional multiple hidden layers + dropout.

        Args:
            input_dim: input feature size (e.g., 1024)
            hidden_dims: list or int, hidden layer sizes (e.g., [512, 256] or 128)
            num_family_classes: output size
            dropout: dropout probability (0.0–0.5 typical)
        """
        super().__init__()

        # Allow single int or list of ints
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_family_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiInputMLP(nn.Module):
    """
    Two-branch MLP:
      - one for embeddings (embed_dim)
      - one for taxonomy (tax_dim)
      then concatenates and passes through a joint head.
    """
    def __init__(
        self,
        embed_dim,
        tax_dim,
        hidden_dims,
        num_classes,
        dropout=0.3,
        tax_hidden_dim=8,
    ):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Embedding branch
        embed_layers = []
        prev = embed_dim
        for h in hidden_dims:
            embed_layers.append(nn.Linear(prev, h))
            embed_layers.append(nn.ReLU())
            embed_layers.append(nn.Dropout(dropout))
            prev = h
        self.embed_net = nn.Sequential(*embed_layers)
        embed_out_dim = prev

        # Taxonomy branch
        self.tax_net = nn.Sequential(
            nn.Linear(tax_dim, tax_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Joint head
        joint_in = embed_out_dim + tax_hidden_dim
        joint_layers = []
        for h in hidden_dims:
            joint_layers.append(nn.Linear(joint_in, h))
            joint_layers.append(nn.ReLU())
            joint_layers.append(nn.Dropout(dropout))
            joint_in = h
        joint_layers.append(nn.Linear(joint_in, num_classes))
        self.joint = nn.Sequential(*joint_layers)

    def forward(self, emb, tax):
        emb_h = self.embed_net(emb)
        tax_h = self.tax_net(tax)
        x = torch.cat([emb_h, tax_h], dim=1)
        return self.joint(x)


class ModularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # --- 1. The Input Projector (Variable Size) ---
        # This layer maps input_dim (56 or 1024) to the first hidden dimension
        first_hidden_dim = hidden_dims[0]
        self.projector = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- 2. The Backbone (Fixed Size) ---
        # These layers will be transferred from Tax training to Embedding training
        layers = []
        prev_dim = first_hidden_dim

        # Add remaining hidden layers if any
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projector(x)
        return self.backbone(x)