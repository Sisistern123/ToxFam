import torch
import torch.nn as nn


class MultiInputMLP(nn.Module):
    """Two-branch MLP: one for embeddings, one for taxonomy,
    concatenated before a joint head."""

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

        self.tax_net = nn.Sequential(
            nn.Linear(tax_dim, tax_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        joint_in = embed_dim + tax_hidden_dim
        joint_layers = []
        for h in hidden_dims:
            joint_layers.append(nn.Linear(joint_in, h))
            joint_layers.append(nn.ReLU())
            joint_layers.append(nn.Dropout(dropout))
            joint_in = h
        joint_layers.append(nn.Linear(joint_in, num_classes))
        self.joint = nn.Sequential(*joint_layers)

    def forward(self, emb, tax):
        tax_h = self.tax_net(tax)
        x = torch.cat([emb, tax_h], dim=1)
        return self.joint(x)


class ModularMLP(nn.Module):
    """MLP with separate projector and backbone."""

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.dropout_rate = dropout
        first_hidden_dim = hidden_dims[0]

        self.projector = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        layers = []
        prev_dim = first_hidden_dim
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projector(x)
        x = self.backbone(x)
        return x
