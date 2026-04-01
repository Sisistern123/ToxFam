from __future__ import annotations

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
            nn.BatchNorm1d(tax_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        joint_in = embed_dim + tax_hidden_dim
        joint_layers = []
        for h in hidden_dims:
            joint_layers.append(nn.Linear(joint_in, h))
            joint_layers.append(nn.BatchNorm1d(h))
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
    """MLP with separate projector and backbone.

    The projector is the first hidden layer (Linear → BN → ReLU → Dropout).
    The backbone contains remaining hidden layers + the final output layer.
    This separation enables transfer learning: a trained projector can be
    extracted and reused in a HierarchicalMLP.
    """

    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        self.dropout_rate = dropout
        first_hidden_dim = hidden_dims[0]

        self.projector = nn.Sequential(
            nn.Linear(input_dim, first_hidden_dim),
            nn.BatchNorm1d(first_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        layers = []
        prev_dim = first_hidden_dim
        for h in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.backbone = nn.Sequential(*layers)

    def forward(self, x):
        x = self.projector(x)
        x = self.backbone(x)
        return x


class HierarchicalMLP(nn.Module):
    """Two-stage architecture: frozen (or fine-tunable) projector from a
    Stage 1 family classifier, with a new binary classification head.

    Stage 1 trains a ModularMLP on family classification.
    Stage 2 loads that model's projector as the backbone here, then trains
    a small binary head (toxic vs nontoxin) on top of it.

    Parameters
    ----------
    projector_state : state_dict from a trained ModularMLP.projector
    projector_out_dim : output dim of the projector (= hidden_dims[0])
    hidden_dim : hidden size of the binary head
    num_binary_classes : output classes for the binary head (default 2)
    freeze_backbone : if True, projector weights are frozen during Stage 2
    """

    def __init__(
        self,
        projector_state: dict,
        projector_out_dim: int,
        hidden_dim: int = 64,
        num_binary_classes: int = 2,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # Reconstruct the projector architecture from its state dict.
        # ModularMLP.projector is always: Linear → BN → ReLU → Dropout
        # State dict keys: 0.weight, 0.bias, 1.weight, 1.bias, 1.running_mean, ...
        input_dim = projector_state["0.weight"].shape[1]
        out_dim = projector_state["0.weight"].shape[0]

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.backbone.load_state_dict(projector_state)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(projector_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_binary_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class MultiTaskMultiInputMLP(nn.Module):
    """Two-branch input (embeddings + taxonomy) with dual task heads.

    Combines MultiInputMLP's input processing (separate taxonomy branch,
    concatenation) with MultiTaskMLP's dual heads (family + binary).

    Parameters
    ----------
    embed_dim : embedding dimension
    tax_dim : taxonomy vector dimension
    hidden_dims : list of hidden layer sizes for the shared backbone
    num_family_classes : number of family classes (e.g. 38)
    num_binary_classes : 2 (toxic, nontoxin)
    dropout : dropout rate
    tax_hidden_dim : hidden size of the taxonomy branch
    """

    def __init__(
        self,
        embed_dim: int,
        tax_dim: int,
        hidden_dims: list[int],
        num_family_classes: int,
        num_binary_classes: int = 2,
        dropout: float = 0.3,
        tax_hidden_dim: int = 8,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Taxonomy branch
        self.tax_net = nn.Sequential(
            nn.Linear(tax_dim, tax_hidden_dim),
            nn.BatchNorm1d(tax_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared backbone (after concatenation)
        joint_in = embed_dim + tax_hidden_dim
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.append(nn.Linear(joint_in, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            joint_in = h
        self.shared = nn.Sequential(*layers)

        # Dual heads
        self.family_head = nn.Linear(joint_in, num_family_classes)
        self.binary_head = nn.Linear(joint_in, num_binary_classes)

    def forward(self, emb, tax):
        tax_h = self.tax_net(tax)
        x = torch.cat([emb, tax_h], dim=1)
        shared_out = self.shared(x)
        return self.family_head(shared_out), self.binary_head(shared_out)


class MultiTaskMLP(nn.Module):
    """Shared backbone with dual heads for family classification and
    binary toxic/nontoxin classification.

    The shared layers learn a representation that serves both tasks.
    The family head predicts the specific toxin family (or nontox).
    The binary head predicts toxic vs nontoxin.

    During training, the total loss is:
        loss = alpha * family_loss + beta * binary_loss
    where alpha and beta are configurable weights.

    Parameters
    ----------
    input_dim : embedding dimension (use config.effective_embedding_dim)
    hidden_dims : list of hidden layer sizes for the shared backbone
    num_family_classes : number of family classes (e.g. 38)
    num_binary_classes : 2 (toxic, nontoxin)
    dropout : dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_family_classes: int,
        num_binary_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Shared backbone
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        self.shared = nn.Sequential(*layers)

        # Task-specific heads
        self.family_head = nn.Linear(prev_dim, num_family_classes)
        self.binary_head = nn.Linear(prev_dim, num_binary_classes)

    def forward(self, x):
        shared_out = self.shared(x)
        return self.family_head(shared_out), self.binary_head(shared_out)
