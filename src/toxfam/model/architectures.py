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


class MultiTaskMLP(nn.Module):
    """Multi-task MLP with shared backbone and two heads: family + binary.

    Shared projector + backbone produce features, then:
    - family_head classifies into N family classes
    - binary_head classifies into toxic/nontoxic (2 classes)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_family_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

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

        self.backbone = nn.Sequential(*layers)

        self.family_head = nn.Linear(prev_dim, num_family_classes)
        self.binary_head = nn.Linear(prev_dim, 2)

    def forward(self, x, return_both: bool = True):
        features = self.projector(x)
        features = self.backbone(features)
        family_logits = self.family_head(features)
        if return_both:
            binary_logits = self.binary_head(features)
            return family_logits, binary_logits
        return family_logits


class HierarchicalMLP(nn.Module):
    """Two-stage model: pretrained backbone (from Stage 1) + binary head (Stage 2).

    The backbone is loaded from a Stage 1 family-classification model's projector
    and can be frozen or fine-tuned with a separate learning rate.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_out_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        head_hidden_dim: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(backbone_out_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, num_classes),
        )

    def forward(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)
