"""Tests for hierarchical training architecture and config."""

from __future__ import annotations

import torch

from toxfam.model.architectures import HierarchicalMLP, ModularMLP


class TestHierarchicalMLP:
    def _make_stage1_projector(self):
        """Create a simple Stage 1 projector for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )

    def test_forward_shape(self):
        projector = self._make_stage1_projector()
        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            freeze_backbone=True,
        )
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 2)

    def test_frozen_backbone_no_grad(self):
        projector = self._make_stage1_projector()
        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            freeze_backbone=True,
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfrozen_backbone_has_grad(self):
        projector = self._make_stage1_projector()
        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            freeze_backbone=False,
        )
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_head_always_has_grad(self):
        projector = self._make_stage1_projector()
        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            freeze_backbone=True,
        )
        for param in model.head.parameters():
            assert param.requires_grad

    def test_custom_head_dim(self):
        projector = self._make_stage1_projector()
        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            head_hidden_dim=32,
        )
        x = torch.randn(4, 1024)
        out = model(x)
        assert out.shape == (4, 2)

    def test_stage1_projector_extraction(self):
        """Verify Stage 1 ModularMLP projector can be extracted and reused."""
        stage1 = ModularMLP(
            input_dim=1024,
            hidden_dims=[256, 256],
            num_classes=50,
            dropout=0.3,
        )
        projector = stage1.projector

        model = HierarchicalMLP(
            backbone=projector,
            backbone_out_dim=256,
            num_classes=2,
            freeze_backbone=True,
        )
        x = torch.randn(4, 1024)
        out = model(x)
        assert out.shape == (4, 2)


class TestHierarchicalConfig:
    def test_hierarchical_strategy_accepted(self):
        from toxfam.config import TrainConfig

        config = TrainConfig(
            input_csv="data/processed/training_data.csv",
            h5_path="data/processed/embeddings.h5",
            output_dir="model/output/test",
            training_strategy="hierarchical",
        )
        assert config.training_strategy == "hierarchical"
        assert config.stage2_freeze_backbone is True
        assert config.stage2_hidden_dim == 64
        assert config.family_min_count == 10

    def test_hierarchical_config_defaults(self):
        from toxfam.config import TrainConfig

        config = TrainConfig(
            input_csv="data/processed/training_data.csv",
            h5_path="data/processed/embeddings.h5",
            output_dir="model/output/test",
            training_strategy="hierarchical",
        )
        assert config.stage1_model_path is None
        assert config.stage2_learning_rate is None
