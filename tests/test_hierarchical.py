"""Tests for hierarchical training architecture and config."""

from __future__ import annotations

import torch

from toxfam.model.architectures import HierarchicalMLP, ModularMLP


class TestHierarchicalMLP:
    def _make_projector_state(self):
        """Create a Stage 1 projector state dict for testing."""
        base = ModularMLP(input_dim=1024, hidden_dims=[256], num_classes=38)
        return base.projector.state_dict()

    def test_forward_shape(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        x = torch.randn(8, 1024)
        out = model(x)
        assert out.shape == (8, 2)

    def test_frozen_backbone_no_grad(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_unfrozen_backbone_has_grad(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            num_binary_classes=2,
            freeze_backbone=False,
        )
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_head_always_has_grad(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        for param in model.head.parameters():
            assert param.requires_grad

    def test_custom_head_dim(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            hidden_dim=32,
            num_binary_classes=2,
        )
        x = torch.randn(4, 1024)
        out = model(x)
        assert out.shape == (4, 2)

    def test_stage1_projector_extraction(self):
        """Verify Stage 1 ModularMLP projector can be extracted and reused."""
        stage1 = ModularMLP(
            input_dim=1024, hidden_dims=[256, 256], num_classes=50, dropout=0.3,
        )
        projector_state = stage1.projector.state_dict()

        model = HierarchicalMLP(
            projector_state=projector_state,
            projector_out_dim=256,
            num_binary_classes=2,
            freeze_backbone=True,
        )
        x = torch.randn(4, 1024)
        out = model(x)
        assert out.shape == (4, 2)

    def test_projector_weights_loaded(self):
        """Verify the backbone actually loads the projector weights."""
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            hidden_dim=64,
        )
        assert torch.allclose(model.backbone[0].weight, state["0.weight"])

    def test_backward_only_updates_head_when_frozen(self):
        state = self._make_projector_state()
        model = HierarchicalMLP(
            projector_state=state,
            projector_out_dim=256,
            hidden_dim=64,
            freeze_backbone=True,
        )
        backbone_weight_before = model.backbone[0].weight.clone()

        x = torch.randn(4, 1024)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert torch.equal(model.backbone[0].weight, backbone_weight_before)
        assert model.head[0].weight.grad is not None


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

    def test_hierarchical_config_defaults(self):
        from toxfam.config import TrainConfig

        config = TrainConfig(
            input_csv="data/processed/training_data.csv",
            h5_path="data/processed/embeddings.h5",
            output_dir="model/output/test",
            training_strategy="hierarchical",
        )
        assert config.stage1_model_path is None
        assert config.stage2_learning_rate == 1e-5
