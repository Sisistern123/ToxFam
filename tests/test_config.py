"""Tests for TrainConfig validation, fields, and YAML loading."""

import pytest
import yaml

from toxfam.config import TrainConfig


@pytest.fixture
def minimal_config(tmp_path):
    """Create a minimal valid config dict with required paths."""
    h5 = tmp_path / "dummy.h5"
    h5.touch()
    return {
        "input_csv": str(tmp_path / "data.csv"),
        "h5_path": str(h5),
        "output_dir": str(tmp_path / "output"),
        "training_strategy": "standard",
    }


class TestStrategyTypes:
    @pytest.mark.parametrize(
        "strategy",
        ["standard", "combined", "binary"],
    )
    def test_all_valid_strategies(self, minimal_config, strategy):
        minimal_config["training_strategy"] = strategy
        cfg = TrainConfig(**minimal_config)
        assert cfg.training_strategy == strategy

    def test_unknown_strategy_raises(self, minimal_config):
        minimal_config["training_strategy"] = "unknown"
        with pytest.raises(Exception):
            TrainConfig(**minimal_config)


class TestExtraFieldsIgnored:
    def test_extra_fields_silently_ignored(self, minimal_config):
        minimal_config["nonexistent_field"] = 42
        cfg = TrainConfig(**minimal_config)
        assert not hasattr(cfg, "nonexistent_field")


class TestFocalLoss:
    def test_defaults(self, minimal_config):
        cfg = TrainConfig(**minimal_config)
        assert cfg.use_focal_loss is False
        assert cfg.focal_loss_gamma == 2.0
        assert cfg.label_smoothing == 0.0

    def test_focal_gamma_must_be_positive_when_enabled(self, minimal_config):
        minimal_config["use_focal_loss"] = True
        minimal_config["focal_loss_gamma"] = -1.0
        with pytest.raises(ValueError, match="focal_loss_gamma"):
            TrainConfig(**minimal_config)


class TestFromYaml:
    def test_loads_correctly(self, minimal_config, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(minimal_config))
        cfg = TrainConfig.from_yaml(yaml_path)
        assert cfg.training_strategy == "standard"
        assert cfg.embedding_dim == 1024


class TestFieldValidation:
    def test_dropout_range(self, minimal_config):
        minimal_config["dropout"] = 1.5
        with pytest.raises(ValueError, match="dropout"):
            TrainConfig(**minimal_config)

    def test_learning_rate_positive(self, minimal_config):
        minimal_config["learning_rate"] = -0.001
        with pytest.raises(ValueError, match="learning_rate"):
            TrainConfig(**minimal_config)

    def test_num_epochs_positive(self, minimal_config):
        minimal_config["num_epochs"] = 0
        with pytest.raises(ValueError, match="num_epochs"):
            TrainConfig(**minimal_config)

    def test_batch_size_positive(self, minimal_config):
        minimal_config["batch_size"] = 0
        with pytest.raises(ValueError, match="batch_size"):
            TrainConfig(**minimal_config)

    def test_patience_positive(self, minimal_config):
        minimal_config["early_stopping_patience"] = 0
        with pytest.raises(ValueError, match="early_stopping_patience"):
            TrainConfig(**minimal_config)

    def test_label_smoothing_out_of_range(self, minimal_config):
        minimal_config["label_smoothing"] = 1.0
        with pytest.raises(ValueError, match="label_smoothing"):
            TrainConfig(**minimal_config)

    def test_weight_decay_negative(self, minimal_config):
        minimal_config["weight_decay"] = -0.1
        with pytest.raises(ValueError, match="weight_decay"):
            TrainConfig(**minimal_config)
