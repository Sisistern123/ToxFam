"""Tests for toxfam.cli."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from toxfam.cli import app

runner = CliRunner()


def test_app_has_expected_commands():
    """Verify all expected CLI commands are registered."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # Top-level commands
    top_level = ["download-data", "preprocess", "embed", "taxonomy", "train", "eval", "plot"]
    for cmd in top_level:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help"

    # Eval subcommands
    eval_result = runner.invoke(app, ["eval", "--help"])
    assert eval_result.exit_code == 0
    for sub in ["hbi", "model", "compare", "binary"]:
        assert sub in eval_result.output, f"Eval subcommand '{sub}' not found"

    # Plot subcommands
    plot_result = runner.invoke(app, ["plot", "--help"])
    assert plot_result.exit_code == 0
    assert "taxonomy" in plot_result.output


@pytest.mark.parametrize(
    "cmd",
    [
        ["download-data", "--help"],
        ["preprocess", "--help"],
        ["embed", "--help"],
        ["taxonomy", "--help"],
        ["train", "--help"],
        ["eval", "hbi", "--help"],
        ["eval", "model", "--help"],
        ["eval", "compare", "--help"],
        ["eval", "binary", "--help"],
        ["plot", "taxonomy", "--help"],
    ],
)
def test_command_help_exits_cleanly(cmd):
    """Every command's --help should exit 0 and show usage text."""
    result = runner.invoke(app, cmd)
    assert result.exit_code == 0
    assert "Usage" in result.output or "usage" in result.output.lower()


def test_train_requires_config_argument():
    """Train without a config file should fail."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0


def test_eval_hbi_requires_dataset():
    """eval hbi without a dataset should fail."""
    result = runner.invoke(app, ["eval", "hbi"])
    assert result.exit_code != 0


def test_eval_binary_requires_model_dir():
    """eval binary without a model dir should fail."""
    result = runner.invoke(app, ["eval", "binary"])
    assert result.exit_code != 0
