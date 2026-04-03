"""Tests for toxfam.cli."""

from __future__ import annotations

from toxfam.cli import app


def test_app_has_expected_commands():
    """Verify all expected CLI commands are registered."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # Top-level commands
    top_level = ["download-data", "preprocess", "embed", "taxonomy", "train", "eval"]
    for cmd in top_level:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help"

    # Eval subcommands
    eval_result = runner.invoke(app, ["eval", "--help"])
    assert eval_result.exit_code == 0
    for sub in ["hbi", "model", "compare", "binary"]:
        assert sub in eval_result.output, f"Eval subcommand '{sub}' not found"
