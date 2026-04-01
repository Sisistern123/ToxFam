"""Tests for toxfam.cli."""

from __future__ import annotations

from toxfam.cli import app


def test_app_has_expected_commands():
    """Verify all expected CLI commands are registered."""
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    expected = [
        "download-data",
        "preprocess",
        "embed",
        "taxonomy",
        "train",
        "eval",           # subcommand group (hbi, model, compare)
        "eval-binary",
        "eval-ensemble",
        "profile-data",
        "cpp",
    ]
    for cmd in expected:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help"
