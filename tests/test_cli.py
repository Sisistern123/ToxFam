"""Tests for toxfam.cli."""

from __future__ import annotations

from toxfam.cli import app


def test_app_has_expected_commands():
    """Verify all expected CLI commands are registered."""
    # Typer stores registered commands in app.registered_commands
    # or we can check via the click group
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
        "eval-test",
        "eval-nonmetazoan",
        "eval-unreviewed",
        "eval-binary",
        "eval-ensemble",
        "profile-data",
        "explore-data",
        "parse-xml",
        "hierarchical-preprocess",
        "cpp",
        "hbi-baseline",
        "fetch-counterparts",
        "benchmark-external",
        "compute-hbi",
    ]
    for cmd in expected:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help"
