"""Tests for toxfam.cli."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from toxfam.cli import app

runner = CliRunner()


def test_app_has_expected_commands():
    """Verify all expected CLI commands are registered."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0

    # Top-level commands
    top_level = ["download-data", "preprocess", "embed", "taxonomy", "train", "predict", "eval", "plot"]
    for cmd in top_level:
        assert cmd in result.output, f"Command '{cmd}' not found in CLI help"

    # Eval subcommands
    eval_result = runner.invoke(app, ["eval", "--help"])
    assert eval_result.exit_code == 0
    for sub in ["hbi", "eat", "model", "compare", "binary"]:
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
        ["predict", "--help"],
        ["eval", "hbi", "--help"],
        ["eval", "eat", "--help"],
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


# ---------------------------------------------------------------------------
# Dispatch tests: the commands are thin delegators whose bodies lazily import
# their target and forward args. `--help`-only tests keep a renamed import
# target or a mis-forwarded kwarg green, so exercise the actual wiring by
# patching each delegate ON ITS SOURCE MODULE (lazy imports re-read it at call
# time) and asserting the forwarded call.
# ---------------------------------------------------------------------------


class _Recorder:
    """Callable that records the (args, kwargs) it was invoked with."""

    def __init__(self, ret=None):
        self.args = None
        self.kwargs = None
        self.called = False
        self._ret = ret

    def __call__(self, *args, **kwargs):
        self.called = True
        self.args = args
        self.kwargs = kwargs
        return self._ret


def test_dataset_enum_matches_registry():
    """The CLI Dataset enum must stay in sync with the runner's registry."""
    from toxfam.cli import Dataset
    from toxfam.data.registry import list_datasets

    assert {d.value for d in Dataset} == set(list_datasets())


def test_eat_metric_enum_matches_source():
    """The CLI EatMetric enum must stay in sync with evaluation.eat.METRICS."""
    from toxfam.cli import EatMetric
    from toxfam.evaluation.eat import METRICS

    assert {m.value for m in EatMetric} == set(METRICS)


def test_predict_forwards_flags(tmp_path, monkeypatch):
    """predict forwards its 9 params, incl. the --embeddings -> embeddings_h5 rename."""
    rec = _Recorder(ret=[])
    monkeypatch.setattr("toxfam.prediction.run_prediction", rec)

    model_dir = tmp_path / "model"  # --model-dir has exists=True
    model_dir.mkdir()

    result = runner.invoke(
        app,
        ["predict", "in.tsv", "--model-dir", str(model_dir),
         "--top-k", "5", "--toxicity-only"],
    )

    assert result.exit_code == 0, result.output
    assert rec.called
    assert rec.args[0] == Path("in.tsv")
    assert rec.args[1] == model_dir
    assert rec.kwargs["top_k"] == 5
    assert rec.kwargs["toxicity_only"] is True
    assert rec.kwargs["embeddings_h5"] is None  # not supplied -> None (rename intact)
    assert rec.kwargs["standard_model_dir"] is None


@pytest.mark.parametrize(
    "sub,target",
    [
        ("hbi", "run_hbi_evaluation"),
        ("compare", "compare_methods"),
    ],
)
def test_eval_single_arg_delegators(sub, target, monkeypatch):
    """eval hbi/compare forward the dataset name (enum .value) positionally."""
    rec = _Recorder()
    monkeypatch.setattr(f"toxfam.evaluation.runner.{target}", rec)

    result = runner.invoke(app, ["eval", sub, "test_set"])

    assert result.exit_code == 0, result.output
    assert rec.args == ("test_set",)


def test_eval_model_forwards_dataset_and_dir(monkeypatch):
    """eval model forwards dataset name + model_dir Path."""
    rec = _Recorder()
    monkeypatch.setattr("toxfam.evaluation.runner.run_model_evaluation", rec)

    result = runner.invoke(app, ["eval", "model", "test_set", "--model-dir", "foo"])

    assert result.exit_code == 0, result.output
    assert rec.args == ("test_set", Path("foo"))


def test_eval_binary_delegates(tmp_path, monkeypatch):
    """eval binary is a thin delegator to the runner entrypoint (post S11/C4)."""
    rec = _Recorder()
    monkeypatch.setattr(
        "toxfam.evaluation.runner.run_binary_evaluation_from_dir", rec
    )
    model_dir = tmp_path / "model"  # positional arg has exists=True
    model_dir.mkdir()

    result = runner.invoke(app, ["eval", "binary", str(model_dir)])

    assert result.exit_code == 0, result.output
    assert rec.args == (model_dir,)


def test_eval_eat_forwards_metric(monkeypatch):
    """eval eat forwards the dataset name and the metric enum .value."""
    rec = _Recorder()
    monkeypatch.setattr("toxfam.evaluation.runner.run_eat_evaluation", rec)

    result = runner.invoke(app, ["eval", "eat", "test_set", "--metric", "euclidean"])

    assert result.exit_code == 0, result.output
    assert rec.args == ("test_set",)
    assert rec.kwargs == {"metric": "euclidean"}


def test_eval_rejects_invalid_dataset():
    """An unknown dataset is rejected at parse time (exit 2), before any import."""
    result = runner.invoke(app, ["eval", "hbi", "bogus"])
    assert result.exit_code == 2


def test_eval_eat_rejects_invalid_metric():
    """An unknown --metric value is rejected at parse time."""
    result = runner.invoke(app, ["eval", "eat", "test_set", "--metric", "manhattan"])
    assert result.exit_code == 2


def test_version_flag():
    """--version prints the package version and exits 0."""
    from toxfam import __version__

    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_verify_command_registered():
    result = runner.invoke(app, ["verify", "--help"])
    assert result.exit_code == 0


def test_train_has_seeds_option():
    """`train` exposes --seeds. Checked on the click params, not rendered help
    text, which wraps by terminal width and is fragile under CI (no TTY)."""
    from typer.main import get_command

    train_cmd = get_command(app).commands["train"]
    assert "seeds" in {p.name for p in train_cmd.params}


def test_verify_command_has_dataset_option():
    from typer.main import get_command

    verify_cmd = get_command(app).commands["verify"]
    assert "dataset" in {p.name for p in verify_cmd.params}


def test_sha256_of_file_matches_hashlib(tmp_path):
    import hashlib

    from toxfam.cli import _sha256_of_file

    f = tmp_path / "blob.bin"
    f.write_bytes(b"toxfam" * 1000)
    assert _sha256_of_file(f) == hashlib.sha256(b"toxfam" * 1000).hexdigest()


def test_fetch_asset_digests_parses_sha256_prefix(monkeypatch):
    """Digests are keyed by asset name with the 'sha256:' prefix stripped."""
    import io
    import json

    from toxfam import cli

    payload = {
        "assets": [
            {"name": "hbi_train_all.csv", "digest": "sha256:abc123"},
            {"name": "weird.bin", "digest": "md5:deadbeef"},  # non-sha256 -> dropped
            {"name": "no_digest.tsv"},  # missing -> dropped
        ]
    }

    def fake_urlopen(url):
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    digests = cli._fetch_asset_digests("owner/repo", "data-v2")
    assert digests == {"hbi_train_all.csv": "abc123"}


def test_fetch_asset_digests_empty_on_failure(monkeypatch):
    """A network/API failure returns {} so download falls back to skip-if-exists."""
    from toxfam import cli

    def boom(url):
        raise OSError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert cli._fetch_asset_digests("owner/repo", "data-v2") == {}
