"""Tests for the paper-tree path helpers (paper._paths)."""

from paper import _paths
from toxfam._paths import get_project_root


def test_paper_root_is_under_project_root():
    assert _paths.paper_root() == get_project_root() / "paper"


def test_figures_output_dir_layout():
    out = _paths.figures_output_dir()
    assert out.name == "output"
    assert out.parent.name == "figures"
    assert out.parent.parent == _paths.paper_root()


def test_curation_files_live_in_paper_data_curation():
    curated, key = _paths.curated_verdicts_tsv(), _paths.curation_key_tsv()
    assert (
        curated.parent == _paths.curation_dir() == _paths.paper_data_dir() / "curation"
    )
    assert curated.name == "confident_errors_curated.tsv"
    assert key.parent == _paths.curation_dir()
    assert key.name == "confident_errors_key.tsv"


def test_manuscript_tex_target_respects_env_override(monkeypatch, tmp_path):
    # With an override dir that exists, the target is <dir>/results_numbers.tex.
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", str(tmp_path))
    target = _paths.manuscript_tex_target()
    assert target == tmp_path / "results_numbers.tex"

    # Pointing at a non-existent dir yields None (callers skip the sync).
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", str(tmp_path / "does_not_exist"))
    assert _paths.manuscript_tex_target() is None


def test_manuscript_tex_target_relative_override_anchors_to_project_root(
    monkeypatch, tmp_path
):
    # A RELATIVE override must resolve under the project root, not the CWD, to honour
    # the module's "stable regardless of the current working directory" contract.
    monkeypatch.setattr(_paths, "get_project_root", lambda: tmp_path)
    (tmp_path / "ms").mkdir()
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", "ms")
    assert _paths.manuscript_tex_target() == tmp_path / "ms" / "results_numbers.tex"


def test_load_preds_returns_canonical_identifier_order(tmp_path, monkeypatch):
    """predictions.csv order must not reach the bootstraps.

    `evaluation.runner` writes the file in whatever order the search produced, and
    every case-bootstrap downstream draws *positions* -- so an unsorted frame makes
    the reported CI a function of that incidental order while the point estimate
    stays put. Canonicalising at load is what stops that, so it is tested here
    rather than at each resampling site.
    """
    import pandas as pd

    from paper.figures import _common

    bench = tmp_path / "bench"
    (bench / "test_set" / "nn_x").mkdir(parents=True)
    shuffled = ["P3", "P1", "P2"]
    pd.DataFrame(
        {
            "identifier": shuffled,
            "actual_label": list("ABC"),
            "predicted_label": list("ABC"),
        }
    ).to_csv(bench / "test_set" / "nn_x" / "predictions.csv", index=False)
    monkeypatch.setattr(_common, "benchmark_dir", lambda: bench)

    out = _common.load_preds("test_set", "nn_x")
    assert out["identifier"].tolist() == ["P1", "P2", "P3"]
    # The row payload must travel with its identifier, not merely be re-indexed.
    assert out.loc[out["identifier"] == "P3", "actual_label"].item() == "A"
    assert out.index.tolist() == [0, 1, 2]
