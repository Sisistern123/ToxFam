"""Tests for the paper-tree path helpers (paper._paths)."""

from toxfam._paths import get_project_root

from paper import _paths


def test_paper_root_is_under_project_root():
    assert _paths.paper_root() == get_project_root() / "paper"


def test_figures_output_dir_layout():
    out = _paths.figures_output_dir()
    assert out.name == "output"
    assert out.parent.name == "figures"
    assert out.parent.parent == _paths.paper_root()


def test_adjudication_csv_lives_in_paper_data():
    csv = _paths.adjudication_csv()
    assert csv.parent == _paths.paper_data_dir()
    assert csv.name == "model_test_wrong_conf_annotated.csv"


def test_manuscript_tex_target_respects_env_override(monkeypatch, tmp_path):
    # With an override dir that exists, the target is <dir>/results_numbers.tex.
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", str(tmp_path))
    target = _paths.manuscript_tex_target()
    assert target == tmp_path / "results_numbers.tex"

    # Pointing at a non-existent dir yields None (callers skip the sync).
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", str(tmp_path / "does_not_exist"))
    assert _paths.manuscript_tex_target() is None


def test_manuscript_tex_target_relative_override_anchors_to_project_root(monkeypatch, tmp_path):
    # A RELATIVE override must resolve under the project root, not the CWD, to honour
    # the module's "stable regardless of the current working directory" contract.
    monkeypatch.setattr(_paths, "get_project_root", lambda: tmp_path)
    (tmp_path / "ms").mkdir()
    monkeypatch.setenv("TOXFAM_MANUSCRIPT_DIR", "ms")
    assert _paths.manuscript_tex_target() == tmp_path / "ms" / "results_numbers.tex"
