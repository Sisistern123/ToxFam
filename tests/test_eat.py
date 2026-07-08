"""Tests for EAT (embedding-based annotation transfer) core search.

EAT = the embedding-space analog of HBI: transfer the family label of the
nearest ProtT5 neighbour (k=1, Euclidean), and derive a P(toxic) score from the
distance margin between the nearest toxic and nearest non-toxic reference.
"""

import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from toxfam.evaluation.eat import EATResult, run_eat_search


def _write_h5(path: Path, vecs: dict[str, list[float]]) -> Path:
    with h5py.File(path, "w") as f:
        for key, vec in vecs.items():
            f.create_dataset(key, data=np.asarray(vec, dtype=np.float32))
    return path


def test_nearest_neighbor_family_transfer(tmp_path):
    ref_h5 = _write_h5(
        tmp_path / "ref.h5",
        {"R1": [10, 0, 0, 0], "R2": [0, 10, 0, 0], "R3": [0, 0, 10, 0]},
    )
    query_h5 = _write_h5(
        tmp_path / "q.h5",
        {"Q1": [9, 0, 0, 0], "Q2": [0, 9, 0, 0], "Q3": [0, 0, 9, 0]},
    )
    ref_df = pd.DataFrame(
        {
            "identifier": ["R1", "R2", "R3"],
            "Protein families": ["alpha", "nontoxin", "beta"],
        }
    )
    res = run_eat_search(query_h5, ref_h5, ref_df, ["Q1", "Q2", "Q3"])

    assert isinstance(res, EATResult)
    assert res.n_queries == 3 and res.n_reference == 3
    preds = res.predictions.set_index("identifier")
    assert preds.loc["Q1", "eat_prediction"] == "alpha"
    assert preds.loc["Q2", "eat_prediction"] == "nontoxin"
    assert preds.loc["Q3", "eat_prediction"] == "beta"
    assert list(res.predictions.columns) == [
        "identifier",
        "eat_prediction",
        "eat_confidence",
        "p_toxic",
    ]


def test_p_toxic_distance_margin_direction(tmp_path):
    ref_h5 = _write_h5(
        tmp_path / "ref.h5",
        {"Rtox": [10, 0, 0, 0], "Rnon": [0, 10, 0, 0]},
    )
    query_h5 = _write_h5(
        tmp_path / "q.h5",
        {"Qtox": [9, 0, 0, 0], "Qnon": [0, 9, 0, 0]},
    )
    ref_df = pd.DataFrame(
        {"identifier": ["Rtox", "Rnon"], "Protein families": ["alpha", "nontoxin"]}
    )
    preds = run_eat_search(
        query_h5, ref_h5, ref_df, ["Qtox", "Qnon"], metric="euclidean"
    ).predictions.set_index("identifier")
    # query closer to a toxic reference -> p_toxic high; closer to nontoxin -> low
    assert preds.loc["Qtox", "p_toxic"] > 0.99
    assert preds.loc["Qnon", "p_toxic"] < 0.01


def test_p_toxic_and_confidence_exact_values(tmp_path):
    # R1 at origin (toxic), R2 at distance 3 (nontoxin); query at distance 1 from R1.
    ref_h5 = _write_h5(tmp_path / "ref.h5", {"R1": [0, 0], "R2": [3, 0]})
    query_h5 = _write_h5(tmp_path / "q.h5", {"Q": [1, 0]})
    ref_df = pd.DataFrame(
        {"identifier": ["R1", "R2"], "Protein families": ["alpha", "nontoxin"]}
    )
    row = run_eat_search(query_h5, ref_h5, ref_df, ["Q"], metric="euclidean").predictions.iloc[0]

    assert row["eat_prediction"] == "alpha"
    # d_nearest_toxic = 1, d_nearest_nontoxin = 2 -> p_toxic = sigmoid(2 - 1) = sigmoid(1)
    assert row["p_toxic"] == pytest.approx(1.0 / (1.0 + math.exp(-1.0)), rel=1e-5)
    # confidence = 1 / (1 + d_nearest) = 1 / (1 + 1)
    assert row["eat_confidence"] == pytest.approx(0.5, rel=1e-5)


def test_all_reference_toxic_gives_p_toxic_one(tmp_path):
    # No non-toxin reference -> nothing pulls the query toward nontoxic -> p_toxic == 1.
    ref_h5 = _write_h5(tmp_path / "ref.h5", {"R1": [0, 0], "R2": [5, 0]})
    query_h5 = _write_h5(tmp_path / "q.h5", {"Q": [1, 0]})
    ref_df = pd.DataFrame(
        {"identifier": ["R1", "R2"], "Protein families": ["alpha", "beta"]}
    )
    row = run_eat_search(query_h5, ref_h5, ref_df, ["Q"]).predictions.iloc[0]
    assert row["p_toxic"] == pytest.approx(1.0)


def test_all_reference_nontoxin_gives_p_toxic_zero(tmp_path):
    ref_h5 = _write_h5(tmp_path / "ref.h5", {"R1": [0, 0], "R2": [5, 0]})
    query_h5 = _write_h5(tmp_path / "q.h5", {"Q": [1, 0]})
    ref_df = pd.DataFrame(
        {"identifier": ["R1", "R2"], "Protein families": ["nontoxin", "nontoxic"]}
    )
    row = run_eat_search(query_h5, ref_h5, ref_df, ["Q"]).predictions.iloc[0]
    assert row["p_toxic"] == pytest.approx(0.0)


def test_confidence_higher_for_closer_neighbor(tmp_path):
    ref_h5 = _write_h5(tmp_path / "ref.h5", {"R1": [0, 0]})
    query_h5 = _write_h5(tmp_path / "q.h5", {"Qnear": [0.5, 0], "Qfar": [5, 0]})
    ref_df = pd.DataFrame({"identifier": ["R1"], "Protein families": ["alpha"]})
    preds = run_eat_search(
        query_h5, ref_h5, ref_df, ["Qnear", "Qfar"], metric="euclidean"
    ).predictions.set_index("identifier")
    assert preds.loc["Qnear", "eat_confidence"] > preds.loc["Qfar", "eat_confidence"]


def test_cosine_uses_angle_not_magnitude(tmp_path):
    # Euclidean nearest is the small mis-angled ref; cosine nearest is the large
    # parallel one. The two metrics must pick different families here.
    ref_h5 = _write_h5(tmp_path / "ref.h5", {"Rpar": [100, 0], "Rorth": [0, 1]})
    query_h5 = _write_h5(tmp_path / "q.h5", {"Q": [1, 0]})
    ref_df = pd.DataFrame(
        {"identifier": ["Rpar", "Rorth"], "Protein families": ["alpha", "nontoxin"]}
    )
    euc = run_eat_search(query_h5, ref_h5, ref_df, ["Q"], metric="euclidean").predictions.iloc[0]
    cos = run_eat_search(query_h5, ref_h5, ref_df, ["Q"], metric="cosine").predictions.iloc[0]
    assert euc["eat_prediction"] == "nontoxin"  # closest by magnitude (dist √2 < 99)
    assert cos["eat_prediction"] == "alpha"  # closest by angle (parallel)


def test_run_eat_evaluation_keeps_nontoxin_for_p_toxic(tmp_path, monkeypatch):
    """Regression: the train-only 'nontox' family must NOT be collapsed to 'other'
    before the toxic mask is derived. If it were, datasets whose queries lack the
    'nontox' label (e.g. non_metazoan) would get a degenerate constant p_toxic=1.0.
    """
    from toxfam.data import registry
    from toxfam.evaluation import runner

    proc = tmp_path / "processed"
    proc.mkdir()
    bench = tmp_path / "benchmark"

    # train carries a non-toxin family; the test (query) split does NOT.
    rows = [
        ("Ttox1", "alpha", "train", [10.0, 0.0]),
        ("Ttox2", "beta", "train", [0.0, 10.0]),
        ("Tnon", "nontox", "train", [-10.0, 0.0]),
        ("Qtox", "alpha", "test", [9.0, 0.0]),   # near a toxic ref  -> p_toxic high
        ("Qnon", "alpha", "test", [-9.0, 0.0]),  # near the nontox ref -> p_toxic low
    ]
    pd.DataFrame(
        {
            "identifier": [r[0] for r in rows],
            "Sequence": ["M" * 5 for _ in rows],
            "Protein families": [r[1] for r in rows],
            "Organism (ID)": [9606 for _ in rows],
            "Split": [r[2] for r in rows],
        }
    ).to_csv(proc / "training_data.csv", index=False)
    with h5py.File(proc / "embeddings.h5", "w") as f:
        for ident, _family, _split, emb in rows:
            f.create_dataset(ident, data=np.asarray(emb, dtype=np.float32))

    # Dataset loading + embeddings-H5 resolution moved to toxfam.data.registry,
    # which holds its own processed_dir reference; patch both call sites.
    monkeypatch.setattr(runner, "processed_dir", lambda: proc)
    monkeypatch.setattr(registry, "processed_dir", lambda: proc)
    monkeypatch.setattr(runner, "benchmark_dir", lambda: bench)
    monkeypatch.setattr(runner, "plot_confusion_matrix", lambda *a, **k: None)

    runner.run_eat_evaluation("test_set")

    preds = pd.read_csv(bench / "test_set" / "eat" / "predictions.csv").set_index(
        "identifier"
    )
    # The non-toxin reference is recognized: p_toxic is NOT a degenerate constant.
    assert preds.loc["Qnon", "p_toxic"] < 0.5
    assert preds.loc["Qtox", "p_toxic"] > 0.5


def test_batching_does_not_change_results(tmp_path):
    # More queries than one batch should give identical results regardless of batch size.
    rng = np.random.default_rng(0)
    ref = {f"R{i}": rng.standard_normal(8).tolist() for i in range(20)}
    qs = {f"Q{i}": rng.standard_normal(8).tolist() for i in range(30)}
    ref_h5 = _write_h5(tmp_path / "ref.h5", ref)
    query_h5 = _write_h5(tmp_path / "q.h5", qs)
    fams = ["nontoxin" if i % 2 else f"fam{i}" for i in range(20)]
    ref_df = pd.DataFrame({"identifier": list(ref), "Protein families": fams})
    qids = list(qs)

    full = run_eat_search(query_h5, ref_h5, ref_df, qids, batch_size=1000).predictions
    batched = run_eat_search(query_h5, ref_h5, ref_df, qids, batch_size=7).predictions
    pd.testing.assert_frame_equal(full, batched)
