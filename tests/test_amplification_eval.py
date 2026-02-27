import json
from pathlib import Path

import pytest

from src.amplification import amplify


def test_compute_normalized_improvement_handles_zero_baseline():
    assert amplify.compute_normalized_improvement(0.5, 0.6) == pytest.approx(0.2)
    assert amplify.compute_normalized_improvement(0.0, 0.0) == 0.0
    assert amplify.compute_normalized_improvement(0.0, 0.1) == float("inf")


def test_load_amplification_eval_dataset_jsonl(tmp_path):
    path = tmp_path / "eval.jsonl"
    position = {"grid": [[0, 0], [0, 0]], "player": 1, "solutions": [(0, 0)]}
    path.write_text("\n".join(json.dumps(position) for _ in range(5)))

    loaded = amplify.load_amplification_eval_dataset(str(path))
    assert len(loaded) == 5
    assert loaded[0]["grid"][0][0] == 0
    assert loaded[0]["solutions"] == [(0, 0)]


def test_compute_accuracy_on_dataset_respects_solutions(monkeypatch):
    dataset = [
        {"grid": [[0] * 4 for _ in range(4)], "player": 1, "solutions": [(0, 0)]},
        {"grid": [[0] * 4 for _ in range(4)], "player": -1, "solutions": [(1, 1)]},
    ]

    def fake_run_mcts(*args, **kwargs):
        return ((0, 0), 0.0)

    monkeypatch.setattr(amplify, "_run_mcts", fake_run_mcts)
    accuracy = amplify.compute_accuracy_on_dataset(
        net=None,
        dataset=dataset,
        n_sims=1,
    )
    assert accuracy == pytest.approx(0.5)


def test_evaluate_concept_on_dataset_reports_normalized_improvement(monkeypatch):
    dataset = [
        {"grid": [[0] * 4 for _ in range(4)], "player": 1, "solutions": [(0, 0)]},
    ]

    def fake_run_mcts(*args, **kwargs):
        return ((0, 0), 0.0)

    monkeypatch.setattr(amplify, "_run_mcts", fake_run_mcts)
    result = amplify.evaluate_concept_on_dataset(
        net=None,
        concept_vector=None,
        layer="bn2",
        eval_dataset=dataset,
        alpha_range=[0.1],
        n_sims=1,
    )

    assert result["baseline_accuracy"] == pytest.approx(1.0)
    assert result["alpha_results"]["0.1"]["normalized_improvement"] == pytest.approx(0.0)
