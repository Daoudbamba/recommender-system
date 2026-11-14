"""Suivi MLflow : noms de metriques, desactivation, enregistrement reel et meilleur run."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from recsys.config import PROJECT_ROOT, Config
from recsys.experiments import collaborative
from recsys.tracking import (
    ExperimentTracker,
    best_runs,
    flatten_params,
    metric_name,
    numeric_metrics,
)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tracked(config: Config, tmp_path: Path, monkeypatch) -> Config:
    """Configuration avec suivi actif, dans une base MLflow propre au test."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config.tracking.enabled = True
    config.tracking.uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config.tracking.artifact_location = str(tmp_path / "mlruns")
    config.tracking.experiment_prefix = "test"
    return config


def test_metric_names_and_params_are_mlflow_compatible():
    assert metric_name("ndcg@10") == "ndcg_at_10"
    assert metric_name("cold_start_ndcg@10_n0") == "cold_start_ndcg_at_10_n0"
    flat = flatten_params({"weights": {"als": 0.5, "tfidf": 0.25}, "k": 50, "name": "x" * 900})
    assert flat == {"weights.als": "0.5", "weights.tfidf": "0.25", "k": "50", "name": "x" * 500}
    row = {"ndcg@10": 0.2, "fit_seconds": 3, "model": "a", "flag": True, "params": {}}
    assert numeric_metrics(row) == {"ndcg_at_10": 0.2, "fit_seconds": 3.0}


def test_disabled_tracker_does_nothing(config: Config, tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = ExperimentTracker(config, "x")
    with tracker.run("x", params={"a": 1}) as run_id:
        tracker.log_candidate("m", "test", {}, {"ndcg@10": 1.0})
        tracker.log_metrics({"a": 1})
    assert run_id is None and not tracker.enabled
    assert not list(tmp_path.iterdir())


def test_experiment_is_recorded_with_one_child_run_per_configuration(synthetic_project, tracked):
    import mlflow

    tracked.collaborative = {"popularity": [{}], "item_knn": [{"k": 5}, {"k": 10}]}
    results = collaborative.run(tracked)

    runs = mlflow.search_runs(experiment_names=["test-collaborative"])
    parents = runs[runs["tags.kind"] == "experiment"]
    children = runs[runs["tags.kind"] != "experiment"]
    assert len(parents) == 1 and parents["status"].iloc[0] == "FINISHED"
    assert len(children) == len(results["validation"]) + len(results["test"]) == 5
    test_runs = children[children["tags.split"] == "test"]
    assert set(test_runs["tags.model"]) == {"popularity", "item_knn"}
    assert (test_runs["tags.selected"] == "true").all()
    knn = next(r for r in results["test"] if r["model"] == "item_knn")
    logged = test_runs[test_runs["tags.model"] == "item_knn"].iloc[0]
    assert logged["metrics.ndcg_at_10"] == pytest.approx(knn["ndcg@10"])
    assert logged["params.k"] == str(knn["params"]["k"])
    artifacts = mlflow.artifacts.list_artifacts(
        run_id=parents["run_id"].iloc[0], artifact_path="reports"
    )
    assert [Path(a.path).name for a in artifacts] == [collaborative.RESULTS_FILE]


def test_failed_experiment_closes_its_run(synthetic_project, tracked):
    import mlflow

    tracked.collaborative = {"inexistant": [{}]}
    with pytest.raises(KeyError):
        collaborative.run(tracked)

    runs = mlflow.search_runs(experiment_names=["test-collaborative"])
    assert runs["status"].tolist() == ["FAILED"]
    assert mlflow.active_run() is None


def test_import_and_best_run_selection(tracked):
    tracked.paths.reports_dir.mkdir(parents=True)
    fold = {"train_end": "2018-01-01"}
    rows = {
        "collaborative": [("item_knn", 0.10), ("als_implicit", 0.14)],
        "ranking": [("xgb_ranker", 0.21), ("xgb_ranker_scores_only", 0.15)],
    }
    for experiment, models in rows.items():
        payload = {
            "seed": 42, "protocol": {"ks": [10]}, "folds": {"test": fold},
            "validation": [{"model": models[0][0], "params": {"k": 1}, "ndcg@10": 0.3}],
            "test": [{"model": m, "params": {"k": 1}, "ndcg@10": v} for m, v in models],
        }  # fmt: skip
        filename = {
            "collaborative": "collaborative_results.json",
            "ranking": "ranking_results.json",
        }
        (tracked.paths.reports_dir / filename[experiment]).write_text(json.dumps(payload))

    counts = _load_script("import_results_to_mlflow").import_results(tracked)
    best = best_runs(tracked, "ndcg@10", "test", limit=3)

    assert counts == {"collaborative": 3, "ranking": 3}
    assert [row["model"] for row in best] == [
        "xgb_ranker",
        "xgb_ranker_scores_only",
        "als_implicit",
    ]
    assert best[0]["metric"] == pytest.approx(0.21)
    assert best[0]["experiment"] == "test-ranking" and best[0]["source"] == "import"
    validation = best_runs(tracked, "ndcg@10", "validation")
    assert [row["metric"] for row in validation] == [0.3, 0.3]


def test_best_runs_on_an_empty_store(tracked):
    assert best_runs(tracked) == []


def test_retrain_step_selection():
    retrain = _load_script("retrain")
    assert retrain.select_steps(None, None, None) == list(retrain.STEPS)
    assert retrain.select_steps("hybrid", "production", None) == ["hybrid", "ranking", "production"]
    assert retrain.select_steps(None, None, ["reports", "prepare"]) == ["prepare", "reports"]
    with pytest.raises(SystemExit, match="inconnues"):
        retrain.select_steps(None, None, ["deploy"])
