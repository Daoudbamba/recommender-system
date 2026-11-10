"""Le protocole d'evaluation, verifie avec des modeles dont on connait le score attendu."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from recsys.config import Config
from recsys.evaluation.protocol import evaluate, evaluate_ranking, evaluate_rating
from recsys.evaluation.split import Fold, build_fold
from recsys.experiments import collaborative as experiment
from recsys.models.base import InteractionMatrix, Recommender


class _FixedScores(Recommender):
    """Modele factice : memes scores pour tous, fournis a la construction."""

    name = "fixed"
    predicts_ratings = True

    def __init__(self, item_scores: list[float]) -> None:
        super().__init__()
        self.item_scores = np.array(item_scores, dtype=np.float32)

    def _fit(self, matrix: InteractionMatrix) -> None: ...

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores, (len(user_idx), 1))

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        return np.full(len(user_idx), 9.0)  # volontairement hors echelle


@pytest.fixture
def fold() -> Fold:
    day = pd.Timestamp("2020-01-01", tz="UTC")
    rows = [
        # entrainement : films 1 a 5
        (1, 1, 5.0, 1), (1, 2, 3.0, 2), (2, 1, 4.0, 3), (2, 3, 2.0, 4),
        (3, 4, 4.0, 5), (3, 5, 4.0, 6), (2, 4, 3.0, 7), (1, 5, 2.0, 8),
        # evaluation : l'utilisateur 1 aime le film 3, l'utilisateur 2 aime le film 5
        (1, 3, 5.0, 11), (1, 4, 2.0, 12), (2, 5, 4.5, 13),
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "d"])
    frame["timestamp"] = day + pd.to_timedelta(frame.pop("d"), "D")
    return build_fold(frame, day + pd.Timedelta(days=10), None, 1, 1, 4.0)


def test_oracle_scores_give_perfect_ranking_metrics(fold: Fold):
    # Les films 3 puis 5 en tete (indices 2 et 4) : chaque utilisateur evalue voit
    # son unique film pertinent au premier rang parmi les films qu'il n'a pas notes.
    model = _FixedScores([0, 0, 2, 0, 1]).fit(fold.matrix)

    result = evaluate_ranking(model, fold, ks=[1, 2])

    assert result.metrics["recall@1"] == pytest.approx(1.0)
    assert result.metrics["ndcg@2"] == pytest.approx(1.0)
    assert result.metrics["map@2"] == pytest.approx(1.0)
    assert result.metrics["precision@2"] == pytest.approx(0.5)
    assert result.recommended.shape == (2, 2)


def test_worst_scores_give_zero_and_seen_movies_are_excluded(fold: Fold):
    model = _FixedScores([9, 8, 0, 7, 0]).fit(fold.matrix)

    result = evaluate_ranking(model, fold, ks=[1])

    assert result.metrics["ndcg@1"] == 0.0
    seen = fold.matrix.ratings
    for row, user in enumerate(fold.eval_users):
        assert seen[user, result.recommended[row, 0]] == 0


def test_coverage_and_popularity_are_reported(fold: Fold):
    model = _FixedScores([0, 0, 2, 0, 1]).fit(fold.matrix)
    summary, _ = evaluate(model, fold, ks=[1], rating_bounds=(0.5, 5.0))

    assert summary["coverage@1"] == pytest.approx(2 / 5)  # films 3 et 5 sur 5
    assert summary["mean_popularity@1"] == pytest.approx(1.5)
    assert summary["ms_per_user"] >= 0


def test_rating_predictions_are_clipped_to_the_rating_scale(fold: Fold):
    model = _FixedScores([0, 0, 0, 0, 0]).fit(fold.matrix)

    out = evaluate_rating(model, fold, rating_min=0.5, rating_max=5.0)

    truth = fold.holdout["rating"].to_numpy()
    assert out["mae"] == pytest.approx(np.abs(truth - 5.0).mean())


def test_selection_respects_metric_direction():
    low, high = {"rmse": 0.8, "ndcg@10": 0.1}, {"rmse": 0.9, "ndcg@10": 0.2}
    assert experiment.is_better(low, high, "rmse")
    assert not experiment.is_better(low, high, "ndcg@10")
    assert experiment.is_better(low, None, "ndcg@10")


def test_experiment_runs_end_to_end_and_never_selects_on_test(config: Config, tmp_path):
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame(
        {
            "userId": rng.integers(1, 61, n),
            "movieId": rng.integers(1, 41, n),
            "rating": rng.integers(1, 11, n) / 2,
            "timestamp": pd.Timestamp("2020-01-01", tz="UTC")
            + pd.to_timedelta(rng.integers(0, 10_000, n), "h"),
        }
    ).drop_duplicates(["userId", "movieId"])
    config.paths.processed_dir.mkdir(parents=True)
    frame.to_parquet(config.paths.processed_dir / "ratings.parquet")
    config.collaborative = {
        "popularity": [{}],
        "item_knn": [{"k": 5}, {"k": 10}],
        "als_explicit": [{"factors": 2, "iterations": 2}],
    }
    config.evaluation.selection = {"default": "ndcg@10", "als_explicit": "rmse"}
    config.evaluation.ks = [5, 10]

    results = experiment.run(config)

    assert [row["model"] for row in results["test"]] == ["popularity", "item_knn", "als_explicit"]
    assert len(results["validation"]) == 4
    val_end = pd.Timestamp(results["folds"]["validation"]["train_end"])
    test_end = pd.Timestamp(results["folds"]["test"]["train_end"])
    assert val_end < test_end
    assert results["folds"]["validation"]["holdout_end"] == results["folds"]["test"]["train_end"]

    item_knn_val = [r for r in results["validation"] if r["model"] == "item_knn"]
    best = max(item_knn_val, key=lambda r: r["ndcg@10"])
    chosen = next(r for r in results["test"] if r["model"] == "item_knn")
    assert chosen["params"]["k"] == best["params"]["k"]
    assert "rmse" in chosen and "rmse" not in results["test"][0]

    on_disk = json.loads((config.paths.reports_dir / experiment.RESULTS_FILE).read_text("utf-8"))
    assert on_disk["test"] == results["test"]
    artifacts = config.paths.artifacts_dir / experiment.ARTIFACT_DIR
    assert {p.name for p in artifacts.glob("*.pkl")} == {
        "popularity.pkl",
        "item_knn.pkl",
        "als_explicit.pkl",
    }
    per_user = pd.read_parquet(artifacts / experiment.PER_USER_FILE)
    assert set(per_user["model"]) == {"popularity", "item_knn", "als_explicit"}
