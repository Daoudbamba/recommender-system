"""Experience de ranking de bout en bout, sur le mini-projet synthetique."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from recsys.config import Config
from recsys.experiments import collaborative, content, hybrid
from recsys.experiments import ranking as experiment
from recsys.experiments.runner import build_folds, load_ratings
from tests.conftest import fake_encoder


def test_ranking_requires_the_hybrid_recipe(synthetic_project: Config):
    collaborative.run(synthetic_project)
    content.run(synthetic_project, encoder=fake_encoder)
    with pytest.raises(FileNotFoundError, match="train_hybrid"):
        experiment.run(synthetic_project, encoder=fake_encoder)


def test_ranking_experiment_runs_end_to_end(synthetic_project: Config):
    config = synthetic_project
    collaborative.run(config)
    content.run(config, encoder=fake_encoder)
    hybrid.run(config, encoder=fake_encoder)

    results = experiment.run(config, encoder=fake_encoder)

    assert [row["model"] for row in results["test"]] == [
        "xgb_ranker",
        "xgb_ranker_no_recency",
        "xgb_ranker_scores_only",
    ]
    assert len(results["validation"]) == 2  # deux schemas de labels x une configuration
    assert {row["params"]["labels"] for row in results["validation"]} == {"graded", "binary"}
    best = max(results["validation"], key=lambda row: row["ndcg@10"])
    assert results["best"]["params"] == best["params"]

    dataset = results["dataset"]
    counts = dataset["label_counts"]
    assert sum(counts.values()) == dataset["train_rows"]
    assert dataset["train_rows"] == dataset["train_users"] * dataset["n_candidates"]
    assert counts["liked"] > 0 and counts["disliked"] > 0 and counts["unknown"] > 0
    assert 0 < dataset["candidate_recall_dev"] <= 1

    full, no_recency, scores_only = results["test"]
    assert full["params"]["n_features"] == len(results["feature_names"]) == 19
    assert no_recency["params"]["n_features"] == 17
    assert scores_only["params"]["n_features"] == 9
    assert "item_age" in results["feature_names"] and "item_year" not in results["feature_names"]
    assert sum(results["feature_importance"].values()) == pytest.approx(1.0)
    # Les gouts suivent le genre : le ranker doit retrouver la plupart des films aimes.
    assert full["recall@10"] > 0.5

    artifacts = config.paths.artifacts_dir / experiment.ARTIFACT_DIR
    assert (artifacts / f"{experiment.MODEL_FILE}.json").is_file()
    spec = json.loads((artifacts / f"{experiment.MODEL_FILE}.spec.json").read_text("utf-8"))
    assert spec["feature_names"] == results["feature_names"]
    per_user = pd.read_parquet(artifacts / experiment.PER_USER_FILE)
    assert set(per_user["model"]) == {
        "xgb_ranker",
        "xgb_ranker_no_recency",
        "xgb_ranker_scores_only",
    }


def test_ranker_training_labels_come_only_from_the_validation_period(synthetic_project: Config):
    """Aucune note de la periode de test ne peut servir de label d'entrainement."""
    config = synthetic_project
    folds = build_folds(load_ratings(config), config)
    train_fold, test_fold = folds["validation"], folds["test"]

    labels = experiment.holdout_ratings(train_fold)

    assert train_fold.holdout_end == test_fold.train_end
    ratings = load_ratings(config)
    late = ratings[ratings["timestamp"] > test_fold.train_end]
    late_pairs = set(zip(late["userId"], late["movieId"], strict=True))
    rows, cols = labels.nonzero()
    used = set(zip(train_fold.matrix.user_ids[rows], train_fold.matrix.item_ids[cols], strict=True))
    assert used and not used & late_pairs
    assert np.isin(rows, train_fold.eval_users).all()
