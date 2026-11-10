"""Protocole de cold start et experience hybride de bout en bout."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from recsys.config import Config
from recsys.evaluation.cold_start import (
    build_cold_user_scenario,
    evaluate_cold_users,
    evaluate_new_items,
)
from recsys.evaluation.split import build_fold
from recsys.experiments import collaborative, content
from recsys.experiments import hybrid as experiment
from recsys.models.collaborative import PopularityRecommender

DAY = pd.Timestamp("2020-01-01", tz="UTC")


def _frame(rows):
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "d"])
    frame["timestamp"] = DAY + pd.to_timedelta(frame.pop("d"), "D")
    return frame


@pytest.fixture
def fold():
    rows = [
        # entrainement : utilisateurs 1 et 2, films 10 a 14
        (1, 10, 5.0, 1), (1, 11, 4.0, 2), (1, 12, 2.0, 3), (2, 10, 4.0, 4),
        (2, 13, 5.0, 5), (2, 14, 3.0, 6),
        # utilisateur 1 (connu) apres la coupure : aime un film inconnu (99) et un connu (13)
        (1, 99, 5.0, 11), (1, 13, 4.5, 12), (1, 98, 2.0, 13),
        # utilisateur 7 (froid) : 4 notes dans l'ordre, puis aime 13 et 14
        (7, 10, 5.0, 20), (7, 11, 1.0, 21), (7, 12, 4.0, 22), (7, 13, 4.5, 23), (7, 14, 5.0, 24),
        # utilisateur 8 (froid) : n'aime rien apres ses deux premieres notes
        (8, 10, 4.0, 30), (8, 11, 4.0, 31), (8, 12, 1.0, 32),
    ]  # fmt: skip
    return build_fold(_frame(rows), DAY + pd.Timedelta(days=10), None, 1, 1, 4.0)


def test_fold_sets_cold_users_and_new_items_aside(fold):
    assert set(fold.cold_user_holdout["userId"]) == {7, 8}
    assert len(fold.cold_user_holdout) == 8
    assert fold.new_item_holdout["movieId"].tolist() == [99, 98]
    assert 7 not in fold.matrix.user_ids


def test_scenario_reveals_first_ratings_and_keeps_one_target_for_every_n(fold):
    scenario = build_cold_user_scenario(fold, [0, 1, 2], max_users=None, seed=0)

    # Seul l'utilisateur 7 aime un film apres ses 2 premieres notes.
    assert scenario.user_ids.tolist() == [7]
    assert scenario.histories[0].nnz == 0
    assert scenario.histories[1].nnz == 1
    items = fold.matrix.item_ids
    assert items[scenario.histories[2].indices].tolist() == [10, 11]
    assert scenario.histories[2].data.tolist() == [5.0, 1.0]
    assert sorted(items[scenario.excluded.indices]) == [10, 11]
    assert sorted(items[scenario.relevant.indices]) == [12, 13, 14]


def test_scenario_sampling_is_reproducible_and_validates_sizes(fold):
    first = build_cold_user_scenario(fold, [0, 1], max_users=1, seed=3)
    second = build_cold_user_scenario(fold, [0, 1], max_users=1, seed=3)
    assert first.user_ids.tolist() == second.user_ids.tolist()
    with pytest.raises(ValueError):
        build_cold_user_scenario(fold, [], None, 0)
    with pytest.raises(ValueError):
        build_cold_user_scenario(fold, [-1], None, 0)


def test_cold_user_evaluation_excludes_revealed_movies(fold):
    scenario = build_cold_user_scenario(fold, [0, 2], max_users=None, seed=0)
    popularity = PopularityRecommender().fit(fold.matrix)

    out = evaluate_cold_users(popularity, scenario, ks=[1, 2])

    # Popularite : film 10 (2 notes >= 4) puis 11 et 13. Les films 10 et 11 sont exclus pour
    # tous les n, donc le premier film recommande est le 13, qui est pertinent.
    assert out[0]["precision@1"] == pytest.approx(1.0)
    assert out[0] == out[2]  # un modele non personnalise ne profite pas de l'historique
    assert set(out) == {0, 2}


def test_new_item_evaluation_ranks_only_new_movies(fold):
    def oracle(users, movie_ids):
        return np.tile(np.where(movie_ids == 99, 1.0, 0.0), (len(users), 1))

    best = evaluate_new_items(oracle, fold, np.array([98, 99]), ks=[1, 5])
    worst = evaluate_new_items(lambda u, m: -oracle(u, m), fold, np.array([98, 99]), ks=[1])

    assert best["n_users"] == 1
    assert best["n_candidates"] == 2
    assert best["recall@1"] == pytest.approx(1.0)
    assert "recall@5" not in best  # k plafonne au nombre de candidats
    assert worst["recall@1"] == 0.0
    assert evaluate_new_items(oracle, fold, np.array([555]), ks=[1])["n_users"] == 0


def test_cold_user_evaluation_without_any_eligible_user_returns_empty_metrics(fold):
    scenario = build_cold_user_scenario(fold, [0, 50], max_users=None, seed=0)
    assert scenario.n_users == 0
    out = evaluate_cold_users(PopularityRecommender().fit(fold.matrix), scenario, ks=[1])
    assert out == {0: {}, 50: {}}


def _fake_encoder(texts):
    return np.array(
        [[1.0, 0.1] if "Sci-Fi" in text else [0.1, 1.0] for text in texts], dtype=np.float32
    )


def test_hybrid_experiment_runs_end_to_end(config: Config):
    rng = np.random.default_rng(2)
    n_movies, n_users = 40, 120
    movies = pd.DataFrame(
        {
            "movieId": range(1, n_movies + 1),
            "title": [f"Film {m} (2000)" for m in range(1, n_movies + 1)],
            "title_clean": [f"Film {m}" for m in range(1, n_movies + 1)],
            "year": pd.array([2000] * n_movies, dtype="Int16"),
            "genres": [["Sci-Fi"] if m % 2 else ["Romance"] for m in range(1, n_movies + 1)],
        }
    )
    rows = []
    for user in range(1, n_users + 1):
        # Un tiers des utilisateurs n'arrive qu'en fin de periode : ce sont les utilisateurs froids.
        late = user % 3 == 0
        for movie in rng.choice(np.arange(1, n_movies + 1), size=24, replace=False):
            hour = int(rng.integers(1950, 2000) if late else rng.integers(0, 2000))
            rows.append((user, movie, 4.5 if movie % 2 == user % 2 else 2.0, hour))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "h"])
    ratings["timestamp"] = DAY + pd.to_timedelta(ratings.pop("h"), "h")
    tags = pd.DataFrame(
        {"userId": [1, 2], "movieId": [1, 3], "tag": ["robots", "robots"], "timestamp": [DAY, DAY]}
    )
    processed = config.paths.processed_dir
    processed.mkdir(parents=True)
    ratings.to_parquet(processed / "ratings.parquet")
    movies.to_parquet(processed / "movies.parquet")
    tags.to_parquet(processed / "tags.parquet")

    config.evaluation.ks = [5, 10]
    config.evaluation.min_user_history = 3
    # Coupures placees avant l'arrivee des utilisateurs tardifs (heure 1950).
    config.evaluation.validation_quantile = 0.4
    config.evaluation.test_quantile = 0.6
    config.preprocessing.min_movie_ratings = 5
    config.content.tag_min_df = 1
    config.content.neighbours = 10
    config.collaborative = {
        "item_knn": [{"k": 10}],
        "als_implicit": [{"factors": 4, "iterations": 5}],
    }
    config.content.grids = {
        "tfidf": [{"genre_weight": 1.0, "tag_weight": 1.0, "aggregation": "knn"}],
        "embeddings": [{"aggregation": "knn"}],
    }
    config.hybrid.components = {
        "als_implicit": "collaborative", "item_knn": "collaborative",
        "tfidf": "content", "embeddings": "content",
    }  # fmt: skip
    config.hybrid.strategies = ["weighted", "rrf"]
    config.hybrid.weight_sets = [{"als_implicit": 1, "tfidf": 1}, {"item_knn": 1, "tfidf": 1}]
    config.hybrid.diversity_pool = 10
    config.hybrid.diversity_lambdas = [1.0, 0.5]
    config.hybrid.popularity_penalties = [0.0, 0.3]
    config.hybrid.cold_start_users = 20
    config.hybrid.cold_start_history_sizes = [0, 2, 5]
    config.hybrid.cold_start_priors = [0.5, 3.0]

    with pytest.raises(FileNotFoundError, match="collaborative"):
        experiment.run(config, encoder=_fake_encoder)
    collaborative.run(config)
    content.run(config, encoder=_fake_encoder)

    results = experiment.run(config, encoder=_fake_encoder)

    assert [row["model"] for row in results["test"]] == ["hybrid_weighted", "hybrid_rrf"]
    assert len(results["validation"]) == 4
    best = max(results["validation"], key=lambda row: row["ndcg@10"])
    assert results["best"]["model"] == best["model"]
    assert results["best"]["params"]["weights"] == best["params"]["weights"]

    diversity = results["diversity"]
    assert [(r["lambda"], r["popularity_penalty"]) for r in diversity] == [
        (1.0, 0.0), (0.5, 0.0), (1.0, 0.3),
    ]  # fmt: skip
    assert diversity[1]["ild_embeddings@10"] >= diversity[0]["ild_embeddings@10"] - 1e-6

    cold = results["cold_users"]
    assert cold["n_users"] > 0
    assert set(cold["models"]["popularity"]) == {"0", "2", "5"}
    pop = cold["models"]["popularity"]
    assert pop["0"]["ndcg@10"] == pop["5"]["ndcg@10"]
    policy = cold["models"]["cold_start"]
    assert policy["0"]["ndcg@10"] == pytest.approx(pop["0"]["ndcg@10"])  # sans note : popularite
    assert policy["5"]["ndcg@10"] > pop["5"]["ndcg@10"]  # avec 5 notes : mieux que la popularite
    selection = cold["prior_selection"]
    assert [row["prior"] for row in selection] == [0.5, 3.0]
    assert cold["prior_strength"] == max(selection, key=lambda row: row["mean"])["prior"]

    assert "random" in results["new_items"]["models"]
    artifacts = config.paths.artifacts_dir / experiment.ARTIFACT_DIR
    assert (artifacts / experiment.BEST_FILE).stat().st_size < 2000
    on_disk = json.loads((config.paths.reports_dir / experiment.RESULTS_FILE).read_text("utf-8"))
    assert on_disk["best"] == results["best"]
