"""Variables, labels, groupes et modele de ranking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.features.ranking import RankingFeatureBuilder, binary_labels, graded_labels
from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.hybrid import HybridRecommender
from recsys.models.ranking import RankingRecommender, group_ids, train_ranker


class _Fixed(Recommender):
    def __init__(self, scores: list[float], name: str) -> None:
        super().__init__()
        self.scores = np.array(scores, dtype=np.float32)
        self.name = name

    def _fit(self, matrix: InteractionMatrix) -> None: ...

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(user_idx), 1))


MOVIES = pd.DataFrame(
    {
        "movieId": [10, 11, 12, 13, 14],
        "year": pd.array([1990, 2000, 2010, pd.NA, 2020], dtype="Int16"),
        "genres": [["Sci-Fi"], ["Sci-Fi", "Action"], ["Romance"], [], ["Romance", "Drama"]],
    }
)


@pytest.fixture
def matrix() -> InteractionMatrix:
    rows = [
        (1, 10, 5.0), (1, 11, 4.0),              # aime la science-fiction
        (2, 12, 4.5), (2, 10, 2.0),              # aime la romance, rejette un film de SF
        (3, 13, 3.0),                            # n'a rien aime
        (4, 14, 3.0),                            # donne une note au film 14
    ]  # fmt: skip
    return InteractionMatrix.from_frame(pd.DataFrame(rows, columns=["userId", "movieId", "rating"]))


@pytest.fixture
def builder(matrix) -> RankingFeatureBuilder:
    components = {
        "a": _Fixed([5, 4, 3, 2, 1], "a").fit(matrix),
        "b": _Fixed([1, 2, 3, 4, 9], "b").fit(matrix),
    }
    hybrid = HybridRecommender(components, {"a": 1, "b": 1}, "rrf").fit(matrix)
    return RankingFeatureBuilder(
        hybrid, components, MOVIES, n_candidates=3, reference_year=2021.5
    ).fit(matrix)


def _column(builder, batch, name):
    return batch.features[:, :, builder.feature_names.index(name)]


def test_candidates_exclude_seen_movies_and_have_one_row_per_pair(builder, matrix):
    batch = builder.build(np.array([0, 1]))

    assert batch.items.shape == (2, 3)
    assert batch.features.shape == (2, 3, len(builder.feature_names))
    assert batch.flat_features.shape == (6, len(builder.feature_names))
    for row, user in enumerate(batch.user_idx):
        seen = set(matrix.ratings[user].indices)
        assert not seen & set(batch.items[row])
    hybrid_score = _column(builder, batch, "score_hybrid")
    assert (np.diff(hybrid_score, axis=1) <= 1e-9).all()  # candidats du meilleur au moins bon


def test_item_and_user_statistics_come_from_the_training_matrix(builder):
    batch = builder.build(np.array([0, 1, 2]))
    items = batch.items

    popularity = _column(builder, batch, "item_log_popularity")
    # Le film 10 (indice 0) a deux notes : 5.0 et 2.0 ; le film 14 n'en a qu'une.
    np.testing.assert_allclose(popularity[items == 0], np.log1p(2))
    np.testing.assert_allclose(popularity[items == 4], np.log1p(1))
    np.testing.assert_allclose(_column(builder, batch, "item_mean_rating")[items == 0], 3.5)
    np.testing.assert_allclose(_column(builder, batch, "item_liked_share")[items == 0], 0.5)
    np.testing.assert_allclose(_column(builder, batch, "user_mean_rating")[0], 4.5)
    np.testing.assert_allclose(_column(builder, batch, "user_liked_share")[1], 0.5)
    np.testing.assert_allclose(_column(builder, batch, "user_log_history")[2], np.log1p(1))


def test_genre_affinity_follows_the_liked_genres(builder):
    batch = builder.build(np.array([0, 1]))
    affinity, items = _column(builder, batch, "genre_affinity"), batch.items

    # Utilisateur 1 : films aimes [Sci-Fi] et [Sci-Fi, Action] -> profil Sci-Fi 2/3, Action 1/3.
    assert affinity[0][items[0] == 2][0] == pytest.approx(0.0)  # film de romance
    # Utilisateur 2 : n'a aime qu'un film de romance -> profil Romance 1.
    assert affinity[1][items[1] == 4][0] == pytest.approx(1.0)  # Romance + Drama
    assert affinity[1][items[1] == 1][0] == pytest.approx(0.0)  # Sci-Fi + Action


def test_missing_information_stays_missing(builder):
    batch = builder.build(np.array([0, 2]))
    age, gap = _column(builder, batch, "item_age"), _column(builder, batch, "year_gap")

    assert np.isnan(age[batch.items == 3]).all()  # film sans annee
    # L'age se mesure a la date de coupure, pas dans l'absolu : film de 2010, coupure mi-2021.
    np.testing.assert_allclose(age[0][batch.items[0] == 2], 11.5)
    assert np.isnan(gap[1]).all()  # l'utilisateur 3 n'a rien aime : pas d'annee de reference
    np.testing.assert_allclose(gap[0][batch.items[0] == 2], 2010 - 1995)


def test_component_scores_are_standardised_and_ranked(builder):
    batch = builder.build(np.array([2]))  # n'a vu que le film 13 (indice 3)
    score_b = _column(builder, batch, "score_b")[0]
    rank_b = _column(builder, batch, "log_rank_b")[0]

    best = batch.items[0] == 4  # le modele b prefere nettement le film d'indice 4
    assert score_b[best][0] == score_b.max()
    assert rank_b[best][0] == pytest.approx(0.0)  # log(1)
    assert builder.feature_names[:4] == ["score_a", "score_b", "log_rank_a", "log_rank_b"]


def test_build_requires_fit_and_valid_candidate_count(matrix):
    components = {"a": _Fixed([1, 2, 3, 4, 5], "a").fit(matrix)}
    hybrid = HybridRecommender(components, {"a": 1}).fit(matrix)
    with pytest.raises(RuntimeError, match="fit"):
        RankingFeatureBuilder(hybrid, components, MOVIES, 3).build(np.array([0]))
    with pytest.raises(ValueError):
        RankingFeatureBuilder(hybrid, components, MOVIES, 0)


def test_graded_labels_separate_unknown_from_disliked():
    items = np.array([[0, 1, 2], [0, 1, 2]])
    holdout = sp.csr_matrix(np.array([[4.5, 2.0, 0.0], [0.0, 4.0, 3.5]], dtype=np.float32))

    labels = graded_labels(items, holdout, np.array([0, 1]), threshold=4.0)

    assert labels.tolist() == [[2, 0, 1], [1, 2, 0]]
    assert binary_labels(labels).tolist() == [[1, 0, 0], [0, 1, 0]]


def test_group_ids_are_contiguous_and_sorted():
    qid = group_ids(3, 4)
    assert qid.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert (np.diff(qid) >= 0).all()


def _synthetic(n_users: int, n_candidates: int, seed: int):
    """La 2e variable determine la pertinence ; la 1re (l'ordre d'origine) est du bruit."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_users * n_candidates, 3)).astype(np.float32)
    y = (x[:, 1] > 0.8).astype(np.int32) * 2
    return x, y, group_ids(n_users, n_candidates)


def test_ranker_learns_the_informative_feature():
    x, y, qid = _synthetic(200, 20, 0)
    dev_x, dev_y, dev_qid = _synthetic(60, 20, 1)

    model = train_ranker(
        x, y, qid, (dev_x, dev_y, dev_qid), {"n_estimators": 60, "max_depth": 3}, seed=0
    )
    scores = model.predict(dev_x).reshape(60, 20)
    top = np.argsort(-scores, axis=1)[:, :3]
    precision = np.take_along_axis(dev_y.reshape(60, 20), top, axis=1).astype(bool).mean()

    assert precision > 0.8
    assert model.best_iteration >= 0


def test_ranker_is_reproducible_and_validates_its_inputs():
    x, y, qid = _synthetic(50, 10, 0)
    first = train_ranker(x, y, qid, None, {"n_estimators": 10}, seed=3).predict(x)
    second = train_ranker(x, y, qid, None, {"n_estimators": 10}, seed=3).predict(x)
    np.testing.assert_array_equal(first, second)

    with pytest.raises(ValueError, match="triees"):
        train_ranker(x, y, qid[::-1].copy(), None, {"n_estimators": 2})
    with pytest.raises(ValueError, match="meme nombre"):
        train_ranker(x, y[:-1], qid, None, {"n_estimators": 2})


def test_ranking_recommender_scores_only_candidates_and_reorders_them(builder, matrix):
    users = np.array([0, 1, 2])
    batch = builder.build(users)
    index = builder.feature_names.index("item_mean_rating")
    # Labels : un film est pertinent si sa note moyenne depasse 3.2 -> le ranker doit l'apprendre.
    well_rated = batch.features[:, :, index] > 3.2
    x = np.tile(batch.flat_features, (40, 1))
    y = np.tile(well_rated.astype(int).ravel() * 2, 40)
    qid = group_ids(40 * len(users), batch.items.shape[1])
    settings = {"n_estimators": 20, "max_depth": 2, "min_child_weight": 0}
    model = train_ranker(x, y, qid, None, settings)

    recommender = RankingRecommender(builder, model)
    scores = recommender.score(users)

    finite = np.isfinite(scores)
    assert finite.sum(axis=1).tolist() == [3, 3, 3]  # seuls les 3 candidats ont un score
    for row in range(len(users)):
        assert set(np.flatnonzero(finite[row])) == set(batch.items[row])
    items, _ = recommender.recommend(users, k=3)
    mean_rating = builder.item_mean_rating_[items] > 3.2
    # Les films bien notes passent devant les autres, pour chaque utilisateur.
    assert (np.diff(mean_rating.astype(int), axis=1) <= 0).all()
    assert well_rated.any() and not well_rated.all()
    importance = recommender.feature_importance()
    assert sum(importance.values()) == pytest.approx(1.0)


def test_ranking_recommender_round_trip_and_feature_subset(builder, tmp_path: Path):
    users = np.array([0, 1, 2])
    batch = builder.build(users)
    columns = [0, 1, 4]  # sous-ensemble de variables
    x = np.tile(batch.flat_features[:, columns], (20, 1))
    y = np.tile(np.array([2, 1, 0] * 3), 20)
    model = train_ranker(
        x, y, group_ids(60, 3), None, {"n_estimators": 5, "max_depth": 2, "min_child_weight": 0}
    )
    recommender = RankingRecommender(builder, model, columns, name="xgb_ranker_scores_only")

    recommender.save(tmp_path / "ranker")
    restored = RankingRecommender.from_artifacts(tmp_path / "ranker", builder)

    assert restored.name == "xgb_ranker_scores_only"
    assert restored.feature_names == ["score_a", "score_b", "score_hybrid"]
    np.testing.assert_allclose(restored.score(users), recommender.score(users), rtol=1e-5)
    with pytest.raises(FileNotFoundError):
        RankingRecommender.from_artifacts(tmp_path / "absent", builder)
    builder.n_candidates = 2
    with pytest.raises(ValueError, match="candidats"):
        RankingRecommender.from_artifacts(tmp_path / "ranker", builder)


def test_rating_table_is_empty_when_no_model_predicts_ratings():
    # Non-regression : un rapport sans aucun modele de prediction de note plantait.
    from recsys.analysis.model_report import rating_table, results_frame

    frame = results_frame([{"model": "xgb_ranker", "params": {"max_depth": 6}, "ndcg@10": 0.2}])
    assert rating_table(frame).empty
    with_rmse = results_frame(
        [
            {"model": "item_knn", "params": {}, "rmse": 0.9, "mae": 0.7},
            {"model": "popularity", "params": {}, "rmse": None, "mae": None},
        ]
    )
    assert rating_table(with_rmse)["modèle"].tolist() == ["Item-kNN"]
