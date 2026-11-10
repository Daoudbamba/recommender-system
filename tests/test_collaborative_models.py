"""Comportement des modeles collaboratifs sur un jeu jouet a deux communautes.

Utilisateurs 0-5 : fans de science-fiction (films 0-3 aimes, films 4-7 rejetes).
Utilisateurs 6-11 : fans de romance (l'inverse).
Chaque utilisateur a un film de son genre qu'il n'a pas encore note : un bon
modele collaboratif doit le lui recommander en premier.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.models.base import InteractionMatrix, Recommender, top_k
from recsys.models.collaborative import (
    MODEL_REGISTRY,
    BiasBaseline,
    ExplicitALS,
    ImplicitALS,
    ItemKNN,
    PopularityRecommender,
    UserKNN,
)

N_USERS, N_ITEMS = 12, 8


def _toy_frame() -> tuple[pd.DataFrame, dict[int, int]]:
    rng = np.random.default_rng(0)
    rows, held_out = [], {}
    for user in range(N_USERS):
        liked = range(0, 4) if user < 6 else range(4, 8)
        missing = list(liked)[user % 4]
        held_out[user] = missing
        for item in range(N_ITEMS):
            if item == missing:
                continue
            base = 4.5 if item in liked else 1.5
            rows.append(
                (user + 100, item + 1000, float(np.clip(base + rng.normal(0, 0.3), 0.5, 5)))
            )
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"]), held_out


@pytest.fixture(scope="module")
def toy() -> tuple[InteractionMatrix, dict[int, int]]:
    frame, held_out = _toy_frame()
    return InteractionMatrix.from_frame(frame), held_out


PERSONALISED = [
    lambda: ItemKNN(k=3),
    lambda: UserKNN(k=3),
    lambda: ExplicitALS(factors=2, reg=0.05, iterations=15, reg_bias=1.0),
    lambda: ImplicitALS(factors=4, reg=0.01, alpha=10, iterations=20),
]


def test_interaction_matrix_maps_ids_and_flags_unknown(toy):
    matrix, _ = toy
    assert (matrix.n_users, matrix.n_items) == (N_USERS, N_ITEMS)
    assert matrix.ratings.nnz == N_USERS * (N_ITEMS - 1)
    assert matrix.user_index([100, 111, 999]).tolist() == [0, 11, -1]
    assert matrix.item_index(1007)[0] == 7
    assert matrix.item_index(42)[0] == -1


def test_top_k_orders_scores():
    items, scores = top_k(np.array([[0.1, 0.9, 0.5, 0.7]]), 3)
    assert items.tolist() == [[1, 3, 2]]
    assert scores.tolist() == [[0.9, 0.7, 0.5]]


@pytest.mark.parametrize("factory", PERSONALISED)
def test_personalised_models_recommend_the_missing_liked_movie(toy, factory):
    matrix, held_out = toy
    model = factory().fit(matrix)
    users = np.arange(N_USERS)

    items, _ = model.recommend(users, k=1)

    expected = [held_out[user] for user in users]
    assert items[:, 0].tolist() == expected


@pytest.mark.parametrize("factory", [*PERSONALISED, PopularityRecommender, BiasBaseline])
def test_recommend_never_returns_an_already_rated_movie(toy, factory):
    matrix, held_out = toy
    model = factory().fit(matrix)
    users = np.arange(N_USERS)

    items, scores = model.recommend(users, k=1)

    # Un seul film reste non note par utilisateur : c'est le seul candidat possible.
    assert items[:, 0].tolist() == [held_out[user] for user in users]
    assert np.isfinite(scores).all()


@pytest.mark.parametrize(
    "factory",
    [
        BiasBaseline,
        lambda: ItemKNN(k=3),
        lambda: UserKNN(k=3),
        lambda: ExplicitALS(factors=2, iterations=15, reg_bias=1.0),
    ],
)
def test_rating_models_predict_high_for_liked_genre_and_low_for_other(toy, factory):
    matrix, held_out = toy
    model = factory().fit(matrix)
    users = np.arange(N_USERS)
    liked = np.array([held_out[user] for user in users])

    predictions = model.predict(users, liked)

    assert model.predicts_ratings
    assert predictions.shape == (N_USERS,)
    if not isinstance(model, BiasBaseline):  # le modele de biais ignore les gouts croises
        assert (predictions > 3.5).all()


def test_bias_baseline_matches_hand_computation():
    frame = pd.DataFrame({"userId": [1, 1, 2], "movieId": [10, 20, 10], "rating": [5.0, 3.0, 4.0]})
    model = BiasBaseline(reg_item=0.0, reg_user=0.0).fit(InteractionMatrix.from_frame(frame))

    assert model.global_mean_ == pytest.approx(4.0)
    # film 10 : moyenne des ecarts (1, 0) = 0.5 ; film 20 : -1
    np.testing.assert_allclose(model.item_bias_, [0.5, -1.0])
    # utilisateur 1 : ecarts residuels (0.5, 0) -> 0.25 ; utilisateur 2 : -0.5
    np.testing.assert_allclose(model.user_bias_, [0.25, -0.5])
    assert model.predict(np.array([1]), np.array([1]))[0] == pytest.approx(4.0 - 0.5 - 1.0)


def test_bias_regularisation_shrinks_biases_towards_zero(toy):
    matrix, _ = toy
    loose = BiasBaseline(reg_item=0.0, reg_user=0.0).fit(matrix)
    tight = BiasBaseline(reg_item=50.0, reg_user=50.0).fit(matrix)
    assert np.abs(tight.item_bias_).max() < np.abs(loose.item_bias_).max()


def test_popularity_ranks_by_number_of_positive_ratings():
    frame = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 3],
            "movieId": [10, 10, 10, 20, 20, 30],
            "rating": [5.0, 4.0, 2.0, 4.0, 1.0, 1.0],
        }
    )
    model = PopularityRecommender(relevance_threshold=4.0).fit(InteractionMatrix.from_frame(frame))
    assert model.item_scores_.tolist() == [2.0, 1.0, 0.0]
    assert not model.predicts_ratings
    with pytest.raises(NotImplementedError):
        model.predict(np.array([0]), np.array([0]))


def test_item_knn_neighbours_stay_in_the_same_community(toy):
    matrix, _ = toy
    model = ItemKNN(k=3).fit(matrix)

    for item in range(N_ITEMS):
        neighbours, similarities = model.similar_items(item, n=3)
        same_side = (neighbours < 4) == (item < 4)
        assert same_side.all()
        assert (np.diff(similarities) <= 0).all()
        assert item not in neighbours
    assert model.similarity_.getnnz(axis=1).max() <= 3


def test_user_knn_neighbours_stay_in_the_same_community(toy):
    matrix, _ = toy
    model = UserKNN(k=3).fit(matrix)

    weights = model.neighbours(np.arange(N_USERS))

    for user in range(N_USERS):
        neighbours = weights[user].indices
        assert user not in neighbours
        assert ((neighbours < 6) == (user < 6)).all()


def test_knn_prediction_falls_back_to_bias_without_neighbours():
    # Deux blocs sans aucun utilisateur en commun : le film 30 n'a pas de voisin
    # note par l'utilisateur 1.
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 10, 20, 30, 40],
            "rating": [5.0, 4.0, 4.0, 5.0, 2.0, 3.0],
        }
    )
    matrix = InteractionMatrix.from_frame(frame)
    model = ItemKNN(k=2).fit(matrix)
    user, item = np.array([0]), np.array([2])

    assert model.predict(user, item)[0] == pytest.approx(model.baseline_.predict(user, item)[0])


def test_explicit_als_training_error_decreases_and_is_reproducible(toy):
    matrix, _ = toy
    first = ExplicitALS(factors=2, iterations=8, reg_bias=1.0, seed=7).fit(matrix)
    second = ExplicitALS(factors=2, iterations=8, reg_bias=1.0, seed=7).fit(matrix)

    assert first.train_rmse_[-1] < first.train_rmse_[0]
    assert all(
        b <= a + 1e-9 for a, b in zip(first.train_rmse_, first.train_rmse_[1:], strict=False)
    )
    np.testing.assert_array_equal(first.user_factors_, second.user_factors_)


def test_implicit_als_ignores_low_ratings_as_positive_signal(toy):
    matrix, _ = toy
    model = ImplicitALS(factors=4, iterations=20, alpha=10).fit(matrix)

    scores = model.score(np.array([0]))[0]  # fan de science-fiction
    assert scores[:4].min() > scores[4:].max()
    neighbours, similarities = model.similar_items(0, n=3)
    assert 0 not in neighbours
    assert (np.diff(similarities) <= 0).all()


def test_unfitted_model_and_out_of_range_user_raise(toy):
    matrix, _ = toy
    with pytest.raises(RuntimeError, match="fit"):
        ItemKNN().recommend(np.array([0]))
    with pytest.raises(IndexError):
        PopularityRecommender().fit(matrix).recommend(np.array([-1]))
    with pytest.raises(ValueError):
        UserKNN(k=0)


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_save_and_load_round_trip_gives_identical_recommendations(toy, tmp_path: Path, name):
    matrix, _ = toy
    params = {"k": 3} if "knn" in name else {"factors": 2} if "als" in name else {}
    model = MODEL_REGISTRY[name](**params).fit(matrix)
    path = model.save(tmp_path / f"{name}.pkl")

    restored = type(model).load(path)

    users = np.arange(N_USERS)
    np.testing.assert_array_equal(model.recommend(users, 3)[0], restored.recommend(users, 3)[0])
    assert restored.params == model.params


def test_load_rejects_missing_file_and_wrong_model_type(toy, tmp_path: Path):
    matrix, _ = toy
    with pytest.raises(FileNotFoundError):
        Recommender.load(tmp_path / "absent.pkl")
    path = PopularityRecommender().fit(matrix).save(tmp_path / "pop.pkl")
    with pytest.raises(TypeError):
        ItemKNN.load(path)


def test_item_knn_explanation_points_to_liked_movies_of_the_same_community(toy):
    matrix, held_out = toy
    model = ItemKNN(k=3).fit(matrix)
    user = 0  # fan de science-fiction

    items, contributions = model.explain(user, held_out[user], n=2)

    assert len(items) >= 1
    assert (items < 4).all()
    assert (contributions > 0).all()
    assert (np.diff(contributions) <= 0).all()
    assert all(matrix.ratings[user, item] > 0 for item in items)
