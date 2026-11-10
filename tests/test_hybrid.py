"""Hybride, re-classement par diversite et strategie de cold start."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.evaluation import metrics
from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative import ImplicitALS, ItemKNN, PopularityRecommender
from recsys.models.hybrid import (
    ColdStartRecommender,
    DiversifiedRecommender,
    HybridRecommender,
    mmr_select,
)
from recsys.models.hybrid.combiner import minmax_rows, normalise, reciprocal_ranks, zscore_rows


class _Fixed(Recommender):
    """Modele factice aux scores connus, identiques pour tous les utilisateurs."""

    def __init__(self, scores: list[float], name: str = "fixed") -> None:
        super().__init__()
        self.scores = np.array(scores, dtype=np.float32)
        self.name = name

    def _fit(self, matrix: InteractionMatrix) -> None: ...

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(user_idx), 1))

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        return np.tile(self.scores, (history.shape[0], 1))


@pytest.fixture
def matrix() -> InteractionMatrix:
    rows = [(1, 10, 5.0), (1, 11, 4.0), (2, 12, 3.0), (3, 13, 4.5), (3, 10, 2.0)]
    return InteractionMatrix.from_frame(pd.DataFrame(rows, columns=["userId", "movieId", "rating"]))


def test_row_normalisations():
    scores = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 2.0]], dtype=np.float32)

    np.testing.assert_allclose(minmax_rows(scores), [[0, 0.5, 1], [0, 0, 0]])
    z = zscore_rows(scores)
    np.testing.assert_allclose(z[0].mean(), 0, atol=1e-6)
    np.testing.assert_allclose(z[0].std(), 1, rtol=1e-5)
    np.testing.assert_allclose(z[1], 0)  # une ligne constante ne produit pas de NaN
    np.testing.assert_allclose(reciprocal_ranks(scores, 60)[0], [1 / 63, 1 / 62, 1 / 61])
    with pytest.raises(ValueError, match="strategie"):
        normalise(scores, "moyenne")


def test_rank_fusion_ignores_score_scale_but_weighted_does_not(matrix):
    # Le modele A est tres sur de lui pour le film 0 ; B prefere legerement le film 1.
    a = _Fixed([100.0, 1.0, 0.0, 0.0], "a").fit(matrix)
    b = _Fixed([0.50, 0.51, 0.0, 0.0], "b").fit(matrix)
    components = {"a": a, "b": b}
    users = np.array([0])

    weighted = HybridRecommender(components, {"a": 1, "b": 1}, "weighted").fit(matrix)
    rrf = HybridRecommender(components, {"a": 1, "b": 1}, "rrf").fit(matrix)

    assert weighted.score(users)[0].argmax() == 0
    rrf_scores = rrf.score(users)[0]
    assert rrf_scores[0] == pytest.approx(rrf_scores[1])  # rangs (1, 2) et (2, 1) : egalite


def test_hybrid_weights_are_normalised_and_zero_weight_components_are_skipped(matrix):
    class _Boom(_Fixed):
        def score(self, user_idx):
            raise AssertionError("un composant de poids nul ne doit pas etre appele")

    components = {"a": _Fixed([1, 0, 0, 0]).fit(matrix), "b": _Boom([0, 1, 0, 0]).fit(matrix)}
    hybrid = HybridRecommender(components, {"a": 3, "b": 0}).fit(matrix)

    assert hybrid.weights == {"a": 1.0}
    assert hybrid.score(np.array([0]))[0].argmax() == 0


def test_hybrid_rejects_invalid_configuration(matrix):
    components = {"a": _Fixed([1, 0, 0, 0])}
    with pytest.raises(KeyError, match="inconnu"):
        HybridRecommender(components, {"inconnu": 1})
    with pytest.raises(ValueError, match="poids"):
        HybridRecommender(components, {"a": -1})
    with pytest.raises(ValueError, match="poids"):
        HybridRecommender(components, {"a": 0})
    with pytest.raises(ValueError, match="strategie"):
        HybridRecommender(components, {"a": 1}, strategy="vote")


def test_hybrid_contributions_sum_to_the_score(matrix):
    components = {
        "a": _Fixed([4, 3, 2, 1], "a").fit(matrix),
        "b": _Fixed([1, 2, 3, 9], "b").fit(matrix),
    }
    hybrid = HybridRecommender(components, {"a": 2, "b": 1}, "zscore").fit(matrix)
    items = np.array([0, 3])

    parts = hybrid.contributions(0, items)

    np.testing.assert_allclose(
        parts["a"] + parts["b"], hybrid.score(np.array([0]))[0][items], rtol=1e-5
    )


def test_hybrid_saves_a_small_recipe_and_reloads_with_components(matrix, tmp_path: Path):
    components = {
        "a": _Fixed([4, 3, 2, 1], "a").fit(matrix),
        "b": _Fixed([1, 2, 3, 9], "b").fit(matrix),
    }
    hybrid = HybridRecommender(components, {"a": 2, "b": 1}, "rrf", name="hybrid_rrf").fit(matrix)

    path = hybrid.save(tmp_path / "hybrid.pkl")
    restored = HybridRecommender.from_spec(path, components)

    assert path.stat().st_size < 1000
    assert restored.name == "hybrid_rrf"
    np.testing.assert_array_equal(
        restored.recommend(np.array([0, 1]), 2)[0], hybrid.recommend(np.array([0, 1]), 2)[0]
    )
    with pytest.raises(KeyError, match="manquants"):
        HybridRecommender.from_spec(path, {"a": components["a"]})
    with pytest.raises(FileNotFoundError):
        HybridRecommender.from_spec(tmp_path / "absent.pkl", components)


def test_mmr_with_lambda_one_keeps_the_relevance_order():
    relevance = np.array([[0.9, 0.8, 0.7, 0.1]], dtype=np.float32)
    similarity = np.ones((1, 4, 4), dtype=np.float32)
    assert mmr_select(relevance, similarity, 3, lam=1.0).tolist() == [[0, 1, 2]]


def test_mmr_skips_a_near_duplicate_when_diversity_matters():
    # Les candidats 0 et 1 sont quasi identiques ; le 2 est different et presque aussi pertinent.
    relevance = np.array([[1.0, 0.95, 0.9]], dtype=np.float32)
    similarity = np.array([[[1, 0.99, 0.0], [0.99, 1, 0.0], [0.0, 0.0, 1]]], dtype=np.float32)

    assert mmr_select(relevance, similarity, 2, lam=1.0).tolist() == [[0, 1]]
    assert mmr_select(relevance, similarity, 2, lam=0.5).tolist() == [[0, 2]]


def test_intra_list_diversity_bounds():
    vectors = np.eye(3, dtype=np.float32)
    identical = np.array([[0, 0, 0]])
    distinct = np.array([[0, 1, 2]])
    assert metrics.intra_list_diversity(identical, vectors, 3)[0] == pytest.approx(0.0)
    assert metrics.intra_list_diversity(distinct, vectors, 3)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.intra_list_diversity(distinct, vectors, 1)


@pytest.fixture
def six_items() -> InteractionMatrix:
    rows = [
        (1, 100, 5.0),
        (2, 101, 4.0),
        (2, 102, 4.0),
        (3, 103, 3.0),
        (3, 104, 5.0),
        (3, 105, 2.0),
    ]
    return InteractionMatrix.from_frame(pd.DataFrame(rows, columns=["userId", "movieId", "rating"]))


def test_diversified_recommender_increases_diversity_and_never_returns_seen(six_items):
    # Films 0-2 quasi identiques et tres bien scores ; 3-5 differents et un peu moins bien.
    vectors = np.array(
        [[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01], [0, 1, 0], [0, 0, 1], [0.6, 0.6, 0.5]],
        dtype=np.float32,
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    base = _Fixed([1.0, 0.98, 0.96, 0.9, 0.88, 0.5]).fit(six_items)
    users = np.array([1])  # a deja vu les films 1 et 2

    plain = DiversifiedRecommender(base, vectors, lam=1.0, pool=6, list_size=3)
    diverse = DiversifiedRecommender(base, vectors, lam=0.3, pool=6, list_size=3)
    plain_items = plain.recommend(users, 3)[0]
    diverse_items = diverse.recommend(users, 3)[0]

    assert plain_items[0].tolist() == [0, 3, 4]  # ordre du modele, films vus exclus
    assert not {1, 2} & set(diverse_items[0])
    newcomer = np.array([0])  # n'a vu que le film 0
    plain_first = plain.recommend(newcomer, 3)[0]
    diverse_first = diverse.recommend(newcomer, 3)[0]
    ild_plain = metrics.intra_list_diversity(plain_first, vectors, 3)
    ild_diverse = metrics.intra_list_diversity(diverse_first, vectors, 3)
    assert ild_diverse[0] > ild_plain[0]


def test_popularity_penalty_demotes_the_most_rated_movie():
    rows = [(u, 100, 4.0) for u in range(1, 6)] + [(6, 101, 4.0), (7, 102, 4.0), (8, 103, 4.0)]
    matrix = InteractionMatrix.from_frame(
        pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    )
    vectors = np.eye(4, dtype=np.float32)
    # Le film 0 est a la fois le mieux score et, de loin, le plus populaire.
    base = _Fixed([1.0, 0.95, 0.5, 0.1]).fit(matrix)
    user = np.array([7])  # n'a vu que le film 3

    without = DiversifiedRecommender(base, vectors, lam=1.0, pool=4, list_size=2)
    penalised = DiversifiedRecommender(
        base, vectors, 1.0, popularity_penalty=0.5, pool=4, list_size=2
    )

    assert without.recommend(user, 1)[0][0, 0] == 0
    assert penalised.recommend(user, 1)[0][0, 0] == 1


def test_diversified_recommender_validates_its_settings(six_items):
    base = _Fixed([1, 2, 3, 4, 5, 6]).fit(six_items)
    vectors = np.eye(6, dtype=np.float32)
    with pytest.raises(ValueError, match="lam"):
        DiversifiedRecommender(base, vectors, lam=1.5)
    with pytest.raises(ValueError, match="list_size"):
        DiversifiedRecommender(base, vectors, pool=5, list_size=10)
    with pytest.raises(ValueError, match="une ligne par film"):
        DiversifiedRecommender(base, np.eye(4, dtype=np.float32), pool=4, list_size=2)


def test_cold_start_weight_decreases_with_history_length(matrix):
    model = ColdStartRecommender(
        _Fixed([0, 0, 0, 1], "perso").fit(matrix),
        _Fixed([1, 0, 0, 0], "pop").fit(matrix),
        prior_strength=3.0,
    )

    weights = model.popularity_weight(np.array([0, 1, 3, 30]))

    np.testing.assert_allclose(weights, [1.0, 0.75, 0.5, 3 / 33])
    assert (np.diff(weights) < 0).all()


def test_cold_start_uses_popularity_alone_without_history_then_personalises(matrix):
    model = ColdStartRecommender(
        _Fixed([0, 0, 0, 1], "perso").fit(matrix),
        _Fixed([1, 0, 0, 0], "pop").fit(matrix),
        prior_strength=3.0,
    )
    empty = sp.csr_matrix((1, 4), dtype=np.float32)
    long_history = sp.csr_matrix(np.array([[5.0, 4.0, 3.0, 0.0]] * 1, dtype=np.float32))
    many = sp.csr_matrix(np.full((1, 4), 4.0, dtype=np.float32))

    assert model.score_history(empty)[0].argmax() == 0  # popularite pure
    assert model.score_history(long_history)[0].argmax() in (0, 3)
    model.prior_strength = 0.5
    assert model.score_history(many)[0].argmax() == 3  # le modele personnalise domine


def _community_matrix() -> InteractionMatrix:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(40):
        liked = range(0, 6) if user < 20 else range(6, 12)
        for item in rng.choice(12, size=9, replace=False):
            rows.append((user, int(item), 4.5 if item in liked else 1.5))
    return InteractionMatrix.from_frame(pd.DataFrame(rows, columns=["userId", "movieId", "rating"]))


@pytest.mark.parametrize(
    "factory",
    [lambda: ItemKNN(k=5), lambda: ImplicitALS(factors=4, iterations=20, alpha=10)],
)
def test_models_serve_an_unknown_user_from_a_few_ratings(factory):
    matrix = _community_matrix()
    model = factory().fit(matrix)
    # Nouvel utilisateur : a aime les films 0 et 1 (premiere communaute), rejete le 6.
    history = sp.csr_matrix(
        (np.array([5.0, 4.5, 1.0], dtype=np.float32), ([0, 0, 0], [0, 1, 6])), shape=(1, 12)
    )

    scores = model.score_history(history)[0]
    scores[[0, 1, 6]] = -np.inf

    assert set(np.argsort(-scores)[:3]) <= set(range(2, 6))


def test_als_fold_in_matches_the_trained_user_vector():
    matrix = _community_matrix()
    model = ImplicitALS(factors=4, iterations=30, alpha=10).fit(matrix)
    users = np.arange(matrix.n_users)

    folded = model.score_history(matrix.ratings[users])
    trained = model.score(users)

    # Reconstituer un utilisateur d'entrainement a partir de ses notes redonne son classement.
    agreement = [
        len(set(np.argsort(-folded[u])[:4]) & set(np.argsort(-trained[u])[:4])) / 4 for u in users
    ]
    assert np.mean(agreement) > 0.9


def test_popularity_and_empty_history(matrix):
    popularity = PopularityRecommender().fit(matrix)
    empty = sp.csr_matrix((2, matrix.n_items), dtype=np.float32)

    scores = popularity.score_history(empty)

    assert scores.shape == (2, matrix.n_items)
    np.testing.assert_array_equal(scores[0], popularity.item_scores_)
    assert ImplicitALS(factors=2, iterations=2).fit(matrix).score_history(empty).any() is np.False_
