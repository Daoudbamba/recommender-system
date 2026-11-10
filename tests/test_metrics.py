"""Metriques verifiees sur des cas calcules a la main."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.evaluation import metrics

# Utilisateur A : 3 films pertinents, trouves aux rangs 1 et 3 d'un top-4.
# Utilisateur B : 1 film pertinent, trouve au rang 2.
# Utilisateur C : 2 films pertinents, aucun trouve.
HITS = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]], dtype=bool)
N_RELEVANT = np.array([3, 1, 2])


def test_rmse_and_mae():
    truth, predicted = np.array([4.0, 2.0, 5.0]), np.array([3.0, 2.0, 3.0])
    assert metrics.rmse(truth, predicted) == pytest.approx(np.sqrt(5 / 3))
    assert metrics.mae(truth, predicted) == pytest.approx(1.0)


def test_precision_recall_f1():
    np.testing.assert_allclose(metrics.precision_at_k(HITS, 4), [0.5, 0.25, 0.0])
    np.testing.assert_allclose(metrics.precision_at_k(HITS, 2), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(metrics.recall_at_k(HITS, N_RELEVANT, 4), [2 / 3, 1.0, 0.0])
    expected_f1_a = 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
    np.testing.assert_allclose(metrics.f1_at_k(HITS, N_RELEVANT, 4), [expected_f1_a, 0.4, 0.0])


def test_average_precision():
    # A : (1/1 + 2/3) / min(3, 4) ; B : (1/2) / min(1, 4)
    expected = [(1 + 2 / 3) / 3, 0.5, 0.0]
    np.testing.assert_allclose(metrics.average_precision_at_k(HITS, N_RELEVANT, 4), expected)


def test_ndcg():
    discounts = 1 / np.log2(np.arange(2, 6))
    ndcg_a = (discounts[0] + discounts[2]) / discounts[:3].sum()
    ndcg_b = discounts[1] / discounts[0]
    np.testing.assert_allclose(metrics.ndcg_at_k(HITS, N_RELEVANT, 4), [ndcg_a, ndcg_b, 0.0])


def test_perfect_ranking_scores_one_even_with_more_relevant_than_k():
    hits = np.ones((1, 5), dtype=bool)
    n_relevant = np.array([12])
    assert metrics.ndcg_at_k(hits, n_relevant, 5)[0] == pytest.approx(1.0)
    assert metrics.average_precision_at_k(hits, n_relevant, 5)[0] == pytest.approx(1.0)
    assert metrics.precision_at_k(hits, 5)[0] == pytest.approx(1.0)


def test_catalog_coverage():
    recommended = np.array([[0, 1, 2], [0, 1, 3]])
    assert metrics.catalog_coverage(recommended, n_items=10, k=3) == pytest.approx(0.4)
    assert metrics.catalog_coverage(recommended, n_items=10, k=2) == pytest.approx(0.2)


def test_ranking_metrics_rejects_k_larger_than_recommendations():
    with pytest.raises(ValueError, match="depasse"):
        metrics.ranking_metrics(HITS, N_RELEVANT, [5])


def test_ranking_metrics_returns_every_metric_for_every_k():
    out = metrics.ranking_metrics(HITS, N_RELEVANT, [2, 4])
    assert set(out) == {
        f"{name}@{k}" for name in ("precision", "recall", "f1", "map", "ndcg") for k in (2, 4)
    }
    assert all(values.shape == (3,) for values in out.values())
