"""La separation temporelle ne doit laisser passer aucune information du futur."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recsys.evaluation.split import build_fold, split_at, temporal_cutoffs


def _ratings(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "day"])
    frame["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(frame["day"], "D")
    return frame.drop(columns="day")


@pytest.fixture
def ratings() -> pd.DataFrame:
    return _ratings(
        [
            # periode d'entrainement (jours 1 a 10)
            (1, 10, 5.0, 1), (1, 20, 4.0, 2), (1, 30, 2.0, 3),
            (2, 10, 4.0, 4), (2, 20, 3.0, 5),
            (3, 10, 4.5, 6), (3, 30, 1.0, 7), (3, 40, 5.0, 8),
            # periode d'evaluation (jours 11 a 20)
            (1, 40, 5.0, 11),   # pertinent, film connu -> evalue
            (1, 50, 5.0, 12),   # film jamais vu a l'entrainement -> cold start film
            (2, 30, 2.0, 13),   # utilisateur connu mais aucune note pertinente
            (4, 10, 5.0, 14),   # utilisateur inconnu -> cold start utilisateur
            (3, 20, 4.0, 15),   # pertinent, film connu -> evalue
        ]
    )  # fmt: skip


CUTOFF = pd.Timestamp("2020-01-11", tz="UTC")


def test_split_is_strictly_temporal(ratings):
    train, holdout = split_at(ratings, CUTOFF)
    assert train["timestamp"].max() <= CUTOFF < holdout["timestamp"].min()
    assert len(train) + len(holdout) == len(ratings)


def test_split_with_end_bound_and_invalid_bounds(ratings):
    end = pd.Timestamp("2020-01-14", tz="UTC")
    _, holdout = split_at(ratings, CUTOFF, end)
    assert holdout["timestamp"].max() <= end
    with pytest.raises(ValueError):
        split_at(ratings, CUTOFF, CUTOFF)


def test_temporal_cutoffs_validate_quantiles(ratings):
    first, second = temporal_cutoffs(ratings, [0.5, 0.8])
    assert first < second
    for bad in ([0.8, 0.5], [0.0, 0.5], [0.5, 1.0]):
        with pytest.raises(ValueError):
            temporal_cutoffs(ratings, bad)


def test_fold_evaluates_only_warm_users_with_a_known_relevant_item(ratings):
    fold = build_fold(
        ratings, CUTOFF, None, min_movie_ratings=1, min_user_history=1, relevance_threshold=4.0
    )

    assert sorted(fold.matrix.user_ids[fold.eval_users]) == [1, 3]
    assert set(fold.holdout["movieId"]) == {40, 20}
    assert 50 not in fold.matrix.item_ids  # film apparu apres la coupure
    assert 4 not in fold.matrix.user_ids  # utilisateur apparu apres la coupure
    assert fold.stats["holdout_cold_users"] == 1
    assert fold.stats["eval_users"] == 2
    assert fold.stats["holdout_positive_share_on_known_items"] == pytest.approx(3 / 4)


def test_movie_filter_uses_training_period_only(ratings):
    # Le film 40 n'a qu'une note avant la coupure (et une apres) : avec un seuil
    # de 2 il doit etre exclu. Compter la note future serait une fuite.
    fold = build_fold(
        ratings, CUTOFF, None, min_movie_ratings=2, min_user_history=1, relevance_threshold=4.0
    )

    assert 40 not in fold.matrix.item_ids
    assert sorted(fold.matrix.item_ids) == [10, 20, 30]
    assert sorted(fold.matrix.user_ids[fold.eval_users]) == [3]


def test_min_user_history_excludes_light_users(ratings):
    fold = build_fold(
        ratings, CUTOFF, None, min_movie_ratings=1, min_user_history=3, relevance_threshold=4.0
    )
    history = np.diff(fold.matrix.ratings.indptr)[fold.eval_users]
    assert (history >= 3).all()
    assert sorted(fold.matrix.user_ids[fold.eval_users]) == [1, 3]


def test_training_matrix_contains_no_holdout_rating(ratings):
    fold = build_fold(
        ratings, CUTOFF, None, min_movie_ratings=1, min_user_history=1, relevance_threshold=4.0
    )
    rated = fold.matrix.ratings
    for row in fold.holdout.itertuples():
        assert rated[row.user_idx, row.item_idx] == 0
    assert rated.nnz == 8
