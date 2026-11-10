"""Evaluation d'un modele entraine sur un `Fold`.

Classement : pour chaque utilisateur evalue, le modele classe tous les films
du catalogue d'entrainement qu'il n'a pas encore notes ; on regarde ou
tombent les films qu'il a reellement aimes apres la coupure.

Limite connue : un film recommande que l'utilisateur n'a jamais note compte
comme une erreur, alors qu'il l'aurait peut-etre aime. Les valeurs absolues
sont donc des bornes basses ; ce sont les ecarts entre modeles, mesures dans
les memes conditions, qui sont interpretables.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.evaluation import metrics
from recsys.evaluation.split import Fold
from recsys.models.base import Recommender, top_k


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    per_user: dict[str, np.ndarray]
    recommended: np.ndarray  # (n_utilisateurs_evalues, K_max) indices de films


def _relevance_matrix(fold: Fold) -> sp.csr_matrix:
    relevant = fold.relevant
    data = np.ones(len(relevant), dtype=bool)
    shape = (fold.matrix.n_users, fold.matrix.n_items)
    return sp.csr_matrix((data, (relevant["user_idx"], relevant["item_idx"])), shape=shape)


def recommend_for_eval_users(model: Recommender, fold: Fold, k: int, batch_size: int) -> np.ndarray:
    """Top-k de chaque utilisateur evalue, films deja notes exclus."""
    seen = fold.matrix.ratings
    out = np.empty((len(fold.eval_users), k), dtype=np.int32)
    for start in range(0, len(fold.eval_users), batch_size):
        batch = fold.eval_users[start : start + batch_size]
        scores = np.array(model.score(batch), dtype=np.float32)
        rows, cols = seen[batch].nonzero()
        scores[rows, cols] = -np.inf
        out[start : start + len(batch)] = top_k(scores, k)[0]
    return out


def evaluate_ranking(
    model: Recommender, fold: Fold, ks: list[int], batch_size: int = 256
) -> EvaluationResult:
    """Metriques Top-K moyennes, valeurs par utilisateur et recommandations."""
    k_max = max(ks)
    recommended = recommend_for_eval_users(model, fold, k_max, batch_size)
    relevance = _relevance_matrix(fold)[fold.eval_users]
    n_relevant = np.asarray(relevance.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(len(fold.eval_users)), k_max)
    hits = np.asarray(relevance[rows, recommended.ravel()]).reshape(len(fold.eval_users), k_max)

    per_user = metrics.ranking_metrics(hits, n_relevant, ks)
    summary = {name: float(values.mean()) for name, values in per_user.items()}
    popularity = np.diff(fold.matrix.ratings.tocsc().indptr)
    for k in ks:
        summary[f"coverage@{k}"] = metrics.catalog_coverage(recommended, fold.matrix.n_items, k)
        summary[f"mean_popularity@{k}"] = float(popularity[recommended[:, :k]].mean())
    return EvaluationResult(summary, per_user, recommended)


def evaluate_rating(model: Recommender, fold: Fold, rating_min: float, rating_max: float):
    """RMSE et MAE sur les notes d'evaluation (utilisateurs et films connus)."""
    if not model.predicts_ratings:
        return {}
    holdout = fold.holdout
    predicted = model.predict(holdout["user_idx"].to_numpy(), holdout["item_idx"].to_numpy())
    predicted = np.clip(predicted, rating_min, rating_max)
    truth = holdout["rating"].to_numpy()
    return {"rmse": metrics.rmse(truth, predicted), "mae": metrics.mae(truth, predicted)}


def evaluate(
    model: Recommender,
    fold: Fold,
    ks: list[int],
    rating_bounds: tuple[float, float],
    batch_size: int = 256,
) -> tuple[dict[str, Any], EvaluationResult]:
    """Evaluation complete d'un modele deja entraine, avec temps de calcul."""
    start = time.perf_counter()
    ranking = evaluate_ranking(model, fold, ks, batch_size)
    ranking_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rating = evaluate_rating(model, fold, *rating_bounds)
    summary = {
        **ranking.metrics,
        **rating,
        "ranking_seconds": ranking_seconds,
        "rating_seconds": time.perf_counter() - start,
        "ms_per_user": 1000 * ranking_seconds / max(len(fold.eval_users), 1),
    }
    return summary, ranking
