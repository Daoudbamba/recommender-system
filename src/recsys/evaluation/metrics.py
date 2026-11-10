"""Metriques de prediction de note et de classement Top-K.

Les metriques de classement prennent une matrice booleenne `hits` de forme
(n_utilisateurs, K_max) : hits[u, r] vaut True si le film recommande au rang
r a l'utilisateur u fait partie de ses films pertinents. Elles retournent
une valeur par utilisateur ; la moyenne est faite par l'appelant, ce qui
permet aussi l'analyse d'erreurs par segment.
"""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erreur absolue moyenne."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_at_k(hits: np.ndarray, k: int) -> np.ndarray:
    """Part des k recommandations qui sont pertinentes."""
    return hits[:, :k].sum(axis=1) / k


def recall_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Part des films pertinents de l'utilisateur retrouves dans le top-k."""
    return hits[:, :k].sum(axis=1) / np.maximum(n_relevant, 1)


def f1_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Moyenne harmonique de la precision et du rappel a k."""
    precision, recall = precision_at_k(hits, k), recall_at_k(hits, n_relevant, k)
    total = precision + recall
    return np.divide(2 * precision * recall, total, out=np.zeros_like(total), where=total > 0)


def average_precision_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """Precision moyenne a k ; sa moyenne sur les utilisateurs est le MAP@k.

    Normalisee par min(k, nombre de films pertinents) : un classement parfait
    vaut 1 meme quand l'utilisateur a plus de k films pertinents.
    """
    top = hits[:, :k].astype(np.float64)
    precision_at_rank = np.cumsum(top, axis=1) / np.arange(1, top.shape[1] + 1)
    best = np.minimum(n_relevant, k)
    return (precision_at_rank * top).sum(axis=1) / np.maximum(best, 1)


def ndcg_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    """NDCG a k avec gains binaires : recompense les bons films places haut."""
    top = hits[:, :k].astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, top.shape[1] + 2))
    dcg = (top * discounts).sum(axis=1)
    ideal = np.concatenate([[0.0], np.cumsum(discounts)])
    idcg = ideal[np.minimum(n_relevant, top.shape[1]).astype(int)]
    return np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)


def catalog_coverage(recommended: np.ndarray, n_items: int, k: int) -> float:
    """Part du catalogue recommandee au moins une fois dans les top-k."""
    return float(len(np.unique(recommended[:, :k])) / n_items)


def ranking_metrics(
    hits: np.ndarray, n_relevant: np.ndarray, ks: list[int]
) -> dict[str, np.ndarray]:
    """Toutes les metriques de classement, par utilisateur, pour chaque k."""
    if max(ks) > hits.shape[1]:
        raise ValueError(f"k={max(ks)} depasse le nombre de recommandations ({hits.shape[1]})")
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(hits, k)
        out[f"recall@{k}"] = recall_at_k(hits, n_relevant, k)
        out[f"f1@{k}"] = f1_at_k(hits, n_relevant, k)
        out[f"map@{k}"] = average_precision_at_k(hits, n_relevant, k)
        out[f"ndcg@{k}"] = ndcg_at_k(hits, n_relevant, k)
    return out
