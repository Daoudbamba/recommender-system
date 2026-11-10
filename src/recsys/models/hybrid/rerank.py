"""Re-classement : diversite (MMR) et frein a la popularite.

Un bon score de pertinence ne suffit pas : dix films presque identiques, ou
dix succes que tout le monde connait, font une mauvaise page d'accueil. Le
re-classement part des meilleurs candidats d'un modele et les reordonne.

MMR (Maximal Marginal Relevance, Carbonell & Goldstein 1998) choisit les
films un par un en maximisant

    lambda x pertinence - (1 - lambda) x similarite max avec les films deja choisis

`lambda = 1` rend le classement d'origine ; plus `lambda` baisse, plus la
liste est variee, au prix de la pertinence. Le compromis est mesure, pas
suppose (voir reports/hybrid_report.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from recsys.models.base import InteractionMatrix, Recommender, top_k
from recsys.models.hybrid.combiner import minmax_rows


def mmr_select(relevance: np.ndarray, similarity: np.ndarray, k: int, lam: float) -> np.ndarray:
    """Ordre MMR des `k` premiers candidats, pour un lot d'utilisateurs.

    relevance : (n, p) pertinence de chaque candidat, de preference dans [0, 1] ;
    similarity : (n, p, p) similarite entre candidats d'un meme utilisateur.
    Retourne (n, k) positions dans la liste des candidats.
    """
    n, p = relevance.shape
    k = min(k, p)
    rows = np.arange(n)
    available = np.ones((n, p), dtype=bool)
    closest = np.zeros((n, p), dtype=np.float32)
    chosen = np.empty((n, k), dtype=np.int64)
    for step in range(k):
        value = lam * relevance - (1.0 - lam) * closest
        value[~available] = -np.inf
        pick = value.argmax(axis=1)
        chosen[:, step] = pick
        available[rows, pick] = False
        closest = np.maximum(closest, similarity[rows, pick, :])
    return chosen


class DiversifiedRecommender(Recommender):
    """Enveloppe un modele et re-classe ses meilleurs candidats."""

    name = "diversified"

    def __init__(
        self,
        base: Recommender,
        item_vectors: np.ndarray,
        lam: float = 0.8,
        popularity_penalty: float = 0.0,
        pool: int = 50,
        list_size: int = 20,
    ) -> None:
        super().__init__()
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam doit etre compris entre 0 et 1")
        if popularity_penalty < 0:
            raise ValueError("popularity_penalty doit etre positif ou nul")
        if list_size > pool:
            raise ValueError("list_size ne peut pas depasser pool")
        self.base = base
        self.item_vectors = np.asarray(item_vectors, dtype=np.float32)
        self.lam = lam
        self.popularity_penalty = popularity_penalty
        self.pool = pool
        self.list_size = list_size
        self.popularity_rank_: np.ndarray | None = None
        if base.matrix is not None:
            self.fit(base.matrix)

    @property
    def params(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "popularity_penalty": self.popularity_penalty,
            "pool": self.pool,
            "base": self.base.name,
        }

    def _fit(self, matrix: InteractionMatrix) -> None:
        if self.base.matrix is not matrix:
            self.base.fit(matrix)
        if self.item_vectors.shape[0] != matrix.n_items:
            raise ValueError("item_vectors doit avoir une ligne par film du catalogue")
        counts = np.diff(matrix.ratings.tocsc().indptr)
        # Rang de popularite dans [0, 1] : 1 pour le film le plus note.
        self.popularity_rank_ = (counts.argsort().argsort() / max(len(counts) - 1, 1)).astype(
            np.float32
        )

    def rerank(self, scores: np.ndarray) -> np.ndarray:
        """Transforme des scores (films vus deja a -inf) en scores re-classes."""
        pool = min(self.pool, scores.shape[1])
        items, pool_scores = top_k(scores, pool)
        finite = np.isfinite(pool_scores)
        relevance = minmax_rows(np.where(finite, pool_scores, pool_scores[finite].min()))
        relevance -= self.popularity_penalty * self.popularity_rank_[items]
        relevance[~finite] = -np.inf
        vectors = self.item_vectors[items]
        similarity = vectors @ vectors.transpose(0, 2, 1)
        order = mmr_select(relevance, similarity, self.list_size, self.lam)

        out = np.full(scores.shape, -np.inf, dtype=np.float32)
        rows = np.arange(scores.shape[0])[:, None]
        # Les candidats non retenus gardent un score fini mais inferieur a toute la liste.
        out[rows, items] = np.where(finite, relevance - 10.0, -np.inf)
        ranks = np.arange(order.shape[1], 0, -1, dtype=np.float32)
        out[rows, np.take_along_axis(items, order, axis=1)] = ranks
        return out

    def _mask_seen(self, scores: np.ndarray, seen_rows: np.ndarray, seen_cols: np.ndarray):
        scores = np.array(scores, dtype=np.float32)
        scores[seen_rows, seen_cols] = -np.inf
        return scores

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        matrix = self._check_fitted()
        rows, cols = matrix.ratings[np.asarray(user_idx)].nonzero()
        return self.rerank(self._mask_seen(self.base.score(user_idx), rows, cols))

    def score_history(self, history) -> np.ndarray:
        self._check_fitted()
        rows, cols = history.nonzero()
        return self.rerank(self._mask_seen(self.base.score_history(history), rows, cols))
