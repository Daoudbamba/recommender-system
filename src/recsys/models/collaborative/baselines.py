"""Modeles de reference : sans eux, impossible de dire si un modele apporte quelque chose."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender


class PopularityRecommender(Recommender):
    """Recommande a tous les films ayant recu le plus de notes positives.

    Non personnalise : c'est le plancher que tout modele doit depasser, et le
    repli naturel pour un utilisateur sans historique.
    """

    name = "popularity"

    def __init__(self, relevance_threshold: float = 4.0) -> None:
        super().__init__()
        self.relevance_threshold = relevance_threshold
        self.item_scores_: np.ndarray | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {"relevance_threshold": self.relevance_threshold}

    def _fit(self, matrix: InteractionMatrix) -> None:
        positives = matrix.ratings.data >= self.relevance_threshold
        counts = np.bincount(matrix.ratings.indices[positives], minlength=matrix.n_items)
        self.item_scores_ = counts.astype(np.float32)

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.tile(self.item_scores_, (len(user_idx), 1))

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        self._check_fitted()
        return np.tile(self.item_scores_, (history.shape[0], 1))


class BiasBaseline(Recommender):
    """Note estimee = moyenne globale + biais utilisateur + biais film.

    Les biais sont amortis (`reg`) : un film note trois fois n'obtient pas un
    biais extreme. Ce modele capture le fait que certains utilisateurs notent
    haut et que certains films sont mieux notes, sans aucune personnalisation
    croisee. Les modeles kNN et ALS apprennent ce qu'il n'explique pas.
    """

    name = "bias_baseline"
    predicts_ratings = True

    def __init__(self, reg_item: float = 10.0, reg_user: float = 10.0) -> None:
        super().__init__()
        self.reg_item = reg_item
        self.reg_user = reg_user
        self.global_mean_: float = 0.0
        self.user_bias_: np.ndarray | None = None
        self.item_bias_: np.ndarray | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {"reg_item": self.reg_item, "reg_user": self.reg_user}

    def _fit(self, matrix: InteractionMatrix) -> None:
        coo = matrix.ratings.tocoo()
        self.global_mean_ = float(coo.data.mean())
        centered = coo.data - self.global_mean_
        item_sum = np.bincount(coo.col, weights=centered, minlength=matrix.n_items)
        item_count = np.bincount(coo.col, minlength=matrix.n_items)
        self.item_bias_ = (item_sum / (self.reg_item + item_count)).astype(np.float32)
        residual = centered - self.item_bias_[coo.col]
        user_sum = np.bincount(coo.row, weights=residual, minlength=matrix.n_users)
        user_count = np.bincount(coo.row, minlength=matrix.n_users)
        self.user_bias_ = (user_sum / (self.reg_user + user_count)).astype(np.float32)

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.global_mean_ + self.user_bias_[user_idx] + self.item_bias_[item_idx]

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.global_mean_ + self.user_bias_[user_idx][:, None] + self.item_bias_[None, :]

    def residuals(self, matrix: InteractionMatrix) -> sp.csr_matrix:
        """Notes observees moins la prediction du modele de biais (meme structure creuse)."""
        coo = matrix.ratings.tocoo()
        data = coo.data - self.predict(coo.row, coo.col)
        return sp.csr_matrix((data.astype(np.float32), (coo.row, coo.col)), shape=coo.shape)
