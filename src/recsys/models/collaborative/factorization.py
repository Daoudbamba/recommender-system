"""Factorisation de matrice par moindres carres alternes (ALS).

Deux variantes, parce qu'elles repondent a deux questions differentes :

- `ExplicitALS` predit une *note* : il factorise les residus du modele de
  biais sur les seules notes observees (implementation NumPy de ce projet) ;
- `ImplicitALS` predit une *preference* : il apprend sur le signal binaire
  « a aime / n'a pas interagi », avec la bibliotheque `implicit`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative.baselines import BiasBaseline


def _solve_side(by_row: sp.csr_matrix, other: np.ndarray, reg: float) -> np.ndarray:
    """Resout, pour chaque ligne, les moindres carres regularises face a `other`.

    Regularisation ponderee par le nombre de notes de la ligne (ALS-WR,
    Zhou et al. 2008) : un utilisateur a 20 notes et un utilisateur a 2 000
    notes sont regularises proportionnellement.
    """
    n_factors = other.shape[1]
    out = np.zeros((by_row.shape[0], n_factors), dtype=np.float64)
    eye = np.eye(n_factors)
    indptr, indices, data = by_row.indptr, by_row.indices, by_row.data
    for row in range(by_row.shape[0]):
        lo, hi = indptr[row], indptr[row + 1]
        if lo == hi:
            continue
        factors = other[indices[lo:hi]]
        gram = factors.T @ factors + reg * (hi - lo) * eye
        out[row] = np.linalg.solve(gram, factors.T @ data[lo:hi])
    return out


class ExplicitALS(Recommender):
    """Note estimee = biais(u, i) + p_u . q_i, appris sur les notes observees."""

    name = "als_explicit"
    predicts_ratings = True

    def __init__(
        self,
        factors: int = 32,
        reg: float = 0.05,
        iterations: int = 10,
        reg_bias: float = 10.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.reg = reg
        self.iterations = iterations
        self.reg_bias = reg_bias
        self.seed = seed
        self.baseline_: BiasBaseline | None = None
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None
        self.train_rmse_: list[float] = []

    @property
    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "reg": self.reg,
            "iterations": self.iterations,
            "reg_bias": self.reg_bias,
            "seed": self.seed,
        }

    def _fit(self, matrix: InteractionMatrix) -> None:
        self.baseline_ = BiasBaseline(self.reg_bias, self.reg_bias).fit(matrix)
        by_user = self.baseline_.residuals(matrix).astype(np.float64)
        by_item = by_user.T.tocsr()
        rng = np.random.default_rng(self.seed)
        items = rng.normal(0.0, 0.1, size=(matrix.n_items, self.factors))
        coo = by_user.tocoo()
        self.train_rmse_ = []
        for _ in range(self.iterations):
            users = _solve_side(by_user, items, self.reg)
            items = _solve_side(by_item, users, self.reg)
            error = coo.data - np.einsum("ij,ij->i", users[coo.row], items[coo.col])
            self.train_rmse_.append(float(np.sqrt(np.mean(error**2))))
        self.user_factors_ = users.astype(np.float32)
        self.item_factors_ = items.astype(np.float32)

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        dots = np.einsum("ij,ij->i", self.user_factors_[user_idx], self.item_factors_[item_idx])
        return self.baseline_.predict(user_idx, item_idx) + dots

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.baseline_.score(user_idx) + self.user_factors_[user_idx] @ self.item_factors_.T


class ImplicitALS(Recommender):
    """ALS pour retours implicites (Hu, Koren, Volinsky 2008).

    Signal d'entree : 1 si l'utilisateur a note le film au moins
    `relevance_threshold`, rien sinon. Une absence de note n'est pas traitee
    comme un avis negatif mais comme une preference inconnue de faible
    confiance ; les notes basses sont ecartees plutot que comptees comme des
    interactions positives.
    """

    name = "als_implicit"

    def __init__(
        self,
        factors: int = 64,
        reg: float = 0.01,
        alpha: float = 10.0,
        iterations: int = 15,
        relevance_threshold: float = 4.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.relevance_threshold = relevance_threshold
        self.seed = seed
        self.user_factors_: np.ndarray | None = None
        self.item_factors_: np.ndarray | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "reg": self.reg,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "relevance_threshold": self.relevance_threshold,
            "seed": self.seed,
        }

    def _fit(self, matrix: InteractionMatrix) -> None:
        from implicit.als import AlternatingLeastSquares
        from threadpoolctl import threadpool_limits

        confidence = matrix.ratings.copy()
        confidence.data = np.where(
            confidence.data >= self.relevance_threshold, self.alpha, 0.0
        ).astype(np.float32)
        confidence.eliminate_zeros()
        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.reg,
            iterations=self.iterations,
            random_state=self.seed,
        )
        # `implicit` parallelise lui-meme : on evite la sur-souscription des threads BLAS.
        with threadpool_limits(1, "blas"):
            model.fit(confidence, show_progress=False)
        self.user_factors_ = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors_ = np.asarray(model.item_factors, dtype=np.float32)

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.user_factors_[user_idx] @ self.item_factors_.T

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        """Calcule le vecteur latent d'un nouvel utilisateur sans toucher aux films.

        C'est l'etape « utilisateur » de l'ALS, appliquee a un seul utilisateur
        avec les facteurs de films figes (Hu, Koren, Volinsky 2008) :
        x = (YtY + Yt (C - I) Y + reg I)^-1 Yt C p, ou C vaut `alpha` sur les
        films aimes et 1 ailleurs.
        """
        self._check_fitted()
        items = self.item_factors_.astype(np.float64)
        gram = items.T @ items + self.reg * np.eye(items.shape[1])
        users = np.zeros((history.shape[0], items.shape[1]))
        for row in range(history.shape[0]):
            lo, hi = history.indptr[row], history.indptr[row + 1]
            liked = history.indices[lo:hi][history.data[lo:hi] >= self.relevance_threshold]
            if len(liked) == 0:
                continue
            factors = items[liked]
            system = gram + (self.alpha - 1.0) * factors.T @ factors
            users[row] = np.linalg.solve(system, self.alpha * factors.sum(axis=0))
        return (users @ items.T).astype(np.float32)

    def similar_items(self, item_idx: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Films les plus proches dans l'espace latent (similarite cosinus)."""
        self._check_fitted()
        unit = self.item_factors_ / (
            np.linalg.norm(self.item_factors_, axis=1, keepdims=True) + 1e-9
        )
        sims = unit @ unit[item_idx]
        sims[item_idx] = -np.inf
        order = np.argsort(-sims)[:n]
        return order, sims[order]
