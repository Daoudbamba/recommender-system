"""Filtrage collaboratif par voisinage : item-based et user-based.

Les deux modeles travaillent sur les *residus* du modele de biais (note
observee moins moyenne, biais utilisateur et biais film). Deux films sont
proches si les memes utilisateurs les notent au-dessus ou en dessous de ce
qu'on attendait ; deux utilisateurs sont proches s'ils s'ecartent de la meme
facon sur les memes films.

Pour une paire (utilisateur u, film i), avec V les voisins retenus :

    numerateur   = somme sur V de similarite x residu
    denominateur = somme sur V de |similarite|

- classement : le numerateur seul (un film soutenu par beaucoup de voisins
  passe devant un film soutenu par un seul) ;
- note estimee : biais(u, i) + numerateur / denominateur.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative.baselines import BiasBaseline


def _unit_rows(matrix: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Retourne la matrice et la norme euclidienne de chaque ligne."""
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return matrix, norms.astype(np.float32)


def top_k_sparse(dense: np.ndarray, k: int) -> sp.csr_matrix:
    """Garde les `k` plus grandes valeurs strictement positives de chaque ligne."""
    k = min(k, dense.shape[1])
    cols = np.argpartition(-dense, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(dense, cols, axis=1)
    rows = np.repeat(np.arange(dense.shape[0]), k)
    keep = values.ravel() > 0
    return sp.csr_matrix(
        (values.ravel()[keep], (rows[keep], cols.ravel()[keep])), shape=dense.shape
    )


class _NeighbourhoodModel(Recommender):
    """Partie commune : residus, biais, passage numerateur / denominateur."""

    predicts_ratings = True

    def __init__(self, k: int, shrinkage: float, reg_bias: float, batch_size: int) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("k doit etre >= 1")
        self.k = k
        self.shrinkage = shrinkage
        self.reg_bias = reg_bias
        self.batch_size = batch_size
        self.baseline_: BiasBaseline | None = None
        self.residuals_: sp.csr_matrix | None = None
        self.observed_: sp.csr_matrix | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {"k": self.k, "shrinkage": self.shrinkage, "reg_bias": self.reg_bias}

    def _fit(self, matrix: InteractionMatrix) -> None:
        self.baseline_ = BiasBaseline(self.reg_bias, self.reg_bias).fit(matrix)
        self.residuals_ = self.baseline_.residuals(matrix)
        self.observed_ = matrix.ratings.astype(bool).astype(np.float32)
        self._fit_similarity()

    @abstractmethod
    def _fit_similarity(self) -> None: ...

    @abstractmethod
    def _num_den(self, user_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numerateur et denominateur denses (n_utilisateurs, n_films)."""

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._num_den(np.asarray(user_idx))[0]

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        user_idx, item_idx = np.asarray(user_idx), np.asarray(item_idx)
        out = self.baseline_.predict(user_idx, item_idx).astype(np.float32)
        unique_users, inverse = np.unique(user_idx, return_inverse=True)
        for start in range(0, len(unique_users), self.batch_size):
            batch = unique_users[start : start + self.batch_size]
            num, den = self._num_den(batch)
            mask = (inverse >= start) & (inverse < start + len(batch))
            rows, cols = inverse[mask] - start, item_idx[mask]
            den_values = den[rows, cols]
            # Sans voisin ayant note le film, on s'en tient au modele de biais.
            adjust = np.divide(
                num[rows, cols], den_values, out=np.zeros_like(den_values), where=den_values > 0
            )
            out[mask] += adjust
        return out


class ItemKNN(_NeighbourhoodModel):
    """Item-based : « ceux qui ont aime ce film ont aussi aime... ».

    La similarite entre films est calculee une fois a l'entrainement et ne
    garde que les `k` films les plus proches de chacun : la recommandation
    est ensuite un simple produit de matrices creuses.
    """

    name = "item_knn"

    def __init__(
        self,
        k: int = 50,
        shrinkage: float = 0.0,
        reg_bias: float = 10.0,
        batch_size: int = 512,
    ) -> None:
        super().__init__(k, shrinkage, reg_bias, batch_size)
        self.similarity_: sp.csr_matrix | None = None
        self.similarity_t_: sp.csr_matrix | None = None
        self.abs_similarity_t_: sp.csr_matrix | None = None

    def _fit_similarity(self) -> None:
        by_item, norms = _unit_rows(self.residuals_.T.tocsr())
        by_user = self.residuals_.tocsr()
        blocks = []
        for start in range(0, by_item.shape[0], self.batch_size):
            stop = min(start + self.batch_size, by_item.shape[0])
            dots = (by_item[start:stop] @ by_user).toarray()
            dots /= np.outer(norms[start:stop], norms) + self.shrinkage + 1e-9
            dots[np.arange(stop - start), np.arange(start, stop)] = 0.0  # pas d'auto-similarite
            blocks.append(top_k_sparse(dots, self.k))
        self.similarity_ = sp.vstack(blocks, format="csr")
        self._cache_transpose()

    def _cache_transpose(self) -> None:
        """Transposee de la similarite, calculee une fois pour servir les requetes."""
        self.similarity_t_ = self.similarity_.T.tocsr()
        self.abs_similarity_t_ = abs(self.similarity_t_)

    def _num_den(self, user_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num = (self.residuals_[user_idx] @ self.similarity_t_).toarray()
        den = (self.observed_[user_idx] @ self.abs_similarity_t_).toarray()
        return num, den

    def similar_items(self, item_idx: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Les `n` films les plus proches d'un film, avec leur similarite."""
        self._check_fitted()
        row = self.similarity_[item_idx]
        order = np.argsort(-row.data)[:n]
        return row.indices[order], row.data[order]

    def explain(self, user_idx: int, item_idx: int, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Films de l'historique qui poussent le plus la recommandation de `item_idx`.

        Retourne les indices de ces films et leur contribution au score
        (similarite x residu de la note que l'utilisateur leur a donnee).
        """
        self._check_fitted()
        neighbours = self.similarity_[item_idx]
        history = self.residuals_[user_idx]
        rated = dict(zip(history.indices, history.data, strict=True))
        pairs = [
            (item, sim * rated[item])
            for item, sim in zip(neighbours.indices, neighbours.data, strict=True)
            if item in rated and sim * rated[item] > 0
        ]
        pairs.sort(key=lambda pair: -pair[1])
        items = np.array([item for item, _ in pairs[:n]], dtype=int)
        return items, np.array([value for _, value in pairs[:n]], dtype=np.float32)


class UserKNN(_NeighbourhoodModel):
    """User-based : « les utilisateurs qui vous ressemblent ont aime... ».

    Rien n'est precalcule : les voisins d'un utilisateur sont cherches a la
    demande parmi tous les utilisateurs d'entrainement. C'est plus couteux a
    servir que l'item-based, ce que l'evaluation mesure.
    """

    name = "user_knn"

    def __init__(
        self,
        k: int = 100,
        shrinkage: float = 0.0,
        reg_bias: float = 10.0,
        batch_size: int = 256,
    ) -> None:
        super().__init__(k, shrinkage, reg_bias, batch_size)
        self.norms_: np.ndarray | None = None
        self.by_item_: sp.csr_matrix | None = None

    def _fit_similarity(self) -> None:
        _, self.norms_ = _unit_rows(self.residuals_)
        self.by_item_ = self.residuals_.T.tocsr()

    def neighbours(self, user_idx: np.ndarray) -> sp.csr_matrix:
        """Matrice creuse (n_utilisateurs_demandes, n_utilisateurs) des k voisins."""
        dots = (self.residuals_[user_idx] @ self.by_item_).toarray()
        dots /= np.outer(self.norms_[user_idx], self.norms_) + self.shrinkage + 1e-9
        dots[np.arange(len(user_idx)), user_idx] = 0.0  # un utilisateur n'est pas son voisin
        return top_k_sparse(dots, self.k)

    def _num_den(self, user_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        parts = [
            self._num_den_batch(user_idx[start : start + self.batch_size])
            for start in range(0, len(user_idx), self.batch_size)
        ]
        return np.vstack([p[0] for p in parts]), np.vstack([p[1] for p in parts])

    def _num_den_batch(self, user_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weights = self.neighbours(user_idx)
        num = (weights @ self.residuals_).toarray()
        den = (weights @ self.observed_).toarray()
        return num, den
