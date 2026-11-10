"""Recommandation par contenu : proposer des films proches de ceux qu'on a aimes.

Le meme modele sert pour TF-IDF (vecteurs creux) et pour les embeddings
(vecteurs denses) : seule la representation des films change, ce qui rend
la comparaison des deux approches equitable.

Deux facons d'agreger l'historique d'un utilisateur :

- `centroid` : on moyenne les vecteurs des films aimes en un « profil », puis
  on classe les films par similarite cosinus avec ce profil. Simple, mais le
  profil d'un utilisateur eclectique devient flou ;
- `knn` : chaque film candidat recoit la somme de ses similarites avec les
  films aimes parmi ses `k` plus proches voisins. Les gouts multiples d'un
  utilisateur restent distincts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.features.item_features import ItemFeatures
from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative.knn import top_k_sparse

AGGREGATIONS = ("centroid", "knn")


class ContentBasedRecommender(Recommender):
    """Recommande selon la similarite de contenu avec les films aimes."""

    name = "content_based"

    def __init__(
        self,
        features: ItemFeatures,
        aggregation: str = "knn",
        k: int = 50,
        relevance_threshold: float = 4.0,
        batch_size: int = 1024,
        name: str | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation doit valoir l'une de {AGGREGATIONS}")
        self.features = features
        self.aggregation = aggregation
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.batch_size = batch_size
        self.extra_params = extra_params or {}
        if name:
            self.name = name
        self.item_vectors_: sp.csr_matrix | np.ndarray | None = None
        self.liked_: sp.csr_matrix | None = None
        self.similarity_: sp.csr_matrix | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {**self.extra_params, "aggregation": self.aggregation, "k": self.k}

    def _fit(self, matrix: InteractionMatrix) -> None:
        self.item_vectors_ = self.features.align(matrix.item_ids).matrix
        liked = matrix.ratings.copy()
        liked.data = (liked.data >= self.relevance_threshold).astype(np.float32)
        liked.eliminate_zeros()
        self.liked_ = liked
        if self.aggregation == "knn":
            self.similarity_ = self._neighbour_matrix()

    def _dots(self, rows: sp.csr_matrix | np.ndarray) -> np.ndarray:
        """Produit scalaire dense entre des vecteurs et tous les films."""
        product = rows @ self.item_vectors_.T
        return product.toarray() if sp.issparse(product) else np.asarray(product)

    def _neighbour_matrix(self) -> sp.csr_matrix:
        blocks = []
        n_items = self.item_vectors_.shape[0]
        for start in range(0, n_items, self.batch_size):
            stop = min(start + self.batch_size, n_items)
            dots = self._dots(self.item_vectors_[start:stop]).astype(np.float32)
            dots[np.arange(stop - start), np.arange(start, stop)] = 0.0
            blocks.append(top_k_sparse(dots, self.k))
        return sp.vstack(blocks, format="csr")

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._aggregate(self.liked_[np.asarray(user_idx)])

    def _aggregate(self, liked: sp.csr_matrix) -> np.ndarray:
        if self.aggregation == "knn":
            return (liked @ self.similarity_.T).toarray()
        return self._dots(liked @ self.item_vectors_)

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        self._check_fitted()
        liked = history.copy().tocsr()
        liked.data = (liked.data >= self.relevance_threshold).astype(np.float32)
        liked.eliminate_zeros()
        return self._aggregate(liked)

    def score_new_items(self, user_idx: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Scores de films absents du catalogue d'entrainement (cold start film).

        Un film que le modele n'a jamais vu est rapproche, par son seul contenu, des films
        que l'utilisateur a aimes : somme des similarites avec ses `k` plus
        proches voisins du catalogue. Retourne (n_utilisateurs, n_films_demandes).
        """
        self._check_fitted()
        new_vectors = self.features.align(np.asarray(movie_ids)).matrix
        blocks = []
        for start in range(0, new_vectors.shape[0], self.batch_size):
            dots = self._dots(new_vectors[start : start + self.batch_size]).astype(np.float32)
            blocks.append(top_k_sparse(dots, self.k))
        neighbours = sp.vstack(blocks, format="csr")
        return (self.liked_[np.asarray(user_idx)] @ neighbours.T).toarray()

    def similar_items(self, item_idx: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Les `n` films au contenu le plus proche d'un film du catalogue d'entrainement."""
        self._check_fitted()
        dots = self._dots(self.item_vectors_[item_idx : item_idx + 1])[0]
        dots[item_idx] = -np.inf
        order = np.argsort(-dots)[:n]
        return order, dots[order]

    def explain(self, user_idx: int, item_idx: int, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Films aimes par l'utilisateur dont le contenu est le plus proche du film recommande."""
        self._check_fitted()
        liked = self.liked_[user_idx].indices
        if len(liked) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        dots = self._dots(self.item_vectors_[item_idx : item_idx + 1])[0][liked]
        order = np.argsort(-dots)[:n]
        keep = dots[order] > 0
        return liked[order][keep], dots[order][keep]

    def shared_terms(self, item_a: int, item_b: int, n: int = 5) -> list[tuple[str, float]]:
        """Termes TF-IDF qui rapprochent deux films (representations creuses uniquement)."""
        self._check_fitted()
        if not sp.issparse(self.item_vectors_) or self.features.feature_names is None:
            raise NotImplementedError("explication par termes reservee aux vecteurs TF-IDF")
        product = self.item_vectors_[item_a].multiply(self.item_vectors_[item_b]).tocoo()
        order = np.argsort(-product.data)[:n]
        names = self.features.feature_names
        return [(str(names[product.col[i]]), float(product.data[i])) for i in order]
