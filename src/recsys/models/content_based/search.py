"""Moteur de recherche de films similaires sur tout le catalogue.

Recherche exacte par produit matriciel. Avec 62 000 films en 384 dimensions,
une requete se compte en millisecondes (mesure dans le notebook 06) : un
index approche de type FAISS n'apporterait rien a cette echelle et
ajouterait une dependance et une approximation. Le choix sera a revoir si le
catalogue depasse le million de vecteurs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.features.item_features import ItemFeatures


class SimilarityIndex:
    """Index de vecteurs normalises, interroge par film ou par vecteur libre."""

    def __init__(self, features: ItemFeatures) -> None:
        self.movie_ids = np.asarray(features.movie_ids)
        self.vectors = features.matrix
        self._positions = pd.Index(self.movie_ids)

    def __len__(self) -> int:
        return len(self.movie_ids)

    def _scores(self, query: sp.csr_matrix | np.ndarray) -> np.ndarray:
        product = query @ self.vectors.T
        scores = product.toarray() if sp.issparse(product) else np.asarray(product)
        return scores.reshape(-1).astype(np.float32)

    @staticmethod
    def _top(scores: np.ndarray, n: int) -> np.ndarray:
        n = min(n, len(scores))
        part = np.argpartition(-scores, n - 1)[:n]
        return part[np.argsort(-scores[part], kind="stable")]

    def position(self, movie_id: int) -> int:
        position = self._positions.get_indexer([movie_id])[0]
        if position < 0:
            raise KeyError(f"film inconnu de l'index : {movie_id}")
        return int(position)

    def similar(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Les `n` films les plus proches de `movie_id` (lui-meme exclu)."""
        position = self.position(movie_id)
        scores = self._scores(self.vectors[position : position + 1])
        scores[position] = -np.inf
        top = self._top(scores, n)
        return pd.DataFrame({"movieId": self.movie_ids[top], "similarity": scores[top]})

    def search_vector(self, vector: np.ndarray, n: int = 10) -> pd.DataFrame:
        """Les `n` films les plus proches d'un vecteur de requete."""
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"dimension {vector.shape[1]} incompatible avec l'index ({self.vectors.shape[1]})"
            )
        scores = self._scores(vector / max(float(np.linalg.norm(vector)), 1e-12))
        top = self._top(scores, n)
        return pd.DataFrame({"movieId": self.movie_ids[top], "similarity": scores[top]})

    def search_text(
        self, query: str, encoder: Callable[[Sequence[str]], np.ndarray], n: int = 10
    ) -> pd.DataFrame:
        """Recherche semantique : le texte est encode par le meme modele que les films."""
        if not query.strip():
            raise ValueError("la requete est vide")
        return self.search_vector(np.asarray(encoder([query]))[0], n)

    def save(self, path: Path) -> Path:
        if sp.issparse(self.vectors):
            raise NotImplementedError("sauvegarde reservee aux index denses")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, movie_ids=self.movie_ids, vectors=self.vectors)
        return path

    @classmethod
    def load(cls, path: Path) -> SimilarityIndex:
        if not path.is_file():
            raise FileNotFoundError(f"Index introuvable : {path}")
        with np.load(path) as data:
            return cls(ItemFeatures(data["movie_ids"], data["vectors"]))
