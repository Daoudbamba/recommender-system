"""Conteneur commun des vecteurs de films, quelle que soit leur origine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class ItemFeatures:
    """Vecteurs de films (une ligne par film), creux ou denses."""

    movie_ids: np.ndarray
    matrix: sp.csr_matrix | np.ndarray
    feature_names: np.ndarray | None = None

    @property
    def is_sparse(self) -> bool:
        return sp.issparse(self.matrix)

    def align(self, movie_ids: np.ndarray) -> ItemFeatures:
        """Reordonne les lignes selon `movie_ids` ; erreur si un film manque."""
        positions = pd.Index(self.movie_ids).get_indexer(movie_ids)
        if (positions < 0).any():
            missing = np.asarray(movie_ids)[positions < 0][:5].tolist()
            raise KeyError(f"films sans vecteur de contenu : {missing}")
        return ItemFeatures(np.asarray(movie_ids), self.matrix[positions], self.feature_names)
