"""Briques communes a tous les modeles de recommandation.

- `InteractionMatrix` : la matrice utilisateur x film et les correspondances
  entre identifiants MovieLens et indices de lignes / colonnes ;
- `Recommender` : le contrat que respecte chaque modele (fit, score,
  recommend, predict, save, load).
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass(frozen=True)
class InteractionMatrix:
    """Matrice creuse des notes (lignes = utilisateurs, colonnes = films)."""

    ratings: sp.csr_matrix
    user_ids: np.ndarray
    item_ids: np.ndarray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> InteractionMatrix:
        """Construit la matrice depuis des colonnes userId, movieId, rating."""
        user_codes, user_ids = pd.factorize(frame["userId"], sort=True)
        item_codes, item_ids = pd.factorize(frame["movieId"], sort=True)
        values = frame["rating"].to_numpy(dtype=np.float32)
        matrix = sp.csr_matrix(
            (values, (user_codes, item_codes)), shape=(len(user_ids), len(item_ids))
        )
        return cls(matrix, np.asarray(user_ids), np.asarray(item_ids))

    @property
    def n_users(self) -> int:
        return self.ratings.shape[0]

    @property
    def n_items(self) -> int:
        return self.ratings.shape[1]

    @property
    def density(self) -> float:
        return self.ratings.nnz / (self.n_users * self.n_items)

    def user_index(self, user_ids: Any) -> np.ndarray:
        """Indices de ligne ; -1 pour un utilisateur inconnu."""
        return pd.Index(self.user_ids).get_indexer(np.atleast_1d(user_ids))

    def item_index(self, item_ids: Any) -> np.ndarray:
        """Indices de colonne ; -1 pour un film inconnu."""
        return pd.Index(self.item_ids).get_indexer(np.atleast_1d(item_ids))


def top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices et valeurs des `k` meilleurs scores de chaque ligne, tries."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(part_scores, order, axis=1)


class Recommender(ABC):
    """Contrat commun des modeles.

    `score` retourne un score de classement par film (plus grand = mieux) ;
    `predict` retourne une note estimee et n'existe que pour les modeles qui
    predisent des notes (`predicts_ratings = True`).
    """

    name: str = "recommender"
    predicts_ratings: bool = False

    def __init__(self) -> None:
        self.matrix: InteractionMatrix | None = None

    @property
    def params(self) -> dict[str, Any]:
        """Hyperparametres, pour les rapports et le suivi d'experiences."""
        return {}

    def fit(self, matrix: InteractionMatrix) -> Self:
        self.matrix = matrix
        self._fit(matrix)
        return self

    @abstractmethod
    def _fit(self, matrix: InteractionMatrix) -> None: ...

    @abstractmethod
    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Scores denses (n_utilisateurs, n_films) pour des indices d'utilisateurs."""

    def predict(self, user_idx: np.ndarray, item_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.name} ne predit pas de notes")

    def _check_fitted(self) -> InteractionMatrix:
        if self.matrix is None:
            raise RuntimeError(f"{self.name} : appeler fit() avant d'utiliser le modele")
        return self.matrix

    def recommend(
        self, user_idx: np.ndarray, k: int = 10, exclude_seen: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k films (indices de colonnes) et leurs scores pour chaque utilisateur."""
        matrix = self._check_fitted()
        user_idx = np.atleast_1d(user_idx)
        if (user_idx < 0).any() or (user_idx >= matrix.n_users).any():
            raise IndexError("indice d'utilisateur hors de la matrice d'entrainement")
        scores = np.array(self.score(user_idx), dtype=np.float32)
        if exclude_seen:
            rows, cols = matrix.ratings[user_idx].nonzero()
            scores[rows, cols] = -np.inf
        return top_k(scores, k)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        """Charge un modele sauvegarde par `save` (fichier produit par ce projet)."""
        if not path.is_file():
            raise FileNotFoundError(f"Artefact de modele introuvable : {path}")
        with path.open("rb") as handle:
            model = pickle.load(handle)  # noqa: S301 - artefact local, jamais telecharge
        if not isinstance(model, cls):
            found = type(model).__name__
            raise TypeError(f"{path.name} contient un {found}, pas un {cls.__name__}")
        return model
