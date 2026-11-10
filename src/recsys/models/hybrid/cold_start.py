"""Strategie de cold start utilisateur.

Un nouvel utilisateur n'a aucune ligne dans la matrice d'entrainement. Les
modeles savent neanmoins le servir a partir de quelques notes fournies a la
volee (`score_history`), sans reentrainement.

Regle retenue : melanger le score personnalise et la popularite, avec un
poids de popularite qui decroit a mesure que l'historique grandit :

    poids_popularite = prior / (prior + nombre de notes)

- 0 note : popularite pure, seule information disponible ;
- quelques notes : la popularite departage et stabilise un signal encore
  fragile (une seule note ne dit presque rien) ;
- beaucoup de notes : le modele personnalise domine.

Le melange se fait sur les rangs (1 / (rrf_k + rang)) et non sur les scores :
la popularite decroit tres vite d'un film au suivant alors que le score d'un
hybride est presque plat en tete de liste ; les additionner tels quels
reviendrait a laisser la popularite decider, quel que soit son poids.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.hybrid.combiner import reciprocal_ranks


class ColdStartRecommender(Recommender):
    """Melange popularite et modele personnalise selon la taille de l'historique."""

    name = "cold_start"

    def __init__(
        self,
        personalised: Recommender,
        popularity: Recommender,
        prior_strength: float = 1.0,
        rrf_k: int = 60,
    ) -> None:
        super().__init__()
        if prior_strength < 0:
            raise ValueError("prior_strength doit etre positif ou nul")
        self.personalised = personalised
        self.popularity = popularity
        self.prior_strength = prior_strength
        self.rrf_k = rrf_k
        if personalised.matrix is not None:
            self.fit(personalised.matrix)

    @property
    def params(self) -> dict[str, Any]:
        return {"prior_strength": self.prior_strength, "personalised": self.personalised.name}

    def _fit(self, matrix: InteractionMatrix) -> None:
        for model in (self.personalised, self.popularity):
            if model.matrix is not matrix:
                model.fit(matrix)

    def popularity_weight(self, n_ratings: np.ndarray) -> np.ndarray:
        n_ratings = np.asarray(n_ratings, dtype=np.float32)
        if self.prior_strength == 0:
            return (n_ratings == 0).astype(np.float32)
        return self.prior_strength / (self.prior_strength + n_ratings)

    def _ranks(self, scores: np.ndarray) -> np.ndarray:
        return reciprocal_ranks(np.asarray(scores, dtype=np.float32), self.rrf_k)

    def _blend(self, personal: np.ndarray, popular: np.ndarray, n_ratings: np.ndarray):
        weight = self.popularity_weight(n_ratings)[:, None]
        return (1.0 - weight) * self._ranks(personal) + weight * self._ranks(popular)

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        matrix = self._check_fitted()
        n_ratings = np.diff(matrix.ratings.indptr)[np.asarray(user_idx)]
        return self._blend(
            self.personalised.score(user_idx), self.popularity.score(user_idx), n_ratings
        )

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        self._check_fitted()
        n_ratings = np.diff(history.indptr)
        popular = self.popularity.score_history(history)
        if history.nnz == 0:
            return np.asarray(popular, dtype=np.float32)  # popularite pure, ex aequo compris
        return self._blend(self.personalised.score_history(history), popular, n_ratings)
