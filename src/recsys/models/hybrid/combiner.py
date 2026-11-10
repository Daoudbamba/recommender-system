"""Systeme hybride : combiner plusieurs modeles deja entraines.

Les modeles produisent des scores d'echelles incomparables (une somme de
similarites, un produit de facteurs latents...). Trois strategies pour les
rendre comparables avant de les additionner avec des poids :

- `weighted` : normalisation min-max par utilisateur, chaque score est ramene
  dans [0, 1]. Sensible aux valeurs extremes d'un modele ;
- `zscore` : centrage-reduction par utilisateur. Conserve l'ecart entre un
  film tres bien note par un modele et la masse des autres ;
- `rrf` : agregation de rangs (Reciprocal Rank Fusion, Cormack et al. 2009).
  Ignore totalement les scores, seul le classement compte : robuste, mais
  perd l'information « ce modele est tres sur de lui ».
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender

STRATEGIES = ("weighted", "zscore", "rrf")


def minmax_rows(scores: np.ndarray) -> np.ndarray:
    low = scores.min(axis=1, keepdims=True)
    span = scores.max(axis=1, keepdims=True) - low
    return np.divide(scores - low, span, out=np.zeros_like(scores), where=span > 0)


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    centered = scores - scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True)
    return np.divide(centered, std, out=np.zeros_like(centered), where=std > 0)


def reciprocal_ranks(scores: np.ndarray, rrf_k: int) -> np.ndarray:
    """1 / (rrf_k + rang), le meilleur film de chaque ligne ayant le rang 1."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, scores.shape[1] + 1)
    return (1.0 / (rrf_k + ranks)).astype(np.float32)


def normalise(scores: np.ndarray, strategy: str, rrf_k: int = 60) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if strategy == "weighted":
        return minmax_rows(scores)
    if strategy == "zscore":
        return zscore_rows(scores)
    if strategy == "rrf":
        return reciprocal_ranks(scores, rrf_k)
    raise ValueError(f"strategie inconnue : {strategy} (attendu : {STRATEGIES})")


class HybridRecommender(Recommender):
    """Somme ponderee des scores normalises de plusieurs modeles."""

    name = "hybrid"

    def __init__(
        self,
        components: dict[str, Recommender],
        weights: dict[str, float],
        strategy: str = "weighted",
        rrf_k: int = 60,
        name: str | None = None,
    ) -> None:
        super().__init__()
        if strategy not in STRATEGIES:
            raise ValueError(f"strategie inconnue : {strategy} (attendu : {STRATEGIES})")
        unknown = set(weights) - set(components)
        if unknown:
            raise KeyError(f"poids sans modele correspondant : {sorted(unknown)}")
        if any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("les poids doivent etre positifs et de somme non nulle")
        total = sum(weights.values())
        self.components = components
        self.weights = {key: value / total for key, value in weights.items() if value > 0}
        self.strategy = strategy
        self.rrf_k = rrf_k
        if name:
            self.name = name

    @property
    def params(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "weights": self.weights}

    def _fit(self, matrix: InteractionMatrix) -> None:
        """N'entraine que les composants qui ne l'ont pas deja ete sur cette matrice."""
        for key in self.weights:
            component = self.components[key]
            if component.matrix is not matrix:
                component.fit(matrix)

    def _combine(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        total: np.ndarray | None = None
        for key, weight in self.weights.items():
            part = weight * normalise(scores[key], self.strategy, self.rrf_k)
            total = part if total is None else total + part
        return total

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._combine({key: self.components[key].score(user_idx) for key in self.weights})

    def score_history(self, history: sp.csr_matrix) -> np.ndarray:
        self._check_fitted()
        return self._combine(
            {key: self.components[key].score_history(history) for key in self.weights}
        )

    def contributions(self, user_idx: int, item_idx: np.ndarray) -> dict[str, np.ndarray]:
        """Part de chaque modele dans le score hybride des films demandes (explication)."""
        self._check_fitted()
        users = np.array([user_idx])
        return {
            key: weight
            * normalise(self.components[key].score(users), self.strategy, self.rrf_k)[0][item_idx]
            for key, weight in self.weights.items()
        }

    def save(self, path: Path) -> Path:
        """Sauvegarde la recette (poids, strategie), pas les composants.

        Chaque composant a deja son propre artefact : les dupliquer ici
        produirait un fichier de plus d'un gigaoctet.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        spec = {
            "name": self.name,
            "weights": self.weights,
            "strategy": self.strategy,
            "rrf_k": self.rrf_k,
        }
        with path.open("wb") as handle:
            pickle.dump(spec, handle)
        return path

    @classmethod
    def from_spec(cls, path: Path, components: dict[str, Recommender]) -> HybridRecommender:
        """Reconstruit l'hybride a partir de sa recette et de composants deja charges."""
        if not path.is_file():
            raise FileNotFoundError(f"Recette d'hybride introuvable : {path}")
        with path.open("rb") as handle:
            spec = pickle.load(handle)  # noqa: S301 - artefact local, jamais telecharge
        missing = set(spec["weights"]) - set(components)
        if missing:
            raise KeyError(f"composants manquants pour l'hybride : {sorted(missing)}")
        model = cls(components, spec["weights"], spec["strategy"], spec["rrf_k"], spec["name"])
        matrices = {id(components[key].matrix) for key in spec["weights"]}
        first = components[next(iter(spec["weights"]))].matrix
        if first is None or (len(matrices) > 1 and not _same_catalogue(components, spec)):
            raise ValueError("les composants ne partagent pas le meme catalogue d'entrainement")
        model.matrix = first
        return model


def _same_catalogue(components: dict[str, Recommender], spec: dict[str, Any]) -> bool:
    matrices = [components[key].matrix for key in spec["weights"]]
    if any(matrix is None for matrix in matrices):
        return False
    reference = matrices[0]
    return all(
        np.array_equal(m.item_ids, reference.item_ids)
        and np.array_equal(m.user_ids, reference.user_ids)
        for m in matrices[1:]
    )
