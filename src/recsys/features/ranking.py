"""Variables du modele de ranking : une ligne par couple (utilisateur, film candidat).

Tout est calcule a partir de la matrice d'entrainement et des modeles deja
entraines sur cette meme matrice. Aucune variable ne regarde au-dela de la
date de coupure : c'est ce qui serait disponible au moment de recommander.

Familles de variables :

- scores des modeles (centres-reduits par utilisateur) et rangs associes ;
- film : popularite, note moyenne, part de notes positives, age a la date de
  coupure (et non annee absolue : « recent » doit garder le meme sens quand la
  date de coupure avance, sinon le modele apprend une epoque, pas une tendance) ;
- utilisateur : taille de l'historique, note moyenne, part de films aimes ;
- croisees : affinite de genres, ecart d'annee et de popularite entre le film
  et ce que l'utilisateur aime d'habitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.models.base import InteractionMatrix, Recommender, top_k
from recsys.models.hybrid.combiner import HybridRecommender, zscore_rows


@dataclass(frozen=True)
class CandidateBatch:
    """Candidats d'un lot d'utilisateurs et leurs variables."""

    user_idx: np.ndarray  # (n,)
    items: np.ndarray  # (n, c) indices de films, du meilleur candidat au moins bon
    features: np.ndarray  # (n, c, f)

    @property
    def flat_features(self) -> np.ndarray:
        return self.features.reshape(-1, self.features.shape[-1])


def _row_ranks(scores: np.ndarray) -> np.ndarray:
    """Rang de chaque film dans sa ligne (1 = meilleur score)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(scores.shape[0])[:, None], order] = np.arange(1, scores.shape[1] + 1)
    return ranks


class RankingFeatureBuilder:
    """Genere les candidats (top de l'hybride) et calcule leurs variables."""

    def __init__(
        self,
        hybrid: HybridRecommender,
        components: dict[str, Recommender],
        movies: pd.DataFrame,
        n_candidates: int = 100,
        relevance_threshold: float = 4.0,
        reference_year: float | None = None,
    ) -> None:
        if n_candidates < 1:
            raise ValueError("n_candidates doit etre >= 1")
        self.hybrid = hybrid
        self.components = components
        self.movies = movies
        self.n_candidates = n_candidates
        self.relevance_threshold = relevance_threshold
        # Annee (fractionnaire) de la date de coupure : sert a calculer l'age des films.
        self.reference_year = reference_year
        self.matrix: InteractionMatrix | None = None

    @property
    def feature_names(self) -> list[str]:
        names = [f"score_{name}" for name in self.components]
        names += [f"log_rank_{name}" for name in self.components]
        return [
            *names,
            "score_hybrid",
            "item_log_popularity",
            "item_mean_rating",
            "item_liked_share",
            "item_age",
            "user_log_history",
            "user_mean_rating",
            "user_liked_share",
            "genre_affinity",
            "year_gap",
            "popularity_gap",
        ]

    def fit(self, matrix: InteractionMatrix) -> RankingFeatureBuilder:
        """Precalcule les statistiques par film et par utilisateur sur l'entrainement seul."""
        self.matrix = matrix
        ratings = matrix.ratings
        liked = ratings.copy()
        liked.data = (liked.data >= self.relevance_threshold).astype(np.float32)

        item_count = np.diff(ratings.tocsc().indptr).astype(np.float32)
        item_sum = np.asarray(ratings.sum(axis=0)).ravel()
        item_liked = np.asarray(liked.sum(axis=0)).ravel()
        self.item_log_popularity_ = np.log1p(item_count)
        self.item_mean_rating_ = item_sum / np.maximum(item_count, 1)
        self.item_liked_share_ = item_liked / np.maximum(item_count, 1)
        catalogue = self.movies.set_index("movieId").loc[matrix.item_ids]
        self.item_year_ = catalogue["year"].astype("float32").to_numpy()
        reference = self.reference_year
        if reference is None:
            reference = float(np.nanmax(self.item_year_)) if len(self.item_year_) else 0.0
        self.item_age_ = (reference - self.item_year_).astype(np.float32)

        user_count = np.diff(ratings.indptr).astype(np.float32)
        user_liked = np.asarray(liked.sum(axis=1)).ravel()
        self.user_log_history_ = np.log1p(user_count)
        self.user_mean_rating_ = np.asarray(ratings.sum(axis=1)).ravel() / np.maximum(user_count, 1)
        self.user_liked_share_ = user_liked / np.maximum(user_count, 1)

        # Profil de genres : part de chaque genre parmi les films aimes par l'utilisateur.
        genres = sorted({genre for row in catalogue["genres"] for genre in row})
        column = {genre: index for index, genre in enumerate(genres)}
        rows, cols = [], []
        for item, values in enumerate(catalogue["genres"]):
            rows.extend([item] * len(values))
            cols.extend(column[genre] for genre in values)
        self.item_genres_ = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(matrix.n_items, len(genres)),
        )
        counts = (liked @ self.item_genres_).toarray()
        totals = counts.sum(axis=1, keepdims=True)
        self.user_genre_profile_ = np.divide(
            counts, totals, out=np.zeros_like(counts), where=totals > 0
        )

        # Ce que l'utilisateur aime d'habitude : annee et popularite moyennes de ses films aimes.
        denominator = np.maximum(user_liked, 1)
        year_known = np.nan_to_num(self.item_year_, nan=0.0)
        has_year = (~np.isnan(self.item_year_)).astype(np.float32)
        liked_with_year = np.maximum(np.asarray(liked @ has_year).ravel(), 1)
        self.user_liked_year_ = np.where(
            user_liked > 0, np.asarray(liked @ year_known).ravel() / liked_with_year, np.nan
        )
        self.user_liked_popularity_ = np.where(
            user_liked > 0, np.asarray(liked @ self.item_log_popularity_).ravel() / denominator,
            np.nan,
        )  # fmt: skip
        return self

    def build(self, user_idx: np.ndarray) -> CandidateBatch:
        """Candidats et variables pour un lot d'utilisateurs de la matrice d'entrainement."""
        if self.matrix is None:
            raise RuntimeError("appeler fit() avant build()")
        user_idx = np.asarray(user_idx)
        raw = {name: np.asarray(model.score(user_idx)) for name, model in self.components.items()}
        hybrid_scores = np.array(
            self.hybrid._combine({name: raw[name] for name in self.hybrid.weights}),
            dtype=np.float32,
        )
        seen_rows, seen_cols = self.matrix.ratings[user_idx].nonzero()
        masked = hybrid_scores.copy()
        masked[seen_rows, seen_cols] = -np.inf
        items, candidate_scores = top_k(masked, min(self.n_candidates, masked.shape[1]))

        rows = np.arange(len(user_idx))[:, None]
        columns: list[np.ndarray] = [zscore_rows(raw[name])[rows, items] for name in raw]
        columns += [np.log(_row_ranks(raw[name])[rows, items]) for name in raw]
        users = np.broadcast_to(user_idx[:, None], items.shape)
        affinity = np.asarray(
            self.item_genres_[items.ravel()]
            .multiply(self.user_genre_profile_[users.ravel()])
            .sum(axis=1)
        ).reshape(items.shape)
        columns += [
            candidate_scores,
            self.item_log_popularity_[items],
            self.item_mean_rating_[items],
            self.item_liked_share_[items],
            self.item_age_[items],
            self.user_log_history_[users],
            self.user_mean_rating_[users],
            self.user_liked_share_[users],
            affinity,
            self.item_year_[items] - self.user_liked_year_[users],
            self.item_log_popularity_[items] - self.user_liked_popularity_[users],
        ]
        features = np.stack([np.asarray(c, dtype=np.float32) for c in columns], axis=-1)
        return CandidateBatch(user_idx, items, features)


def graded_labels(
    items: np.ndarray, holdout_ratings: sp.csr_matrix, user_idx: np.ndarray, threshold: float
) -> np.ndarray:
    """Labels a trois niveaux, pour ne pas confondre « inconnu » et « rejete ».

    2 : film note >= seuil apres la coupure (aime) ;
    1 : film non note (preference inconnue : ni positif, ni negatif) ;
    0 : film note sous le seuil (rejet explicite).
    """
    rows = np.repeat(user_idx, items.shape[1])
    ratings = np.asarray(holdout_ratings[rows, items.ravel()]).reshape(items.shape)
    labels = np.ones(items.shape, dtype=np.int32)
    labels[(ratings > 0) & (ratings < threshold)] = 0
    labels[ratings >= threshold] = 2
    return labels


def binary_labels(graded: np.ndarray) -> np.ndarray:
    """Variante de reference : 1 si aime, 0 sinon (inconnu et rejete confondus)."""
    return (graded == 2).astype(np.int32)
