"""Separation temporelle entrainement / evaluation, sans fuite de donnees.

Principe : une date de coupure globale. Tout ce qui est note avant sert a
apprendre, tout ce qui est note apres sert a evaluer. Aucune information
posterieure a la coupure n'entre dans l'entrainement : ni les notes, ni le
choix des films conserves (le seuil de popularite est calcule sur la periode
d'entrainement seule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from recsys.models.base import InteractionMatrix


@dataclass(frozen=True)
class Fold:
    """Un couple (entrainement, evaluation) pret a l'emploi."""

    matrix: InteractionMatrix
    holdout: pd.DataFrame  # userId, movieId, rating, user_idx, item_idx
    eval_users: np.ndarray  # indices de lignes des utilisateurs evalues
    relevance_threshold: float
    train_end: pd.Timestamp
    holdout_end: pd.Timestamp
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def relevant(self) -> pd.DataFrame:
        """Notes d'evaluation considerees comme pertinentes."""
        return self.holdout[self.holdout["rating"] >= self.relevance_threshold]


def temporal_cutoffs(ratings: pd.DataFrame, quantiles: list[float]) -> list[pd.Timestamp]:
    """Dates en dessous desquelles se trouve la part `q` des notes."""
    if any(not 0 < q < 1 for q in quantiles) or sorted(quantiles) != list(quantiles):
        raise ValueError("les quantiles doivent etre croissants et strictement entre 0 et 1")
    return [ratings["timestamp"].quantile(q) for q in quantiles]


def split_at(
    ratings: pd.DataFrame, train_end: pd.Timestamp, holdout_end: pd.Timestamp | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Notes jusqu'a `train_end` inclus, puis notes dans ]train_end, holdout_end]."""
    stamps = ratings["timestamp"]
    after = stamps > train_end
    if holdout_end is not None:
        if holdout_end <= train_end:
            raise ValueError("holdout_end doit etre posterieur a train_end")
        after &= stamps <= holdout_end
    return ratings[stamps <= train_end], ratings[after]


def build_fold(
    ratings: pd.DataFrame,
    train_end: pd.Timestamp,
    holdout_end: pd.Timestamp | None,
    min_movie_ratings: int,
    min_user_history: int,
    relevance_threshold: float,
) -> Fold:
    """Construit la matrice d'entrainement et le jeu d'evaluation associe.

    Sont evalues les utilisateurs ayant au moins `min_user_history` notes
    avant la coupure et au moins un film pertinent, connu du modele, apres.
    Les utilisateurs et films apparus apres la coupure relevent du cold start
    (Sprint 4) : ils sont comptes dans `stats`, pas evalues ici.
    """
    train, holdout = split_at(ratings, train_end, holdout_end)
    movie_counts = train["movieId"].value_counts()
    kept_movies = movie_counts.index[movie_counts >= min_movie_ratings]
    train_kept = train[train["movieId"].isin(kept_movies)]
    matrix = InteractionMatrix.from_frame(train_kept)

    history = np.diff(matrix.ratings.indptr)
    known = holdout.assign(
        user_idx=matrix.user_index(holdout["userId"].to_numpy()),
        item_idx=matrix.item_index(holdout["movieId"].to_numpy()),
    )
    warm_user = known["user_idx"] >= 0
    enough = warm_user.to_numpy().copy()
    enough[enough] = history[known.loc[warm_user, "user_idx"].to_numpy()] >= min_user_history
    usable = known[enough & (known["item_idx"] >= 0).to_numpy()]
    relevant = usable[usable["rating"] >= relevance_threshold]
    eval_users = np.sort(relevant["user_idx"].unique())
    usable = usable[usable["user_idx"].isin(eval_users)]

    positives = holdout[holdout["rating"] >= relevance_threshold]
    stats = {
        "train_end": train_end.isoformat(),
        "holdout_end": None if holdout_end is None else holdout_end.isoformat(),
        "train_ratings": int(len(train_kept)),
        "train_ratings_dropped_by_movie_filter": int(len(train) - len(train_kept)),
        "train_users": matrix.n_users,
        "train_items": matrix.n_items,
        "train_density": matrix.density,
        "holdout_ratings": int(len(holdout)),
        "holdout_users": int(holdout["userId"].nunique()),
        "holdout_cold_users": int(holdout.loc[~warm_user.to_numpy(), "userId"].nunique()),
        "holdout_positive_share_on_known_items": float(
            positives["movieId"].isin(kept_movies).mean() if len(positives) else 0.0
        ),
        "eval_users": int(len(eval_users)),
        "eval_ratings": int(len(usable)),
        "eval_relevant": int((usable["rating"] >= relevance_threshold).sum()),
        "eval_median_history": float(np.median(history[eval_users])) if len(eval_users) else 0.0,
    }
    return Fold(
        matrix=matrix,
        holdout=usable.reset_index(drop=True),
        eval_users=eval_users,
        relevance_threshold=relevance_threshold,
        train_end=train_end,
        holdout_end=holdout_end if holdout_end is not None else holdout["timestamp"].max(),
        stats=stats,
    )
