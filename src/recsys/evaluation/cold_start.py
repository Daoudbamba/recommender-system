"""Evaluation du cold start, sur de vrais nouveaux venus.

Utilisateurs froids : ceux qui n'apparaissent qu'apres la date de coupure.
Le modele ne les a jamais vus. On rejoue leur arrivee : on lui revele leurs
`n` premieres notes (par ordre chronologique), et on regarde s'il retrouve
les films qu'ils ont aimes ensuite.

Pour que les resultats soient comparables d'un `n` a l'autre, la cible est
la meme pour tous les `n` : les films aimes apres les `max(n)` premieres
notes. Ces `max(n)` premiers films sont exclus des candidats dans tous les
cas, meme quand le modele n'en a vu qu'une partie.

Films froids : ceux qui sont absents du catalogue d'entrainement, faute de notes
suffisantes avant la coupure (films sortis ensuite compris). Seul leur
contenu permet de les classer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.evaluation import metrics
from recsys.evaluation.split import Fold
from recsys.models.base import Recommender, top_k


@dataclass(frozen=True)
class ColdUserScenario:
    """Historiques reveles et cible commune pour un echantillon d'utilisateurs froids."""

    user_ids: np.ndarray
    histories: dict[int, sp.csr_matrix]  # n -> notes revelees (utilisateurs x films)
    excluded: sp.csr_matrix  # les max(n) premiers films, exclus des candidats
    relevant: sp.csr_matrix  # films aimes ensuite

    @property
    def n_users(self) -> int:
        return len(self.user_ids)


def build_cold_user_scenario(
    fold: Fold, history_sizes: list[int], max_users: int | None, seed: int
) -> ColdUserScenario:
    """Prepare les historiques tronques des utilisateurs absents de l'entrainement."""
    if not history_sizes or min(history_sizes) < 0:
        raise ValueError("history_sizes doit contenir des entiers positifs ou nuls")
    max_n = max(history_sizes)
    frame = fold.cold_user_holdout.sort_values(["userId", "timestamp"], kind="stable")
    position = frame.groupby("userId", sort=False).cumcount()
    later = frame[position >= max_n]
    liked_later = later[later["rating"] >= fold.relevance_threshold]
    eligible = np.sort(liked_later["userId"].unique())
    if max_users is not None and len(eligible) > max_users:
        rng = np.random.default_rng(seed)
        eligible = np.sort(rng.choice(eligible, size=max_users, replace=False))

    rows = pd.Index(eligible)
    shape = (len(eligible), fold.matrix.n_items)

    def to_matrix(part: pd.DataFrame, values: np.ndarray) -> sp.csr_matrix:
        row_idx = rows.get_indexer(part["userId"])
        keep = row_idx >= 0
        return sp.csr_matrix(
            (values[keep], (row_idx[keep], part["item_idx"].to_numpy()[keep])), shape=shape
        )

    histories = {}
    for n in sorted(set(history_sizes)):
        part = frame[position < n]
        histories[n] = to_matrix(part, part["rating"].to_numpy(dtype=np.float32))
    first = frame[position < max_n]
    return ColdUserScenario(
        user_ids=np.asarray(eligible),
        histories=histories,
        excluded=to_matrix(first, np.ones(len(first), dtype=bool)),
        relevant=to_matrix(liked_later, np.ones(len(liked_later), dtype=bool)),
    )


def evaluate_cold_users(
    model: Recommender, scenario: ColdUserScenario, ks: list[int], batch_size: int = 256
) -> dict[int, dict[str, float]]:
    """Metriques Top-K par taille d'historique revele."""
    if scenario.n_users == 0:
        # Aucun nouvel utilisateur exploitable : rien a mesurer, mais pas d'erreur.
        return {n: {} for n in scenario.histories}
    k_max = max(ks)
    n_relevant = np.asarray(scenario.relevant.sum(axis=1)).ravel()
    out: dict[int, dict[str, float]] = {}
    for n, history in scenario.histories.items():
        recommended = np.empty((scenario.n_users, k_max), dtype=np.int64)
        for start in range(0, scenario.n_users, batch_size):
            stop = min(start + batch_size, scenario.n_users)
            scores = np.array(model.score_history(history[start:stop]), dtype=np.float32)
            rows, cols = scenario.excluded[start:stop].nonzero()
            scores[rows, cols] = -np.inf
            recommended[start:stop] = top_k(scores, k_max)[0]
        rows = np.repeat(np.arange(scenario.n_users), k_max)
        hits = np.asarray(scenario.relevant[rows, recommended.ravel()]).reshape(-1, k_max)
        per_user = metrics.ranking_metrics(hits, n_relevant, ks)
        summary = {name: float(values.mean()) for name, values in per_user.items()}
        for k in ks:
            summary[f"coverage@{k}"] = metrics.catalog_coverage(
                recommended, scenario.relevant.shape[1], k
            )
        out[n] = summary
    return out


def evaluate_new_items(
    score_new_items, fold: Fold, movie_ids: np.ndarray, ks: list[int], batch_size: int = 256
) -> dict[str, float]:
    """Classement des films absents du catalogue d'entrainement, pour les utilisateurs evalues.

    `score_new_items(user_idx, movie_ids)` retourne (n_utilisateurs, n_films).
    Candidats : `movie_ids`. Cible : ceux que l'utilisateur a notes >= seuil.
    """
    movie_ids = np.asarray(movie_ids)
    holdout = fold.new_item_holdout
    liked = holdout[
        (holdout["rating"] >= fold.relevance_threshold) & holdout["movieId"].isin(movie_ids)
    ]
    users = np.sort(liked["user_idx"].unique())
    if len(users) == 0:
        return {"n_users": 0, "n_candidates": int(len(movie_ids))}
    relevant = sp.csr_matrix(
        (
            np.ones(len(liked), dtype=bool),
            (
                pd.Index(users).get_indexer(liked["user_idx"]),
                pd.Index(movie_ids).get_indexer(liked["movieId"]),
            ),
        ),
        shape=(len(users), len(movie_ids)),
    )
    k_max = min(max(ks), len(movie_ids))
    recommended = np.empty((len(users), k_max), dtype=np.int64)
    for start in range(0, len(users), batch_size):
        batch = users[start : start + batch_size]
        scores = np.asarray(score_new_items(batch, movie_ids), dtype=np.float32)
        recommended[start : start + len(batch)] = top_k(scores, k_max)[0]
    rows = np.repeat(np.arange(len(users)), k_max)
    hits = np.asarray(relevant[rows, recommended.ravel()]).reshape(-1, k_max)
    n_relevant = np.asarray(relevant.sum(axis=1)).ravel()
    usable = [k for k in ks if k <= k_max]
    per_user = metrics.ranking_metrics(hits, n_relevant, usable)
    summary = {name: float(values.mean()) for name, values in per_user.items()}
    summary.update(
        n_users=int(len(users)),
        n_candidates=int(len(movie_ids)),
        mean_relevant_per_user=float(n_relevant.mean()),
    )
    return summary
