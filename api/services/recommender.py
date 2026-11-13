"""Service metier de l'API : traduit des identifiants MovieLens en appels aux modeles.

Aucune logique de modele ici : le service choisit le bon modele selon la
situation (utilisateur connu ou non, strategie demandee), gere les
identifiants inconnus et met en forme les reponses.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.serving.bundle import ServingBundle

STRATEGIES = ("ranker", "hybrid", "popularity")
SOURCE_LABELS = {
    "als_implicit": "les goûts d'utilisateurs proches",
    "item_knn": "les films notés de la même façon que vos préférés",
    "tfidf": "son contenu (genres et tags)",
    "embeddings": "sa proximité sémantique avec vos films",
}


class UnknownMovieError(KeyError):
    """Le film demande n'existe pas dans le catalogue."""


class UnknownUserError(KeyError):
    """L'utilisateur demande n'a aucun historique connu."""


class RecommenderService:
    """Facade thread-safe au-dessus du bundle de modeles."""

    def __init__(self, bundle: ServingBundle) -> None:
        self.bundle = bundle
        self._lock = threading.Lock()  # les modeles partagent des tampons : un calcul a la fois
        self._user_index = pd.Index(bundle.matrix.user_ids)
        self._item_index = pd.Index(bundle.matrix.item_ids)
        counts = np.diff(bundle.matrix.ratings.indptr)
        self._history_size = counts

    # ------------------------------------------------------------------ films
    def _movie(self, movie_id: int) -> dict[str, Any]:
        movies = self.bundle.movies
        if movie_id not in movies.index:
            raise UnknownMovieError(movie_id)
        row = movies.loc[movie_id]
        return {
            "movie_id": int(movie_id),
            "title": str(row["title"]),
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": [str(g) for g in row["genres"]],
            "n_ratings": int(row["n_ratings"]),
            "mean_rating": None
            if pd.isna(row["mean_rating"])
            else round(float(row["mean_rating"]), 3),
            "in_model": bool(row["in_model"]),
        }

    def get_movie(self, movie_id: int) -> dict[str, Any]:
        return self._movie(movie_id)

    def list_movies(
        self, query: str | None, genre: str | None, limit: int, offset: int
    ) -> dict[str, Any]:
        """Catalogue filtre, trie par nombre de notes decroissant."""
        movies = self.bundle.movies
        mask = np.ones(len(movies), dtype=bool)
        if query:
            mask &= movies["title"].str.contains(query, case=False, regex=False).to_numpy()
        if genre:
            wanted = genre.lower()
            mask &= movies["genres"].map(lambda gs: any(g.lower() == wanted for g in gs)).to_numpy()
        selected = movies.index[mask]
        page = selected[offset : offset + limit]
        return {
            "total": int(len(selected)),
            "limit": limit,
            "offset": offset,
            "items": [self._movie(int(movie_id)) for movie_id in page],
        }

    def genres(self) -> list[str]:
        return sorted(row["genre"] for row in self.bundle.stats["genres"])

    # ---------------------------------------------------------- utilisateurs
    def _user_idx(self, user_id: int) -> int:
        position = self._user_index.get_indexer([user_id])[0]
        return int(position)

    def knows_user(self, user_id: int) -> bool:
        return self._user_idx(user_id) >= 0

    def sample_users(self, limit: int) -> list[dict[str, int]]:
        """Utilisateurs d'exemple, repartis sur toutes les tailles d'historique."""
        order = np.argsort(self._history_size, kind="stable")
        picks = order[np.linspace(0, len(order) - 1, num=min(limit, len(order)), dtype=int)]
        return [
            {
                "user_id": int(self.bundle.matrix.user_ids[i]),
                "n_ratings": int(self._history_size[i]),
            }
            for i in picks
        ]

    def user_history(self, user_id: int, limit: int) -> dict[str, Any]:
        idx = self._user_idx(user_id)
        if idx < 0:
            raise UnknownUserError(user_id)
        row = self.bundle.matrix.ratings[idx]
        order = np.argsort(-row.data, kind="stable")
        movie_ids = self.bundle.matrix.item_ids[row.indices]
        liked = row.data >= self.bundle.manifest["relevance_threshold"]
        genre_counts: dict[str, int] = {}
        for movie_id in movie_ids[liked]:
            for genre in self.bundle.movies.at[int(movie_id), "genres"]:
                genre_counts[str(genre)] = genre_counts.get(str(genre), 0) + 1
        total = max(int(liked.sum()), 1)
        favourites = sorted(genre_counts.items(), key=lambda kv: -kv[1])[:6]
        return {
            "user_id": user_id,
            "n_ratings": int(row.nnz),
            "mean_rating": round(float(row.data.mean()), 3),
            "n_liked": int(liked.sum()),
            "favourite_genres": [
                {"genre": genre, "share": round(count / total, 3)} for genre, count in favourites
            ],
            "items": [
                {"movie": self._movie(int(movie_ids[i])), "rating": float(row.data[i])}
                for i in order[:limit]
            ],
        }

    # ------------------------------------------------------- recommandations
    def _explain(self, user_idx: int, items: np.ndarray) -> list[dict[str, Any]]:
        """Pour chaque film : part de chaque modele et films de l'historique qui l'expliquent."""
        bundle = self.bundle
        parts = bundle.hybrid.contributions(user_idx, items)
        totals = sum(parts.values())
        item_ids = bundle.matrix.item_ids
        out = []
        for position, item in enumerate(items):
            shares = {
                name: round(float(values[position] / totals[position]), 3)
                if totals[position]
                else 0.0
                for name, values in parts.items()
            }
            because, _ = bundle.components["item_knn"].explain(user_idx, int(item), n=2)
            if len(because) == 0:
                because, _ = bundle.components["tfidf"].explain(user_idx, int(item), n=2)
            titles = [str(bundle.movies.at[int(item_ids[i]), "title"]) for i in because]
            main_source = max(shares, key=shares.get)
            reasons = [f"Recommandé surtout par {SOURCE_LABELS.get(main_source, main_source)}"]
            if titles:
                reasons.append("Parce que vous avez aimé : " + " ; ".join(titles))
            out.append({"contributions": shares, "because_you_liked": titles, "reasons": reasons})
        return out

    def recommend(
        self, user_id: int, k: int, strategy: str = "ranker", diversity: float = 1.0
    ) -> dict[str, Any]:
        """Top-k d'un utilisateur ; repli sur la popularite s'il est inconnu."""
        if strategy not in STRATEGIES:
            raise ValueError(f"strategie inconnue : {strategy}")
        bundle = self.bundle
        idx = self._user_idx(user_id)
        start = time.perf_counter()
        with self._lock:
            if idx < 0:
                scores = bundle.popularity.score_history(
                    sp.csr_matrix((1, bundle.matrix.n_items), dtype=np.float32)
                )[0]
                items = np.argsort(-scores, kind="stable")[:k]
                values, explanations, used = scores[items], None, "popularity"
            else:
                users = np.array([idx])
                if strategy == "popularity":
                    model = bundle.popularity
                elif diversity < 1.0:
                    model = bundle.diversified[strategy]
                    model.lam = diversity
                else:
                    model = bundle.ranker if strategy == "ranker" else bundle.hybrid
                found, found_scores = model.recommend(users, k)
                items, values, used = found[0], found_scores[0], strategy
                keep = np.isfinite(values)
                items, values = items[keep], values[keep]
                explanations = self._explain(idx, items) if strategy != "popularity" else None
        elapsed = 1000 * (time.perf_counter() - start)
        item_ids = bundle.matrix.item_ids
        results = []
        for rank, (item, value) in enumerate(zip(items, values, strict=True), start=1):
            entry = {"rank": rank, "movie": self._movie(int(item_ids[item])), "score": float(value)}
            if explanations is None:
                entry.update(
                    contributions={}, because_you_liked=[],
                    reasons=["Parmi les films les plus appréciés de la plateforme"],
                )  # fmt: skip
            else:
                entry.update(explanations[rank - 1])
            results.append(entry)
        return {
            "user_id": user_id,
            "known_user": idx >= 0,
            "strategy": used,
            "diversity": diversity if (idx >= 0 and strategy != "popularity") else 1.0,
            "n_ratings": int(self._history_size[idx]) if idx >= 0 else 0,
            "items": results,
            "elapsed_ms": round(elapsed, 1),
        }

    def recommend_from_ratings(self, ratings: dict[int, float], k: int) -> dict[str, Any]:
        """Cold start : recommandations pour un visiteur a partir de quelques notes."""
        bundle = self.bundle
        positions = self._item_index.get_indexer(list(ratings))
        known = positions >= 0
        ignored = [movie_id for movie_id, ok in zip(ratings, known, strict=True) if not ok]
        cols = positions[known]
        values = np.array(list(ratings.values()), dtype=np.float32)[known]
        history = sp.csr_matrix(
            (values, (np.zeros(len(cols), dtype=int), cols)), shape=(1, bundle.matrix.n_items)
        )
        start = time.perf_counter()
        with self._lock:
            scores = np.array(bundle.cold_start.score_history(history)[0], dtype=np.float32)
        scores[cols] = -np.inf
        items = np.argsort(-scores, kind="stable")[:k]
        weight = float(bundle.cold_start.popularity_weight(np.array([len(cols)]))[0])
        item_ids = bundle.matrix.item_ids
        reason = (
            "Parmi les films les plus appréciés de la plateforme"
            if len(cols) == 0
            else f"D'après vos {len(cols)} note(s), popularité pondérée à {weight:.0%}"
        )
        return {
            "n_ratings_used": int(len(cols)),
            "ignored_movie_ids": ignored,
            "popularity_weight": round(weight, 3),
            "items": [
                {
                    "rank": rank,
                    "movie": self._movie(int(item_ids[item])),
                    "score": float(scores[item]),
                    "contributions": {},
                    "because_you_liked": [],
                    "reasons": [reason],
                }
                for rank, item in enumerate(items, start=1)
            ],
            "elapsed_ms": round(1000 * (time.perf_counter() - start), 1),
        }

    # --------------------------------------------------------------- similarite
    def similar_movies(self, movie_id: int, k: int, engine: str = "embeddings") -> dict[str, Any]:
        """Films proches, par contenu (tout le catalogue) ou par comportement (films notes)."""
        bundle = self.bundle
        reference = self._movie(movie_id)
        if engine == "embeddings":
            found = bundle.index.similar(movie_id, k)
            pairs = list(zip(found["movieId"].tolist(), found["similarity"].tolist(), strict=True))
        else:
            position = self._item_index.get_indexer([movie_id])[0]
            if position < 0:
                raise UnknownMovieError(movie_id)
            model = bundle.components["item_knn" if engine == "collaborative" else "tfidf"]
            with self._lock:
                items, sims = model.similar_items(int(position), k)
            pairs = [
                (int(bundle.matrix.item_ids[i]), float(s)) for i, s in zip(items, sims, strict=True)
            ]
        genres = set(reference["genres"])
        return {
            "movie": reference,
            "engine": engine,
            "items": [
                {
                    "movie": (movie := self._movie(int(other))),
                    "similarity": round(float(similarity), 4),
                    "shared_genres": sorted(genres & set(movie["genres"])),
                }
                for other, similarity in pairs
            ],
        }

    # ------------------------------------------------------------------ notes
    def predict_rating(self, user_id: int, movie_id: int) -> dict[str, Any]:
        """Note estimee par l'item-kNN ; repli sur les moyennes si l'utilisateur est inconnu."""
        bundle = self.bundle
        self._movie(movie_id)  # 404 si le film n'existe pas
        low, high = bundle.manifest["rating_bounds"]
        item = self._item_index.get_indexer([movie_id])[0]
        user = self._user_idx(user_id)
        model = bundle.components["item_knn"]
        baseline = model.baseline_
        if item < 0:
            value, basis = (
                baseline.global_mean_,
                "moyenne globale (film trop peu noté pour le modèle)",
            )
        elif user < 0:
            value = baseline.global_mean_ + float(baseline.item_bias_[item])
            basis = "moyenne du film (utilisateur inconnu)"
        else:
            with self._lock:
                value = float(model.predict(np.array([user]), np.array([item]))[0])
            basis = "voisinage de films (item-kNN)"
        already = None
        if user >= 0 and item >= 0:
            existing = bundle.matrix.ratings[user, item]
            already = float(existing) if existing else None
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "predicted_rating": round(float(np.clip(value, low, high)), 2),
            "basis": basis,
            "known_user": user >= 0,
            "movie_in_model": item >= 0,
            "existing_rating": already,
        }
