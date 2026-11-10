"""Nettoyage des tables MovieLens.

Fonctions pures : elles ne modifient jamais le DataFrame recu et ne touchent
pas au disque. L'orchestration (lecture / ecriture) est dans pipeline.py.
"""

from __future__ import annotations

import pandas as pd

NO_GENRE = "(no genres listed)"
_YEAR_PATTERN = r"\((\d{4})\)\s*$"


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Separe l'annee du titre et transforme les genres en liste.

    - `year` : annee de sortie extraite de la fin du titre (nullable) ;
    - `title_clean` : titre sans l'annee ;
    - `genres` : liste de genres, vide quand MovieLens indique
      "(no genres listed)" (on n'invente pas de genre).
    """
    out = movies.drop_duplicates(subset="movieId", keep="first").copy()
    title = out["title"].str.strip()
    out["title"] = title
    out["year"] = title.str.extract(_YEAR_PATTERN, expand=False).astype("Int16")
    out["title_clean"] = title.str.replace(_YEAR_PATTERN, "", regex=True).str.strip()
    genres = out["genres"].fillna(NO_GENRE)
    out["genres"] = [[] if value == NO_GENRE else value.split("|") for value in genres]
    out["n_genres"] = out["genres"].str.len().astype("int8")
    columns = ["movieId", "title", "title_clean", "year", "genres", "n_genres"]
    return out[columns].reset_index(drop=True)


def clean_ratings(ratings: pd.DataFrame, rating_min: float, rating_max: float) -> pd.DataFrame:
    """Supprime les notes invalides et les doublons (utilisateur, film).

    En cas de doublon, on garde la note la plus recente : c'est le dernier
    avis exprime par l'utilisateur. L'horodatage Unix devient une date UTC.
    """
    out = ratings.dropna(subset=["userId", "movieId", "rating", "timestamp"])
    out = out[out["rating"].between(rating_min, rating_max)]
    out = out.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    out = out.drop_duplicates(subset=["userId", "movieId"], keep="last").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out.reset_index(drop=True)


def clean_tags(tags: pd.DataFrame) -> pd.DataFrame:
    """Normalise les tags libres : espaces, casse, vides et doublons."""
    out = tags.dropna(subset=["tag"]).copy()
    out["tag"] = out["tag"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    out = out[out["tag"].str.len() > 0]
    out = out.sort_values(["userId", "movieId", "timestamp"], kind="stable")
    out = out.drop_duplicates(subset=["userId", "movieId", "tag"], keep="last").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True)
    return out.reset_index(drop=True)


def clean_links(links: pd.DataFrame) -> pd.DataFrame:
    """Dedoublonne les liens externes ; les tmdbId manquants restent manquants."""
    return links.drop_duplicates(subset="movieId", keep="first").reset_index(drop=True)


def restrict_to_catalog(df: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Ecarte les lignes dont le movieId n'existe pas dans le catalogue."""
    return df[df["movieId"].isin(movies["movieId"])].reset_index(drop=True)


def filter_min_activity(
    ratings: pd.DataFrame, min_user_ratings: int, min_movie_ratings: int
) -> pd.DataFrame:
    """Garde les utilisateurs et films suffisamment actifs (filtrage k-core).

    Retirer des films peut faire passer un utilisateur sous son seuil, et
    inversement : on itere donc jusqu'a stabilite.
    """
    out = ratings
    while True:
        movie_counts = out["movieId"].value_counts()
        keep_movies = movie_counts.index[movie_counts >= min_movie_ratings]
        step = out[out["movieId"].isin(keep_movies)]
        user_counts = step["userId"].value_counts()
        keep_users = user_counts.index[user_counts >= min_user_ratings]
        step = step[step["userId"].isin(keep_users)]
        if len(step) == len(out):
            return step.reset_index(drop=True)
        out = step
