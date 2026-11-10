"""Statistiques descriptives du jeu de donnees nettoye.

Chaque fonction retourne un DataFrame ou un dict : le notebook, le rapport
Markdown et les tests partagent exactement les memes calculs.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def overview(ratings: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame) -> dict[str, Any]:
    """Chiffres cles : volumes, densite de la matrice, periode couverte."""
    n_users = int(ratings["userId"].nunique())
    n_rated = int(ratings["movieId"].nunique())
    n_ratings = int(len(ratings))
    return {
        "n_users": n_users,
        "n_movies_catalog": int(len(movies)),
        "n_movies_rated": n_rated,
        "n_ratings": n_ratings,
        "n_tags": int(len(tags)),
        "density": n_ratings / (n_users * n_rated) if n_users and n_rated else 0.0,
        "mean_rating": float(ratings["rating"].mean()),
        "median_rating": float(ratings["rating"].median()),
        "first_rating": ratings["timestamp"].min(),
        "last_rating": ratings["timestamp"].max(),
    }


def rating_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    """Nombre et part de chaque valeur de note."""
    counts = ratings["rating"].value_counts().sort_index()
    return pd.DataFrame(
        {"rating": counts.index, "count": counts.to_numpy(), "share": counts / counts.sum()}
    ).reset_index(drop=True)


def user_activity(ratings: pd.DataFrame) -> pd.DataFrame:
    """Par utilisateur : volume, note moyenne, dispersion, periode d'activite."""
    grouped = ratings.groupby("userId", sort=False)
    out = grouped.agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        first_rating=("timestamp", "min"),
        last_rating=("timestamp", "max"),
    )
    out["active_days"] = (out["last_rating"] - out["first_rating"]).dt.days
    return out


def movie_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    """Par film : nombre de notes et note moyenne."""
    return ratings.groupby("movieId", sort=False).agg(
        n_ratings=("rating", "size"), mean_rating=("rating", "mean")
    )


def describe_counts(counts: pd.Series) -> pd.Series:
    """Resume d'une distribution a longue traine (quantiles extremes inclus)."""
    return counts.describe(percentiles=QUANTILES)


def low_activity(counts: pd.Series, thresholds: list[int]) -> pd.DataFrame:
    """Combien d'entites ont au plus `seuil` notes, et quelle part des notes."""
    total_entities = len(counts)
    total_ratings = counts.sum()
    rows = []
    for threshold in thresholds:
        mask = counts <= threshold
        rows.append(
            {
                "max_ratings": threshold,
                "n_entities": int(mask.sum()),
                "share_entities": float(mask.sum() / total_entities),
                "share_ratings": float(counts[mask].sum() / total_ratings),
            }
        )
    return pd.DataFrame(rows)


def concentration(counts: pd.Series, top_shares: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)):
    """Part des notes captee par les x % d'entites les plus actives."""
    ordered = counts.sort_values(ascending=False).to_numpy()
    total = ordered.sum()
    rows = []
    for share in top_shares:
        n_top = max(1, int(round(len(ordered) * share)))
        rows.append(
            {
                "top_share": share,
                "n_entities": n_top,
                "share_ratings": float(ordered[:n_top].sum() / total),
            }
        )
    return pd.DataFrame(rows)


def genre_stats(movies: pd.DataFrame, popularity: pd.DataFrame) -> pd.DataFrame:
    """Par genre : films au catalogue, notes recues, note moyenne ponderee.

    Un film a plusieurs genres compte dans chacun d'eux. On agrege a partir
    des totaux par film pour ne pas dupliquer les 25 M de notes en memoire.
    """
    per_movie = movies[["movieId", "genres"]].merge(
        popularity, left_on="movieId", right_index=True, how="left"
    )
    per_movie["n_ratings"] = per_movie["n_ratings"].fillna(0)
    per_movie["rating_sum"] = per_movie["n_ratings"] * per_movie["mean_rating"].fillna(0)
    exploded = per_movie.explode("genres").dropna(subset=["genres"])
    out = exploded.groupby("genres").agg(
        n_movies=("movieId", "size"),
        n_ratings=("n_ratings", "sum"),
        rating_sum=("rating_sum", "sum"),
    )
    out["mean_rating"] = out["rating_sum"] / out["n_ratings"]
    out["n_ratings"] = out["n_ratings"].astype("int64")
    out = out.drop(columns="rating_sum").sort_values("n_ratings", ascending=False)
    return out.rename_axis("genre").reset_index()


def ratings_per_year(ratings: pd.DataFrame) -> pd.DataFrame:
    """Par annee civile : volume de notes, note moyenne, utilisateurs actifs."""
    year = ratings["timestamp"].dt.year.rename("year")
    out = ratings.groupby(year).agg(
        n_ratings=("rating", "size"),
        mean_rating=("rating", "mean"),
        active_users=("userId", "nunique"),
    )
    return out.reset_index()


def movies_per_release_year(movies: pd.DataFrame) -> pd.DataFrame:
    """Nombre de films du catalogue par annee de sortie."""
    counts = movies["year"].dropna().astype(int).value_counts().sort_index()
    return pd.DataFrame({"year": counts.index, "n_movies": counts.to_numpy()})


def top_movies(movies: pd.DataFrame, popularity: pd.DataFrame, n: int) -> pd.DataFrame:
    """Les `n` films les plus notes, avec leur note moyenne."""
    top = popularity.nlargest(n, "n_ratings").reset_index()
    return top.merge(movies[["movieId", "title"]], on="movieId")[
        ["title", "n_ratings", "mean_rating"]
    ]


def tag_stats(tags: pd.DataFrame, movies: pd.DataFrame) -> dict[str, Any]:
    """Couverture des tags : c'est elle qui borne l'approche content-based."""
    tagged_movies = int(tags["movieId"].nunique())
    return {
        "n_tag_applications": int(len(tags)),
        "n_distinct_tags": int(tags["tag"].nunique()),
        "n_tagging_users": int(tags["userId"].nunique()),
        "n_tagged_movies": tagged_movies,
        "share_catalog_tagged": tagged_movies / len(movies) if len(movies) else 0.0,
    }


def top_tags(tags: pd.DataFrame, n: int) -> pd.DataFrame:
    """Les `n` tags les plus appliques et le nombre de films concernes."""
    out = tags.groupby("tag").agg(n_applications=("tag", "size"), n_movies=("movieId", "nunique"))
    return out.nlargest(n, "n_applications").reset_index()
