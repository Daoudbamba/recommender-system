"""Feature engineering du contenu des films : titre, genres, tags.

MovieLens ne fournit aucune description de film ; on n'en invente pas. Le
contenu d'un film se limite a ce que le dataset contient reellement.

Les tags sont horodates : pour une evaluation sans fuite, seuls les tags
poses avant la date de coupure sont utilises (`until`).
"""

from __future__ import annotations

import pandas as pd


def aggregate_tags(tags: pd.DataFrame, until: pd.Timestamp | None = None) -> pd.DataFrame:
    """Nombre d'utilisateurs distincts ayant pose chaque tag sur chaque film.

    Compter des utilisateurs plutot que des applications evite qu'un seul
    utilisateur tres actif n'impose sa description d'un film.
    """
    if until is not None:
        tags = tags[tags["timestamp"] <= until]
    counts = (
        tags.groupby(["movieId", "tag"], sort=False)["userId"]
        .nunique()
        .rename("n_users")
        .reset_index()
    )
    return counts.sort_values(["movieId", "n_users", "tag"], ascending=[True, False, True])


def top_tags_per_movie(tag_counts: pd.DataFrame, max_tags: int) -> dict[int, list[str]]:
    """Les `max_tags` tags les plus consensuels de chaque film (film -> liste de tags).

    Retourne un dictionnaire de listes Python : selon le type de chaine de la
    colonne `tag` (objet ou Arrow), une Series de listes change de nature et
    se prete mal aux tests de type.
    """
    top = tag_counts.groupby("movieId", sort=False).head(max_tags)
    out: dict[int, list[str]] = {}
    for movie_id, tag in zip(top["movieId"].tolist(), top["tag"].tolist(), strict=True):
        out.setdefault(movie_id, []).append(str(tag))
    return out


def format_text(title: str, year: object, genres: list[str], tags: list[str]) -> str:
    """Phrase descriptive d'un film, destinee au modele d'embeddings."""
    parts = [f"{title} ({int(year)})" if pd.notna(year) else title]
    if len(genres):
        parts.append("Genres: " + ", ".join(genres))
    if len(tags):
        parts.append("Tags: " + ", ".join(tags))
    return ". ".join(parts) + "."


def build_item_documents(
    movies: pd.DataFrame, tag_counts: pd.DataFrame, max_tags: int
) -> pd.DataFrame:
    """Une ligne par film du catalogue : genres, tags retenus et texte descriptif."""
    docs = movies[["movieId", "title", "title_clean", "year", "genres"]].copy()
    docs["genres"] = docs["genres"].map(list)
    top = top_tags_per_movie(tag_counts, max_tags)
    docs["tags"] = [top.get(movie_id, []) for movie_id in docs["movieId"].tolist()]
    docs["text"] = [
        format_text(title, year, genres, tags)
        for title, year, genres, tags in zip(
            docs["title_clean"], docs["year"], docs["genres"], docs["tags"], strict=True
        )
    ]
    return docs.reset_index(drop=True)


def content_coverage(documents: pd.DataFrame) -> dict[str, float]:
    """Part des films disposant de chaque type d'information."""
    has_genre = documents["genres"].map(len) > 0
    has_tag = documents["tags"].map(len) > 0
    return {
        "n_movies": int(len(documents)),
        "share_with_genres": float(has_genre.mean()),
        "share_with_tags": float(has_tag.mean()),
        "share_with_year": float(documents["year"].notna().mean()),
        "share_title_only": float((~has_genre & ~has_tag).mean()),
        "median_tags_when_tagged": float(documents.loc[has_tag, "tags"].map(len).median())
        if has_tag.any()
        else 0.0,
    }
