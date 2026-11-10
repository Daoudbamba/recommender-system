"""Controles de qualite sur les donnees brutes.

Chaque fonction retourne des nombres simples (dict) : le rapport est ainsi
serialisable en JSON et reutilisable par le notebook, le rapport Markdown et
les tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recsys.data.schemas import SCHEMAS


def table_quality(df: pd.DataFrame, key: tuple[str, ...]) -> dict[str, Any]:
    """Volumetrie, valeurs manquantes et doublons d'une table."""
    report: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "missing": {col: int(n) for col, n in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if key:
        report["key"] = list(key)
        report["duplicate_keys"] = int(df.duplicated(subset=list(key)).sum())
    return report


def rating_checks(ratings: pd.DataFrame, rating_min: float, rating_max: float) -> dict[str, Any]:
    """Controles specifiques aux notes : bornes, pas de 0.5, horodatage."""
    values = ratings["rating"].to_numpy()
    out_of_range = (values < rating_min) | (values > rating_max)
    doubled = values * 2
    off_grid = ~np.isclose(doubled, np.round(doubled))
    stamps = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    return {
        "out_of_range": int(np.nansum(out_of_range)),
        "not_half_star": int(np.nansum(off_grid & ~np.isnan(values))),
        "distinct_values": sorted(float(v) for v in pd.unique(ratings["rating"].dropna())),
        "non_positive_timestamps": int((ratings["timestamp"] <= 0).sum()),
        "first_rating": stamps.min().isoformat(),
        "last_rating": stamps.max().isoformat(),
    }


def referential_integrity(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Coherence entre tables : chaque movieId reference existe-t-il au catalogue ?"""
    catalog = pd.Index(tables["movies"]["movieId"].unique())
    rated = pd.Index(tables["ratings"]["movieId"].unique())
    tagged = pd.Index(tables["tags"]["movieId"].unique())
    linked = pd.Index(tables["links"]["movieId"].unique())
    return {
        "rated_movies_missing_from_catalog": int(len(rated.difference(catalog))),
        "tagged_movies_missing_from_catalog": int(len(tagged.difference(catalog))),
        "linked_movies_missing_from_catalog": int(len(linked.difference(catalog))),
        "catalog_movies_without_rating": int(len(catalog.difference(rated))),
        "catalog_movies_without_tag": int(len(catalog.difference(tagged))),
        "catalog_movies_without_link": int(len(catalog.difference(linked))),
    }


def movie_checks(movies: pd.DataFrame) -> dict[str, int]:
    """Particularites du catalogue qui comptent pour le nettoyage."""
    titles = movies["title"].str.strip()
    return {
        "no_genres_listed": int((movies["genres"] == "(no genres listed)").sum()),
        "titles_without_year": int((~titles.str.contains(r"\(\d{4}\)\s*$", na=False)).sum()),
        "duplicate_titles": int(titles.duplicated().sum()),
    }


def quality_report(
    tables: dict[str, pd.DataFrame], rating_min: float, rating_max: float
) -> dict[str, Any]:
    """Rapport de qualite complet des quatre tables brutes."""
    return {
        "tables": {name: table_quality(df, SCHEMAS[name].key) for name, df in tables.items()},
        "ratings": rating_checks(tables["ratings"], rating_min, rating_max),
        "movies": movie_checks(tables["movies"]),
        "integrity": referential_integrity(tables),
    }
