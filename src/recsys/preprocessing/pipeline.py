"""Orchestration : CSV bruts -> controle qualite -> nettoyage -> Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.config import Config
from recsys.data.loaders import load_raw_tables
from recsys.data.validation import quality_report
from recsys.preprocessing import cleaning
from recsys.utils.logging import get_logger

logger = get_logger(__name__)

QUALITY_FILE = "data_quality.json"
SUMMARY_FILE = "preprocessing_summary.json"


def clean_tables(tables: dict[str, pd.DataFrame], config: Config) -> dict[str, pd.DataFrame]:
    """Applique le nettoyage aux quatre tables et construit `interactions`."""
    prep = config.preprocessing
    movies = cleaning.clean_movies(tables["movies"])
    ratings = cleaning.clean_ratings(tables["ratings"], prep.rating_min, prep.rating_max)
    ratings = cleaning.restrict_to_catalog(ratings, movies)
    tags = cleaning.restrict_to_catalog(cleaning.clean_tags(tables["tags"]), movies)
    links = cleaning.restrict_to_catalog(cleaning.clean_links(tables["links"]), movies)
    interactions = cleaning.filter_min_activity(
        ratings, prep.min_user_ratings, prep.min_movie_ratings
    )
    return {
        "movies": movies,
        "ratings": ratings,
        "tags": tags,
        "links": links,
        "interactions": interactions,
    }


def _shape(df: pd.DataFrame) -> dict[str, int]:
    shape = {"n_rows": int(len(df))}
    for column in ("userId", "movieId"):
        if column in df.columns:
            shape[f"n_{column[:-2]}s"] = int(df[column].nunique())
    return shape


def summarize(raw: dict[str, pd.DataFrame], clean: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Volumes avant / apres nettoyage, pour tracer ce qui a ete retire."""
    summary: dict[str, Any] = {
        name: {"raw": _shape(raw[name]), "clean": _shape(clean[name])} for name in raw
    }
    summary["interactions"] = _shape(clean["interactions"])
    return summary


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run(config: Config) -> dict[str, Any]:
    """Execute le pipeline complet et retourne le resume des volumes."""
    logger.info("Chargement des CSV depuis %s", config.raw_dataset_dir)
    raw = load_raw_tables(config.raw_dataset_dir)

    prep = config.preprocessing
    quality = quality_report(raw, prep.rating_min, prep.rating_max)
    write_json(quality, config.paths.reports_dir / QUALITY_FILE)

    logger.info("Nettoyage")
    clean = clean_tables(raw, config)
    summary = summarize(raw, clean)
    summary["thresholds"] = {
        "min_user_ratings": prep.min_user_ratings,
        "min_movie_ratings": prep.min_movie_ratings,
    }
    write_json(summary, config.paths.reports_dir / SUMMARY_FILE)

    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in clean.items():
        target = config.paths.processed_dir / f"{name}.parquet"
        df.to_parquet(target, index=False)
        logger.info("%-13s %10d lignes -> %s", name, len(df), target.name)
    return summary
