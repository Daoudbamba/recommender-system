"""Test de bout en bout : CSV -> Parquet -> rapport, sur le mini-jeu."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from recsys.analysis.report import build_report
from recsys.config import Config
from recsys.preprocessing import pipeline


@pytest.fixture
def processed(config: Config, raw_dir: Path) -> dict:
    return pipeline.run(config)


def test_pipeline_writes_parquet_and_reports(config: Config, processed: dict):
    for name in ("ratings", "movies", "tags", "links", "interactions"):
        assert (config.paths.processed_dir / f"{name}.parquet").is_file()

    quality = json.loads((config.paths.reports_dir / pipeline.QUALITY_FILE).read_text("utf-8"))
    assert quality["tables"]["ratings"]["duplicate_keys"] == 1
    assert processed["ratings"] == {
        "raw": {"n_rows": 9, "n_users": 3, "n_movies": 4},
        "clean": {"n_rows": 6, "n_users": 3, "n_movies": 3},
    }


def test_parquet_round_trip_preserves_types(config: Config, processed: dict):
    movies = pd.read_parquet(config.paths.processed_dir / "movies.parquet")
    ratings = pd.read_parquet(config.paths.processed_dir / "ratings.parquet")

    assert list(movies.loc[movies.movieId == 10, "genres"].iloc[0]) == [
        "Adventure",
        "Animation",
        "Children",
    ]
    assert str(ratings["timestamp"].dt.tz) == "UTC"
    assert ratings["rating"].dtype == "float32"


def test_interactions_respect_thresholds(config: Config, processed: dict):
    interactions = pd.read_parquet(config.paths.processed_dir / "interactions.parquet")
    prep = config.preprocessing

    assert interactions["userId"].value_counts().min() >= prep.min_user_ratings
    assert interactions["movieId"].value_counts().min() >= prep.min_movie_ratings
    assert processed["interactions"]["n_rows"] == len(interactions)


def test_report_is_generated_with_figures(config: Config, processed: dict):
    report = build_report(config)
    text = report.read_text(encoding="utf-8")

    assert "## 3. Chiffres clés" in text
    assert "Notes : **6**" in text
    figures = sorted((config.paths.reports_dir / "figures").glob("*.png"))
    assert len(figures) == 11
    for figure in figures:
        assert f"figures/{figure.name}" in text


def test_report_requires_prepared_data(config: Config):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        build_report(config)
