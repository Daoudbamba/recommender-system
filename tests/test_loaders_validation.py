from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.data import loaders, validation
from recsys.data.schemas import SCHEMAS


def test_load_table_applies_declared_dtypes(raw_dir: Path):
    ratings = loaders.load_table(raw_dir, "ratings")
    assert dict(ratings.dtypes.astype(str)) == SCHEMAS["ratings"].dtypes


def test_imdb_id_keeps_leading_zeros(raw_dir: Path):
    links = loaders.load_table(raw_dir, "links")
    assert links.loc[0, "imdbId"] == "0114709"
    assert links["tmdbId"].isna().sum() == 1


def test_missing_column_raises_schema_error(raw_dir: Path):
    path = raw_dir / "ratings.csv"
    pd.read_csv(path).drop(columns="timestamp").to_csv(path, index=False)
    with pytest.raises(loaders.SchemaError, match="timestamp"):
        loaders.load_table(raw_dir, "ratings")


def test_extra_columns_are_reported_not_loaded(raw_dir: Path):
    path = raw_dir / "movies.csv"
    pd.read_csv(path).assign(extra=1).to_csv(path, index=False)
    assert loaders.check_schema(path, SCHEMAS["movies"]) == ["extra"]
    assert "extra" not in loaders.load_table(raw_dir, "movies").columns


def test_unknown_table_and_missing_file(raw_dir: Path, tmp_path: Path):
    with pytest.raises(KeyError):
        loaders.load_table(raw_dir, "genome")
    with pytest.raises(FileNotFoundError):
        loaders.load_table(tmp_path, "ratings")


def test_quality_report_counts_known_defects(raw_tables):
    report = validation.quality_report(raw_tables, rating_min=0.5, rating_max=5.0)

    assert report["tables"]["ratings"]["n_rows"] == 9
    assert report["tables"]["ratings"]["duplicate_keys"] == 1
    assert report["tables"]["tags"]["missing"]["tag"] == 1
    assert report["tables"]["links"]["missing"]["tmdbId"] == 1
    assert report["ratings"]["out_of_range"] == 1
    assert report["ratings"]["not_half_star"] == 0
    assert report["movies"] == {
        "no_genres_listed": 1,
        "titles_without_year": 1,
        "duplicate_titles": 0,
    }
    assert report["integrity"]["rated_movies_missing_from_catalog"] == 1
    assert report["integrity"]["catalog_movies_without_rating"] == 1


def test_rating_checks_detect_off_grid_values(raw_tables):
    ratings = raw_tables["ratings"].copy()
    ratings.loc[0, "rating"] = 3.3
    assert validation.rating_checks(ratings, 0.5, 5.0)["not_half_star"] == 1
