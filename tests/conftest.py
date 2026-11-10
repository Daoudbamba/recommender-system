"""Jeux de donnees miniatures, au format exact des CSV MovieLens."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys.config import Config
from recsys.data.schemas import SCHEMAS

T0 = 1_500_000_000  # 2017-07-14


@pytest.fixture
def raw_tables() -> dict[str, pd.DataFrame]:
    """Quatre tables brutes avec les defauts que le nettoyage doit traiter."""
    ratings = pd.DataFrame(
        [
            (1, 10, 4.0, T0),
            (1, 20, 3.5, T0 + 10),
            (1, 30, 5.0, T0 + 20),
            (2, 10, 2.0, T0 + 30),
            (2, 20, 4.5, T0 + 40),
            (2, 10, 5.0, T0 + 50),  # doublon (2, 10) plus recent
            (3, 10, 7.0, T0 + 60),  # note hors bornes
            (3, 20, 1.0, T0 + 70),
            (3, 99, 3.0, T0 + 80),  # film absent du catalogue
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    ).astype(SCHEMAS["ratings"].dtypes)
    movies = pd.DataFrame(
        [
            (10, "Toy Story (1995)", "Adventure|Animation|Children"),
            (20, "Heat (1995) ", "Action|Crime|Thriller"),
            (30, "Mystery Film", "(no genres listed)"),
            (40, "Never Rated (2001)", "Drama"),
        ],
        columns=["movieId", "title", "genres"],
    ).astype(SCHEMAS["movies"].dtypes)
    tags = pd.DataFrame(
        [
            (1, 10, "Pixar", T0),
            (1, 10, " pixar ", T0 + 5),  # meme tag apres normalisation
            (2, 20, "Al  Pacino", T0 + 6),
            (2, 20, None, T0 + 7),  # tag manquant
            (3, 20, "   ", T0 + 8),  # tag vide
        ],
        columns=["userId", "movieId", "tag", "timestamp"],
    ).astype(SCHEMAS["tags"].dtypes)
    links = pd.DataFrame(
        [(10, "0114709", 862), (20, "0113277", 949), (30, "0000001", None), (40, "0000002", 5)],
        columns=["movieId", "imdbId", "tmdbId"],
    ).astype(SCHEMAS["links"].dtypes)
    return {"ratings": ratings, "movies": movies, "tags": tags, "links": links}


@pytest.fixture
def raw_dir(tmp_path: Path, raw_tables: dict[str, pd.DataFrame]) -> Path:
    """Les memes tables ecrites en CSV, comme apres extraction de l'archive."""
    folder = tmp_path / "raw" / "ml-test"
    folder.mkdir(parents=True)
    for name, df in raw_tables.items():
        df.to_csv(folder / SCHEMAS[name].filename, index=False)
    return folder


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Configuration pointant vers un dossier temporaire, seuils adaptes au mini-jeu."""
    return Config.model_validate(
        {
            "dataset": {"name": "ml-test", "url": "file:///unused", "md5": "0" * 32},
            "paths": {
                "raw_dir": tmp_path / "raw",
                "processed_dir": tmp_path / "processed",
                "features_dir": tmp_path / "features",
                "reports_dir": tmp_path / "reports",
                "artifacts_dir": tmp_path / "artifacts",
            },
            "preprocessing": {"min_user_ratings": 2, "min_movie_ratings": 2},
            "eda": {"activity_thresholds": [1, 2], "top_n": 3},
        }
    )
