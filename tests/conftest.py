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


def make_config(tmp_path: Path) -> Config:
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


def fake_encoder(texts):
    """Encodeur factice : le vecteur ne depend que du genre present dans le texte."""
    import numpy as np

    return np.array(
        [[1.0, 0.1] if "Sci-Fi" in text else [0.1, 1.0] for text in texts], dtype=np.float32
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return make_config(tmp_path)


@pytest.fixture
def synthetic_project(config: Config) -> Config:
    return build_synthetic_project(config)


def build_synthetic_project(config: Config) -> Config:
    """Mini-projet complet : deux genres, des gouts qui suivent le genre, des arrivees tardives.

    Ecrit ratings / movies / tags dans le dossier temporaire et regle la configuration pour
    que toutes les experiences (collaboratif, contenu, hybride, ranking) tournent en secondes.
    """
    import numpy as np

    rng = np.random.default_rng(2)
    n_movies, n_users = 40, 120
    start = pd.Timestamp("2020-01-01", tz="UTC")
    movies = pd.DataFrame(
        {
            "movieId": range(1, n_movies + 1),
            "title": [f"Film {m} (2000)" for m in range(1, n_movies + 1)],
            "title_clean": [f"Film {m}" for m in range(1, n_movies + 1)],
            "year": pd.array([1990 + m % 30 for m in range(1, n_movies + 1)], dtype="Int16"),
            "genres": [["Sci-Fi"] if m % 2 else ["Romance"] for m in range(1, n_movies + 1)],
        }
    )
    rows = []
    for user in range(1, n_users + 1):
        late = user % 3 == 0  # un tiers des utilisateurs n'arrive qu'en fin de periode
        for movie in rng.choice(np.arange(1, n_movies + 1), size=24, replace=False):
            hour = int(rng.integers(1950, 2000) if late else rng.integers(0, 2000))
            rows.append((user, movie, 4.5 if movie % 2 == user % 2 else 2.0, hour))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "h"])
    ratings["timestamp"] = start + pd.to_timedelta(ratings.pop("h"), "h")
    tags = pd.DataFrame(
        {"userId": [1, 2], "movieId": [1, 3], "tag": ["robots", "robots"], "timestamp": [start] * 2}
    )
    processed = config.paths.processed_dir
    processed.mkdir(parents=True)
    ratings.to_parquet(processed / "ratings.parquet")
    movies.to_parquet(processed / "movies.parquet")
    tags.to_parquet(processed / "tags.parquet")

    config.evaluation.ks = [5, 10]
    config.evaluation.min_user_history = 3
    config.evaluation.validation_quantile = 0.4
    config.evaluation.test_quantile = 0.6
    config.preprocessing.min_movie_ratings = 5
    config.content.tag_min_df = 1
    config.content.neighbours = 10
    config.collaborative = {
        "item_knn": [{"k": 10}],
        "als_implicit": [{"factors": 4, "iterations": 5}],
    }
    config.content.grids = {
        "tfidf": [{"genre_weight": 1.0, "tag_weight": 1.0, "aggregation": "knn"}],
        "embeddings": [{"aggregation": "knn"}],
    }
    config.hybrid.components = {
        "als_implicit": "collaborative",
        "item_knn": "collaborative",
        "tfidf": "content",
        "embeddings": "content",
    }
    config.hybrid.strategies = ["rrf"]
    config.hybrid.weight_sets = [{"als_implicit": 1, "tfidf": 1}]
    config.hybrid.diversity_pool = 10
    config.hybrid.diversity_lambdas = [1.0]
    config.hybrid.popularity_penalties = [0.0]
    config.hybrid.cold_start_users = 20
    config.hybrid.cold_start_history_sizes = [0, 2]
    config.hybrid.cold_start_priors = [0.5]
    config.ranking.n_candidates = 10
    config.ranking.grid = [{"max_depth": 2, "n_estimators": 30, "min_child_weight": 0}]
    return config


@pytest.fixture(scope="session")
def trained_config(tmp_path_factory) -> Config:
    """Mini-projet entierement entraine, bundle de production compris (une fois par session)."""
    from recsys.experiments import collaborative, content, hybrid, ranking
    from recsys.serving import training

    config = build_synthetic_project(make_config(tmp_path_factory.mktemp("serving")))
    collaborative.run(config)
    content.run(config, encoder=fake_encoder)
    hybrid.run(config, encoder=fake_encoder)
    ranking.run(config, encoder=fake_encoder)
    training.train_production(config, encoder=fake_encoder)
    return config


@pytest.fixture(scope="session")
def client(trained_config: Config):
    """Client HTTP de test branche sur l'application, modeles charges."""
    from fastapi.testclient import TestClient

    from api.main import create_app

    with TestClient(create_app(trained_config)) as test_client:
        yield test_client
