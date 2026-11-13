"""Entrainement des modeles de production.

L'evaluation des sprints 2 a 5 reserve les notes les plus recentes au test.
Une fois les choix arretes, on reentraine sur TOUTES les donnees pour
servir : les hyperparametres, les poids de l'hybride et le prior de cold
start sont ceux retenus pendant l'evaluation, rien n'est re-regle ici.

Seule exception, le modele de ranking : il est repris tel qu'evalue. Le
reentrainer sur des labels plus recents le rendrait probablement meilleur,
mais plus aucune donnee ne resterait pour le mesurer.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.analysis import eda
from recsys.config import Config
from recsys.experiments import collaborative, content, hybrid, ranking
from recsys.experiments.runner import load_ratings
from recsys.features.embeddings import Encoder, sentence_transformer_encoder
from recsys.features.item_text import aggregate_tags, build_item_documents
from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative import MODEL_REGISTRY, PopularityRecommender
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger(__name__)

BUNDLE_DIR = "production"
MANIFEST_FILE = "manifest.json"
STATS_FILE = "stats.json"
METRICS_FILE = "metrics.json"
MOVIES_FILE = "movies.parquet"
INDEX_FILE = "embedding_index.npz"
RANKER_FILE = "xgb_ranker"
ACTIVITY_EDGES = (30, 50, 100, 200, 500, 1000, 5000)
RESULT_FILES = {
    "collaborative": collaborative.RESULTS_FILE,
    "content": content.RESULTS_FILE,
    "hybrid": hybrid.RESULTS_FILE,
    "ranking": ranking.RESULTS_FILE,
}


def fractional_year(stamp: pd.Timestamp) -> float:
    return stamp.year + (stamp.dayofyear - 1) / 365.25


def dataset_stats(
    ratings: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame, matrix: InteractionMatrix
) -> dict[str, Any]:
    """Chiffres de la page « Data Insights », calcules une fois a l'entrainement."""
    overview = eda.overview(ratings, movies, tags)
    popularity = eda.movie_popularity(ratings)
    users = eda.user_activity(ratings)
    # Bornes adaptees aux donnees : sur un petit jeu, le maximum peut etre sous 5 000.
    low, top = int(users["n_ratings"].min()), int(users["n_ratings"].max()) + 1
    inner = [edge for edge in ACTIVITY_EDGES if low < edge < top]
    edges = [low, *inner, top]
    counts, _ = np.histogram(users["n_ratings"], bins=edges)
    return {
        "overview": {
            **{k: v for k, v in overview.items() if not isinstance(v, pd.Timestamp)},
            "first_rating": overview["first_rating"].isoformat(),
            "last_rating": overview["last_rating"].isoformat(),
            "n_movies_in_model": matrix.n_items,
            "n_users_in_model": matrix.n_users,
        },
        "rating_distribution": eda.rating_distribution(ratings).to_dict("records"),
        "genres": eda.genre_stats(movies, popularity).to_dict("records"),
        "ratings_per_year": eda.ratings_per_year(ratings).to_dict("records"),
        "user_activity": [
            {"from": int(lo), "to": int(hi) - 1, "users": int(n)}
            for lo, hi, n in zip(edges[:-1], edges[1:], counts, strict=True)
        ],
        "user_activity_median": float(users["n_ratings"].median()),
        "top_movies": eda.top_movies(movies, popularity, 20).to_dict("records"),
    }


def collect_metrics(config: Config) -> dict[str, Any]:
    """Resultats de test des quatre experiences, pour la page « Model Performance »."""
    out: dict[str, Any] = {"models": [], "protocol": None, "test_fold": None}
    for source, filename in RESULT_FILES.items():
        path = config.paths.reports_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"{path} absent : lancer d'abord l'experience {source}")
        results = json.loads(path.read_text(encoding="utf-8"))
        out["protocol"] = results["protocol"]
        out["test_fold"] = results["folds"]["test"]
        for row in results["test"]:
            out["models"].append({**row, "source": source, "params": str(row["params"])})
    return out


def movie_catalogue(
    movies: pd.DataFrame, ratings: pd.DataFrame, matrix: InteractionMatrix
) -> pd.DataFrame:
    """Catalogue complet enrichi : nombre de notes, note moyenne, presence dans les modeles."""
    stats = eda.movie_popularity(ratings)
    out = movies[["movieId", "title", "title_clean", "year", "genres"]].merge(
        stats, left_on="movieId", right_index=True, how="left"
    )
    out["n_ratings"] = out["n_ratings"].fillna(0).astype("int32")
    out["in_model"] = out["movieId"].isin(matrix.item_ids)
    return out.sort_values("n_ratings", ascending=False).reset_index(drop=True)


def train_production(config: Config, encoder: Encoder | None = None) -> Path:
    """Entraine tous les modeles sur l'ensemble des donnees et ecrit le bundle de service."""
    set_seed(config.project.seed)
    encoder = encoder or sentence_transformer_encoder(
        config.content.embedding_model, config.content.embedding_batch_size
    )
    params = hybrid.selected_params(config)
    recipe = ranking.hybrid_recipe(config)
    hybrid_results = json.loads(
        (config.paths.reports_dir / hybrid.RESULTS_FILE).read_text(encoding="utf-8")
    )
    metrics = collect_metrics(config)
    ranker_source = config.paths.artifacts_dir / ranking.ARTIFACT_DIR / ranking.MODEL_FILE
    for suffix in (".json", ".spec.json"):
        if not ranker_source.with_suffix(suffix).is_file():
            raise FileNotFoundError("modele de ranking absent : lancer scripts/train_ranking.py")

    ratings = load_ratings(config)
    movies, tags = content.load_content_tables(config)
    counts = ratings["movieId"].value_counts()
    kept = counts.index[counts >= config.preprocessing.min_movie_ratings]
    matrix = InteractionMatrix.from_frame(ratings[ratings["movieId"].isin(kept)])
    logger.info(
        "matrice de production : %d utilisateurs x %d films", matrix.n_users, matrix.n_items
    )

    tag_counts = aggregate_tags(tags)  # tous les tags : plus de coupure a respecter
    documents = build_item_documents(movies, tag_counts, config.content.max_tags)
    bundle = config.paths.artifacts_dir / BUNDLE_DIR
    bundle.mkdir(parents=True, exist_ok=True)

    models: dict[str, Recommender] = {
        "popularity": PopularityRecommender(config.evaluation.relevance_threshold)
    }
    for name, source in config.hybrid.components.items():
        if source == "collaborative":
            models[name] = MODEL_REGISTRY[name](**params[name])
        else:
            models[name] = content.build_content_model(
                name, params[name], documents, tag_counts, config, encoder
            )
    for name, model in models.items():
        model.fit(matrix).save(bundle / f"{name}.pkl")
        logger.info("modele %-12s entraine et sauvegarde", name)

    index, _ = content.build_catalog_index(movies, tags, config, encoder)
    index.save(bundle / INDEX_FILE)
    movie_catalogue(movies, ratings, matrix).to_parquet(bundle / MOVIES_FILE)
    for suffix in (".json", ".spec.json"):
        shutil.copyfile(
            ranker_source.with_suffix(suffix), (bundle / RANKER_FILE).with_suffix(suffix)
        )
    stats = dataset_stats(ratings, movies, tags, matrix)
    (bundle / STATS_FILE).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (bundle / METRICS_FILE).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    last_rating = ratings["timestamp"].max()
    manifest = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": config.project.seed,
        "data": {
            "last_rating": last_rating.isoformat(),
            "n_ratings": int(matrix.ratings.nnz),
            "n_users": matrix.n_users,
            "n_items": matrix.n_items,
        },
        "components": {name: {"source": config.hybrid.components[name], "params": params[name]}
                       for name in config.hybrid.components},
        "hybrid": {**recipe, "rrf_k": config.hybrid.rrf_k},
        "cold_start_prior": hybrid_results["cold_users"]["prior_strength"],
        "ranking": {
            "n_candidates": config.ranking.n_candidates,
            "reference_year": fractional_year(last_rating),
            "note": "modele repris de l'evaluation (labels 2016-2018), non reentraine",
        },
        "diversity_pool": config.hybrid.diversity_pool,
        "relevance_threshold": config.evaluation.relevance_threshold,
        "rating_bounds": [config.preprocessing.rating_min, config.preprocessing.rating_max],
    }  # fmt: skip
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("bundle de production ecrit dans %s", bundle)
    return bundle
