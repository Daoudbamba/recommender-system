"""Experience du Sprint 3 : filtrage par contenu, TF-IDF contre embeddings.

Les deux approches utilisent le meme modele de recommandation et le meme
protocole que le filtrage collaboratif : seule la representation des films
change. Les tags sont limites a ceux poses avant la date de coupure du fold.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from recsys.config import Config
from recsys.evaluation.split import Fold
from recsys.experiments.runner import Builder, build_folds, load_ratings, run_grid
from recsys.features.embeddings import (
    Encoder,
    build_embedding_features,
    sentence_transformer_encoder,
)
from recsys.features.item_features import ItemFeatures
from recsys.features.item_text import aggregate_tags, build_item_documents, content_coverage
from recsys.features.tfidf import build_tfidf_features
from recsys.models.content_based import ContentBasedRecommender, SimilarityIndex
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger(__name__)

RESULTS_FILE = "content_results.json"
PER_USER_FILE = "content_per_user.parquet"
ARTIFACT_DIR = "content"
INDEX_FILE = "embedding_index.npz"
DOCUMENTS_FILE = "item_documents.parquet"
TFIDF_FIELDS = ("genre_weight", "tag_weight", "title_weight")


class FoldContent:
    """Documents et tags d'un fold, calcules une fois et partages entre configurations."""

    def __init__(self, movies: pd.DataFrame, tags: pd.DataFrame, fold: Fold, config: Config):
        self.tag_counts = aggregate_tags(tags, until=fold.train_end)
        self.documents = build_item_documents(movies, self.tag_counts, config.content.max_tags)


def load_content_tables(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = config.paths.processed_dir
    for name in ("movies", "tags"):
        if not (processed / f"{name}.parquet").is_file():
            raise FileNotFoundError(f"{name}.parquet absent : lancer scripts/prepare_data.py")
    return pd.read_parquet(processed / "movies.parquet"), pd.read_parquet(
        processed / "tags.parquet"
    )


def make_builders(
    contents: dict[int, FoldContent], config: Config, encoder: Encoder
) -> dict[str, Builder]:
    """Constructeurs de modeles ; `contents` associe chaque fold a son contenu."""
    cfg = config.content
    cache_dir = config.paths.features_dir / "embeddings"
    threshold = config.evaluation.relevance_threshold

    def recommender(features: ItemFeatures, name: str, params: dict[str, Any], extra: dict):
        return ContentBasedRecommender(
            features,
            aggregation=params.get("aggregation", "knn"),
            k=params.get("k", cfg.neighbours),
            relevance_threshold=threshold,
            name=name,
            extra_params=extra,
        )

    def tfidf(params: dict[str, Any], fold: Fold) -> ContentBasedRecommender:
        content = contents[id(fold)]
        weights = {field: params.get(field, 0.0) for field in TFIDF_FIELDS}
        features = build_tfidf_features(
            content.documents, content.tag_counts, tag_min_df=cfg.tag_min_df, **weights
        )
        return recommender(features, "tfidf", params, weights)

    def embeddings(params: dict[str, Any], fold: Fold) -> ContentBasedRecommender:
        content = contents[id(fold)]
        features = build_embedding_features(
            content.documents, cfg.embedding_model, encoder, cache_dir
        )
        extra = {"model": cfg.embedding_model, "max_tags": cfg.max_tags}
        return recommender(features, "embeddings", params, extra)

    return {"tfidf": tfidf, "embeddings": embeddings}


def build_catalog_index(
    movies: pd.DataFrame, tags: pd.DataFrame, config: Config, encoder: Encoder
) -> tuple[SimilarityIndex, pd.DataFrame]:
    """Index d'embeddings de tout le catalogue, avec tous les tags disponibles.

    Sert le moteur de films similaires et la recherche semantique : ici il
    n'y a pas d'evaluation, donc pas de coupure temporelle a respecter.
    """
    documents = build_item_documents(movies, aggregate_tags(tags), config.content.max_tags)
    features = build_embedding_features(
        documents,
        config.content.embedding_model,
        encoder,
        config.paths.features_dir / "embeddings",
    )
    return SimilarityIndex(features), documents


def run(
    config: Config,
    sample_users: int | None = None,
    models: list[str] | None = None,
    encoder: Encoder | None = None,
) -> dict[str, Any]:
    """Execute l'experience complete et ecrit resultats, modeles et index."""
    set_seed(config.project.seed)
    encoder = encoder or sentence_transformer_encoder(
        config.content.embedding_model, config.content.embedding_batch_size
    )
    movies, tags = load_content_tables(config)
    folds = build_folds(load_ratings(config, sample_users), config)
    contents = {id(fold): FoldContent(movies, tags, fold, config) for fold in folds.values()}
    coverage = {
        name: content_coverage(
            contents[id(fold)].documents[
                contents[id(fold)].documents["movieId"].isin(fold.matrix.item_ids)
            ]
        )
        for name, fold in folds.items()
    }
    artifact_dir = config.paths.artifacts_dir / ARTIFACT_DIR
    results = run_grid(
        config,
        folds,
        builders=make_builders(contents, config, encoder),
        grids=config.content.grids,
        results_path=config.paths.reports_dir / RESULTS_FILE,
        artifact_dir=artifact_dir,
        extra={
            "sample_users": sample_users,
            "content": config.content.model_dump(exclude={"grids"}),
            "content_coverage": coverage,
        },
        models=models,
        per_user_file=PER_USER_FILE,
    )

    if not models or "embeddings" in models:
        index, documents = build_catalog_index(movies, tags, config, encoder)
        index.save(artifact_dir / INDEX_FILE)
        documents.to_parquet(artifact_dir / DOCUMENTS_FILE)
        logger.info(
            "Index du catalogue : %d films, dimension %d", len(index), index.vectors.shape[1]
        )
    return results
