"""Chargement des modeles de production, une seule fois au demarrage du service."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.features.ranking import RankingFeatureBuilder
from recsys.models.base import InteractionMatrix, Recommender
from recsys.models.collaborative import ImplicitALS, ItemKNN, PopularityRecommender
from recsys.models.content_based import ContentBasedRecommender, SimilarityIndex
from recsys.models.hybrid import ColdStartRecommender, DiversifiedRecommender, HybridRecommender
from recsys.models.ranking import RankingRecommender
from recsys.serving import training

COMPONENT_CLASSES: dict[str, type[Recommender]] = {
    "als_implicit": ImplicitALS,
    "item_knn": ItemKNN,
    "tfidf": ContentBasedRecommender,
    "embeddings": ContentBasedRecommender,
}
REQUIRED_FILES = (
    training.MANIFEST_FILE,
    training.STATS_FILE,
    training.METRICS_FILE,
    training.MOVIES_FILE,
    training.INDEX_FILE,
    "popularity.pkl",
    f"{training.RANKER_FILE}.json",
    f"{training.RANKER_FILE}.spec.json",
)


class BundleError(RuntimeError):
    """Le bundle de production est absent ou incomplet."""


def artifact_status(directory: Path) -> dict[str, bool]:
    """Presence de chaque fichier attendu (pour /health, sans rien charger)."""
    names = list(REQUIRED_FILES)
    manifest = directory / training.MANIFEST_FILE
    if manifest.is_file():
        components = json.loads(manifest.read_text(encoding="utf-8"))["components"]
        names += [f"{name}.pkl" for name in components]
    return {name: (directory / name).is_file() for name in names}


@dataclass
class ServingBundle:
    """Tout ce dont le service a besoin, charge en memoire."""

    manifest: dict[str, Any]
    stats: dict[str, Any]
    metrics: dict[str, Any]
    movies: pd.DataFrame  # index = movieId, tout le catalogue
    matrix: InteractionMatrix
    components: dict[str, Recommender]
    popularity: PopularityRecommender
    hybrid: HybridRecommender
    ranker: RankingRecommender
    cold_start: ColdStartRecommender
    diversified: dict[str, DiversifiedRecommender]
    index: SimilarityIndex

    @classmethod
    def load(cls, directory: Path) -> ServingBundle:
        missing = [name for name, present in artifact_status(directory).items() if not present]
        if missing:
            raise BundleError(
                f"bundle incomplet dans {directory} (manquants : {missing}). "
                "Lancer scripts/train_production.py"
            )
        manifest = json.loads((directory / training.MANIFEST_FILE).read_text(encoding="utf-8"))
        components = {
            name: COMPONENT_CLASSES[name].load(directory / f"{name}.pkl")
            for name in manifest["components"]
        }
        popularity = PopularityRecommender.load(directory / "popularity.pkl")
        matrix = popularity.matrix
        for model in components.values():
            if not np.array_equal(model.matrix.item_ids, matrix.item_ids):
                raise BundleError("les modeles du bundle n'ont pas le meme catalogue")
            model.matrix = matrix  # une seule copie de la matrice en memoire

        recipe = manifest["hybrid"]
        hybrid = HybridRecommender(
            components, recipe["weights"], recipe["strategy"], recipe["rrf_k"]
        )
        hybrid.matrix = matrix
        movies = pd.read_parquet(directory / training.MOVIES_FILE)
        builder = RankingFeatureBuilder(
            hybrid,
            components,
            movies,
            manifest["ranking"]["n_candidates"],
            manifest["relevance_threshold"],
            reference_year=manifest["ranking"]["reference_year"],
        ).fit(matrix)
        ranker = RankingRecommender.from_artifacts(directory / training.RANKER_FILE, builder)
        cold_start = ColdStartRecommender(
            hybrid, popularity, manifest["cold_start_prior"], recipe["rrf_k"]
        )
        vectors = np.asarray(components["embeddings"].item_vectors_, dtype=np.float32)
        pool = manifest["diversity_pool"]
        diversified = {
            "ranker": DiversifiedRecommender(ranker, vectors, 1.0, pool=pool, list_size=pool),
            "hybrid": DiversifiedRecommender(hybrid, vectors, 1.0, pool=pool, list_size=pool),
        }
        return cls(
            manifest=manifest,
            stats=json.loads((directory / training.STATS_FILE).read_text(encoding="utf-8")),
            metrics=json.loads((directory / training.METRICS_FILE).read_text(encoding="utf-8")),
            movies=movies.set_index("movieId"),
            matrix=matrix,
            components=components,
            popularity=popularity,
            hybrid=hybrid,
            ranker=ranker,
            cold_start=cold_start,
            diversified=diversified,
            index=SimilarityIndex.load(directory / training.INDEX_FILE),
        )
