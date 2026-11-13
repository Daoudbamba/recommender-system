"""Configuration centralisee : un fichier YAML valide par Pydantic.

Tous les chemins sont resolus par rapport a la racine du depot, ce qui rend
les scripts independants du repertoire depuis lequel ils sont lances.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CONFIG_ENV_VAR = "RECSYS_CONFIG"


class ProjectConfig(BaseModel):
    seed: int = 42


class DatasetConfig(BaseModel):
    name: str
    url: str
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class PathsConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    features_dir: Path
    reports_dir: Path
    artifacts_dir: Path

    @model_validator(mode="after")
    def _resolve(self) -> PathsConfig:
        for name in type(self).model_fields:
            value: Path = getattr(self, name)
            if not value.is_absolute():
                setattr(self, name, PROJECT_ROOT / value)
        return self


class PreprocessingConfig(BaseModel):
    rating_min: float = 0.5
    rating_max: float = 5.0
    min_user_ratings: int = Field(default=20, ge=1)
    min_movie_ratings: int = Field(default=20, ge=1)

    @model_validator(mode="after")
    def _check_range(self) -> PreprocessingConfig:
        if self.rating_min >= self.rating_max:
            raise ValueError("rating_min doit etre strictement inferieur a rating_max")
        return self


class EdaConfig(BaseModel):
    activity_thresholds: list[int] = [1, 5, 10, 20, 50, 100]
    top_n: int = 15


class EvaluationConfig(BaseModel):
    """Protocole d'evaluation : coupures temporelles, pertinence, metriques."""

    validation_quantile: float = Field(default=0.8, gt=0, lt=1)
    test_quantile: float = Field(default=0.9, gt=0, lt=1)
    relevance_threshold: float = 4.0
    min_user_history: int = Field(default=5, ge=1)
    ks: list[int] = [5, 10, 20]
    batch_size: int = Field(default=256, ge=1)
    # Metrique de selection des hyperparametres, par modele (`default` sinon).
    selection: dict[str, str] = {"default": "ndcg@10"}

    @model_validator(mode="after")
    def _check_order(self) -> EvaluationConfig:
        if self.validation_quantile >= self.test_quantile:
            raise ValueError("validation_quantile doit preceder test_quantile")
        return self

    def selection_metric(self, model_name: str) -> str:
        return self.selection.get(model_name, self.selection["default"])


class ContentConfig(BaseModel):
    """Filtrage par contenu : construction des textes, modele d'embeddings, grilles."""

    max_tags: int = Field(default=15, ge=0)
    tag_min_df: int = Field(default=3, ge=1)
    neighbours: int = Field(default=50, ge=1)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = Field(default=128, ge=1)
    grids: dict[str, list[dict[str, Any]]] = {}


class HybridConfig(BaseModel):
    """Systeme hybride : composants, combinaisons de poids, cold start, diversite."""

    # Nom du composant -> experience qui l'a selectionne ("collaborative" ou "content").
    components: dict[str, str] = {}
    strategies: list[str] = ["weighted", "zscore", "rrf"]
    weight_sets: list[dict[str, float]] = []
    rrf_k: int = Field(default=60, ge=1)
    diversity_pool: int = Field(default=50, ge=2)
    diversity_lambdas: list[float] = [1.0, 0.9, 0.8, 0.7, 0.5]
    popularity_penalties: list[float] = [0.0, 0.1, 0.2, 0.4]
    cold_start_users: int = Field(default=2000, ge=1)
    cold_start_history_sizes: list[int] = [0, 1, 3, 5, 10]
    cold_start_priors: list[float] = [0.5, 1.0, 3.0]


class RankingConfig(BaseModel):
    """Learning to rank : candidats, jeu de dev, grille XGBoost, schemas de labels."""

    n_candidates: int = Field(default=100, ge=10)
    dev_share: float = Field(default=0.2, gt=0, lt=1)
    label_schemes: list[str] = ["graded", "binary"]
    grid: list[dict[str, Any]] = [{"max_depth": 6}]


class Config(BaseModel):
    project: ProjectConfig = ProjectConfig()
    dataset: DatasetConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    eda: EdaConfig = EdaConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    # Grilles d'hyperparametres : nom du modele -> liste de configurations.
    collaborative: dict[str, list[dict[str, Any]]] = {}
    content: ContentConfig = ContentConfig()
    hybrid: HybridConfig = HybridConfig()
    ranking: RankingConfig = RankingConfig()

    @property
    def raw_dataset_dir(self) -> Path:
        """Dossier contenant les CSV extraits (ex. data/raw/ml-25m)."""
        return self.paths.raw_dir / self.dataset.name


def load_config(path: str | Path | None = None) -> Config:
    """Charge et valide la configuration.

    Ordre de priorite : argument `path`, variable d'environnement
    RECSYS_CONFIG, puis configs/config.yaml.
    """
    resolved = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not resolved.is_file():
        raise FileNotFoundError(f"Fichier de configuration introuvable : {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        return Config.model_validate(yaml.safe_load(handle))


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Configuration par defaut, chargee une seule fois par processus."""
    return load_config()
