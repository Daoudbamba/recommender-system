"""Configuration centralisee : un fichier YAML valide par Pydantic.

Tous les chemins sont resolus par rapport a la racine du depot, ce qui rend
les scripts independants du repertoire depuis lequel ils sont lances.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

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


class Config(BaseModel):
    project: ProjectConfig = ProjectConfig()
    dataset: DatasetConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    eda: EdaConfig = EdaConfig()

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
