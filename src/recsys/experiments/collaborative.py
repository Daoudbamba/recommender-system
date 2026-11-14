"""Experience du Sprint 2 : entrainer, selectionner et comparer les modeles collaboratifs."""

from __future__ import annotations

from typing import Any

from recsys.config import Config
from recsys.experiments.runner import (
    Builder,
    build_folds,
    is_better,
    load_ratings,
    run_grid,
)
from recsys.models.collaborative import MODEL_REGISTRY
from recsys.tracking import ExperimentTracker
from recsys.utils.seed import set_seed

RESULTS_FILE = "collaborative_results.json"
PER_USER_FILE = "collaborative_per_user.parquet"
ARTIFACT_DIR = "collaborative"

__all__ = [
    "ARTIFACT_DIR",
    "PER_USER_FILE",
    "RESULTS_FILE",
    "build_folds",
    "is_better",
    "load_ratings",
    "run",
]


def _builder(name: str) -> Builder:
    return lambda params, _fold: MODEL_REGISTRY[name](**params)


def run(
    config: Config, sample_users: int | None = None, models: list[str] | None = None
) -> dict[str, Any]:
    """Execute l'experience complete et ecrit resultats et artefacts."""
    set_seed(config.project.seed)
    folds = build_folds(load_ratings(config, sample_users), config)
    artifact_dir = config.paths.artifacts_dir / ARTIFACT_DIR
    results_path = config.paths.reports_dir / RESULTS_FILE
    tracker = ExperimentTracker(config, "collaborative")
    with tracker.run(
        "collaborative",
        params={"seed": config.project.seed, "sample_users": sample_users,
                "protocol": config.evaluation.model_dump()},
    ):  # fmt: skip
        results = _run(config, folds, artifact_dir, results_path, sample_users, models, tracker)
        tracker.log_artifacts([results_path])
    return results


def _run(config, folds, artifact_dir, results_path, sample_users, models, tracker):
    results = run_grid(
        config,
        folds,
        builders={name: _builder(name) for name in config.collaborative},
        grids=config.collaborative,
        results_path=results_path,
        artifact_dir=artifact_dir,
        extra={"sample_users": sample_users},
        models=models,
        per_user_file=PER_USER_FILE,
        tracker=tracker,
    )
    return results
