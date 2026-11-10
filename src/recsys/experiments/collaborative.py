"""Experience du Sprint 2 : entrainer, selectionner et comparer les modeles collaboratifs.

Deroule :
1. deux coupures temporelles definissent validation et test ;
2. chaque configuration de la grille est entrainee sur la periode
   d'entrainement et mesuree sur la validation ;
3. la meilleure configuration de chaque modele est reentrainee sur
   entrainement + validation, mesuree une seule fois sur le test, puis
   sauvegardee dans artifacts/.

Le test ne sert jamais a choisir un hyperparametre.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.evaluation.protocol import EvaluationResult, evaluate
from recsys.evaluation.split import Fold, build_fold, temporal_cutoffs
from recsys.models.base import Recommender
from recsys.models.collaborative import MODEL_REGISTRY
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger(__name__)

RESULTS_FILE = "collaborative_results.json"
PER_USER_FILE = "collaborative_per_user.parquet"
ARTIFACT_DIR = "collaborative"
LOWER_IS_BETTER = {"rmse", "mae"}


def load_ratings(config: Config, sample_users: int | None = None) -> pd.DataFrame:
    """Notes nettoyees ; `sample_users` tire un sous-ensemble pour un essai rapide."""
    path = config.paths.processed_dir / "ratings.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path} absent : lancer d'abord scripts/prepare_data.py")
    ratings = pd.read_parquet(path)
    if sample_users is not None:
        rng = np.random.default_rng(config.project.seed)
        users = ratings["userId"].unique()
        chosen = rng.choice(users, size=min(sample_users, len(users)), replace=False)
        ratings = ratings[ratings["userId"].isin(chosen)]
    return ratings


def build_folds(ratings: pd.DataFrame, config: Config) -> dict[str, Fold]:
    """Folds de validation et de test, construits avec les memes regles."""
    ev = config.evaluation
    val_end, test_end = temporal_cutoffs(ratings, [ev.validation_quantile, ev.test_quantile])
    common = {
        "min_movie_ratings": config.preprocessing.min_movie_ratings,
        "min_user_history": ev.min_user_history,
        "relevance_threshold": ev.relevance_threshold,
    }
    return {
        "validation": build_fold(ratings, val_end, test_end, **common),
        "test": build_fold(ratings, test_end, None, **common),
    }


def fit_and_evaluate(
    name: str, params: dict[str, Any], fold: Fold, config: Config
) -> tuple[dict[str, Any], Recommender, EvaluationResult]:
    """Entraine une configuration sur un fold et retourne sa ligne de resultats."""
    prep, ev = config.preprocessing, config.evaluation
    model = MODEL_REGISTRY[name](**params)
    start = time.perf_counter()
    model.fit(fold.matrix)
    fit_seconds = time.perf_counter() - start
    summary, ranking = evaluate(
        model, fold, ev.ks, (prep.rating_min, prep.rating_max), ev.batch_size
    )
    row = {"model": name, "params": model.params, "fit_seconds": fit_seconds, **summary}
    return row, model, ranking


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None, metric: str) -> bool:
    if best is None:
        return True
    if metric in LOWER_IS_BETTER:
        return candidate[metric] < best[metric]
    return candidate[metric] > best[metric]


def _per_user_frame(name: str, fold: Fold, ranking: EvaluationResult) -> pd.DataFrame:
    history = np.diff(fold.matrix.ratings.indptr)[fold.eval_users]
    frame = pd.DataFrame({key: values for key, values in ranking.per_user.items()})
    frame.insert(0, "model", name)
    frame.insert(1, "userId", fold.matrix.user_ids[fold.eval_users])
    frame.insert(2, "history", history)
    return frame


def write_results(results: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    target = reports_dir / RESULTS_FILE
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def run(
    config: Config, sample_users: int | None = None, models: list[str] | None = None
) -> dict[str, Any]:
    """Execute l'experience complete et ecrit resultats et artefacts."""
    set_seed(config.project.seed)
    ratings = load_ratings(config, sample_users)
    folds = build_folds(ratings, config)
    del ratings
    for fold_name, fold in folds.items():
        logger.info("%s : %s", fold_name, json.dumps(fold.stats))

    results: dict[str, Any] = {
        "seed": config.project.seed,
        "sample_users": sample_users,
        "protocol": config.evaluation.model_dump(),
        "folds": {name: fold.stats for name, fold in folds.items()},
        "validation": [],
        "test": [],
    }
    artifact_dir = config.paths.artifacts_dir / ARTIFACT_DIR
    per_user: list[pd.DataFrame] = []

    for name, grid in config.collaborative.items():
        if models and name not in models:
            continue
        metric = config.evaluation.selection_metric(name)
        best: dict[str, Any] | None = None
        for params in grid:
            row, _, _ = fit_and_evaluate(name, params, folds["validation"], config)
            results["validation"].append(row)
            logger.info(
                "validation | %-14s %s | %s=%.4f | fit %.0fs",
                name,
                params,
                metric,
                row[metric],
                row["fit_seconds"],
            )
            if is_better(row, best, metric):
                best = row

        row, model, ranking = fit_and_evaluate(name, best["params"], folds["test"], config)
        row["selected_on"] = metric
        results["test"].append(row)
        per_user.append(_per_user_frame(name, folds["test"], ranking))
        model.save(artifact_dir / f"{name}.pkl")
        logger.info(
            "TEST       | %-14s | ndcg@10=%.4f recall@10=%.4f coverage@10=%.3f%s",
            name,
            row["ndcg@10"],
            row["recall@10"],
            row["coverage@10"],
            f" rmse={row['rmse']:.4f}" if "rmse" in row else "",
        )
        write_results(results, config.paths.reports_dir)  # ecriture incrementale

    if per_user:
        pd.concat(per_user, ignore_index=True).to_parquet(artifact_dir / PER_USER_FILE)
    write_results(results, config.paths.reports_dir)
    return results
