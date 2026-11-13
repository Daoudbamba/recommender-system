"""Experience du Sprint 5 : learning to rank avec XGBoost.

Chronologie, pour qu'aucune information du futur n'entre dans l'entrainement :

    ---- T0 (juin 2016) ---------- T1 (janvier 2018) ---------- fin (2019)
    modeles de base A      labels d'entrainement du ranker
    modeles de base B (jusqu'a T1)                       labels de test

- entrainement du ranker : candidats et variables calcules par les modeles
  de base entraines jusqu'a T0 ; labels = ce que les utilisateurs ont note
  entre T0 et T1 ;
- choix des hyperparametres : 20 % de ces utilisateurs, mis de cote ;
- test : candidats et variables calcules par les modeles entraines jusqu'a
  T1 ; labels = notes posterieures a T1. Le test ne sert qu'a la mesure.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from recsys.config import Config
from recsys.evaluation import metrics
from recsys.evaluation.protocol import _relevance_matrix, evaluate
from recsys.evaluation.split import Fold
from recsys.experiments import content, hybrid
from recsys.experiments.runner import build_folds, load_ratings, per_user_frame, write_results
from recsys.features.embeddings import Encoder, sentence_transformer_encoder
from recsys.features.ranking import (
    RankingFeatureBuilder,
    binary_labels,
    graded_labels,
)
from recsys.models.hybrid import HybridRecommender
from recsys.models.ranking import RankingRecommender, group_ids, train_ranker
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger(__name__)

RESULTS_FILE = "ranking_results.json"
PER_USER_FILE = "ranking_per_user.parquet"
ARTIFACT_DIR = "ranking"
MODEL_FILE = "xgb_ranker"
SCORE_PREFIXES = ("score_", "log_rank_")
RECENCY_FEATURES = ("item_age", "year_gap")


def hybrid_recipe(config: Config) -> dict[str, Any]:
    """Strategie et poids de l'hybride retenu au Sprint 4."""
    path = config.paths.reports_dir / hybrid.RESULTS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} absent : lancer d'abord scripts/train_hybrid.py")
    return json.loads(path.read_text(encoding="utf-8"))["best"]["params"]


def holdout_ratings(fold: Fold) -> sp.csr_matrix:
    frame = fold.holdout
    shape = (fold.matrix.n_users, fold.matrix.n_items)
    return sp.csr_matrix(
        (frame["rating"].to_numpy(dtype=np.float32), (frame["user_idx"], frame["item_idx"])),
        shape=shape,
    )


def build_dataset(
    builder: RankingFeatureBuilder, users: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Candidats (n, c) et variables (n * c, f) pour une liste d'utilisateurs, dans l'ordre."""
    items, features = [], []
    for start in range(0, len(users), batch_size):
        batch = builder.build(users[start : start + batch_size])
        items.append(batch.items)
        features.append(batch.flat_features)
    return np.vstack(items), np.vstack(features)


def top_k_metrics(
    scores: np.ndarray, items: np.ndarray, users: np.ndarray, relevance: sp.csr_matrix, k: int
) -> dict[str, float]:
    """NDCG, rappel et couverture a k d'un classement des candidats (pertinence binaire)."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    recommended = np.take_along_axis(items, order, axis=1)
    rows = np.repeat(users, recommended.shape[1])
    hits = np.asarray(relevance[rows, recommended.ravel()]).reshape(recommended.shape)
    n_relevant = np.asarray(relevance[users].sum(axis=1)).ravel()
    return {
        f"ndcg@{k}": float(metrics.ndcg_at_k(hits, n_relevant, k).mean()),
        f"recall@{k}": float(metrics.recall_at_k(hits, n_relevant, k).mean()),
        f"coverage@{k}": metrics.catalog_coverage(recommended, relevance.shape[1], k),
    }


def candidate_recall(items: np.ndarray, users: np.ndarray, relevance: sp.csr_matrix) -> float:
    """Part des films pertinents presents parmi les candidats : plafond du rappel du ranker."""
    rows = np.repeat(users, items.shape[1])
    hits = np.asarray(relevance[rows, items.ravel()]).reshape(items.shape)
    n_relevant = np.asarray(relevance[users].sum(axis=1)).ravel()
    return float((hits.sum(axis=1) / np.maximum(n_relevant, 1)).mean())


def make_builder(
    fold: Fold, components: dict, recipe: dict[str, Any], movies: pd.DataFrame, config: Config
) -> RankingFeatureBuilder:
    model = HybridRecommender(
        components, recipe["weights"], recipe["strategy"], config.hybrid.rrf_k
    ).fit(fold.matrix)
    return RankingFeatureBuilder(
        model, components, movies, config.ranking.n_candidates,
        config.evaluation.relevance_threshold,
        reference_year=fold.train_end.year + (fold.train_end.dayofyear - 1) / 365.25,
    ).fit(fold.matrix)  # fmt: skip


def run(
    config: Config, sample_users: int | None = None, encoder: Encoder | None = None
) -> dict[str, Any]:
    """Execute l'experience complete et ecrit resultats et modele."""
    set_seed(config.project.seed)
    cfg, ev = config.ranking, config.evaluation
    encoder = encoder or sentence_transformer_encoder(
        config.content.embedding_model, config.content.embedding_batch_size
    )
    movies, tags = content.load_content_tables(config)
    folds = build_folds(load_ratings(config, sample_users), config)
    contents = {id(f): content.FoldContent(movies, tags, f, config) for f in folds.values()}
    content_builders = content.make_builders(contents, config, encoder)
    params = hybrid.selected_params(config)
    recipe = hybrid_recipe(config)
    builders = {
        name: make_builder(
            fold, hybrid.fit_components(fold, params, config, content_builders), recipe, movies,
            config,
        )
        for name, fold in folds.items()
    }  # fmt: skip
    names = builders["validation"].feature_names

    # --- Jeu d'entrainement du ranker : utilisateurs evalues de la periode de validation.
    train_fold = folds["validation"]
    rng = np.random.default_rng(config.project.seed)
    users = rng.permutation(train_fold.eval_users)
    n_dev = max(1, int(round(len(users) * cfg.dev_share)))
    dev_users, fit_users = np.sort(users[:n_dev]), np.sort(users[n_dev:])
    start = time.perf_counter()
    fit_items, fit_x = build_dataset(builders["validation"], fit_users, ev.batch_size)
    dev_items, dev_x = build_dataset(builders["validation"], dev_users, ev.batch_size)
    feature_seconds = time.perf_counter() - start
    ratings = holdout_ratings(train_fold)
    relevance = _relevance_matrix(train_fold)
    threshold = ev.relevance_threshold
    labels = {
        "graded": (
            graded_labels(fit_items, ratings, fit_users, threshold),
            graded_labels(dev_items, ratings, dev_users, threshold),
        )
    }
    labels["binary"] = tuple(binary_labels(part) for part in labels["graded"])
    n_candidates = fit_items.shape[1]
    fit_qid, dev_qid = (
        group_ids(len(fit_users), n_candidates),
        group_ids(len(dev_users), n_candidates),
    )
    graded_fit = labels["graded"][0]

    results: dict[str, Any] = {
        "seed": config.project.seed,
        "sample_users": sample_users,
        "protocol": ev.model_dump(),
        "folds": {name: fold.stats for name, fold in folds.items()},
        "hybrid": recipe,
        "feature_names": names,
        "dataset": {
            "n_candidates": int(n_candidates),
            "train_users": int(len(fit_users)),
            "dev_users": int(len(dev_users)),
            "train_rows": int(fit_x.shape[0]),
            "label_counts": {
                "liked": int((graded_fit == 2).sum()),
                "unknown": int((graded_fit == 1).sum()),
                "disliked": int((graded_fit == 0).sum()),
            },
            "users_without_positive_candidate": int(((graded_fit == 2).sum(axis=1) == 0).sum()),
            "candidate_recall_dev": candidate_recall(dev_items, dev_users, relevance),
            "feature_seconds": feature_seconds,
        },
        "validation": [],
        "test": [],
    }

    # --- Reference sur le jeu de dev : l'ordre de l'hybride, sans ranker.
    hybrid_column = names.index("score_hybrid")
    hybrid_dev = dev_x[:, hybrid_column].reshape(dev_items.shape)
    results["dev_reference"] = top_k_metrics(hybrid_dev, dev_items, dev_users, relevance, 10)
    logger.info("dev | ordre de l'hybride | %s", results["dev_reference"])

    best: dict[str, Any] | None = None
    for scheme in cfg.label_schemes:
        fit_y, dev_y = (part.ravel() for part in labels[scheme])
        for grid_params in cfg.grid:
            start = time.perf_counter()
            model = train_ranker(
                fit_x, fit_y, fit_qid, (dev_x, dev_y, dev_qid), grid_params, config.project.seed
            )
            seconds = time.perf_counter() - start
            scores = model.predict(dev_x).reshape(dev_items.shape)
            row = {
                "model": "xgb_ranker",
                "params": {"labels": scheme, **grid_params, "best_iteration": model.best_iteration},
                "fit_seconds": seconds,
                **top_k_metrics(scores, dev_items, dev_users, relevance, 10),
            }
            results["validation"].append(row)
            logger.info(
                "dev | %-7s %s | ndcg@10=%.4f (%d arbres)",
                scheme, grid_params, row["ndcg@10"], model.best_iteration + 1,
            )  # fmt: skip
            if best is None or row["ndcg@10"] > best["ndcg@10"]:
                best = row

    # --- Modeles finaux : meme reglage, reentraines sur tous les utilisateurs disponibles.
    best_params = {k: v for k, v in best["params"].items() if k not in {"labels", "best_iteration"}}
    final_params = {**best_params, "n_estimators": best["params"]["best_iteration"] + 1}
    all_x = np.vstack([fit_x, dev_x])
    all_y = np.concatenate([part.ravel() for part in labels[best["params"]["labels"]]])
    all_qid = group_ids(len(fit_users) + len(dev_users), n_candidates)
    score_columns = [i for i, name in enumerate(names) if name.startswith(SCORE_PREFIXES)]
    no_recency = [i for i, name in enumerate(names) if name not in RECENCY_FEATURES]
    variants = {
        "xgb_ranker": None,
        "xgb_ranker_no_recency": no_recency,
        "xgb_ranker_scores_only": score_columns,
    }
    results["best"] = {"params": best["params"], "dev_ndcg@10": best["ndcg@10"]}

    test_fold = folds["test"]
    artifact_dir = config.paths.artifacts_dir / ARTIFACT_DIR
    per_user = []
    for name, columns in variants.items():
        x = all_x if columns is None else all_x[:, columns]
        start = time.perf_counter()
        model = train_ranker(x, all_y, all_qid, None, final_params, config.project.seed)
        seconds = time.perf_counter() - start
        recommender = RankingRecommender(builders["test"], model, columns, name, ev.batch_size)
        summary, ranking = evaluate(
            recommender, test_fold, ev.ks, (config.preprocessing.rating_min,
                                            config.preprocessing.rating_max), ev.batch_size,
        )  # fmt: skip
        results["test"].append(
            {"model": name, "params": recommender.params, "fit_seconds": seconds, **summary}
        )
        per_user.append(per_user_frame(name, test_fold, ranking))
        logger.info(
            "TEST | %-24s | ndcg@10=%.4f recall@10=%.4f coverage@10=%.3f",
            name, summary["ndcg@10"], summary["recall@10"], summary["coverage@10"],
        )  # fmt: skip
        if columns is None:
            recommender.save(artifact_dir / MODEL_FILE)
            results["feature_importance"] = recommender.feature_importance()

    test_items, _ = build_dataset(
        builders["test"], test_fold.eval_users[: 4 * ev.batch_size], ev.batch_size
    )
    results["dataset"]["candidate_recall_test_sample"] = candidate_recall(
        test_items, test_fold.eval_users[: 4 * ev.batch_size], _relevance_matrix(test_fold)
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(per_user, ignore_index=True).to_parquet(artifact_dir / PER_USER_FILE)
    write_results(results, config.paths.reports_dir / RESULTS_FILE)
    return results
