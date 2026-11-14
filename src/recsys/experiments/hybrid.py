"""Experience du Sprint 4 : hybride, diversite et cold start.

1. Les composants (meilleure configuration de chaque modele des sprints 2 et
   3) sont reentraines sur chaque fold ;
2. chaque strategie de combinaison est reglee sur la validation (choix des
   poids) puis mesuree une fois sur le test ;
3. sur le test, le meilleur hybride est re-classe avec plusieurs niveaux de
   diversite et de frein a la popularite : le compromis est mesure, aucun
   niveau n'est « selectionne » sur le test ;
4. le cold start est evalue sur les utilisateurs et les films absents de
   l'entrainement.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.evaluation import metrics
from recsys.evaluation.cold_start import (
    build_cold_user_scenario,
    evaluate_cold_users,
    evaluate_new_items,
)
from recsys.evaluation.protocol import evaluate
from recsys.evaluation.split import Fold
from recsys.experiments import collaborative, content
from recsys.experiments.runner import Builder, build_folds, load_ratings, run_grid, write_results
from recsys.features.embeddings import Encoder, sentence_transformer_encoder
from recsys.models.base import Recommender
from recsys.models.collaborative import MODEL_REGISTRY, PopularityRecommender
from recsys.models.hybrid import ColdStartRecommender, DiversifiedRecommender, HybridRecommender
from recsys.tracking import ExperimentTracker
from recsys.utils.logging import get_logger
from recsys.utils.seed import set_seed

logger = get_logger(__name__)

RESULTS_FILE = "hybrid_results.json"
PER_USER_FILE = "hybrid_per_user.parquet"
ARTIFACT_DIR = "hybrid"
BEST_FILE = "hybrid.pkl"
SOURCES = {"collaborative": collaborative.RESULTS_FILE, "content": content.RESULTS_FILE}


def selected_params(config: Config) -> dict[str, dict[str, Any]]:
    """Hyperparametres retenus aux sprints precedents pour chaque composant."""
    out = {}
    for name, source in config.hybrid.components.items():
        if source not in SOURCES:
            raise ValueError(f"source inconnue pour {name} : {source}")
        path = config.paths.reports_dir / SOURCES[source]
        if not path.is_file():
            raise FileNotFoundError(f"{path} absent : lancer d'abord l'experience {source}")
        rows = json.loads(path.read_text(encoding="utf-8"))["test"]
        match = [row["params"] for row in rows if row["model"] == name]
        if not match:
            raise KeyError(f"{name} absent de {path.name}")
        out[name] = match[0]
    return out


def fit_components(
    fold: Fold, params: dict[str, dict[str, Any]], config: Config, content_builders: dict
) -> dict[str, Recommender]:
    """Entraine sur un fold tous les composants de l'hybride."""
    components: dict[str, Recommender] = {}
    for name, source in config.hybrid.components.items():
        if source == "collaborative":
            model = MODEL_REGISTRY[name](**params[name])
        else:
            model = content_builders[name](params[name], fold)
        components[name] = model.fit(fold.matrix)
        logger.info("composant %-12s entraine (%d films)", name, fold.matrix.n_items)
    return components


def _hybrid_builder(components: dict[int, dict[str, Recommender]], strategy: str, rrf_k: int):
    def build(params: dict[str, Any], fold: Fold) -> HybridRecommender:
        return HybridRecommender(
            components[id(fold)], params["weights"], strategy, rrf_k, name=f"hybrid_{strategy}"
        )

    return build


def genre_vectors(movies: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    """Vecteurs de genres normalises (mesure de diversite independante des embeddings)."""
    genres = movies.set_index("movieId").loc[item_ids, "genres"]
    names = sorted({genre for row in genres for genre in row})
    index = {name: column for column, name in enumerate(names)}
    out = np.zeros((len(item_ids), len(names)), dtype=np.float32)
    for row, values in enumerate(genres):
        for genre in values:
            out[row, index[genre]] = 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=out, where=norms > 0)


def diversity_sweep(
    best: Recommender,
    fold: Fold,
    embedding_vectors: np.ndarray,
    genres: np.ndarray,
    config: Config,
) -> list[dict[str, Any]]:
    """Pertinence, diversite et popularite pour chaque reglage de re-classement."""
    prep, ev, cfg = config.preprocessing, config.evaluation, config.hybrid
    settings = [(lam, 0.0) for lam in cfg.diversity_lambdas]
    settings += [(1.0, penalty) for penalty in cfg.popularity_penalties if penalty > 0]
    rows = []
    for lam, penalty in settings:
        model = DiversifiedRecommender(
            best, embedding_vectors, lam, penalty, cfg.diversity_pool, list_size=max(ev.ks)
        )
        summary, ranking = evaluate(
            model, fold, ev.ks, (prep.rating_min, prep.rating_max), ev.batch_size
        )
        recommended = ranking.recommended
        rows.append(
            {
                "lambda": lam,
                "popularity_penalty": penalty,
                **{key: summary[key] for key in summary if key.endswith("@10")},
                "ild_embeddings@10": float(
                    metrics.intra_list_diversity(recommended, embedding_vectors, 10).mean()
                ),
                "ild_genres@10": float(
                    metrics.intra_list_diversity(recommended, genres, 10).mean()
                ),
                "ms_per_user": summary["ms_per_user"],
            }
        )
        logger.info(
            "diversite  | lambda=%.1f penalite=%.1f | ndcg@10=%.4f ild=%.3f couverture=%.3f",
            lam, penalty, rows[-1]["ndcg@10"], rows[-1]["ild_embeddings@10"],
            rows[-1]["coverage@10"],
        )  # fmt: skip
    return rows


def select_cold_start_prior(
    best: HybridRecommender, fold: Fold, config: Config
) -> tuple[float, list[dict[str, Any]]]:
    """Choisit la force du prior de popularite sur les nouveaux venus de la validation.

    Critere : NDCG@10 moyen sur toutes les tailles d'historique. Les
    utilisateurs froids du test ne servent qu'a la mesure finale.
    """
    cfg = config.hybrid
    scenario = build_cold_user_scenario(
        fold, cfg.cold_start_history_sizes, cfg.cold_start_users, config.project.seed
    )
    popularity = PopularityRecommender(config.evaluation.relevance_threshold).fit(fold.matrix)
    rows = []
    for prior in cfg.cold_start_priors:
        model = ColdStartRecommender(best, popularity, prior, cfg.rrf_k)
        per_n = evaluate_cold_users(model, scenario, [10])
        values = {str(n): v.get("ndcg@10", float("nan")) for n, v in per_n.items()}
        rows.append({"prior": prior, **values, "mean": float(np.mean(list(values.values())))})
        logger.info("prior %.2f | validation ndcg@10 moyen = %.4f", prior, rows[-1]["mean"])
    best_row = max(rows, key=lambda row: row["mean"])
    return float(best_row["prior"]), rows


def cold_user_results(
    components: dict[str, Recommender],
    best: HybridRecommender,
    fold: Fold,
    config: Config,
    prior: float,
) -> dict[str, Any]:
    """NDCG et rappel des nouveaux utilisateurs selon le nombre de notes revelees."""
    cfg = config.hybrid
    scenario = build_cold_user_scenario(
        fold, cfg.cold_start_history_sizes, cfg.cold_start_users, config.project.seed
    )
    popularity = PopularityRecommender(config.evaluation.relevance_threshold).fit(fold.matrix)
    candidates: dict[str, Recommender] = {
        "popularity": popularity,
        **{name: components[name] for name in best.weights},
        best.name: best,
        "cold_start": ColdStartRecommender(best, popularity, prior, cfg.rrf_k),
    }
    out: dict[str, Any] = {
        "n_users": scenario.n_users,
        "history_sizes": sorted(scenario.histories),
        "prior_strength": prior,
        "models": {},
    }
    for name, model in candidates.items():
        per_n = evaluate_cold_users(model, scenario, config.evaluation.ks)
        out["models"][name] = {str(n): values for n, values in per_n.items()}
        logger.info(
            "cold start | %-16s | ndcg@10 par n : %s",
            name, {n: round(v.get("ndcg@10", float("nan")), 4) for n, v in per_n.items()},
        )  # fmt: skip
    return out


def new_item_results(
    components: dict[str, Recommender], fold: Fold, config: Config
) -> dict[str, Any]:
    """Classement des films absents du catalogue d'entrainement, par le contenu seul."""
    holdout = fold.new_item_holdout
    out: dict[str, Any] = {"models": {}}
    rng = np.random.default_rng(config.project.seed)
    for name, source in config.hybrid.components.items():
        if source != "content":
            continue
        model = components[name]
        known = pd.Index(model.features.movie_ids)
        movie_ids = np.sort(holdout.loc[holdout["movieId"].isin(known), "movieId"].unique())
        out["models"][name] = evaluate_new_items(
            model.score_new_items, fold, movie_ids, config.evaluation.ks
        )
        if "random" not in out["models"]:
            out["models"]["random"] = evaluate_new_items(
                lambda users, ids: rng.random((len(users), len(ids))),
                fold, movie_ids, config.evaluation.ks,
            )  # fmt: skip
        logger.info("films froids | %-10s | %s", name, out["models"][name])
    return out


def _run(
    config: Config,
    sample_users: int | None = None,
    encoder: Encoder | None = None,
    *,
    tracker: ExperimentTracker,
) -> dict[str, Any]:
    """Execute l'experience complete et ecrit resultats et recette du meilleur hybride."""
    set_seed(config.project.seed)
    cfg = config.hybrid
    encoder = encoder or sentence_transformer_encoder(
        config.content.embedding_model, config.content.embedding_batch_size
    )
    movies, tags = content.load_content_tables(config)
    folds = build_folds(load_ratings(config, sample_users), config)
    contents = {id(f): content.FoldContent(movies, tags, f, config) for f in folds.values()}
    content_builders = content.make_builders(contents, config, encoder)
    params = selected_params(config)
    components = {
        id(fold): fit_components(fold, params, config, content_builders) for fold in folds.values()
    }

    names = [f"hybrid_{strategy}" for strategy in cfg.strategies]
    builders: dict[str, Builder] = {
        name: _hybrid_builder(components, strategy, cfg.rrf_k)
        for name, strategy in zip(names, cfg.strategies, strict=True)
    }
    grids = {name: [{"weights": weights} for weights in cfg.weight_sets] for name in names}
    artifact_dir = config.paths.artifacts_dir / ARTIFACT_DIR
    results_path = config.paths.reports_dir / RESULTS_FILE
    tracker.log_params({"sample_users": sample_users, "components": params, "rrf_k": cfg.rrf_k})
    results = run_grid(
        config, folds, builders, grids, results_path, artifact_dir,
        extra={"sample_users": sample_users, "components": params},
        per_user_file=PER_USER_FILE, tracker=tracker,
    )  # fmt: skip

    # Meilleure strategie : choisie sur la VALIDATION, jamais sur le test.
    best_validation = max(results["validation"], key=lambda row: row["ndcg@10"])
    test_fold = folds["test"]
    test_components = components[id(test_fold)]
    best = HybridRecommender(
        test_components,
        best_validation["params"]["weights"],
        best_validation["params"]["strategy"],
        cfg.rrf_k,
        name=best_validation["model"],
    ).fit(test_fold.matrix)
    best.save(artifact_dir / BEST_FILE)
    results["best"] = {
        "model": best.name,
        "params": best.params,
        "validation_ndcg@10": best_validation["ndcg@10"],
    }

    embedding_vectors = np.asarray(test_components["embeddings"].item_vectors_, dtype=np.float32)
    genres = genre_vectors(movies, test_fold.matrix.item_ids)
    results["diversity"] = diversity_sweep(best, test_fold, embedding_vectors, genres, config)
    validation_fold = folds["validation"]
    validation_best = HybridRecommender(
        components[id(validation_fold)], best.weights, best.strategy, cfg.rrf_k, name=best.name
    ).fit(validation_fold.matrix)
    prior, prior_rows = select_cold_start_prior(validation_best, validation_fold, config)
    results["cold_users"] = cold_user_results(test_components, best, test_fold, config, prior)
    results["cold_users"]["prior_selection"] = prior_rows
    results["new_items"] = new_item_results(test_components, test_fold, config)
    write_results(results, results_path)
    cold = results["cold_users"]
    tracker.log_metrics(
        {f"cold_start_ndcg@10_n{n}": values.get("ndcg@10", 0.0)
         for n, values in cold["models"]["cold_start"].items()}
        | {"cold_start_prior": cold["prior_strength"]}
    )  # fmt: skip
    tracker.log_artifacts([results_path])
    return results


def run(
    config: Config, sample_users: int | None = None, encoder: Encoder | None = None
) -> dict[str, Any]:
    """Execute l'experience dans un run MLflow parent, ferme meme en cas d'erreur."""
    tracker = ExperimentTracker(config, "hybrid")
    with tracker.run("hybrid", params={"seed": config.project.seed}):
        return _run(config=config, sample_users=sample_users, encoder=encoder, tracker=tracker)
