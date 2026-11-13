"""Tableaux et figures specifiques au Sprint 5 : jeu d'entrainement et variables du ranker."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from recsys.analysis import plots
from recsys.analysis.model_report import MODEL_LABELS, _md

MODEL_LABELS.update(
    {
        "xgb_ranker": "Ranking XGBoost",
        "xgb_ranker_no_recency": "Ranking XGBoost (sans variables de récence)",
        "xgb_ranker_scores_only": "Ranking XGBoost (scores des modèles seuls)",
    }
)

FEATURE_LABELS = {
    "score_hybrid": "score de l'hybride",
    "item_log_popularity": "popularité du film",
    "item_mean_rating": "note moyenne du film",
    "item_liked_share": "part de notes positives du film",
    "item_age": "âge du film à la date de coupure",
    "user_log_history": "taille de l'historique",
    "user_mean_rating": "note moyenne de l'utilisateur",
    "user_liked_share": "part de films aimés par l'utilisateur",
    "genre_affinity": "affinité de genres",
    "year_gap": "écart d'année avec les films aimés",
    "popularity_gap": "écart de popularité avec les films aimés",
}


def feature_label(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    kind, _, model = name.partition("_rank_" if "_rank_" in name else "score_")
    prefix = "rang" if name.startswith("log_rank_") else "score"
    return f"{prefix} {MODEL_LABELS.get(model, model)}" if kind is not None else name


def importance_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {"variable": feature_label(name), "nom technique": name, "part du gain": share}
        for name, share in results["feature_importance"].items()
    ]
    return pd.DataFrame(rows)


def plot_importance(results: dict[str, Any], top: int = 12) -> plt.Figure:
    frame = importance_table(results).head(top)
    return plots.plot_horizontal_bars(
        frame, "variable", "part du gain", "Variables les plus utiles au ranker",
        "Part du gain total", fmt="float",
    )  # fmt: skip


def dataset_table(results: dict[str, Any]) -> pd.DataFrame:
    data = results["dataset"]
    counts = data["label_counts"]
    total = data["train_rows"]
    rows = [
        ("2 — aimé (note ≥ seuil)", counts["liked"]),
        ("1 — non noté (inconnu)", counts["unknown"]),
        ("0 — noté sous le seuil (rejet)", counts["disliked"]),
    ]
    return pd.DataFrame(
        [{"label": label, "lignes": n, "part": f"{n / total:.2%}"} for label, n in rows]
    )


def ranking_sections(results: dict[str, Any], figures_dir: Path) -> list[str]:
    """Sections Markdown ajoutees au rapport de ranking, figure comprise."""
    plots.save(plot_importance(results), figures_dir / "ranking_feature_importance.png")
    data, best, reference = results["dataset"], results["best"], results["dev_reference"]
    return [
        "## Jeu d'entraînement du ranker",
        "",
        f"- {data['train_users'] + data['dev_users']:,} utilisateurs "
        f"({data['dev_users']:,} gardés pour le réglage), {data['n_candidates']} candidats chacun, "
        f"{len(results['feature_names'])} variables.",
        "- Variables calculées avec les modèles entraînés jusqu'à la première coupure ; labels "
        "pris dans la période de validation.",
        f"- Plafond du rappel : {data['candidate_recall_dev']:.1%} des films aimés figurent parmi "
        "les candidats (jeu de réglage). Le ranker ne peut pas faire remonter un film absent.",
        f"- {data['users_without_positive_candidate']:,} utilisateurs d'entraînement n'ont aucun "
        "film aimé parmi leurs candidats.",
        "",
        _md(dataset_table(results)),
        "",
        "## Réglage sur le jeu de dev",
        "",
        f"- Référence, ordre de l'hybride sans ranker : NDCG@10 = {reference['ndcg@10']:.4f}",
        f"- Meilleure configuration : `{best['params']}`, NDCG@10 = {best['dev_ndcg@10']:.4f}",
        "",
        "## Importance des variables",
        "",
        "![Importance des variables](figures/ranking_feature_importance.png)",
        "",
        _md(importance_table(results)),
        "",
    ]
