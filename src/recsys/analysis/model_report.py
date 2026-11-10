"""Mise en forme des resultats d'experience : tableaux, figures, rapport Markdown.

Lit exclusivement le fichier JSON ecrit par l'experience : aucun score n'est
saisi a la main.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from recsys.analysis import plots
from recsys.config import Config

REPORT_FILE = "collaborative_report.md"
MODEL_LABELS = {
    "popularity": "Popularité",
    "bias_baseline": "Biais (moyennes)",
    "item_knn": "Item-kNN",
    "user_knn": "User-kNN",
    "als_explicit": "ALS explicite",
    "als_implicit": "ALS implicite",
}


def load_results(reports_dir: Path, filename: str) -> dict[str, Any]:
    path = reports_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"{path} absent : lancer d'abord l'experience correspondante")
    return json.loads(path.read_text(encoding="utf-8"))


def results_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Une ligne par modele (ou configuration), colonnes = metriques."""
    frame = pd.DataFrame(rows)
    frame.insert(1, "label", frame["model"].map(MODEL_LABELS).fillna(frame["model"]))
    frame["params"] = frame["params"].map(
        lambda p: ", ".join(f"{k}={v}" for k, v in p.items() if k not in {"seed", "reg_bias"})
    )
    return frame


def ranking_table(frame: pd.DataFrame, k: int) -> pd.DataFrame:
    columns = [f"{m}@{k}" for m in ("precision", "recall", "f1", "map", "ndcg", "coverage")]
    return frame[["label", *columns]].rename(columns={"label": "modèle"})


def rating_table(frame: pd.DataFrame) -> pd.DataFrame:
    rated = frame.dropna(subset=["rmse"]) if "rmse" in frame else frame.iloc[0:0]
    return rated[["label", "rmse", "mae"]].rename(columns={"label": "modèle"})


def cost_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[["label", "params", "fit_seconds", "ms_per_user", "mean_popularity@10"]]
    return out.rename(
        columns={
            "label": "modèle",
            "params": "hyperparamètres retenus",
            "fit_seconds": "entraînement (s)",
            "ms_per_user": "ms / utilisateur",
            "mean_popularity@10": "popularité moyenne du top-10",
        }
    )


def plot_metric(frame: pd.DataFrame, metric: str, title: str, fmt: str = "float") -> plt.Figure:
    data = frame.dropna(subset=[metric])
    return plots.plot_horizontal_bars(data, "label", metric, title, metric, fmt=fmt)


def plot_metric_by_k(frame: pd.DataFrame, metric: str, ks: list[int], title: str) -> plt.Figure:
    """Une courbe par modele : comment la metrique evolue avec k."""
    plots.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for (_, row), color in zip(frame.iterrows(), plots.CATEGORICAL, strict=False):
        values = [row[f"{metric}@{k}"] for k in ks]
        ax.plot(ks, values, marker="o", markersize=5, linewidth=2, color=color, label=row["label"])
        ax.annotate(
            row["label"],
            (ks[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=plots.MUTED,
            va="center",
        )
    ax.set_xticks(ks)
    ax.set(xlabel="k", ylabel=f"{metric}@k", title=title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.12, 1.0))
    return fig


def segment_table(per_user: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """Analyse d'erreurs : la metrique selon la taille de l'historique utilisateur."""
    bins = [0, 50, 150, 400, 1000, float("inf")]
    labels = ["5-50", "51-150", "151-400", "401-1000", "> 1000"]
    segment = pd.cut(per_user["history"], bins=bins, labels=labels)
    table = per_user.assign(historique=segment).pivot_table(
        index="historique", columns="model", values=metric, aggfunc="mean", observed=True
    )
    counts = per_user[per_user["model"] == per_user["model"].iloc[0]]
    table.insert(0, "utilisateurs", counts.assign(h=segment).groupby("h", observed=True).size())
    return table.rename(columns=MODEL_LABELS).reset_index()


def _md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    from recsys.analysis.report import _table

    return _table(df, floatfmt=floatfmt)


def build_collaborative_report(config: Config, results_file: str, per_user: pd.DataFrame) -> Path:
    """Ecrit reports/collaborative_report.md et ses figures."""
    reports_dir = config.paths.reports_dir
    results = load_results(reports_dir, results_file)
    test = results_frame(results["test"])
    validation = results_frame(results["validation"])
    ks = results["protocol"]["ks"]
    figures_dir = reports_dir / "figures"

    figures = {
        "cf_ndcg_at_10": plot_metric(test, "ndcg@10", "NDCG@10 sur le test"),
        "cf_recall_by_k": plot_metric_by_k(test, "recall", ks, "Recall@k sur le test"),
        "cf_coverage_at_10": plot_metric(
            test, "coverage@10", "Part du catalogue recommandée (top-10)"
        ),
        "cf_rmse": plot_metric(test, "rmse", "RMSE sur le test (plus bas = mieux)"),
    }
    for name, fig in figures.items():
        plots.save(fig, figures_dir / f"{name}.png")

    folds = pd.DataFrame(results["folds"]).T.rename_axis("jeu").reset_index()
    fold_columns = [
        "jeu", "train_end", "train_ratings", "train_users", "train_items",
        "holdout_ratings", "holdout_users", "holdout_cold_users", "eval_users", "eval_relevant",
    ]  # fmt: skip
    val_columns = ["label", "params", "ndcg@10", "recall@10", "coverage@10", "fit_seconds"]
    if "rmse" in validation:
        val_columns.insert(5, "rmse")

    lines = [
        "# Filtrage collaboratif — résultats",
        "",
        "> Fichier généré par `python scripts/build_model_report.py` à partir de",
        f"> `reports/{results_file}`. Ne pas éditer à la main.",
        "",
        "## Protocole",
        "",
        f"- Coupures temporelles globales aux quantiles "
        f"{results['protocol']['validation_quantile']} et {results['protocol']['test_quantile']} "
        "des notes classées par date.",
        f"- Film pertinent : note ≥ {results['protocol']['relevance_threshold']}.",
        "- Candidats : tous les films du catalogue d'entraînement non encore notés par "
        "l'utilisateur.",
        f"- Graine aléatoire : {results['seed']}.",
        "",
        _md(folds[fold_columns]),
        "",
        "## Résultats sur le test",
        "",
        *[part for k in ks for part in (f"### Top-{k}", "", _md(ranking_table(test, k)), "")],
        "### Prédiction de note",
        "",
        _md(rating_table(test)),
        "",
        "### Coût et biais de popularité",
        "",
        _md(cost_table(test), floatfmt=",.1f"),
        "",
        *[f"![{name}](figures/{name}.png)\n" for name in figures],
        "## Analyse d'erreurs : NDCG@10 selon la taille de l'historique",
        "",
        _md(segment_table(per_user)),
        "",
        "## Sélection des hyperparamètres (validation)",
        "",
        _md(validation[val_columns].rename(columns={"label": "modèle"})),
        "",
    ]
    target = reports_dir / REPORT_FILE
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
