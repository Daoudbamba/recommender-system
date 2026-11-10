"""Tableaux et figures specifiques au Sprint 4 : diversite et cold start."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from recsys.analysis import plots
from recsys.analysis.model_report import MODEL_LABELS, _md

HYBRID_LABELS = {
    "hybrid_weighted": "Hybride pondéré (min-max)",
    "hybrid_zscore": "Hybride scores centrés-réduits",
    "hybrid_rrf": "Hybride agrégation de rangs (RRF)",
    "cold_start": "Stratégie cold start (hybride + popularité)",
    "random": "Hasard",
}
MODEL_LABELS.update(HYBRID_LABELS)


def diversity_table(results: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(results["diversity"])
    columns = {
        "lambda": "lambda (MMR)",
        "popularity_penalty": "pénalité de popularité",
        "ndcg@10": "NDCG@10",
        "recall@10": "Recall@10",
        "ild_embeddings@10": "diversité (embeddings)",
        "ild_genres@10": "diversité (genres)",
        "coverage@10": "couverture@10",
        "mean_popularity@10": "popularité moyenne du top-10",
    }
    return frame[list(columns)].rename(columns=columns)


def cold_user_table(results: dict[str, Any], metric: str = "ndcg@10") -> pd.DataFrame:
    """Une ligne par modele, une colonne par nombre de notes revelees."""
    cold = results["cold_users"]
    rows = []
    for name, per_n in cold["models"].items():
        row = {"modèle": MODEL_LABELS.get(name, name)}
        for n in cold["history_sizes"]:
            row[f"{n} note{'s' if n > 1 else ''}"] = per_n[str(n)].get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def prior_table(results: dict[str, Any]) -> pd.DataFrame:
    """Selection du prior de popularite : une ligne par valeur essayee."""
    cold = results["cold_users"]
    frame = pd.DataFrame(cold["prior_selection"])
    names = {str(n): f"{n} note{'s' if n > 1 else ''}" for n in cold["history_sizes"]}
    frame["prior"] = frame["prior"].map(lambda value: f"{value:g}")
    return frame.rename(columns={**names, "mean": "moyenne"})


def new_item_table(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, values in results["new_items"]["models"].items():
        rows.append(
            {
                "modèle": MODEL_LABELS.get(name, name),
                **{key: values.get(key) for key in ("recall@10", "ndcg@10", "recall@20")},
            }
        )
    return pd.DataFrame(rows)


def plot_cold_users(results: dict[str, Any], metric: str = "ndcg@10") -> plt.Figure:
    """Une courbe par modele : la pertinence selon le nombre de notes revelees."""
    plots.apply_style()
    cold = results["cold_users"]
    sizes = cold["history_sizes"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for (name, per_n), color in zip(cold["models"].items(), plots.CATEGORICAL, strict=False):
        values = [per_n[str(n)].get(metric) for n in sizes]
        label = MODEL_LABELS.get(name, name)
        ax.plot(range(len(sizes)), values, marker="o", markersize=5, linewidth=2, color=color,
                label=label)  # fmt: skip
    ax.set_xticks(range(len(sizes)), [str(n) for n in sizes])
    ax.set(
        xlabel="Notes révélées au modèle",
        ylabel=metric,
        title="Nouveaux utilisateurs : pertinence selon l'historique disponible",
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    return fig


def plot_diversity_tradeoff(results: dict[str, Any]) -> plt.Figure:
    """Pertinence contre diversite pour chaque valeur de lambda (penalite nulle)."""
    plots.apply_style()
    frame = pd.DataFrame(results["diversity"])
    frame = frame[frame["popularity_penalty"] == 0].sort_values("lambda", ascending=False)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(frame["ild_embeddings@10"], frame["ndcg@10"], color=plots.SERIES, linewidth=2,
            marker="o", markersize=6)  # fmt: skip
    for _, row in frame.iterrows():
        ax.annotate(
            f"λ = {row['lambda']:g}", (row["ild_embeddings@10"], row["ndcg@10"]),
            xytext=(6, 6), textcoords="offset points", fontsize=8, color=plots.MUTED,
        )  # fmt: skip
    ax.set(
        xlabel="Diversité intra-liste du top-10 (embeddings)",
        ylabel="NDCG@10",
        title="Compromis pertinence / diversité (MMR)",
    )
    return fig


def hybrid_sections(results: dict[str, Any], figures_dir: Path) -> list[str]:
    """Sections Markdown ajoutees au rapport hybride, figures comprises."""
    plots.save(plot_diversity_tradeoff(results), figures_dir / "hybrid_diversity_tradeoff.png")
    plots.save(plot_cold_users(results), figures_dir / "hybrid_cold_users.png")
    best, cold = results["best"], results["cold_users"]
    weights = ", ".join(
        f"{MODEL_LABELS.get(k, k)} {v:.2f}" for k, v in best["params"]["weights"].items()
    )
    return [
        "## Hybride retenu",
        "",
        f"- Stratégie : **{MODEL_LABELS[best['model']]}** (choisie sur la validation, "
        f"NDCG@10 = {best['validation_ndcg@10']:.4f})",
        f"- Poids normalisés : {weights}",
        "",
        "## Diversité et popularité (re-classement du meilleur hybride, test)",
        "",
        "Descriptif : aucun réglage n'est sélectionné sur le test. `lambda = 1` et pénalité nulle",
        "correspondent à l'hybride sans re-classement.",
        "",
        _md(diversity_table(results), floatfmt=",.4f"),
        "",
        "![Compromis pertinence / diversité](figures/hybrid_diversity_tradeoff.png)",
        "",
        "## Cold start utilisateur",
        "",
        f"{cold['n_users']:,} utilisateurs absents de l'entraînement, tirés au hasard. On révèle",
        "leurs premières notes dans l'ordre chronologique ; la cible, identique pour toutes les",
        f"colonnes, est l'ensemble des films aimés après les {max(cold['history_sizes'])} "
        "premières notes. NDCG@10 :",
        "",
        _md(cold_user_table(results)),
        "",
        "Recall@10 :",
        "",
        _md(cold_user_table(results, "recall@10")),
        "",
        "![Cold start utilisateur](figures/hybrid_cold_users.png)",
        "",
        "Choix du poids de la popularité, fait sur les nouveaux venus de la **validation** "
        f"(prior retenu : **{cold['prior_strength']:g}**, NDCG@10) :",
        "",
        _md(prior_table(results)),
        "",
        "## Cold start film",
        "",
        "Films absents du catalogue d'entraînement (moins de 20 notes avant la coupure, films",
        "sortis ensuite compris), classés par leur seul contenu pour les utilisateurs évalués",
        "qui en ont aimé au moins un.",
        "",
        _md(new_item_table(results)),
        "",
        *[
            f"- {MODEL_LABELS.get(name, name)} : {values.get('n_users', 0):,} utilisateurs, "
            f"{values.get('n_candidates', 0):,} films candidats"
            for name, values in results["new_items"]["models"].items()
            if name != "random"
        ],
        "",
    ]
