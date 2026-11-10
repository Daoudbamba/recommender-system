"""Graphiques de l'analyse exploratoire.

Regles suivies : une seule mesure par graphique (jamais de double axe), une
seule teinte quand il n'y a qu'une serie, grille et axes discrets, et le
titre porte le constat plutot que le nom de la variable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

SERIES = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
# Palette categorielle validee (ordre fixe : une couleur suit toujours le meme modele).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def apply_style() -> None:
    """Style commun a toutes les figures du projet."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def _compact(value: float, _pos: int | None = None) -> str:
    """1 500 000 -> 1.5M, pour des axes lisibles."""
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.3g}{suffix}"
    return f"{value:.3g}"


def _new_axes(figsize: tuple[float, float] = (8, 4.2)) -> tuple[Figure, plt.Axes]:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_rating_distribution(distribution: pd.DataFrame) -> Figure:
    fig, ax = _new_axes()
    labels = [f"{value:g}" for value in distribution["rating"]]
    bars = ax.bar(labels, distribution["count"], color=SERIES, width=0.72)
    ax.bar_label(bars, labels=[f"{s:.1%}" for s in distribution["share"]], color=MUTED, fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="Note", ylabel="Nombre de notes", title="Répartition des notes")
    return fig


def plot_count_histogram(counts: pd.Series, entity: str, title: str) -> Figure:
    """Histogramme en echelle logarithmique d'une distribution a longue traine."""
    fig, ax = _new_axes()
    # Bornes entieres : les effectifs sont discrets, des classes plus fines qu'une
    # unite laisseraient des trous artificiels a gauche du graphique.
    bins = np.unique(np.round(np.logspace(0, np.log10(counts.max() + 1), 45)))
    ax.hist(counts, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(counts.median(), color=TEXT, linewidth=1, linestyle="--")
    ax.annotate(
        f"médiane = {counts.median():.0f}",
        xy=(counts.median(), 0.94),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        color=TEXT,
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_compact))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.set(xlabel=f"Nombre de notes par {entity} (échelle log)", ylabel=f"{entity.title()}s")
    ax.set_title(title)
    return fig


def plot_lorenz(counts: pd.Series, entity: str) -> Figure:
    """Courbe de concentration : part des notes captee par les plus actifs."""
    fig, ax = _new_axes((6.2, 4.6))
    ordered = np.sort(counts.to_numpy())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    share = np.arange(1, len(ordered) + 1) / len(ordered)
    ax.plot(share, cumulative, color=SERIES, linewidth=2)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set(
        xlabel=f"Part des {entity}s, du plus noté au moins noté",
        ylabel="Part cumulée des notes",
        title=f"Concentration des notes sur les {entity}s",
    )
    return fig


def plot_horizontal_bars(
    data: pd.DataFrame, label: str, value: str, title: str, xlabel: str, fmt: str = "compact"
) -> Figure:
    """Barres horizontales triees : pour comparer des categories nommees."""
    ordered = data.sort_values(value)
    fig, ax = _new_axes((8, max(3.2, 0.3 * len(ordered) + 1)))
    bars = ax.barh(ordered[label], ordered[value], color=SERIES, height=0.7)
    texts = [_compact(v) if fmt == "compact" else f"{v:.2f}" for v in ordered[value]]
    ax.bar_label(bars, labels=texts, color=MUTED, fontsize=8, padding=3)
    if fmt == "compact":
        ax.xaxis.set_major_formatter(FuncFormatter(_compact))
    ax.grid(axis="y", visible=False)
    ax.set(xlabel=xlabel, title=title)
    return fig


def plot_yearly(data: pd.DataFrame, value: str, title: str, ylabel: str, kind: str = "bar"):
    """Evolution annuelle d'une seule mesure."""
    fig, ax = _new_axes()
    if kind == "bar":
        ax.bar(data["year"], data[value], color=SERIES, width=0.75)
        ax.yaxis.set_major_formatter(FuncFormatter(_compact))
        ax.grid(axis="x", visible=False)
    else:
        ax.plot(data["year"], data[value], color=SERIES, linewidth=2, marker="o", markersize=4)
    ax.set(xlabel="Année", ylabel=ylabel, title=title)
    return fig


def plot_distribution(values: pd.Series, title: str, xlabel: str, bins: int = 40) -> Figure:
    """Histogramme lineaire d'une mesure bornee (ex. note moyenne)."""
    fig, ax = _new_axes()
    ax.hist(values.dropna(), bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.6)
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.set(xlabel=xlabel, ylabel="Effectif", title=title)
    return fig


def save(fig: Figure, path: Path) -> Path:
    """Enregistre la figure en PNG et libere la memoire."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
