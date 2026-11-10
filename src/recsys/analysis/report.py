"""Generation du rapport de donnees (Markdown + figures).

Tous les chiffres du rapport sont calcules ici a partir des fichiers
reellement presents : rien n'est saisi a la main.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any

import pandas as pd

from recsys.analysis import eda, plots
from recsys.config import Config
from recsys.preprocessing.pipeline import QUALITY_FILE, SUMMARY_FILE

REPORT_FILE = "data_report.md"
FIGURES_DIR = "figures"

COLUMN_DOCS: dict[str, dict[str, str]] = {
    "ratings": {
        "userId": "identifiant anonymisé de l'utilisateur",
        "movieId": "identifiant MovieLens du film",
        "rating": "note de 0.5 à 5.0 par pas de 0.5",
        "timestamp": "date de la note (secondes Unix, UTC)",
    },
    "movies": {
        "movieId": "identifiant MovieLens du film",
        "title": "titre suivi de l'année de sortie entre parenthèses",
        "genres": "genres séparés par une barre verticale",
    },
    "tags": {
        "userId": "identifiant de l'utilisateur qui a posé le tag",
        "movieId": "identifiant MovieLens du film",
        "tag": "texte libre saisi par l'utilisateur",
        "timestamp": "date du tag (secondes Unix, UTC)",
    },
    "links": {
        "movieId": "identifiant MovieLens du film",
        "imdbId": "identifiant IMDb",
        "tmdbId": "identifiant TMDB",
    },
}


PLAIN_INT_COLUMNS = {"year", "max_ratings"}


def _cell(value: Any, column: str, floatfmt: str) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, numbers.Integral):
        return str(int(value)) if column in PLAIN_INT_COLUMNS else f"{int(value):,}"
    if isinstance(value, numbers.Real):
        return "" if pd.isna(value) else format(float(value), floatfmt)
    return str(value).replace("|", "/")


def _table(df: pd.DataFrame, floatfmt: str = ",.3f") -> str:
    """Tableau Markdown : entiers avec separateur de milliers, flottants formates."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = []
    for row in df.itertuples(index=False, name=None):
        cells = [_cell(v, c, floatfmt) for c, v in zip(df.columns, row, strict=True)]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _percent(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = out[column].map(lambda v: f"{v:.2%}")
    return out


def _describe(counts: pd.Series, label: str) -> pd.DataFrame:
    stats = eda.describe_counts(counts).rename(label).to_frame().T
    return stats.rename_axis("mesure").reset_index()


def load_processed(config: Config) -> dict[str, pd.DataFrame]:
    """Charge les Parquet produits par le pipeline de preparation."""
    tables = {}
    for name in ("ratings", "movies", "tags", "links", "interactions"):
        path = config.paths.processed_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} absent : lancer d'abord scripts/prepare_data.py")
        tables[name] = pd.read_parquet(path)
    return tables


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} absent : lancer d'abord scripts/prepare_data.py")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_statistics(tables: dict[str, pd.DataFrame], config: Config) -> dict[str, Any]:
    """Calcule toutes les statistiques du rapport (reutilise par le notebook)."""
    ratings, movies, tags = tables["ratings"], tables["movies"], tables["tags"]
    users = eda.user_activity(ratings)
    popularity = eda.movie_popularity(ratings)
    thresholds = config.eda.activity_thresholds
    return {
        "overview": eda.overview(ratings, movies, tags),
        "rating_distribution": eda.rating_distribution(ratings),
        "users": users,
        "popularity": popularity,
        "user_low": eda.low_activity(users["n_ratings"], thresholds),
        "movie_low": eda.low_activity(popularity["n_ratings"], thresholds),
        "movie_concentration": eda.concentration(popularity["n_ratings"]),
        "user_concentration": eda.concentration(users["n_ratings"]),
        "genres": eda.genre_stats(movies, popularity),
        "per_year": eda.ratings_per_year(ratings),
        "release_years": eda.movies_per_release_year(movies),
        "top_movies": eda.top_movies(movies, popularity, config.eda.top_n),
        "tag_stats": eda.tag_stats(tags, movies),
        "top_tags": eda.top_tags(tags, config.eda.top_n),
    }


def make_figures(stats: dict[str, Any], figures_dir: Path) -> dict[str, Path]:
    """Produit les figures PNG et retourne leurs chemins."""
    users, popularity, genres = stats["users"], stats["popularity"], stats["genres"]
    figures = {
        "rating_distribution": plots.plot_rating_distribution(stats["rating_distribution"]),
        "user_activity": plots.plot_count_histogram(
            users["n_ratings"], "utilisateur", "Activité des utilisateurs"
        ),
        "movie_popularity": plots.plot_count_histogram(
            popularity["n_ratings"], "film", "Popularité des films"
        ),
        "movie_concentration": plots.plot_lorenz(popularity["n_ratings"], "film"),
        "user_mean_rating": plots.plot_distribution(
            users["mean_rating"], "Note moyenne donnée par utilisateur", "Note moyenne"
        ),
        "genres_ratings": plots.plot_horizontal_bars(
            genres, "genre", "n_ratings", "Notes reçues par genre", "Nombre de notes"
        ),
        "genres_movies": plots.plot_horizontal_bars(
            genres, "genre", "n_movies", "Films au catalogue par genre", "Nombre de films"
        ),
        "genres_mean_rating": plots.plot_horizontal_bars(
            genres, "genre", "mean_rating", "Note moyenne par genre", "Note moyenne", fmt="float"
        ),
        "ratings_per_year": plots.plot_yearly(
            stats["per_year"], "n_ratings", "Notes déposées par année", "Nombre de notes"
        ),
        "mean_rating_per_year": plots.plot_yearly(
            stats["per_year"], "mean_rating", "Note moyenne par année", "Note moyenne", kind="line"
        ),
        "release_years": plots.plot_yearly(
            stats["release_years"], "n_movies", "Films par année de sortie", "Nombre de films"
        ),
    }
    return {name: plots.save(fig, figures_dir / f"{name}.png") for name, fig in figures.items()}


def _quality_section(quality: dict[str, Any]) -> list[str]:
    lines = ["## 2. Tables, colonnes et qualité des données brutes", ""]
    for name, info in quality["tables"].items():
        lines += [f"### `{name}.csv` — {info['n_rows']:,} lignes", ""]
        rows = [
            {"colonne": col, "description": COLUMN_DOCS[name][col], "valeurs manquantes": missing}
            for col, missing in info["missing"].items()
        ]
        lines += [_table(pd.DataFrame(rows)), ""]
        key = ", ".join(info.get("key", []))
        lines += [
            f"- Lignes strictement dupliquées : **{info['duplicate_rows']:,}**",
            f"- Doublons sur la clé ({key}) : **{info.get('duplicate_keys', 0):,}**",
            "",
        ]
    checks, movies, integrity = quality["ratings"], quality["movies"], quality["integrity"]
    lines += [
        "### Contrôles de cohérence",
        "",
        f"- Notes hors bornes : **{checks['out_of_range']:,}**",
        f"- Notes hors pas de 0.5 : **{checks['not_half_star']:,}**",
        f"- Valeurs de note observées : {checks['distinct_values']}",
        f"- Horodatages non positifs : **{checks['non_positive_timestamps']:,}**",
        f"- Films sans genre (`(no genres listed)`) : **{movies['no_genres_listed']:,}**",
        f"- Titres sans année : **{movies['titles_without_year']:,}**",
        f"- Titres en double (movieId différents) : **{movies['duplicate_titles']:,}**",
        "",
        _table(
            pd.DataFrame(
                [{"contrôle d'intégrité": key, "films": value} for key, value in integrity.items()]
            )
        ),
        "",
    ]
    return lines


def _figure(figures: dict[str, Path], name: str, alt: str) -> list[str]:
    return [f"![{alt}]({FIGURES_DIR}/{figures[name].name})", ""]


def _cleaning_section(summary: dict[str, Any]) -> list[str]:
    rows = []
    for name in ("ratings", "movies", "tags", "links"):
        raw, clean = summary[name]["raw"]["n_rows"], summary[name]["clean"]["n_rows"]
        rows.append(
            {
                "table": name,
                "lignes brutes": raw,
                "lignes nettoyées": clean,
                "retirées": raw - clean,
            }
        )
    inter, ratings = summary["interactions"], summary["ratings"]["clean"]
    thresholds = summary["thresholds"]
    kept = inter["n_rows"] / ratings["n_rows"] if ratings["n_rows"] else 0.0
    return [
        "## 9. Effet du nettoyage",
        "",
        _table(pd.DataFrame(rows)),
        "",
        "### Jeu `interactions` destiné aux modèles",
        "",
        f"Filtrage itératif : au moins **{thresholds['min_user_ratings']}** notes par "
        f"utilisateur et **{thresholds['min_movie_ratings']}** notes par film.",
        "",
        _table(
            pd.DataFrame(
                [
                    {
                        "jeu": "ratings nettoyé",
                        "notes": ratings["n_rows"],
                        "utilisateurs": ratings["n_users"],
                        "films": ratings["n_movies"],
                    },
                    {
                        "jeu": "interactions",
                        "notes": inter["n_rows"],
                        "utilisateurs": inter["n_users"],
                        "films": inter["n_movies"],
                    },
                ]
            )
        ),
        "",
        f"Le filtrage conserve **{kept:.2%}** des notes.",
        "",
    ]


def render_markdown(
    config: Config,
    stats: dict[str, Any],
    quality: dict[str, Any],
    summary: dict[str, Any],
    figures: dict[str, Path],
) -> str:
    """Assemble le rapport Markdown."""
    ov, tag_stats = stats["overview"], stats["tag_stats"]
    users, popularity = stats["users"], stats["popularity"]
    lines = [
        "# Rapport sur les données — MovieLens",
        "",
        "> Fichier généré par `python scripts/build_data_report.py`. Ne pas éditer à la main :",
        "> tous les chiffres sont recalculés à partir des données téléchargées.",
        "",
        "## 1. Provenance",
        "",
        f"- Jeu de données : **{config.dataset.name}** (GroupLens Research)",
        f"- Source : <{config.dataset.url}>",
        f"- Somme MD5 vérifiée au téléchargement : `{config.dataset.md5}`",
        "- Conditions d'utilisation : voir `docs/DATASET.md`",
        "",
        *_quality_section(quality),
        "## 3. Chiffres clés (après nettoyage)",
        "",
        f"- Utilisateurs : **{ov['n_users']:,}**",
        f"- Films au catalogue : **{ov['n_movies_catalog']:,}**, dont "
        f"**{ov['n_movies_rated']:,}** ont au moins une note",
        f"- Notes : **{ov['n_ratings']:,}**",
        f"- Densité de la matrice utilisateur × film : **{ov['density']:.4%}**",
        f"- Note moyenne : **{ov['mean_rating']:.3f}** (médiane {ov['median_rating']:.1f})",
        f"- Période : du **{ov['first_rating']:%Y-%m-%d}** au **{ov['last_rating']:%Y-%m-%d}**",
        "",
        "## 4. Distribution des notes",
        "",
        *_figure(figures, "rating_distribution", "Répartition des notes"),
        _table(_percent(stats["rating_distribution"], ["share"]).astype({"rating": str})),
        "",
        "## 5. Comportement des utilisateurs",
        "",
        *_figure(figures, "user_activity", "Activité des utilisateurs"),
        _table(_describe(users["n_ratings"], "notes par utilisateur"), floatfmt=",.1f"),
        "",
        "Utilisateurs peu actifs (au plus *n* notes) :",
        "",
        _table(_percent(stats["user_low"], ["share_entities", "share_ratings"])),
        "",
        "Part des notes déposée par les utilisateurs les plus actifs :",
        "",
        _table(_percent(stats["user_concentration"], ["top_share", "share_ratings"])),
        "",
        *_figure(figures, "user_mean_rating", "Note moyenne par utilisateur"),
        _table(_describe(users["mean_rating"], "note moyenne par utilisateur")),
        "",
        _table(
            _describe(users["active_days"], "jours entre première et dernière note"),
            floatfmt=",.1f",
        ),
        "",
        "## 6. Popularité des films",
        "",
        *_figure(figures, "movie_popularity", "Popularité des films"),
        _table(_describe(popularity["n_ratings"], "notes par film"), floatfmt=",.1f"),
        "",
        "Films peu notés (au plus *n* notes) :",
        "",
        _table(_percent(stats["movie_low"], ["share_entities", "share_ratings"])),
        "",
        *_figure(figures, "movie_concentration", "Concentration des notes"),
        _table(_percent(stats["movie_concentration"], ["top_share", "share_ratings"])),
        "",
        f"Les {len(stats['top_movies'])} films les plus notés :",
        "",
        _table(stats["top_movies"]),
        "",
        *_figure(figures, "release_years", "Films par année de sortie"),
        "## 7. Genres",
        "",
        "Un film à plusieurs genres compte dans chacun d'eux.",
        "",
        _table(stats["genres"]),
        "",
        *_figure(figures, "genres_ratings", "Notes par genre"),
        *_figure(figures, "genres_movies", "Films par genre"),
        *_figure(figures, "genres_mean_rating", "Note moyenne par genre"),
        "## 8. Dimension temporelle et tags",
        "",
        *_figure(figures, "ratings_per_year", "Notes par année"),
        *_figure(figures, "mean_rating_per_year", "Note moyenne par année"),
        _table(stats["per_year"]),
        "",
        "### Tags",
        "",
        f"- Applications de tags : **{tag_stats['n_tag_applications']:,}**",
        f"- Tags distincts (après normalisation) : **{tag_stats['n_distinct_tags']:,}**",
        f"- Utilisateurs ayant posé au moins un tag : **{tag_stats['n_tagging_users']:,}**",
        f"- Films avec au moins un tag : **{tag_stats['n_tagged_movies']:,}** "
        f"(**{tag_stats['share_catalog_tagged']:.1%}** du catalogue)",
        "",
        _table(stats["top_tags"]),
        "",
        *_cleaning_section(summary),
    ]
    return "\n".join(lines)


def build_report(config: Config) -> Path:
    """Calcule les statistiques, produit les figures et ecrit le rapport."""
    reports_dir = config.paths.reports_dir
    tables = load_processed(config)
    stats = compute_statistics(tables, config)
    figures = make_figures(stats, reports_dir / FIGURES_DIR)
    markdown = render_markdown(
        config,
        stats,
        _load_json(reports_dir / QUALITY_FILE),
        _load_json(reports_dir / SUMMARY_FILE),
        figures,
    )
    target = reports_dir / REPORT_FILE
    target.write_text(markdown, encoding="utf-8")
    return target
