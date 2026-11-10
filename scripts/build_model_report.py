"""Genere les rapports de resultats des modeles a partir des fichiers d'experience.

Usage : python scripts/build_model_report.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from recsys.analysis import similarity
from recsys.analysis.hybrid_report import hybrid_sections
from recsys.analysis.model_report import MODEL_LABELS, _md, build_model_report, load_results
from recsys.config import Config, load_config
from recsys.experiments import collaborative, content, hybrid
from recsys.models.collaborative import ItemKNN
from recsys.models.content_based import ContentBasedRecommender, SimilarityIndex

SIMILARITY_FILE = "similarity_comparison.json"
N_NEIGHBOURS = 10
N_SAMPLED_MOVIES = 2000


def collaborative_report(config: Config) -> None:
    directory = config.paths.artifacts_dir / collaborative.ARTIFACT_DIR
    per_user = pd.read_parquet(directory / collaborative.PER_USER_FILE)
    path = build_model_report(
        config,
        collaborative.RESULTS_FILE,
        per_user,
        title="Filtrage collaboratif — résultats",
        report_file="collaborative_report.md",
        figure_prefix="cf",
    )
    print(f"Rapport ecrit : {path}")


def compare_similarity_engines(config: Config) -> dict:
    """TF-IDF contre embeddings : accord avec les voisins collaboratifs et coherence de genres."""
    artifacts = config.paths.artifacts_dir
    item_knn = ItemKNN.load(artifacts / collaborative.ARTIFACT_DIR / "item_knn.pkl")
    engines = {
        name: ContentBasedRecommender.load(artifacts / content.ARTIFACT_DIR / f"{name}.pkl")
        for name in ("tfidf", "embeddings")
    }
    item_ids = item_knn.matrix.item_ids
    if any(not np.array_equal(m.matrix.item_ids, item_ids) for m in engines.values()):
        raise ValueError("les modeles compares n'ont pas le meme catalogue d'entrainement")
    movies = pd.read_parquet(config.paths.processed_dir / "movies.parquet").set_index("movieId")
    genre_sets = [frozenset(genres) for genres in movies.loc[item_ids, "genres"]]

    rng = np.random.default_rng(config.project.seed)
    sample = np.sort(
        rng.choice(len(item_ids), size=min(N_SAMPLED_MOVIES, len(item_ids)), replace=False)
    )
    reference = similarity.neighbour_table(item_knn, sample, N_NEIGHBOURS)
    tables = {
        name: similarity.neighbour_table(m, sample, N_NEIGHBOURS) for name, m in engines.items()
    }
    index = SimilarityIndex.load(artifacts / content.ARTIFACT_DIR / content.INDEX_FILE)
    result = {
        "n_sampled_movies": len(sample),
        "n_neighbours": N_NEIGHBOURS,
        "engines": {
            name: {
                "overlap_with_collaborative": similarity.mean_overlap(table, reference),
                "genre_jaccard": similarity.mean_genre_jaccard(table, sample, genre_sets),
            }
            for name, table in tables.items()
        },
        "collaborative_genre_jaccard": similarity.mean_genre_jaccard(reference, sample, genre_sets),
        "overlap_tfidf_embeddings": similarity.mean_overlap(tables["tfidf"], tables["embeddings"]),
        "exact_search": similarity.search_latency_ms(index),
    }
    target = config.paths.reports_dir / SIMILARITY_FILE
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def similarity_section(result: dict) -> list[str]:
    rows = [
        {
            "moteur": MODEL_LABELS[name],
            "voisins communs avec l'item-kNN": values["overlap_with_collaborative"],
            "Jaccard des genres": values["genre_jaccard"],
        }
        for name, values in result["engines"].items()
    ]
    rows.append(
        {
            "moteur": "Item-kNN (référence collaborative)",
            "voisins communs avec l'item-kNN": 1.0,
            "Jaccard des genres": result["collaborative_genre_jaccard"],
        }
    )
    search = result["exact_search"]
    return [
        "## Films similaires : TF-IDF contre embeddings",
        "",
        f"Mesuré sur {result['n_sampled_movies']:,} films tirés au hasard, "
        f"{result['n_neighbours']} voisins par film.",
        "",
        _md(pd.DataFrame(rows)),
        "",
        f"- Voisins communs entre TF-IDF et embeddings : "
        f"**{result['overlap_tfidf_embeddings']:.1%}**",
        f"- Recherche exacte dans l'index d'embeddings ({search['n_vectors']:,} films, "
        f"dimension {search['dimension']}) : **{search['mean_ms']:.1f} ms** en moyenne, "
        f"{search['p95_ms']:.1f} ms au 95ᵉ centile.",
        "",
    ]


def content_report(config: Config) -> None:
    directory = config.paths.artifacts_dir / content.ARTIFACT_DIR
    per_user = pd.concat(
        [
            pd.read_parquet(
                config.paths.artifacts_dir
                / collaborative.ARTIFACT_DIR
                / collaborative.PER_USER_FILE
            ).query("model in ['popularity', 'item_knn', 'als_implicit']"),
            pd.read_parquet(directory / content.PER_USER_FILE),
        ],
        ignore_index=True,
    )
    path = build_model_report(
        config,
        content.RESULTS_FILE,
        per_user,
        title="Filtrage par contenu et embeddings — résultats",
        report_file="content_report.md",
        figure_prefix="content",
        reference_file=collaborative.RESULTS_FILE,
        reference_models=("popularity", "item_knn", "als_implicit"),
        extra_sections=similarity_section(compare_similarity_engines(config)),
    )
    print(f"Rapport ecrit : {path}")


def hybrid_report(config: Config) -> None:
    artifacts = config.paths.artifacts_dir
    per_user = pd.concat(
        [
            pd.read_parquet(
                artifacts / collaborative.ARTIFACT_DIR / collaborative.PER_USER_FILE
            ).query("model in ['popularity', 'item_knn', 'als_implicit']"),
            pd.read_parquet(artifacts / content.ARTIFACT_DIR / content.PER_USER_FILE).query(
                "model == 'tfidf'"
            ),
            pd.read_parquet(artifacts / hybrid.ARTIFACT_DIR / hybrid.PER_USER_FILE),
        ],
        ignore_index=True,
    )
    results = load_results(config.paths.reports_dir, hybrid.RESULTS_FILE)
    path = build_model_report(
        config,
        hybrid.RESULTS_FILE,
        per_user,
        title="Système hybride, diversité et cold start — résultats",
        report_file="hybrid_report.md",
        figure_prefix="hybrid",
        reference_file=collaborative.RESULTS_FILE,
        reference_models=("popularity", "item_knn", "als_implicit"),
        extra_reference=(content.RESULTS_FILE, ("tfidf", "embeddings")),
        extra_sections=hybrid_sections(results, config.paths.reports_dir / "figures"),
    )
    print(f"Rapport ecrit : {path}")


def main() -> None:
    config = load_config()
    reports = {
        collaborative.RESULTS_FILE: collaborative_report,
        content.RESULTS_FILE: content_report,
        hybrid.RESULTS_FILE: hybrid_report,
    }
    for results_file, build in reports.items():
        if (config.paths.reports_dir / results_file).is_file():
            build(config)
        else:
            print(f"{results_file} absent : rapport ignore")


if __name__ == "__main__":
    main()
