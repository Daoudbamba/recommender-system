"""Modele par contenu, moteur de similarite et experience de bout en bout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from recsys.config import Config
from recsys.experiments import content as experiment
from recsys.features.item_features import ItemFeatures
from recsys.models.base import InteractionMatrix
from recsys.models.content_based import ContentBasedRecommender, SimilarityIndex

# Six films : 0-2 de science-fiction, 3-5 de romance. Les vecteurs sont choisis
# pour que la similarite a l'interieur d'un genre soit forte et nulle entre genres.
VECTORS = np.array(
    [[1, 0.2, 0, 0], [1, 0.1, 0, 0], [0.9, 0.3, 0, 0],
     [0, 0, 1, 0.2], [0, 0, 1, 0.1], [0, 0, 0.9, 0.3]],
    dtype=np.float32,
)  # fmt: skip
VECTORS /= np.linalg.norm(VECTORS, axis=1, keepdims=True)
MOVIE_IDS = np.array([100, 101, 102, 103, 104, 105])
NAMES = np.array(["tag:space", "tag:robot", "tag:love", "tag:wedding"])


@pytest.fixture
def matrix() -> InteractionMatrix:
    rows = [
        (1, 100, 5.0), (1, 101, 4.5), (1, 103, 1.0),   # fan de science-fiction
        (2, 103, 5.0), (2, 104, 4.0), (2, 100, 2.0),   # fan de romance
        (3, 102, 2.0), (3, 105, 1.5),                  # n'a rien aime
    ]  # fmt: skip
    return InteractionMatrix.from_frame(pd.DataFrame(rows, columns=["userId", "movieId", "rating"]))


def _features(sparse: bool) -> ItemFeatures:
    # Lignes melangees : le modele doit realigner les vecteurs sur la matrice.
    order = np.array([5, 0, 3, 1, 4, 2])
    values = VECTORS[order]
    return ItemFeatures(MOVIE_IDS[order], sp.csr_matrix(values) if sparse else values, NAMES)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("aggregation", ["centroid", "knn"])
def test_recommends_the_unseen_movie_of_the_liked_genre(matrix, sparse, aggregation):
    model = ContentBasedRecommender(_features(sparse), aggregation=aggregation, k=3).fit(matrix)

    items, scores = model.recommend(np.array([0, 1]), k=1)

    assert matrix.item_ids[items[:, 0]].tolist() == [102, 105]
    assert (scores > 0).all()


def test_low_ratings_do_not_feed_the_profile(matrix):
    model = ContentBasedRecommender(_features(False), aggregation="centroid").fit(matrix)

    assert model.liked_[2].nnz == 0  # l'utilisateur 3 n'a aucune note >= 4
    np.testing.assert_array_equal(model.score(np.array([2])), np.zeros((1, 6), dtype=np.float32))
    assert model.liked_[0].indices.tolist() == [0, 1]  # la note 1.0 de l'utilisateur 1 est ignoree


def test_similar_items_explain_and_shared_terms(matrix):
    model = ContentBasedRecommender(_features(True), aggregation="knn", k=2).fit(matrix)

    neighbours, similarities = model.similar_items(0, n=2)
    assert set(neighbours) == {1, 2}
    assert (np.diff(similarities) <= 0).all()

    because, weights = model.explain(user_idx=0, item_idx=2, n=2)
    assert set(because) == {0, 1}
    assert (weights > 0).all()
    assert model.explain(user_idx=2, item_idx=2)[0].size == 0

    terms = model.shared_terms(0, 2)
    assert terms[0][0] == "tag:space"
    assert model.shared_terms(0, 3) == []


def test_shared_terms_requires_sparse_features_and_aggregation_is_validated(matrix):
    dense = ContentBasedRecommender(_features(False)).fit(matrix)
    with pytest.raises(NotImplementedError):
        dense.shared_terms(0, 1)
    with pytest.raises(ValueError, match="aggregation"):
        ContentBasedRecommender(_features(False), aggregation="moyenne")


def test_missing_content_vector_is_an_explicit_error(matrix):
    partial = ItemFeatures(MOVIE_IDS[:4], VECTORS[:4])
    with pytest.raises(KeyError, match="films sans vecteur"):
        ContentBasedRecommender(partial).fit(matrix)


def test_similarity_index_queries():
    index = SimilarityIndex(ItemFeatures(MOVIE_IDS, VECTORS))

    similar = index.similar(100, n=2)
    assert set(similar["movieId"]) == {101, 102}
    assert 100 not in similar["movieId"].to_numpy()
    assert similar["similarity"].is_monotonic_decreasing

    by_vector = index.search_vector(np.array([0, 0, 5.0, 1.0]), n=3)
    assert set(by_vector["movieId"]) == {103, 104, 105}
    assert by_vector["similarity"].max() <= 1.0 + 1e-5

    by_text = index.search_text("love story", lambda texts: np.array([[0, 0, 1.0, 0]]), n=1)
    assert by_text["movieId"].iloc[0] in {103, 104, 105}


def test_similarity_index_errors_and_round_trip(tmp_path: Path):
    index = SimilarityIndex(ItemFeatures(MOVIE_IDS, VECTORS))
    with pytest.raises(KeyError):
        index.similar(999)
    with pytest.raises(ValueError, match="dimension"):
        index.search_vector(np.ones(7))
    with pytest.raises(ValueError, match="vide"):
        index.search_text("   ", lambda texts: np.ones((1, 4)))
    with pytest.raises(FileNotFoundError):
        SimilarityIndex.load(tmp_path / "absent.npz")

    restored = SimilarityIndex.load(index.save(tmp_path / "index.npz"))
    pd.testing.assert_frame_equal(restored.similar(103, 2), index.similar(103, 2))
    assert len(restored) == 6


def _fake_encoder(texts):
    """Vecteurs deterministes derives du genre present dans le texte."""
    return np.array(
        [[1.0, 0.1] if "Sci-Fi" in text else [0.1, 1.0] for text in texts], dtype=np.float32
    )


def test_content_experiment_runs_end_to_end_without_future_tags(config: Config):
    rng = np.random.default_rng(1)
    n_movies, n_users = 30, 60
    genres = [["Sci-Fi"] if movie % 2 else ["Romance"] for movie in range(1, n_movies + 1)]
    movies = pd.DataFrame(
        {
            "movieId": range(1, n_movies + 1),
            "title": [f"Film {m} (2000)" for m in range(1, n_movies + 1)],
            "title_clean": [f"Film {m}" for m in range(1, n_movies + 1)],
            "year": pd.array([2000] * n_movies, dtype="Int16"),
            "genres": genres,
        }
    )
    start = pd.Timestamp("2020-01-01", tz="UTC")
    rows = []
    for user in range(1, n_users + 1):
        for movie in rng.choice(np.arange(1, n_movies + 1), size=22, replace=False):
            likes = (movie % 2) == (user % 2)
            rows.append((user, movie, 4.5 if likes else 2.0, int(rng.integers(0, 2000))))
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating", "h"])
    ratings["timestamp"] = start + pd.to_timedelta(ratings.pop("h"), "h")
    tags = pd.DataFrame(
        {
            "userId": [1, 2, 3, 4],
            "movieId": [1, 3, 1, 2],
            "tag": ["robots", "robots", "leak", "love"],
            "timestamp": [start, start, start + pd.Timedelta(days=4000), start],
        }
    )
    processed = config.paths.processed_dir
    processed.mkdir(parents=True)
    ratings.to_parquet(processed / "ratings.parquet")
    movies.to_parquet(processed / "movies.parquet")
    tags.to_parquet(processed / "tags.parquet")
    config.evaluation.ks = [5, 10]
    config.content.tag_min_df = 1
    config.content.grids = {
        "tfidf": [
            {"genre_weight": 1.0, "tag_weight": 1.0, "aggregation": "knn"},
            {"genre_weight": 1.0, "tag_weight": 0.0, "aggregation": "centroid"},
        ],
        "embeddings": [{"aggregation": "knn"}],
    }

    results = experiment.run(config, encoder=_fake_encoder)

    assert [row["model"] for row in results["test"]] == ["tfidf", "embeddings"]
    assert len(results["validation"]) == 3
    # Les gouts suivent le genre : un modele par contenu doit retrouver les films aimes.
    assert all(row["recall@10"] > 0.9 for row in results["test"])
    assert "rmse" not in results["test"][0]

    artifacts = config.paths.artifacts_dir / experiment.ARTIFACT_DIR
    tfidf = ContentBasedRecommender.load(artifacts / "tfidf.pkl")
    assert "tag:leak" not in set(tfidf.features.feature_names)  # tag pose apres la coupure
    index = SimilarityIndex.load(artifacts / experiment.INDEX_FILE)
    assert len(index) == n_movies
    documents = pd.read_parquet(artifacts / experiment.DOCUMENTS_FILE)
    assert "leak" in documents.loc[documents["movieId"] == 1, "tags"].iloc[0]  # index sans coupure
    on_disk = json.loads((config.paths.reports_dir / experiment.RESULTS_FILE).read_text("utf-8"))
    assert on_disk["content_coverage"]["test"]["share_with_genres"] == 1.0
    assert (artifacts / experiment.PER_USER_FILE).is_file()
