"""Feature engineering du contenu : documents, TF-IDF, cache d'embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recsys.features import embeddings as emb
from recsys.features.item_text import (
    aggregate_tags,
    build_item_documents,
    content_coverage,
    format_text,
)
from recsys.features.tfidf import build_tfidf_features

DAY = pd.Timestamp("2020-01-01", tz="UTC")


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Alien (1979)", "Aliens (1986)", "Notting Hill (1999)", "Untitled"],
            "title_clean": ["Alien", "Aliens", "Notting Hill", "Untitled"],
            "year": pd.array([1979, 1986, 1999, pd.NA], dtype="Int16"),
            "genres": [["Horror", "Sci-Fi"], ["Action", "Sci-Fi"], ["Comedy", "Romance"], []],
        }
    )


@pytest.fixture
def tags() -> pd.DataFrame:
    rows = [
        (10, 1, "space", 1), (11, 1, "space", 2), (10, 1, "monster", 3),
        (10, 2, "space", 4), (12, 2, "monster", 5), (13, 2, "marines", 6),
        (10, 3, "london", 7), (11, 3, "romance", 8),
        (14, 3, "space", 30),  # pose apres la coupure du jour 20
    ]  # fmt: skip
    frame = pd.DataFrame(rows, columns=["userId", "movieId", "tag", "d"])
    frame["timestamp"] = DAY + pd.to_timedelta(frame.pop("d"), "D")
    return frame


def test_aggregate_tags_counts_distinct_users(tags):
    counts = aggregate_tags(tags).set_index(["movieId", "tag"])["n_users"]
    assert counts[(1, "space")] == 2
    assert counts[(1, "monster")] == 1


def test_aggregate_tags_ignores_tags_after_the_cutoff(tags):
    counts = aggregate_tags(tags, until=DAY + pd.Timedelta(days=20))
    assert not ((counts["movieId"] == 3) & (counts["tag"] == "space")).any()
    assert len(aggregate_tags(tags)) == len(counts) + 1


def test_documents_keep_most_consensual_tags_and_invent_nothing(movies, tags):
    docs = build_item_documents(movies, aggregate_tags(tags), max_tags=1).set_index("movieId")

    assert docs.loc[1, "tags"] == ["space"]
    assert docs.loc[1, "text"] == "Alien (1979). Genres: Horror, Sci-Fi. Tags: space."
    assert docs.loc[4, "tags"] == []
    assert docs.loc[4, "text"] == "Untitled."


@pytest.mark.parametrize("dtype", ["object", "string[pyarrow]", "str"])
def test_documents_contain_tags_whatever_the_string_dtype(movies, tags, dtype):
    # Non-regression : avec le type `string[pyarrow]` produit par le nettoyage, les tags
    # disparaissaient silencieusement des documents (et donc des embeddings).
    typed = tags.astype({"tag": dtype})

    docs = build_item_documents(movies, aggregate_tags(typed), max_tags=5).set_index("movieId")

    assert docs.loc[1, "tags"] == ["space", "monster"]
    assert all(isinstance(value, list) for value in docs["tags"])
    assert "Tags: space, monster" in docs.loc[1, "text"]
    assert content_coverage(docs.reset_index())["share_with_tags"] == pytest.approx(0.75)


def test_format_text_skips_missing_parts():
    assert format_text("Heat", 1995, ["Crime"], []) == "Heat (1995). Genres: Crime."
    assert format_text("Heat", pd.NA, [], ["tense"]) == "Heat. Tags: tense."


def test_content_coverage(movies, tags):
    docs = build_item_documents(movies, aggregate_tags(tags), max_tags=5)
    coverage = content_coverage(docs)
    assert coverage["share_with_genres"] == pytest.approx(0.75)
    assert coverage["share_with_tags"] == pytest.approx(0.75)
    assert coverage["share_title_only"] == pytest.approx(0.25)


def test_tfidf_rows_are_unit_norm_and_features_are_named(movies, tags):
    counts = aggregate_tags(tags)
    docs = build_item_documents(movies, counts, max_tags=5)

    features = build_tfidf_features(docs, counts, tag_min_df=2, title_weight=0.5, title_min_df=1)

    norms = np.sqrt(features.matrix.multiply(features.matrix).sum(axis=1)).A1
    np.testing.assert_allclose(norms[:3], 1.0, rtol=1e-5)
    names = set(features.feature_names)
    assert {"genre:Sci-Fi", "tag:space", "tag:monster", "title:alien"} <= names
    assert "tag:marines" not in names  # un seul film : sous le seuil min_df
    assert "tag:london" not in names


def test_tfidf_similarity_follows_shared_content(movies, tags):
    counts = aggregate_tags(tags)
    docs = build_item_documents(movies, counts, max_tags=5)
    matrix = build_tfidf_features(docs, counts, tag_min_df=2).matrix

    similarity = (matrix @ matrix.T).toarray()

    assert similarity[0, 1] > similarity[0, 2]  # Alien est plus proche d'Aliens
    assert similarity[0, 3] == 0  # un film sans genre ni tag n'est proche de rien


def test_tfidf_field_weights_change_the_representation(movies, tags):
    counts = aggregate_tags(tags)
    docs = build_item_documents(movies, counts, max_tags=5)

    genres_only = build_tfidf_features(docs, counts, tag_weight=0.0)
    assert all(name.startswith("genre:") for name in genres_only.feature_names)
    with pytest.raises(ValueError):
        build_tfidf_features(docs, counts, genre_weight=0, tag_weight=0, title_weight=0)
    with pytest.raises(ValueError):
        build_tfidf_features(docs, counts, genre_weight=-1)


def test_item_features_align_reorders_and_rejects_unknown_movies(movies, tags):
    counts = aggregate_tags(tags)
    features = build_tfidf_features(build_item_documents(movies, counts, 5), counts, tag_min_df=2)

    aligned = features.align(np.array([3, 1]))

    assert aligned.movie_ids.tolist() == [3, 1]
    np.testing.assert_allclose(aligned.matrix[1].toarray(), features.matrix[0].toarray())
    with pytest.raises(KeyError, match="999"):
        features.align(np.array([1, 999]))


def _counting_encoder(calls: list[int]):
    def encode(texts):
        calls.append(len(texts))
        rng = np.random.default_rng(len(texts))
        return rng.normal(size=(len(texts), 6)) * 5  # volontairement non normalise

    return encode


def test_embeddings_are_normalised_and_cached(tmp_path: Path):
    calls: list[int] = []
    texts = ["Alien (1979).", "Heat (1995)."]

    first = emb.encode_with_cache(texts, "fake-model", _counting_encoder(calls), tmp_path)
    second = emb.encode_with_cache(texts, "fake-model", _counting_encoder(calls), tmp_path)

    assert calls == [2]  # le second appel relit le cache
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-5)
    assert first.dtype == np.float32


def test_cache_encodes_only_unseen_texts_and_is_separate_per_model(tmp_path: Path):
    calls: list[int] = []
    encoder = _counting_encoder(calls)
    first = emb.encode_with_cache(["a", "b"], "model-1", encoder, tmp_path)
    second = emb.encode_with_cache(["c", "a", "a"], "model-1", encoder, tmp_path)
    emb.encode_with_cache(["a", "b"], "model-2", encoder, tmp_path)

    assert calls == [2, 1, 2]  # seul "c" est encode au deuxieme appel
    np.testing.assert_array_equal(second[1], first[0])  # meme texte, meme vecteur
    np.testing.assert_array_equal(second[1], second[2])
    assert second.shape == (3, 6)
    assert emb.cache_path(tmp_path, "model-1") != emb.cache_path(tmp_path, "model-2")


def test_encoder_returning_wrong_number_of_vectors_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="vecteurs"):
        emb.encode_with_cache(["a", "b"], "m", lambda texts: np.ones((1, 4)), tmp_path)
