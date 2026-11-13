"""Tests fonctionnels de l'API, sur un bundle entraine a partir du mini-projet synthetique.

Toute la chaine est rejouee une fois pour la session (fixtures `trained_config` et `client`
de conftest.py) : quatre experiences, entrainement de production, chargement du bundle par
l'application, puis requetes HTTP reelles.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from recsys.config import Config
from recsys.serving import training
from recsys.serving.bundle import BundleError, ServingBundle, artifact_status


@pytest.fixture(scope="module")
def known_user(client: TestClient) -> int:
    return client.get("/users", params={"limit": 5}).json()[-1]["user_id"]


# ------------------------------------------------------------------ bundle
def test_bundle_contains_every_expected_artifact(trained_config: Config):
    directory = trained_config.paths.artifacts_dir / training.BUNDLE_DIR
    status = artifact_status(directory)

    assert all(status.values())
    assert {"als_implicit.pkl", "item_knn.pkl", "tfidf.pkl", "embeddings.pkl"} <= set(status)
    manifest = json.loads((directory / training.MANIFEST_FILE).read_text("utf-8"))
    assert manifest["hybrid"]["strategy"] == "rrf"
    assert manifest["ranking"]["reference_year"] > 2020


def test_incomplete_bundle_is_reported_not_loaded(tmp_path: Path):
    with pytest.raises(BundleError, match="train_production"):
        ServingBundle.load(tmp_path)
    assert not any(artifact_status(tmp_path).values())


def test_api_starts_degraded_without_models(config: Config):
    with TestClient(create_app(config)) as degraded:
        health = degraded.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded" and not body["models_loaded"]
        assert "train_production" in body["detail"]
        for path in ("/movies", "/recommendations/1", "/stats"):
            response = degraded.get(path)
            assert response.status_code == 503
            assert "train_production" in response.json()["detail"]


# ------------------------------------------------------------------ sante
def test_health_reports_loaded_models(client: TestClient):
    body = client.get("/health").json()

    assert body["status"] == "ok" and body["models_loaded"]
    assert all(body["artifacts"].values())
    assert body["n_movies_catalogue"] == 40
    assert 0 < body["n_movies_in_model"] <= 40
    assert body["trained_at"]


def test_openapi_documents_every_required_endpoint(client: TestClient):
    paths = client.get("/openapi.json").json()["paths"]
    required = {
        "/health", "/movies", "/movies/{movie_id}", "/recommendations/{user_id}",
        "/similar-movies/{movie_id}", "/predict-rating", "/users/{user_id}/history",
    }  # fmt: skip
    assert required <= set(paths)
    assert client.get("/docs").status_code == 200


# ------------------------------------------------------------------ films
def test_movies_are_paginated_and_sorted_by_popularity(client: TestClient):
    first = client.get("/movies", params={"limit": 5}).json()
    second = client.get("/movies", params={"limit": 5, "offset": 5}).json()

    assert first["total"] == 40 and len(first["items"]) == 5
    counts = [m["n_ratings"] for m in first["items"] + second["items"]]
    assert counts == sorted(counts, reverse=True)
    assert not {m["movie_id"] for m in first["items"]} & {m["movie_id"] for m in second["items"]}


def test_movies_filter_by_title_and_genre(client: TestClient):
    by_title = client.get("/movies", params={"q": "film 12"}).json()
    assert [m["title"] for m in by_title["items"]] == ["Film 12 (2000)"]

    by_genre = client.get("/movies", params={"genre": "sci-fi", "limit": 100}).json()
    assert by_genre["total"] == 20
    assert all(m["genres"] == ["Sci-Fi"] for m in by_genre["items"])
    assert client.get("/movies", params={"q": "introuvable"}).json()["total"] == 0
    assert client.get("/genres").json() == ["Romance", "Sci-Fi"]


def test_movie_detail_and_unknown_movie(client: TestClient):
    movie = client.get("/movies/7").json()
    assert movie["movie_id"] == 7 and movie["title"] == "Film 7 (2000)"
    assert movie["genres"] == ["Sci-Fi"]

    missing = client.get("/movies/99999")
    assert missing.status_code == 404
    assert "99999" in missing.json()["detail"]


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/movies", {"limit": 0}),
        ("/movies", {"limit": 1000}),
        ("/movies", {"offset": -1}),
        ("/movies/abc", {}),
        ("/recommendations/1", {"k": 0}),
        ("/recommendations/1", {"strategy": "magie"}),
        ("/recommendations/1", {"diversity": 1.5}),
        ("/similar-movies/1", {"engine": "inconnu"}),
        ("/users/1/history", {"limit": 0}),
    ],
)
def test_invalid_parameters_are_rejected(client: TestClient, path: str, params: dict):
    assert client.get(path, params=params).status_code == 422


# ------------------------------------------------------- recommandations
@pytest.mark.parametrize("strategy", ["ranker", "hybrid", "popularity"])
def test_recommendations_for_a_known_user(client: TestClient, known_user: int, strategy: str):
    body = client.get(
        f"/recommendations/{known_user}", params={"k": 5, "strategy": strategy}
    ).json()

    assert body["known_user"] and body["strategy"] == strategy
    assert [item["rank"] for item in body["items"]] == list(range(1, len(body["items"]) + 1))
    assert 1 <= len(body["items"]) <= 5
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)
    history = client.get(f"/users/{known_user}/history", params={"limit": 500}).json()
    seen = {entry["movie"]["movie_id"] for entry in history["items"]}
    assert not seen & {item["movie"]["movie_id"] for item in body["items"]}


def test_recommendations_are_explained(client: TestClient, known_user: int):
    body = client.get(f"/recommendations/{known_user}", params={"k": 3}).json()

    for item in body["items"]:
        assert item["reasons"] and item["reasons"][0].startswith("Recommandé surtout par")
        assert sum(item["contributions"].values()) == pytest.approx(1.0, abs=0.01)
        assert set(item["contributions"]) <= {"als_implicit", "item_knn", "tfidf", "embeddings"}


def test_recommendations_follow_the_users_taste(client: TestClient, known_user: int):
    history = client.get(f"/users/{known_user}/history").json()
    favourite = history["favourite_genres"][0]["genre"]

    body = client.get(f"/recommendations/{known_user}", params={"k": 5}).json()

    genres = [item["movie"]["genres"][0] for item in body["items"]]
    assert genres.count(favourite) >= 4


def test_unknown_user_falls_back_to_popular_movies(client: TestClient):
    body = client.get("/recommendations/987654", params={"k": 5}).json()
    popular = client.get("/recommendations/987654", params={"k": 5, "strategy": "popularity"})

    assert not body["known_user"] and body["strategy"] == "popularity"
    assert body["n_ratings"] == 0 and len(body["items"]) == 5
    assert body["items"] == popular.json()["items"]
    assert "plus appréciés" in body["items"][0]["reasons"][0]


def test_diversity_parameter_is_applied_and_reported(client: TestClient, known_user: int):
    plain = client.get(f"/recommendations/{known_user}", params={"k": 5}).json()
    diverse = client.get(f"/recommendations/{known_user}", params={"k": 5, "diversity": 0.3}).json()

    assert plain["diversity"] == 1.0 and diverse["diversity"] == 0.3
    assert len(diverse["items"]) == len(plain["items"])
    again = client.get(f"/recommendations/{known_user}", params={"k": 5}).json()
    assert [i["movie"]["movie_id"] for i in again["items"]] == [
        i["movie"]["movie_id"] for i in plain["items"]
    ]  # le reglage d'une requete ne deborde pas sur la suivante


# -------------------------------------------------------------- cold start
def test_cold_start_without_ratings_returns_popular_movies(client: TestClient):
    body = client.post("/recommendations/cold-start", json={"k": 4}).json()
    assert body["n_ratings_used"] == 0 and body["popularity_weight"] == 1.0
    assert len(body["items"]) == 4


def test_cold_start_personalises_from_a_few_ratings(client: TestClient):
    sci_fi = client.get("/movies", params={"genre": "Sci-Fi", "limit": 3}).json()["items"]
    payload = {"ratings": [{"movie_id": m["movie_id"], "rating": 5.0} for m in sci_fi], "k": 5}

    body = client.post("/recommendations/cold-start", json=payload).json()

    assert body["n_ratings_used"] == 3 and body["popularity_weight"] < 1.0
    rated = {m["movie_id"] for m in sci_fi}
    assert not rated & {item["movie"]["movie_id"] for item in body["items"]}
    genres = [item["movie"]["genres"][0] for item in body["items"]]
    assert genres.count("Sci-Fi") >= 4


def test_cold_start_ignores_unknown_movies_and_validates_ratings(client: TestClient):
    body = client.post(
        "/recommendations/cold-start",
        json={"ratings": [{"movie_id": 424242, "rating": 4.0}, {"movie_id": 1, "rating": 4.5}]},
    ).json()
    assert body["ignored_movie_ids"] == [424242] and body["n_ratings_used"] == 1

    for bad in (
        {"ratings": [{"movie_id": 1, "rating": 7}]},
        {"ratings": [{"movie_id": 1, "rating": 3.3}]},
        {"ratings": [{"movie_id": 1, "rating": 4}, {"movie_id": 1, "rating": 5}]},
        {"ratings": [{"movie_id": -1, "rating": 4}]},
        {"k": 0},
    ):
        assert client.post("/recommendations/cold-start", json=bad).status_code == 422


# --------------------------------------------------------------- similarite
@pytest.mark.parametrize("engine", ["embeddings", "tfidf", "collaborative"])
def test_similar_movies(client: TestClient, engine: str):
    movie_id = client.get("/movies", params={"limit": 1}).json()["items"][0]["movie_id"]

    body = client.get(f"/similar-movies/{movie_id}", params={"k": 4, "engine": engine}).json()

    assert body["movie"]["movie_id"] == movie_id and body["engine"] == engine
    assert 1 <= len(body["items"]) <= 4
    assert movie_id not in {item["movie"]["movie_id"] for item in body["items"]}
    similarities = [item["similarity"] for item in body["items"]]
    assert similarities == sorted(similarities, reverse=True)
    for item in body["items"]:
        assert set(item["shared_genres"]) <= set(body["movie"]["genres"])


def test_similar_movies_unknown_movie(client: TestClient):
    assert client.get("/similar-movies/99999").status_code == 404


# ------------------------------------------------------------------- notes
def test_predict_rating_for_known_and_unknown_user(client: TestClient, known_user: int):
    history = client.get(f"/users/{known_user}/history", params={"limit": 500}).json()
    seen = {entry["movie"]["movie_id"] for entry in history["items"]}
    unseen = next(
        m["movie_id"]
        for m in client.get("/movies", params={"limit": 100}).json()["items"]
        if m["movie_id"] not in seen and m["in_model"]
    )

    known = client.post("/predict-rating", json={"user_id": known_user, "movie_id": unseen}).json()
    stranger = client.post("/predict-rating", json={"user_id": 987654, "movie_id": unseen}).json()

    assert 0.5 <= known["predicted_rating"] <= 5.0
    assert known["known_user"] and "item-kNN" in known["basis"]
    assert known["existing_rating"] is None
    assert not stranger["known_user"] and "utilisateur inconnu" in stranger["basis"]
    rated = history["items"][0]
    again = client.post(
        "/predict-rating", json={"user_id": known_user, "movie_id": rated["movie"]["movie_id"]}
    ).json()
    assert again["existing_rating"] == rated["rating"]


def test_predict_rating_validation_and_unknown_movie(client: TestClient):
    assert client.post("/predict-rating", json={"user_id": 1, "movie_id": 99999}).status_code == 404
    assert client.post("/predict-rating", json={"user_id": 1}).status_code == 422
    assert client.post("/predict-rating", json={"user_id": 0, "movie_id": 1}).status_code == 422
    assert client.post("/predict-rating", json={"user_id": "x", "movie_id": 1}).status_code == 422


# ------------------------------------------------------------- utilisateurs
def test_user_history(client: TestClient, known_user: int):
    body = client.get(f"/users/{known_user}/history", params={"limit": 10}).json()

    assert body["user_id"] == known_user and len(body["items"]) == min(10, body["n_ratings"])
    ratings = [entry["rating"] for entry in body["items"]]
    assert ratings == sorted(ratings, reverse=True)
    assert body["favourite_genres"][0]["share"] == pytest.approx(1.0)
    assert 0.5 <= body["mean_rating"] <= 5.0

    missing = client.get("/users/987654/history")
    assert missing.status_code == 404 and "987654" in missing.json()["detail"]


def test_sample_users_cover_small_and_large_histories(client: TestClient):
    users = client.get("/users", params={"limit": 10}).json()
    sizes = [user["n_ratings"] for user in users]
    assert len(users) == 10 and sizes == sorted(sizes)


# --------------------------------------------------------------- dashboard
def test_stats_metrics_and_model_info(client: TestClient):
    stats = client.get("/stats").json()
    assert stats["overview"]["n_users"] == 120
    assert (
        sum(row["count"] for row in stats["rating_distribution"]) == stats["overview"]["n_ratings"]
    )
    assert {row["genre"] for row in stats["genres"]} == {"Romance", "Sci-Fi"}

    metrics = client.get("/metrics").json()
    names = {row["model"] for row in metrics["models"]}
    assert {"als_implicit", "tfidf", "hybrid_rrf", "xgb_ranker"} <= names
    assert all("ndcg@10" in row for row in metrics["models"])

    info = client.get("/model-info").json()
    assert set(info["components"]) == {"als_implicit", "item_knn", "tfidf", "embeddings"}
