"""Dashboard : client HTTP et pages Streamlit, branches sur la vraie application de test."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from dashboard.components import ui
from dashboard.components.api_client import ApiClient, ApiError

PAGES = Path(__file__).resolve().parents[1] / "dashboard" / "pages"


@pytest.fixture(scope="module")
def api(client: TestClient) -> ApiClient:
    """Le client du dashboard, relie a l'API de test sans passer par le reseau.

    Chaque requete emise par le dashboard est relayee telle quelle a l'application :
    c'est le vrai code des deux cotes, seul le cable est remplace.
    """

    def relay(request: httpx.Request) -> httpx.Response:
        response = client.request(
            request.method,
            request.url.raw_path.decode(),
            content=request.content,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers={"content-type": response.headers.get("content-type", "application/json")},
        )

    return ApiClient(base_url="http://testserver", transport=httpx.MockTransport(relay))


def test_client_reads_every_resource_the_dashboard_needs(api: ApiClient):
    assert api.health()["models_loaded"]
    assert api.stats()["overview"]["n_users"] == 120
    assert {row["model"] for row in api.metrics()["models"]} >= {"xgb_ranker", "hybrid_rrf"}
    assert api.genres() == ["Romance", "Sci-Fi"]
    assert api.movies("film 12")["items"][0]["title"] == "Film 12 (2000)"
    assert api.movies(genre="Romance", limit=5)["total"] == 20
    assert api.movie(3)["movie_id"] == 3

    user = api.sample_users(5)[-1]["user_id"]
    assert api.history(user, 5)["user_id"] == user
    recommended = api.recommendations(user, 5, "hybrid", 0.5)
    assert recommended["strategy"] == "hybrid" and recommended["diversity"] == 0.5
    assert api.similar(1, 3, "tfidf")["engine"] == "tfidf"
    assert 0.5 <= api.predict_rating(user, 1)["predicted_rating"] <= 5.0
    assert api.cold_start({1: 5.0, 3: 4.5}, 4)["n_ratings_used"] == 2


def test_client_turns_http_errors_into_readable_messages(api: ApiClient):
    with pytest.raises(ApiError, match="404.*Film inconnu"):
        api.movie(99999)
    with pytest.raises(ApiError, match="422"):
        api.recommendations(1, k=0)


def test_client_explains_how_to_start_an_unreachable_api():
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusee", request=request)

    offline = ApiClient(base_url="http://localhost:1", transport=httpx.MockTransport(refuse))
    with pytest.raises(ApiError, match="uvicorn api.main:app"):
        offline.health()


def test_api_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("RECSYS_API_URL", "http://api.interne:9000/")
    assert ApiClient().base_url == "http://api.interne:9000"
    monkeypatch.delenv("RECSYS_API_URL")
    assert ApiClient().base_url == "http://localhost:8000"


def test_movie_card_escapes_text_and_shows_real_fields():
    card = ui.movie_card(
        {"title": "<b>Alien</b>", "year": 1979, "genres": ["Horror", "Sci-Fi"], "n_ratings": 1234,
         "mean_rating": 4.05},
        rank=1, badge="score 0.9", why="Parce que vous avez aimé Aliens",
    )  # fmt: skip
    assert "&lt;b&gt;Alien&lt;/b&gt;" in card and "<b>Alien" not in card
    assert "1979" in card and "Horror, Sci-Fi" in card and "4.05" in card and "1 234 notes" in card
    assert "Parce que vous avez aimé Aliens" in card
    bare = ui.movie_card({"title": "Sans info", "genres": [], "n_ratings": 0})
    assert "Sans info" in bare and "★" not in bare


@pytest.mark.parametrize(
    ("page", "expected"),
    [
        ("0_Home.py", "Les plus populaires"),
        ("1_Personalized_Recommendations.py", "recommandé"),
        ("2_Similar_Movies.py", "Films les plus proches"),
        ("3_Model_Performance.py", "Tableau complet"),
        ("4_Data_Insights.py", "Répartition des notes"),
    ],
)
def test_pages_render_against_the_api_without_error(api: ApiClient, monkeypatch, page, expected):
    monkeypatch.setattr(ui, "get_client", lambda: api)

    app = AppTest.from_file(str(PAGES / page), default_timeout=60).run()
    if page == "2_Similar_Movies.py":
        # La recherche par defaut ("Matrix") ne trouve rien dans le mini-catalogue :
        # la page l'annonce sans erreur, puis on cherche un film qui existe.
        assert "Aucun film" in app.info[0].value
        app.text_input[0].set_value("Film").run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert not app.error, [error.value for error in app.error]
    headings = " ".join(element.value for element in app.subheader)
    assert expected.lower() in headings.lower()


def test_page_shows_a_clear_message_when_the_api_is_down(monkeypatch):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connexion refusee", request=request)

    offline = ApiClient(base_url="http://localhost:1", transport=httpx.MockTransport(refuse))
    monkeypatch.setattr(ui, "get_client", lambda: offline)

    app = AppTest.from_file(str(PAGES / "0_Home.py"), default_timeout=30).run()

    assert not app.exception
    assert len(app.error) == 1 and "API injoignable" in app.error[0].value
