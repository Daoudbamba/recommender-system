"""Client HTTP de l'API : le dashboard n'accede aux donnees et aux modeles que par elle."""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:8000"
ENV_VAR = "RECSYS_API_URL"


class ApiError(RuntimeError):
    """Erreur lisible par un humain, a afficher telle quelle dans l'interface."""


class ApiClient:
    """Enveloppe fine autour des routes de l'API."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get(ENV_VAR) or DEFAULT_URL).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApiError(
                f"API injoignable à l'adresse {self.base_url}. Lancez-la avec "
                "`uvicorn api.main:app --port 8000`."
            ) from error
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(f"Erreur {response.status_code} : {detail}")
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/stats")

    def metrics(self) -> dict[str, Any]:
        return self._request("GET", "/metrics")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def genres(self) -> list[str]:
        return self._request("GET", "/genres")

    def movies(
        self, query: str | None = None, genre: str | None = None, limit: int = 20, offset: int = 0
    ) -> dict[str, Any]:
        params = {"limit": limit, "offset": offset}
        if query:
            params["q"] = query
        if genre:
            params["genre"] = genre
        return self._request("GET", "/movies", params=params)

    def movie(self, movie_id: int) -> dict[str, Any]:
        return self._request("GET", f"/movies/{movie_id}")

    def sample_users(self, limit: int = 30) -> list[dict[str, int]]:
        return self._request("GET", "/users", params={"limit": limit})

    def history(self, user_id: int, limit: int = 50) -> dict[str, Any]:
        return self._request("GET", f"/users/{user_id}/history", params={"limit": limit})

    def recommendations(
        self, user_id: int, k: int = 10, strategy: str = "ranker", diversity: float = 1.0
    ) -> dict[str, Any]:
        params = {"k": k, "strategy": strategy, "diversity": diversity}
        return self._request("GET", f"/recommendations/{user_id}", params=params)

    def cold_start(self, ratings: dict[int, float], k: int = 10) -> dict[str, Any]:
        payload = {
            "ratings": [{"movie_id": movie, "rating": value} for movie, value in ratings.items()],
            "k": k,
        }
        return self._request("POST", "/recommendations/cold-start", json=payload)

    def similar(self, movie_id: int, k: int = 10, engine: str = "embeddings") -> dict[str, Any]:
        return self._request(
            "GET", f"/similar-movies/{movie_id}", params={"k": k, "engine": engine}
        )

    def predict_rating(self, user_id: int, movie_id: int) -> dict[str, Any]:
        return self._request(
            "POST", "/predict-rating", json={"user_id": user_id, "movie_id": movie_id}
        )
