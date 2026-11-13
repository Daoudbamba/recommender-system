"""Schemas Pydantic des requetes et reponses de l'API."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Strategy(StrEnum):
    ranker = "ranker"
    hybrid = "hybrid"
    popularity = "popularity"


class SimilarityEngine(StrEnum):
    embeddings = "embeddings"
    tfidf = "tfidf"
    collaborative = "collaborative"


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str]
    n_ratings: int
    mean_rating: float | None = None
    in_model: bool = Field(description="Le film a assez de notes pour être recommandé")


class MoviePage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[Movie]


class Recommendation(BaseModel):
    rank: int
    movie: Movie
    score: float
    contributions: dict[str, float] = Field(
        default_factory=dict, description="Part de chaque modèle dans le score hybride"
    )
    because_you_liked: list[str] = []
    reasons: list[str] = []


class RecommendationResponse(BaseModel):
    user_id: int
    known_user: bool
    strategy: Strategy
    diversity: float
    n_ratings: int
    items: list[Recommendation]
    elapsed_ms: float


class RatingInput(BaseModel):
    movie_id: int = Field(gt=0)
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def _half_star(cls, value: float) -> float:
        if (value * 2) % 1 != 0:
            raise ValueError("la note doit être un multiple de 0.5")
        return value


class ColdStartRequest(BaseModel):
    ratings: list[RatingInput] = Field(default_factory=list, max_length=200)
    k: int = Field(default=10, ge=1, le=50)

    @field_validator("ratings")
    @classmethod
    def _unique_movies(cls, value: list[RatingInput]) -> list[RatingInput]:
        ids = [item.movie_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("un même film ne peut être noté qu'une fois")
        return value


class ColdStartResponse(BaseModel):
    n_ratings_used: int
    ignored_movie_ids: list[int]
    popularity_weight: float
    items: list[Recommendation]
    elapsed_ms: float


class SimilarMovie(BaseModel):
    movie: Movie
    similarity: float
    shared_genres: list[str]


class SimilarMoviesResponse(BaseModel):
    movie: Movie
    engine: SimilarityEngine
    items: list[SimilarMovie]


class RatingPredictionRequest(BaseModel):
    user_id: int = Field(gt=0)
    movie_id: int = Field(gt=0)


class RatingPrediction(BaseModel):
    user_id: int
    movie_id: int
    predicted_rating: float
    basis: str
    known_user: bool
    movie_in_model: bool
    existing_rating: float | None = None


class GenreShare(BaseModel):
    genre: str
    share: float


class HistoryItem(BaseModel):
    movie: Movie
    rating: float


class UserHistory(BaseModel):
    user_id: int
    n_ratings: int
    mean_rating: float
    n_liked: int
    favourite_genres: list[GenreShare]
    items: list[HistoryItem]


class SampleUser(BaseModel):
    user_id: int
    n_ratings: int


class Health(BaseModel):
    status: str = Field(description="ok, ou degraded si les modèles ne sont pas chargés")
    models_loaded: bool
    artifacts: dict[str, bool]
    detail: str | None = None
    trained_at: str | None = None
    n_users: int | None = None
    n_movies_in_model: int | None = None
    n_movies_catalogue: int | None = None


class ErrorResponse(BaseModel):
    detail: Any
