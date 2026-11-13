"""Catalogue de films et films similaires."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.schemas.models import (
    ErrorResponse,
    Movie,
    MoviePage,
    SimilarityEngine,
    SimilarMoviesResponse,
)
from api.services.recommender import RecommenderService

router = APIRouter(tags=["Films"])
Service = Annotated[RecommenderService, Depends(get_service)]
NOT_FOUND = {404: {"model": ErrorResponse, "description": "Film inconnu"}}


@router.get("/movies", response_model=MoviePage, summary="Rechercher dans le catalogue")
def list_movies(
    service: Service,
    q: Annotated[str | None, Query(max_length=100, description="Texte du titre")] = None,
    genre: Annotated[str | None, Query(max_length=40)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> MoviePage:
    """Films triés par nombre de notes décroissant, filtrables par titre et par genre."""
    return MoviePage(**service.list_movies(q, genre, limit, offset))


@router.get("/genres", response_model=list[str], summary="Genres du catalogue")
def list_genres(service: Service) -> list[str]:
    return service.genres()


@router.get("/movies/{movie_id}", response_model=Movie, responses=NOT_FOUND, summary="Un film")
def get_movie(movie_id: int, service: Service) -> Movie:
    return Movie(**service.get_movie(movie_id))


@router.get(
    "/similar-movies/{movie_id}",
    response_model=SimilarMoviesResponse,
    responses=NOT_FOUND,
    summary="Films similaires",
)
def similar_movies(
    movie_id: int,
    service: Service,
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    engine: SimilarityEngine = SimilarityEngine.embeddings,
) -> SimilarMoviesResponse:
    """`embeddings` couvre tout le catalogue ; les deux autres exigent un film assez noté."""
    return SimilarMoviesResponse(**service.similar_movies(movie_id, k, engine.value))
