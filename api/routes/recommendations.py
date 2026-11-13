"""Recommandations personnalisees et cold start."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.schemas.models import (
    ColdStartRequest,
    ColdStartResponse,
    RecommendationResponse,
    Strategy,
)
from api.services.recommender import RecommenderService

router = APIRouter(tags=["Recommandations"])
Service = Annotated[RecommenderService, Depends(get_service)]
Diversity = Annotated[
    float, Query(ge=0.0, le=1.0, description="1 = pertinence seule ; plus bas = liste plus variée")
]


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    summary="Recommandations d'un utilisateur",
)
def recommend(
    user_id: int,
    service: Service,
    k: Annotated[int, Query(ge=1, le=50)] = 10,
    strategy: Strategy = Strategy.ranker,
    diversity: Diversity = 1.0,
) -> RecommendationResponse:
    """Un utilisateur inconnu reçoit les films populaires (`known_user: false`), pas une erreur."""
    return RecommendationResponse(**service.recommend(user_id, k, strategy.value, diversity))


@router.post(
    "/recommendations/cold-start",
    response_model=ColdStartResponse,
    summary="Recommandations pour un nouveau visiteur",
)
def recommend_cold_start(payload: ColdStartRequest, service: Service) -> ColdStartResponse:
    """À partir de quelques notes fournies dans la requête, sans compte ni réentraînement."""
    ratings = {item.movie_id: item.rating for item in payload.ratings}
    return ColdStartResponse(**service.recommend_from_ratings(ratings, payload.k))
