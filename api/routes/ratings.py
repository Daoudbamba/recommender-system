"""Prediction de note."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas.models import ErrorResponse, RatingPrediction, RatingPredictionRequest
from api.services.recommender import RecommenderService

router = APIRouter(tags=["Notes"])


@router.post(
    "/predict-rating",
    response_model=RatingPrediction,
    responses={404: {"model": ErrorResponse, "description": "Film inconnu"}},
    summary="Note qu'un utilisateur donnerait à un film",
)
def predict_rating(
    payload: RatingPredictionRequest,
    service: Annotated[RecommenderService, Depends(get_service)],
) -> RatingPrediction:
    """Le champ `basis` indique sur quoi repose l'estimation (voisinage, moyenne du film...)."""
    return RatingPrediction(**service.predict_rating(payload.user_id, payload.movie_id))
