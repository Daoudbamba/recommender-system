"""Historique et profils des utilisateurs."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.schemas.models import ErrorResponse, SampleUser, UserHistory
from api.services.recommender import RecommenderService

router = APIRouter(tags=["Utilisateurs"])
Service = Annotated[RecommenderService, Depends(get_service)]


@router.get("/users", response_model=list[SampleUser], summary="Utilisateurs d'exemple")
def sample_users(service: Service, limit: Annotated[int, Query(ge=1, le=200)] = 30) -> list:
    """Échantillon réparti du plus petit au plus gros historique, pour la démonstration."""
    return service.sample_users(limit)


@router.get(
    "/users/{user_id}/history",
    response_model=UserHistory,
    responses={404: {"model": ErrorResponse, "description": "Utilisateur inconnu"}},
    summary="Historique de notes d'un utilisateur",
)
def user_history(
    user_id: int, service: Service, limit: Annotated[int, Query(ge=1, le=500)] = 50
) -> UserHistory:
    """Films notés, du mieux noté au moins bien noté, et genres préférés."""
    return UserHistory(**service.user_history(user_id, limit))
