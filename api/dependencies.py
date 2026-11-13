"""Dependances FastAPI partagees par les routes."""

from __future__ import annotations

from fastapi import HTTPException, Request

from api.services.recommender import RecommenderService


def get_service(request: Request) -> RecommenderService:
    """Service de recommandation ; 503 tant que les modeles ne sont pas charges."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=getattr(request.app.state, "load_error", None) or "modèles non chargés",
        )
    return service
