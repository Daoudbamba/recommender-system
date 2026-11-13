"""Chiffres du jeu de donnees et performances des modeles (pour le dashboard)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.services.recommender import RecommenderService

router = APIRouter(tags=["Données et modèles"])
Service = Annotated[RecommenderService, Depends(get_service)]


@router.get("/stats", summary="Statistiques du jeu de données")
def dataset_stats(service: Service) -> dict[str, Any]:
    """Volumes, distribution des notes, genres, activité : calculés à l'entraînement."""
    return service.bundle.stats


@router.get("/metrics", summary="Performances mesurées des modèles")
def model_metrics(service: Service) -> dict[str, Any]:
    """Résultats de test des expériences, tels qu'écrits dans `reports/*_results.json`."""
    return service.bundle.metrics


@router.get("/model-info", summary="Modèles en service")
def model_info(service: Service) -> dict[str, Any]:
    """Date d'entraînement, composants, poids de l'hybride, réglages."""
    return service.bundle.manifest
