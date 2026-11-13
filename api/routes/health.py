"""Etat du service et disponibilite des artefacts."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas.models import Health
from recsys.serving.bundle import artifact_status

router = APIRouter(tags=["Santé"])


@router.get("/health", response_model=Health, summary="État du service")
def health(request: Request) -> Health:
    """Toujours 200 : indique si les modèles sont chargés et quels artefacts manquent."""
    state = request.app.state
    artifacts = artifact_status(state.bundle_dir)
    service = state.service
    if service is None:
        return Health(
            status="degraded", models_loaded=False, artifacts=artifacts, detail=state.load_error
        )
    bundle = service.bundle
    return Health(
        status="ok",
        models_loaded=True,
        artifacts=artifacts,
        trained_at=bundle.manifest["trained_at"],
        n_users=bundle.matrix.n_users,
        n_movies_in_model=bundle.matrix.n_items,
        n_movies_catalogue=len(bundle.movies),
    )
