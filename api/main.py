"""Application FastAPI du systeme de recommandation.

Lancement : uvicorn api.main:app --port 8000   (documentation interactive sur /docs)

Les modeles sont charges une seule fois, au demarrage. Si le bundle de
production est absent, l'API demarre quand meme en mode degrade : /health
explique ce qui manque et les autres routes repondent 503.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import health, insights, movies, ratings, recommendations, users
from api.services.recommender import RecommenderService, UnknownMovieError, UnknownUserError
from recsys.config import Config, load_config
from recsys.serving.bundle import BundleError, ServingBundle
from recsys.serving.training import BUNDLE_DIR
from recsys.utils.logging import get_logger

logger = get_logger("api")

DESCRIPTION = """
Recommandations de films personnalisées, films similaires et prédiction de notes,
servis par les modèles entraînés sur MovieLens 25M.

- **Utilisateur connu** : candidats de l'hybride réordonnés par un modèle XGBoost.
- **Utilisateur inconnu** : repli sur la popularité, ou recommandations à partir de
  quelques notes via `POST /recommendations/cold-start`.
"""


def bundle_directory(config: Config) -> Path:
    return config.paths.artifacts_dir / BUNDLE_DIR


def create_app(config: Config | None = None) -> FastAPI:
    """Fabrique l'application ; `config` permet aux tests de pointer vers un autre bundle."""
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.bundle_dir = bundle_directory(config)
        app.state.service = None
        app.state.load_error = None
        try:
            bundle = ServingBundle.load(app.state.bundle_dir)
            app.state.service = RecommenderService(bundle)
            logger.info(
                "modeles charges : %d utilisateurs, %d films",
                bundle.matrix.n_users, bundle.matrix.n_items,
            )  # fmt: skip
        except (BundleError, FileNotFoundError, ValueError) as error:
            app.state.load_error = str(error)
            logger.error("API en mode degrade : %s", error)
        yield

    app = FastAPI(
        title="Recommender System API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    for module in (health, movies, recommendations, ratings, users, insights):
        app.include_router(module.router)

    @app.exception_handler(UnknownMovieError)
    async def _unknown_movie(_: Request, error: UnknownMovieError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Film inconnu : {error.args[0]}"})

    @app.exception_handler(UnknownUserError)
    async def _unknown_user(_: Request, error: UnknownUserError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Utilisateur inconnu : {error.args[0]} (aucun historique)"},
        )

    return app


app = create_app()
