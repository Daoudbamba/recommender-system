# Image unique pour l'API et le dashboard (la commande change selon le service).
#
# L'image ne contient ni les donnees MovieLens (licence : pas de redistribution) ni les
# modeles : le bundle artifacts/production est monte en lecture seule au lancement.
# Elle n'embarque que les dependances de service, sans PyTorch ni MLflow.

FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenMP : requis par XGBoost et implicit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependances d'abord : couche reutilisee tant que les versions ne changent pas.
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY configs ./configs
COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit

# Echec de construction si un module du service ne s'importe pas avec ces dependances.
RUN python -c "import api.main, dashboard.components.ui; print('imports ok')"

RUN useradd --create-home --uid 1000 app && mkdir -p /app/artifacts && chown -R app /app
USER app

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
