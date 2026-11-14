"""Suivi des experiences avec MLflow.

Chaque execution d'une experience cree un run parent (une ligne par lancement)
et un run enfant par configuration evaluee :

    recsys-collaborative                  <- experience MLflow
    └── collaborative 2026-11-14 10:02    <- run parent : protocole, graine, resultats JSON
        ├── item_knn | validation         <- parametres + metriques d'une configuration
        ├── item_knn | validation
        └── item_knn | test  (selected)   <- configuration retenue, mesuree sur le test

Les noms de metriques MLflow n'acceptent pas « @ » : `ndcg@10` devient `ndcg_at_10`.

Stockage : base SQLite `mlflow.db` et artefacts dans `mlruns/` a la racine du depot
(le stockage sur fichiers seul n'est plus accepte par MLflow 3). La variable
d'environnement MLFLOW_TRACKING_URI prend le dessus, par exemple pour envoyer les
runs au serveur MLflow de docker-compose (http://localhost:5000).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from recsys.config import PROJECT_ROOT, Config
from recsys.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DB = PROJECT_ROOT / "mlflow.db"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "mlruns"
MAX_PARAM_LENGTH = 500
NUMERIC = (int, float)


def metric_name(name: str) -> str:
    """Nom de metrique accepte par MLflow (`ndcg@10` -> `ndcg_at_10`)."""
    return re.sub(r"[^0-9A-Za-z_\-. /]", "_", name.replace("@", "_at_"))


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Parametres a plat, en texte court : `{"weights": {"a": 1}}` -> `{"weights.a": "1"}`."""
    out: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_params(value, f"{name}."))
        else:
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            out[name] = text[:MAX_PARAM_LENGTH]
    return out


def numeric_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Metriques numeriques d'une ligne de resultats, noms convertis pour MLflow."""
    return {
        metric_name(key): float(value)
        for key, value in row.items()
        if isinstance(value, NUMERIC) and not isinstance(value, bool)
    }


def _utf8_console() -> None:
    """Evite un plantage sous Windows : MLflow ecrit des emojis que la console cp1252 refuse."""
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if encoding not in {"utf-8", "utf8"} and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def tracking_uri(config: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or config.tracking.uri or default_uri()


def default_uri() -> str:
    return f"sqlite:///{DEFAULT_DB.as_posix()}"


class ExperimentTracker:
    """Facade minimale au-dessus de MLflow ; ne fait rien si le suivi est desactive."""

    def __init__(self, config: Config, experiment: str) -> None:
        self.enabled = config.tracking.enabled
        self.name = f"{config.tracking.experiment_prefix}-{experiment}"
        self._mlflow = None
        if not self.enabled:
            return
        import mlflow

        _utf8_console()
        self._mlflow = mlflow
        uri = tracking_uri(config)
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(self.name) is None:
            artifacts = config.tracking.artifact_location
            location = None
            if uri.startswith("sqlite:"):
                root = Path(artifacts) if artifacts else DEFAULT_ARTIFACTS
                location = root.resolve().as_uri()
            mlflow.create_experiment(self.name, artifact_location=location)
        mlflow.set_experiment(self.name)

    @contextmanager
    def run(
        self, run_name: str, params: dict[str, Any] | None = None, tags: dict | None = None
    ) -> Iterator[str | None]:
        """Run parent d'une execution d'experience ; retourne son identifiant."""
        if not self.enabled:
            yield None
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self._mlflow.start_run(run_name=f"{run_name} {stamp}") as active:
            self._mlflow.set_tags({"kind": "experiment", **(tags or {})})
            if params:
                self._mlflow.log_params(flatten_params(params))
            yield active.info.run_id

    def log_candidate(
        self,
        model: str,
        split: str,
        params: dict[str, Any],
        row: dict[str, Any],
        selected: bool = False,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Run enfant : une configuration mesuree sur un jeu (validation ou test)."""
        if not self.enabled:
            return
        name = f"{model} | {split}" + (" (selected)" if selected else "")
        with self._mlflow.start_run(run_name=name, nested=True):
            self._mlflow.set_tags(
                {"model": model, "split": split, "selected": str(selected).lower(), **(tags or {})}
            )
            self._mlflow.log_params(flatten_params(params))
            self._mlflow.log_metrics(numeric_metrics(row))

    def log_params(self, params: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_params(flatten_params(params))

    def log_metrics(self, values: dict[str, Any]) -> None:
        if self.enabled:
            self._mlflow.log_metrics(numeric_metrics(values))

    def log_artifacts(self, paths: list[Path], folder: str = "reports") -> None:
        if not self.enabled:
            return
        for path in paths:
            if path.is_file():
                self._mlflow.log_artifact(str(path), artifact_path=folder)

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        if self.enabled:
            self._mlflow.log_dict(payload, filename)


def best_runs(
    config: Config, metric: str = "ndcg@10", split: str = "test", limit: int = 10
) -> list[dict[str, Any]]:
    """Meilleurs runs enfants, toutes experiences confondues, selon une metrique.

    Le critere par defaut est le NDCG@10 sur le test des configurations retenues :
    c'est la metrique de selection du projet, mesuree une seule fois par modele.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri(config))
    prefix = f"{config.tracking.experiment_prefix}-"
    names = [e.name for e in mlflow.search_experiments() if e.name.startswith(prefix)]
    if not names:
        return []
    column = f"metrics.{metric_name(metric)}"
    frame = mlflow.search_runs(
        experiment_names=names,
        filter_string=f"tags.split = '{split}'",
        order_by=[f"{column} DESC"],
    )
    if frame.empty or column not in frame:
        return []
    frame = frame.dropna(subset=[column]).head(limit)
    return [
        {
            "run_id": row["run_id"],
            "experiment": next(
                (n for n in names if mlflow.get_experiment_by_name(n).experiment_id
                 == row["experiment_id"]),
                row["experiment_id"],
            ),
            "model": row.get("tags.model"),
            "metric": float(row[column]),
            "source": row.get("tags.source"),
            "started": str(row["start_time"]),
        }
        for _, row in frame.iterrows()
    ]  # fmt: skip
