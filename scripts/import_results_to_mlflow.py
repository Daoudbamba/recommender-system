"""Importe dans MLflow les resultats d'experiences deja executees.

Les experiences des sprints 2 a 5 ont ete executees avant l'ajout du suivi MLflow.
Plutot que de les relancer (environ deux heures), ce script rejoue leurs fichiers
`reports/*_results.json` dans MLflow, a l'identique : memes parametres, memes
metriques, fichier JSON joint en artefact. Chaque run importe porte le tag
`source=import` pour ne pas le confondre avec un run enregistre en direct.

Usage :
    python scripts/import_results_to_mlflow.py
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/import_results_to_mlflow.py
"""

from __future__ import annotations

import json

from recsys.config import Config, load_config
from recsys.serving.training import RESULT_FILES
from recsys.tracking import ExperimentTracker, tracking_uri


def import_results(config: Config) -> dict[str, int]:
    """Cree un run parent par experience et un run enfant par ligne de resultats."""
    counts: dict[str, int] = {}
    for experiment, filename in RESULT_FILES.items():
        path = config.paths.reports_dir / filename
        if not path.is_file():
            print(f"{filename} absent : experience {experiment} ignoree")
            continue
        results = json.loads(path.read_text(encoding="utf-8"))
        tracker = ExperimentTracker(config, experiment)
        tags = {"source": "import"}
        params = {"seed": results["seed"], "protocol": results["protocol"]}
        with tracker.run(f"{experiment} (import)", params=params, tags=tags):
            for row in results["validation"]:
                tracker.log_candidate(row["model"], "validation", row["params"], row, tags=tags)
            for row in results["test"]:
                selected = experiment != "ranking" or row["model"] == "xgb_ranker"
                tracker.log_candidate(
                    row["model"], "test", row["params"], row, selected=selected, tags=tags
                )
            tracker.log_artifacts([path])
        counts[experiment] = len(results["validation"]) + len(results["test"])
    return counts


def main() -> None:
    config = load_config()
    config.tracking.enabled = True
    counts = import_results(config)
    print(f"Runs importes dans {tracking_uri(config)} : {counts}")


if __name__ == "__main__":
    main()
