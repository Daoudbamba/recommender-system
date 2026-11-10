"""Genere reports/collaborative_report.md a partir des resultats d'experience.

Usage : python scripts/build_model_report.py
"""

from __future__ import annotations

import pandas as pd

from recsys.analysis.model_report import build_collaborative_report
from recsys.config import load_config
from recsys.experiments.collaborative import ARTIFACT_DIR, PER_USER_FILE, RESULTS_FILE


def main() -> None:
    config = load_config()
    per_user_path = config.paths.artifacts_dir / ARTIFACT_DIR / PER_USER_FILE
    if not per_user_path.is_file():
        raise FileNotFoundError(f"{per_user_path} absent : lancer scripts/train_collaborative.py")
    report = build_collaborative_report(config, RESULTS_FILE, pd.read_parquet(per_user_path))
    print(f"Rapport ecrit : {report}")


if __name__ == "__main__":
    main()
