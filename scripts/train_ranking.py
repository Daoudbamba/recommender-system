"""Entraine et evalue le modele de ranking XGBoost.

Prerequis : scripts/train_collaborative.py, train_content.py et train_hybrid.py.

Usage :
    python scripts/train_ranking.py
    python scripts/train_ranking.py --sample-users 5000   # essai rapide
"""

from __future__ import annotations

import argparse

from recsys.config import load_config
from recsys.experiments.ranking import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-users", type=int, default=None, help="sous-echantillon")
    args = parser.parse_args()
    config = load_config()
    if args.sample_users:
        # Hyperparametres et recette de l'hybride viennent des resultats officiels ;
        # seules les sorties de l'essai sont ecrites a part.
        official = config.paths.reports_dir
        config.paths.reports_dir = official / "sample"
        config.paths.artifacts_dir = config.paths.artifacts_dir / "sample"
        config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
        for name in ("collaborative_results.json", "content_results.json", "hybrid_results.json"):
            (config.paths.reports_dir / name).write_bytes((official / name).read_bytes())
    run(config, sample_users=args.sample_users)


if __name__ == "__main__":
    main()
