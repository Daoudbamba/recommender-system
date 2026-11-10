"""Entraine, selectionne et evalue les modeles par contenu (TF-IDF et embeddings).

Usage :
    python scripts/train_content.py                      # experience complete
    python scripts/train_content.py --sample-users 5000  # essai rapide
    python scripts/train_content.py --models tfidf       # sans les embeddings
"""

from __future__ import annotations

import argparse

from recsys.config import load_config
from recsys.experiments.content import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-users", type=int, default=None, help="sous-echantillon")
    parser.add_argument("--models", nargs="*", default=None, help="modeles a lancer")
    args = parser.parse_args()
    config = load_config()
    if args.sample_users:
        # Un essai sur echantillon ne doit pas ecraser les resultats officiels.
        config.paths.reports_dir = config.paths.reports_dir / "sample"
        config.paths.artifacts_dir = config.paths.artifacts_dir / "sample"
    run(config, sample_users=args.sample_users, models=args.models)


if __name__ == "__main__":
    main()
