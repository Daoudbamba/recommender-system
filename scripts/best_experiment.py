"""Affiche les meilleurs modeles enregistres dans MLflow, selon un critere explicite.

Critere par defaut : NDCG@10 mesure sur le test, pour les configurations retenues
sur la validation. C'est la metrique de selection du projet.

Usage :
    python scripts/best_experiment.py
    python scripts/best_experiment.py --metric recall@20 --limit 5
"""

from __future__ import annotations

import argparse

from recsys.config import load_config
from recsys.tracking import best_runs, tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="ndcg@10")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    config = load_config()

    rows = best_runs(config, args.metric, args.split, args.limit)
    if not rows:
        print(f"Aucun run trouve dans {tracking_uri(config)}.")
        return
    print(f"Meilleurs runs selon {args.metric} ({args.split}) — {tracking_uri(config)}\n")
    for rank, row in enumerate(rows, start=1):
        print(
            f"{rank:>2}. {row['model']:<24} {row['metric']:.4f}  "
            f"{row['experiment']:<22} run {row['run_id']}  ({row['source'] or 'direct'})"
        )


if __name__ == "__main__":
    main()
