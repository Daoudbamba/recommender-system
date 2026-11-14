"""Workflow de reentrainement complet et reproductible, des donnees au bundle servi.

Chaque etape lit les sorties de la precedente ; toutes utilisent la graine et la
configuration de configs/config.yaml, et chaque experience est enregistree dans MLflow.

Etapes, dans l'ordre :
    download    telechargement verifie de MovieLens (ignore si deja present)
    prepare     nettoyage vers data/processed
    collaborative, content, hybrid, ranking    experiences (validation puis test)
    production  reentrainement sur toutes les donnees -> artifacts/production
    reports     rapports Markdown et figures

Usage :
    python scripts/retrain.py                       # tout (environ 2 h 30)
    python scripts/retrain.py --from hybrid         # reprendre a une etape
    python scripts/retrain.py --only production     # une seule etape
    python scripts/retrain.py --sample-users 5000 --until ranking   # essai rapide
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS: dict[str, list[str]] = {
    "download": ["scripts/download_data.py"],
    "prepare": ["scripts/prepare_data.py"],
    "collaborative": ["scripts/train_collaborative.py"],
    "content": ["scripts/train_content.py"],
    "hybrid": ["scripts/train_hybrid.py"],
    "ranking": ["scripts/train_ranking.py"],
    "production": ["scripts/train_production.py"],
    "reports": ["scripts/build_data_report.py", "scripts/build_model_report.py"],
}
SAMPLED = {"collaborative", "content", "hybrid", "ranking"}


def select_steps(start: str | None, until: str | None, only: list[str] | None) -> list[str]:
    names = list(STEPS)
    if only:
        unknown = set(only) - set(names)
        if unknown:
            raise SystemExit(f"etapes inconnues : {sorted(unknown)} (disponibles : {names})")
        return [name for name in names if name in only]
    first = names.index(start) if start else 0
    last = names.index(until) if until else len(names) - 1
    return names[first : last + 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--from", dest="start", choices=list(STEPS))
    parser.add_argument("--until", choices=list(STEPS))
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--sample-users", type=int, default=None)
    args = parser.parse_args()

    steps = select_steps(args.start, args.until, args.only)
    if args.sample_users and {"production", "reports"} & set(steps):
        raise SystemExit(
            "--sample-users ne s'applique qu'aux experiences : ajouter --until ranking"
        )
    print("Etapes :", " -> ".join(steps))
    total = time.perf_counter()
    for step in steps:
        for script in STEPS[step]:
            command = [sys.executable, str(ROOT / script)]
            if args.sample_users and step in SAMPLED:
                command += ["--sample-users", str(args.sample_users)]
            print(f"\n=== {step} : {' '.join(command[1:])}", flush=True)
            start = time.perf_counter()
            subprocess.run(command, cwd=ROOT, check=True)
            print(f"=== {step} termine en {time.perf_counter() - start:,.0f} s", flush=True)
    print(f"\nWorkflow termine en {(time.perf_counter() - total) / 60:,.1f} min")


if __name__ == "__main__":
    main()
