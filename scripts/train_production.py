"""Entraine les modeles de production sur toutes les donnees et ecrit le bundle de service.

Prerequis : les quatre experiences (collaboratif, contenu, hybride, ranking), dont les
resultats fournissent hyperparametres, poids de l'hybride et modele de ranking.

Usage : python scripts/train_production.py
"""

from __future__ import annotations

from recsys.config import load_config
from recsys.serving.training import train_production


def main() -> None:
    print(f"Bundle ecrit : {train_production(load_config())}")


if __name__ == "__main__":
    main()
