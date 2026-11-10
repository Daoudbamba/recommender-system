"""Controle qualite + nettoyage : data/raw -> data/processed (Parquet).

Usage : python scripts/prepare_data.py
"""

from __future__ import annotations

import json

from recsys.config import load_config
from recsys.preprocessing.pipeline import run


def main() -> None:
    summary = run(load_config())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
