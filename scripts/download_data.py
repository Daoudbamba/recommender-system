"""Telecharge MovieLens, verifie sa somme MD5 et controle le schema des CSV.

Usage : python scripts/download_data.py [--force]
"""

from __future__ import annotations

import argparse

from recsys.config import load_config
from recsys.data.download import ensure_dataset
from recsys.data.loaders import check_schema, read_header
from recsys.data.schemas import SCHEMAS
from recsys.utils.logging import get_logger

logger = get_logger("scripts.download_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retelecharge meme si present")
    args = parser.parse_args()

    config = load_config()
    raw_dir = ensure_dataset(config, force=args.force)

    for path in sorted(raw_dir.iterdir()):
        logger.info("%-22s %8.1f Mo", path.name, path.stat().st_size / 1e6)
    for name, schema in SCHEMAS.items():
        path = raw_dir / schema.filename
        extra = check_schema(path, schema)
        logger.info("%s : colonnes %s%s", name, read_header(path), f" (+{extra})" if extra else "")
    logger.info("Schema conforme pour les %d tables.", len(SCHEMAS))


if __name__ == "__main__":
    main()
