"""Chargement des CSV bruts avec verification du schema reel."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recsys.data.schemas import SCHEMAS, TableSchema


class SchemaError(ValueError):
    """Le fichier ne contient pas les colonnes attendues."""


def read_header(path: Path) -> list[str]:
    """Colonnes reellement presentes dans le fichier, sans charger les donnees."""
    return list(pd.read_csv(path, nrows=0).columns)


def check_schema(path: Path, schema: TableSchema) -> list[str]:
    """Verifie les colonnes du fichier et retourne les colonnes supplementaires."""
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    actual = read_header(path)
    missing = [col for col in schema.columns if col not in actual]
    if missing:
        raise SchemaError(
            f"{path.name} : colonnes manquantes {missing} (colonnes presentes : {actual})"
        )
    return [col for col in actual if col not in schema.columns]


def load_table(raw_dir: Path, name: str, nrows: int | None = None) -> pd.DataFrame:
    """Charge une table brute avec des types explicites et compacts."""
    if name not in SCHEMAS:
        raise KeyError(f"Table inconnue : {name} (tables connues : {sorted(SCHEMAS)})")
    schema = SCHEMAS[name]
    path = raw_dir / schema.filename
    check_schema(path, schema)
    return pd.read_csv(path, usecols=schema.columns, dtype=schema.dtypes, nrows=nrows)


def load_raw_tables(raw_dir: Path, nrows: int | None = None) -> dict[str, pd.DataFrame]:
    """Charge les quatre tables principales."""
    return {name: load_table(raw_dir, name, nrows=nrows) for name in SCHEMAS}
