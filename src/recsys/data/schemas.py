"""Schemas attendus des fichiers MovieLens.

Ces schemas sont une *hypothese* confrontee au fichier reel a chaque
chargement (voir loaders.py) : une colonne manquante leve une erreur au lieu
de produire un resultat silencieusement faux.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSchema:
    filename: str
    dtypes: dict[str, str]
    key: tuple[str, ...] = field(default_factory=tuple)

    @property
    def columns(self) -> list[str]:
        return list(self.dtypes)


SCHEMAS: dict[str, TableSchema] = {
    "ratings": TableSchema(
        filename="ratings.csv",
        dtypes={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
        key=("userId", "movieId"),
    ),
    "movies": TableSchema(
        filename="movies.csv",
        dtypes={"movieId": "int32", "title": "string", "genres": "string"},
        key=("movieId",),
    ),
    "tags": TableSchema(
        filename="tags.csv",
        dtypes={"userId": "int32", "movieId": "int32", "tag": "string", "timestamp": "int64"},
        key=("userId", "movieId", "tag"),
    ),
    # imdbId est lu comme texte pour conserver les zeros de tete ;
    # tmdbId est un entier nullable car certains films n'en ont pas.
    "links": TableSchema(
        filename="links.csv",
        dtypes={"movieId": "int32", "imdbId": "string", "tmdbId": "Int64"},
        key=("movieId",),
    ),
}
