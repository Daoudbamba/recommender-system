"""Comparaison de moteurs de films similaires.

Il n'existe pas de verite terrain pour « ces deux films se ressemblent ».
On utilise donc deux mesures indirectes, a lire ensemble :

- l'accord avec le comportement : part des voisins de contenu qui sont aussi
  des voisins collaboratifs (films aimes par les memes utilisateurs). Les
  notes n'entrent pas dans les vecteurs de contenu : c'est un controle
  independant ;
- la coherence de genres : indice de Jaccard moyen entre les genres d'un
  film et ceux de ses voisins. Mesure biaisee en faveur de toute
  representation qui contient les genres : elle sert de garde-fou, pas de
  classement.
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from recsys.models.content_based.search import SimilarityIndex


class HasSimilarItems(Protocol):
    def similar_items(self, item_idx: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]: ...


def neighbour_table(model: HasSimilarItems, items: np.ndarray, n: int) -> np.ndarray:
    """Indices des `n` voisins de chaque film ; -1 quand il y en a moins de `n`."""
    out = np.full((len(items), n), -1, dtype=np.int64)
    for row, item in enumerate(items):
        neighbours = model.similar_items(int(item), n)[0]
        out[row, : len(neighbours)] = neighbours
    return out


def mean_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Part moyenne des voisins de `first` presents dans `second` (meme film, meme n)."""
    shared = [
        len(set(a[a >= 0]) & set(b[b >= 0])) / max((a >= 0).sum(), 1)
        for a, b in zip(first, second, strict=True)
    ]
    return float(np.mean(shared))


def mean_genre_jaccard(
    neighbours: np.ndarray, items: np.ndarray, genre_sets: list[frozenset[str]]
) -> float:
    """Jaccard moyen entre les genres d'un film et ceux de ses voisins."""
    values = []
    for item, row in zip(items, neighbours, strict=True):
        own = genre_sets[int(item)]
        for neighbour in row[row >= 0]:
            other = genre_sets[int(neighbour)]
            union = own | other
            values.append(len(own & other) / len(union) if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def search_latency_ms(index: SimilarityIndex, n_queries: int = 200, seed: int = 42) -> dict:
    """Temps d'une recherche exacte des 10 plus proches voisins dans l'index."""
    rng = np.random.default_rng(seed)
    movie_ids = rng.choice(index.movie_ids, size=min(n_queries, len(index)), replace=False)
    timings = []
    for movie_id in movie_ids:
        start = time.perf_counter()
        index.similar(int(movie_id), 10)
        timings.append(1000 * (time.perf_counter() - start))
    return {
        "n_vectors": len(index),
        "dimension": int(index.vectors.shape[1]),
        "n_queries": len(timings),
        "mean_ms": float(np.mean(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }
