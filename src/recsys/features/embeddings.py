"""Embeddings de phrases des films (Sentence Transformers), avec cache disque.

Encoder 62 000 textes prend plusieurs minutes sur CPU. Le cache associe a
chaque texte son vecteur, dans un fichier par modele : seuls les textes
jamais vus sont encodes. Passer d'une date de coupure a une autre ne
reencode donc que les films dont les tags ont change, et modifier un texte
ou changer de modele invalide exactement ce qu'il faut.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.features.item_features import ItemFeatures
from recsys.utils.logging import get_logger

logger = get_logger(__name__)

Encoder = Callable[[Sequence[str]], np.ndarray]


def sentence_transformer_encoder(model_name: str, batch_size: int = 128) -> Encoder:
    """Encodeur base sur un modele Sentence Transformers (charge a la demande)."""

    loaded: dict[str, object] = {}

    def encode(texts: Sequence[str]) -> np.ndarray:
        if "model" not in loaded:
            from sentence_transformers import SentenceTransformer

            loaded["model"] = SentenceTransformer(model_name, device="cpu")
        return loaded["model"].encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    return encode


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()  # noqa: S324 - cle de cache


def cache_path(cache_dir: Path, model_name: str) -> Path:
    slug = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:12]  # noqa: S324
    return cache_dir / f"{slug}.npz"


def _load_store(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if not path.is_file():
        return np.array([], dtype=str), None
    with np.load(path) as data:
        return data["keys"], data["vectors"]


def encode_with_cache(
    texts: Sequence[str], model_name: str, encoder: Encoder, cache_dir: Path
) -> np.ndarray:
    """Vecteurs normalises des textes ; n'encode que ceux absents du cache."""
    path = cache_path(cache_dir, model_name)
    keys = np.array([text_key(text) for text in texts])
    stored_keys, stored_vectors = _load_store(path)
    known = set(stored_keys.tolist())
    missing = {
        key: text for key, text in zip(keys.tolist(), texts, strict=True) if key not in known
    }
    if missing:
        logger.info(
            "Encodage de %d textes avec %s (%d deja en cache)",
            len(missing), model_name, len(texts) - len(missing),
        )  # fmt: skip
        vectors = np.asarray(encoder(list(missing.values())), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(missing):
            raise ValueError(
                f"l'encodeur a retourne {vectors.shape[0]} vecteurs pour {len(missing)} textes"
            )
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        stored_keys = np.concatenate([stored_keys, np.array(list(missing))])
        stored_vectors = vectors if stored_vectors is None else np.vstack([stored_vectors, vectors])
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(path, keys=stored_keys, vectors=stored_vectors)
    else:
        logger.info("Embeddings relus depuis le cache (%d textes)", len(texts))
    return stored_vectors[pd.Index(stored_keys).get_indexer(keys)]


def build_embedding_features(
    documents: pd.DataFrame, model_name: str, encoder: Encoder, cache_dir: Path
) -> ItemFeatures:
    """Matrice dense (films x dimensions), lignes de norme 1."""
    vectors = encode_with_cache(documents["text"].tolist(), model_name, encoder, cache_dir)
    return ItemFeatures(documents["movieId"].to_numpy(), vectors)
