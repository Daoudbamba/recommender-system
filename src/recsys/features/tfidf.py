"""Representation TF-IDF des films, par champ pondere.

Trois blocs independants, chacun avec son propre vocabulaire et son propre
IDF, puis concatenes avec un poids par champ :

- genres : 19 valeurs possibles, peu discriminantes mais presentes partout ;
- tags : texte libre des utilisateurs, tres discriminant mais absent d'un
  film sur quatre. Un tag est garde entier (« based on a book » reste un
  seul terme) et pondere par le nombre d'utilisateurs qui l'ont pose ;
- titre : mots du titre, utile surtout pour les suites et les franchises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

from recsys.features.item_features import ItemFeatures


def _tfidf(counts: sp.csr_matrix) -> sp.csr_matrix:
    if counts.shape[1] == 0:
        return counts.astype(np.float32)
    return TfidfTransformer(sublinear_tf=True, norm="l2").fit_transform(counts).astype(np.float32)


def genre_block(documents: pd.DataFrame) -> tuple[sp.csr_matrix, list[str]]:
    exploded = documents["genres"].explode().dropna()
    codes, names = pd.factorize(exploded, sort=True)
    counts = sp.csr_matrix(
        (np.ones(len(codes)), (exploded.index.to_numpy(), codes)),
        shape=(len(documents), len(names)),
    )
    return _tfidf(counts), [f"genre:{name}" for name in names]


def tag_block(
    documents: pd.DataFrame, tag_counts: pd.DataFrame, min_df: int
) -> tuple[sp.csr_matrix, list[str]]:
    """Bloc des tags ; un tag doit decrire au moins `min_df` films pour etre garde."""
    rows = pd.Index(documents["movieId"]).get_indexer(tag_counts["movieId"])
    known = tag_counts[rows >= 0]
    document_frequency = known.groupby("tag")["movieId"].nunique()
    vocabulary = document_frequency.index[document_frequency >= min_df].sort_values()
    cols = pd.Index(vocabulary).get_indexer(known["tag"])
    keep = cols >= 0
    counts = sp.csr_matrix(
        (known["n_users"].to_numpy()[keep], (rows[rows >= 0][keep], cols[keep])),
        shape=(len(documents), len(vocabulary)),
        dtype=np.float64,
    )
    return _tfidf(counts), [f"tag:{name}" for name in vocabulary]


def title_block(documents: pd.DataFrame, min_df: int) -> tuple[sp.csr_matrix, list[str]]:
    vectorizer = CountVectorizer(stop_words="english", min_df=min_df, token_pattern=r"[a-z]{3,}")
    counts = vectorizer.fit_transform(documents["title_clean"].fillna("").str.lower())
    return _tfidf(counts), [f"title:{name}" for name in vectorizer.get_feature_names_out()]


def build_tfidf_features(
    documents: pd.DataFrame,
    tag_counts: pd.DataFrame,
    genre_weight: float = 1.0,
    tag_weight: float = 1.0,
    title_weight: float = 0.0,
    tag_min_df: int = 3,
    title_min_df: int = 2,
) -> ItemFeatures:
    """Matrice TF-IDF (films x termes), lignes de norme 1."""
    if min(genre_weight, tag_weight, title_weight) < 0:
        raise ValueError("les poids de champ doivent etre positifs ou nuls")
    if genre_weight + tag_weight + title_weight == 0:
        raise ValueError("au moins un champ doit avoir un poids non nul")
    documents = documents.reset_index(drop=True)
    # Un champ de poids nul n'est pas calcule du tout.
    builders = (
        (genre_weight, lambda: genre_block(documents)),
        (tag_weight, lambda: tag_block(documents, tag_counts, tag_min_df)),
        (title_weight, lambda: title_block(documents, title_min_df)),
    )
    blocks, names = [], []
    for weight, build in builders:
        if weight > 0:
            block, block_names = build()
            blocks.append(block * weight)
            names.extend(block_names)
    matrix = normalize(sp.hstack(blocks, format="csr"), norm="l2").astype(np.float32)
    return ItemFeatures(documents["movieId"].to_numpy(), matrix, np.array(names))
