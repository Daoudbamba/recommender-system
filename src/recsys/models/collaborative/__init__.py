"""Modeles de filtrage collaboratif."""

from recsys.models.collaborative.baselines import BiasBaseline, PopularityRecommender
from recsys.models.collaborative.factorization import ExplicitALS, ImplicitALS
from recsys.models.collaborative.knn import ItemKNN, UserKNN

MODEL_REGISTRY = {
    model.name: model
    for model in (PopularityRecommender, BiasBaseline, ItemKNN, UserKNN, ExplicitALS, ImplicitALS)
}

__all__ = [
    "MODEL_REGISTRY",
    "BiasBaseline",
    "ExplicitALS",
    "ImplicitALS",
    "ItemKNN",
    "PopularityRecommender",
    "UserKNN",
]
