"""Modeles de recommandation par contenu."""

from recsys.models.content_based.recommender import ContentBasedRecommender
from recsys.models.content_based.search import SimilarityIndex

__all__ = ["ContentBasedRecommender", "SimilarityIndex"]
