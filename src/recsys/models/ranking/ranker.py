"""Learning to rank : XGBoost reordonne les candidats de l'hybride.

Deux etages, comme dans un systeme de recommandation en production :

1. generation de candidats : l'hybride propose les `n` meilleurs films ;
2. ranking : un modele supervise les reordonne a partir de variables que
   l'hybride ne voit pas (popularite, profil de l'utilisateur, affinite de
   genres...) et apprend lui-meme comment ponderer les modeles.

XGBRanker attend les lignes groupees par requete (ici : par utilisateur) et
les groupes contigus. Les lots produits par `RankingFeatureBuilder` sont deja
dans cet ordre : chaque utilisateur occupe `n_candidats` lignes consecutives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from recsys.features.ranking import RankingFeatureBuilder
from recsys.models.base import InteractionMatrix, Recommender


def group_ids(n_users: int, n_candidates: int) -> np.ndarray:
    """Identifiant de requete de chaque ligne : n_candidates lignes consecutives par utilisateur."""
    return np.repeat(np.arange(n_users), n_candidates)


def train_ranker(
    features: np.ndarray,
    labels: np.ndarray,
    qid: np.ndarray,
    dev: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> xgb.XGBRanker:
    """Entraine un XGBRanker ; `dev` (features, labels, qid) active l'arret anticipe."""
    if not (len(features) == len(labels) == len(qid)):
        raise ValueError("features, labels et qid doivent avoir le meme nombre de lignes")
    if np.any(np.diff(qid) < 0):
        raise ValueError("les lignes doivent etre triees par identifiant de requete (qid)")
    settings: dict[str, Any] = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@10",
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "random_state": seed,
        **(params or {}),
    }
    if dev is not None:
        settings.setdefault("early_stopping_rounds", 30)
    model = xgb.XGBRanker(**settings)
    fit_args: dict[str, Any] = {"qid": qid, "verbose": False}
    if dev is not None:
        fit_args.update(eval_set=[(dev[0], dev[1])], eval_qid=[dev[2]])
    model.fit(features, labels, **fit_args)
    return model


class RankingRecommender(Recommender):
    """Candidats de l'hybride reordonnes par un XGBRanker."""

    name = "xgb_ranker"

    def __init__(
        self,
        builder: RankingFeatureBuilder,
        model: xgb.XGBRanker,
        feature_idx: list[int] | None = None,
        name: str | None = None,
        batch_size: int = 256,
    ) -> None:
        super().__init__()
        self.builder = builder
        self.model = model
        self.feature_idx = feature_idx
        self.batch_size = batch_size
        if name:
            self.name = name
        if builder.matrix is not None:
            self.matrix = builder.matrix

    @property
    def feature_names(self) -> list[str]:
        names = self.builder.feature_names
        return names if self.feature_idx is None else [names[i] for i in self.feature_idx]

    @property
    def params(self) -> dict[str, Any]:
        keys = ("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree")
        settings = {key: self.model.get_params().get(key) for key in keys}
        return {
            **settings,
            "best_iteration": getattr(self.model, "best_iteration", None),
            "n_candidates": self.builder.n_candidates,
            "n_features": len(self.feature_names),
        }

    def _fit(self, matrix: InteractionMatrix) -> None:
        if self.builder.matrix is not matrix:
            self.builder.fit(matrix)

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Score du ranker pour les candidats, -inf pour tous les autres films."""
        matrix = self._check_fitted()
        user_idx = np.asarray(user_idx)
        out = np.full((len(user_idx), matrix.n_items), -np.inf, dtype=np.float32)
        for start in range(0, len(user_idx), self.batch_size):
            batch = self.builder.build(user_idx[start : start + self.batch_size])
            features = batch.flat_features
            if self.feature_idx is not None:
                features = features[:, self.feature_idx]
            predicted = self.model.predict(features).reshape(batch.items.shape)
            rows = np.arange(start, start + len(batch.user_idx))[:, None]
            out[rows, batch.items] = predicted
        return out

    def feature_importance(self) -> dict[str, float]:
        """Gain total apporte par chaque variable, normalise a 1."""
        gains = self.model.get_booster().get_score(importance_type="total_gain")
        named = {
            name: float(gains.get(f"f{index}", 0.0))
            for index, name in enumerate(self.feature_names)
        }
        total = sum(named.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in named.items()), key=lambda kv: -kv[1]))

    def save(self, path: Path) -> Path:
        """Sauvegarde le modele XGBoost (format JSON natif) et sa fiche a cote.

        Les composants et l'hybride ont deja leurs artefacts : seul le ranker est ecrit.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        model_path = path.with_suffix(".json")
        self.model.save_model(model_path)
        spec = {
            "name": self.name,
            "feature_names": self.feature_names,
            "feature_idx": self.feature_idx,
            "n_candidates": self.builder.n_candidates,
            "best_iteration": getattr(self.model, "best_iteration", None),
        }
        path.with_suffix(".spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return model_path

    @classmethod
    def from_artifacts(cls, path: Path, builder: RankingFeatureBuilder) -> RankingRecommender:
        """Recharge un ranker sauvegarde et le branche sur un constructeur de variables."""
        model_path, spec_path = path.with_suffix(".json"), path.with_suffix(".spec.json")
        for required in (model_path, spec_path):
            if not required.is_file():
                raise FileNotFoundError(f"Artefact de ranking introuvable : {required}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        expected = spec["feature_names"]
        names = builder.feature_names
        actual = names if spec["feature_idx"] is None else [names[i] for i in spec["feature_idx"]]
        if actual != expected:
            raise ValueError(
                "les variables du constructeur ne correspondent pas a celles du modele"
            )
        if builder.n_candidates != spec["n_candidates"]:
            raise ValueError("le nombre de candidats differe de celui de l'entrainement")
        model = xgb.XGBRanker()
        model.load_model(model_path)
        return cls(builder, model, spec["feature_idx"], spec["name"])
