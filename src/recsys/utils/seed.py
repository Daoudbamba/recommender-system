"""Reproductibilite : une seule fonction pour fixer les graines aleatoires."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> np.random.Generator:
    """Fixe les graines de `random` et NumPy et retourne un generateur dedie.

    Les bibliotheques ajoutees dans les sprints suivants (PyTorch, XGBoost)
    recevront leur graine ici ou via leurs propres parametres `random_state`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)
