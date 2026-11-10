from __future__ import annotations

import random

import numpy as np

from recsys.utils.seed import set_seed


def test_same_seed_gives_same_draws():
    first = set_seed(42).random(3), random.random(), np.random.rand()
    second = set_seed(42).random(3), random.random(), np.random.rand()

    np.testing.assert_array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_different_seeds_give_different_draws():
    assert not np.array_equal(set_seed(1).random(3), set_seed(2).random(3))
