from __future__ import annotations

import pandas as pd
import pytest

from recsys.analysis import eda
from recsys.preprocessing.pipeline import clean_tables


@pytest.fixture
def clean(raw_tables, config):
    return clean_tables(raw_tables, config)


def test_overview_density_matches_definition(clean):
    ov = eda.overview(clean["ratings"], clean["movies"], clean["tags"])

    assert (ov["n_users"], ov["n_movies_rated"], ov["n_ratings"]) == (3, 3, 6)
    assert ov["n_movies_catalog"] == 4
    assert ov["density"] == pytest.approx(6 / 9)


def test_rating_distribution_shares_sum_to_one(clean):
    distribution = eda.rating_distribution(clean["ratings"])

    assert distribution["count"].sum() == len(clean["ratings"])
    assert distribution["share"].sum() == pytest.approx(1.0)
    assert distribution["rating"].is_monotonic_increasing


def test_low_activity_and_concentration():
    counts = pd.Series([1, 1, 2, 6])

    low = eda.low_activity(counts, [1, 2]).set_index("max_ratings")
    assert low.loc[1, "n_entities"] == 2
    assert low.loc[2, "share_entities"] == pytest.approx(0.75)
    assert low.loc[2, "share_ratings"] == pytest.approx(0.4)

    top = eda.concentration(counts, top_shares=(0.25,))
    assert top.loc[0, "n_entities"] == 1
    assert top.loc[0, "share_ratings"] == pytest.approx(0.6)


def test_genre_stats_weights_mean_by_number_of_ratings(clean):
    popularity = eda.movie_popularity(clean["ratings"])
    genres = eda.genre_stats(clean["movies"], popularity).set_index("genre")

    # Toy Story (film 10) : notes 4.0 et 5.0 -> Animation = 2 notes, moyenne 4.5
    assert genres.loc["Animation", "n_ratings"] == 2
    assert genres.loc["Animation", "mean_rating"] == pytest.approx(4.5)
    # Drama n'a qu'un film jamais note : present au catalogue, zero note
    assert genres.loc["Drama", "n_movies"] == 1
    assert genres.loc["Drama", "n_ratings"] == 0
    assert "(no genres listed)" not in genres.index


def test_user_activity_and_yearly_aggregates(clean):
    users = eda.user_activity(clean["ratings"])
    assert users.loc[1, "n_ratings"] == 3
    assert users.loc[1, "mean_rating"] == pytest.approx((4.0 + 3.5 + 5.0) / 3)

    per_year = eda.ratings_per_year(clean["ratings"])
    assert per_year["year"].tolist() == [2017]
    assert per_year.loc[0, "active_users"] == 3


def test_tag_stats_and_top_movies(clean):
    stats = eda.tag_stats(clean["tags"], clean["movies"])
    assert stats["n_distinct_tags"] == 2
    assert stats["share_catalog_tagged"] == pytest.approx(0.5)

    popularity = eda.movie_popularity(clean["ratings"])
    top = eda.top_movies(clean["movies"], popularity, n=1)
    assert top.loc[0, "title"] == "Heat (1995)"
    assert top.loc[0, "n_ratings"] == 3
