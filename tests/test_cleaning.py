from __future__ import annotations

import pandas as pd

from recsys.preprocessing import cleaning


def test_clean_movies_extracts_year_title_and_genres(raw_tables):
    movies = cleaning.clean_movies(raw_tables["movies"]).set_index("movieId")

    assert movies.loc[10, "year"] == 1995
    assert movies.loc[10, "title_clean"] == "Toy Story"
    assert list(movies.loc[10, "genres"]) == ["Adventure", "Animation", "Children"]
    assert movies.loc[20, "title"] == "Heat (1995)"  # espace final retire
    assert movies.loc[20, "year"] == 1995


def test_clean_movies_does_not_invent_missing_information(raw_tables):
    movies = cleaning.clean_movies(raw_tables["movies"]).set_index("movieId")

    assert pd.isna(movies.loc[30, "year"])
    assert movies.loc[30, "title_clean"] == "Mystery Film"
    assert list(movies.loc[30, "genres"]) == []
    assert movies.loc[30, "n_genres"] == 0


def test_clean_ratings_drops_invalid_and_keeps_latest_duplicate(raw_tables):
    ratings = cleaning.clean_ratings(raw_tables["ratings"], 0.5, 5.0)

    assert not ratings.duplicated(["userId", "movieId"]).any()
    assert ratings["rating"].between(0.5, 5.0).all()
    kept = ratings[(ratings.userId == 2) & (ratings.movieId == 10)]
    assert kept["rating"].tolist() == [5.0]
    assert str(ratings["timestamp"].dt.tz) == "UTC"
    assert len(ratings) == 7


def test_clean_ratings_does_not_mutate_input(raw_tables):
    before = raw_tables["ratings"].copy()
    cleaning.clean_ratings(raw_tables["ratings"], 0.5, 5.0)
    pd.testing.assert_frame_equal(raw_tables["ratings"], before)


def test_clean_tags_normalises_and_deduplicates(raw_tables):
    tags = cleaning.clean_tags(raw_tables["tags"])

    assert sorted(tags["tag"]) == ["al pacino", "pixar"]


def test_restrict_to_catalog_removes_unknown_movies(raw_tables):
    movies = cleaning.clean_movies(raw_tables["movies"])
    ratings = cleaning.clean_ratings(raw_tables["ratings"], 0.5, 5.0)

    restricted = cleaning.restrict_to_catalog(ratings, movies)

    assert 99 not in set(restricted["movieId"])
    assert len(restricted) == len(ratings) - 1


def test_filter_min_activity_iterates_until_stable():
    # Le film 3 n'a qu'une note : son retrait fait tomber l'utilisateur C a 1 note,
    # dont le retrait fait tomber le film 2 a 2 notes. Un seul passage ne suffirait pas.
    rows = [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", 2), ("C", 3), ("D", 1), ("D", 4)]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId"]).assign(rating=4.0)

    filtered = cleaning.filter_min_activity(ratings, min_user_ratings=2, min_movie_ratings=2)

    assert set(filtered["userId"]) == {"A", "B"}
    assert filtered["userId"].value_counts().min() >= 2
    assert filtered["movieId"].value_counts().min() >= 2


def test_filter_min_activity_with_threshold_one_keeps_everything(raw_tables):
    ratings = cleaning.clean_ratings(raw_tables["ratings"], 0.5, 5.0)
    assert len(cleaning.filter_min_activity(ratings, 1, 1)) == len(ratings)
