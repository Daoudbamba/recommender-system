"""Films similaires : trois moteurs comparables, scores et genres communs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import ui  # noqa: E402

client = ui.setup_page("Similar Movies")
ui.hero(
    "Similar Movies",
    "Choisissez un film : le système retrouve ceux qui lui ressemblent, par leur contenu ou par "
    "le comportement des utilisateurs.",
)
ENGINES = {
    "embeddings": "Embeddings (sens du texte, tout le catalogue)",
    "tfidf": "TF-IDF (genres et tags communs)",
    "collaborative": "Collaboratif (aimés par les mêmes personnes)",
}

left, right = st.columns([3, 2])
query = left.text_input("Rechercher un film", value="Matrix")
engine = right.selectbox("Moteur de similarité", list(ENGINES), format_func=ENGINES.get)
found = ui.guard(client.movies, query or None, None, 20)["items"]
if not found:
    st.info("Aucun film ne correspond à cette recherche.")
    st.stop()
movie = st.selectbox(
    "Film de référence",
    found,
    format_func=lambda m: f"{m['title']} — {ui.fmt_int(m['n_ratings'])} notes",
)
if engine != "embeddings" and not movie["in_model"]:
    st.warning(
        "Ce film a trop peu de notes pour les moteurs TF-IDF et collaboratif. "
        "Le moteur par embeddings, lui, couvre tout le catalogue."
    )
    st.stop()

result = ui.guard(client.similar, movie["movie_id"], 12, engine)
ui.movie_grid([ui.movie_card(result["movie"], badge="référence")])
st.subheader("Films les plus proches")
ui.movie_grid(
    [
        ui.movie_card(
            item["movie"],
            rank=rank,
            badge=f"similarité {item['similarity']:.2f}",
            why="Genres communs : " + ", ".join(item["shared_genres"])
            if item["shared_genres"]
            else "Aucun genre commun",
        )
        for rank, item in enumerate(result["items"], start=1)
    ]
)

left, right = st.columns(2)
with left:
    st.subheader("Scores de similarité")
    frame = pd.DataFrame(
        [{"Film": i["movie"]["title"], "Similarité": i["similarity"]} for i in result["items"]]
    )
    fig = px.bar(frame.iloc[::-1], x="Similarité", y="Film", orientation="h")
    fig.update_traces(marker_color=ui.PALETTE[0])
    fig.update_layout(yaxis_title="")
    ui.show(fig, 420)
with right:
    st.subheader("Comparaison des genres")
    reference = set(result["movie"]["genres"])
    counts: dict[str, int] = {}
    for item in result["items"]:
        for genre in item["movie"]["genres"]:
            counts[genre] = counts.get(genre, 0) + 1
    rows = [
        {
            "Genre": genre,
            "Films voisins": count,
            "Présent dans le film de référence": "oui" if genre in reference else "non",
        }
        for genre, count in sorted(counts.items(), key=lambda kv: kv[1])
    ]
    fig = px.bar(
        pd.DataFrame(rows), x="Films voisins", y="Genre", orientation="h",
        color="Présent dans le film de référence",
        color_discrete_map={"oui": ui.PALETTE[0], "non": "#5a5a5a"},
    )  # fmt: skip
    fig.update_layout(yaxis_title="")
    ui.show(fig, 420)
