"""Le jeu de donnees en chiffres : volumes, notes, genres, activite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import ui  # noqa: E402

client = ui.setup_page("Data Insights")
ui.hero(
    "Data Insights",
    "MovieLens 25M (GroupLens) : des notes déposées par de vrais utilisateurs entre 1995 et 2019.",
)
stats = ui.guard(client.stats)
overview = stats["overview"]
ui.kpis(
    [
        ("Utilisateurs", ui.fmt_int(overview["n_users"])),
        ("Films au catalogue", ui.fmt_int(overview["n_movies_catalog"])),
        ("Interactions (notes)", ui.fmt_int(overview["n_ratings"])),
        ("Note moyenne", f"{overview['mean_rating']:.2f}"),
        ("Matrice remplie à", f"{overview['density']:.2%}"),
    ]
)
st.caption(f"Période couverte : {overview['first_rating'][:10]} → {overview['last_rating'][:10]}")

left, right = st.columns(2)
with left:
    st.subheader("Répartition des notes")
    ratings = pd.DataFrame(stats["rating_distribution"])
    ratings["Note"] = ratings["rating"].map(lambda value: f"{value:g}")
    fig = px.bar(ratings, x="Note", y="count", text=ratings["share"].map(lambda v: f"{v:.1%}"))
    fig.update_traces(marker_color=ui.PALETTE[0], textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title="Nombre de notes", xaxis_type="category")
    ui.show(fig, 400)
with right:
    st.subheader("Activité des utilisateurs")
    activity = pd.DataFrame(stats["user_activity"])
    activity["Notes déposées"] = activity.apply(lambda r: f"{r['from']}–{r['to']}", axis=1)
    fig = px.bar(activity, x="Notes déposées", y="users")
    fig.update_traces(marker_color=ui.PALETTE[0])
    fig.update_layout(yaxis_title="Utilisateurs", xaxis_type="category")
    ui.show(fig, 400)
    st.caption(f"Un utilisateur médian a déposé {stats['user_activity_median']:.0f} notes.")

genres = pd.DataFrame(stats["genres"])
left, right = st.columns(2)
with left:
    st.subheader("Genres les plus populaires")
    fig = px.bar(genres.sort_values("n_ratings"), x="n_ratings", y="genre", orientation="h")
    fig.update_traces(marker_color=ui.PALETTE[0])
    fig.update_layout(xaxis_title="Notes reçues", yaxis_title="")
    ui.show(fig, 520)
with right:
    st.subheader("Note moyenne par genre")
    fig = px.bar(genres.sort_values("mean_rating"), x="mean_rating", y="genre", orientation="h")
    fig.update_traces(marker_color=ui.PALETTE[2])
    low = max(0.0, float(genres["mean_rating"].min()) - 0.3)
    fig.update_layout(xaxis_title="Note moyenne", yaxis_title="", xaxis_range=[low, 4.1])
    ui.show(fig, 520)
    st.caption("L'axe ne part pas de zéro : les écarts entre genres sont faibles.")

st.subheader("Notes déposées par année")
yearly = pd.DataFrame(stats["ratings_per_year"])
fig = px.bar(yearly, x="year", y="n_ratings")
fig.update_traces(marker_color=ui.PALETTE[0])
fig.update_layout(xaxis_title="", yaxis_title="Nombre de notes")
ui.show(fig, 360)

st.subheader("Les films les plus notés")
top = pd.DataFrame(stats["top_movies"]).rename(
    columns={"title": "Film", "n_ratings": "Notes", "mean_rating": "Note moyenne"}
)
st.dataframe(
    top, hide_index=True, use_container_width=True,
    column_config={"Note moyenne": st.column_config.NumberColumn(format="%.2f")},
)  # fmt: skip
