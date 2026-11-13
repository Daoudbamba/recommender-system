"""Accueil : presentation, recherche, films populaires, recommandations d'un utilisateur."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from dashboard.components import ui  # noqa: E402

client = ui.setup_page("Home")
health = ui.guard(client.health)
if not health["models_loaded"]:
    st.error(f"L'API répond mais ses modèles ne sont pas chargés : {health.get('detail')}")
    st.stop()
stats = ui.guard(client.stats)
overview = stats["overview"]

ui.hero(
    "Intelligent Movie Recommendation Platform",
    "Un système de recommandation complet sur MovieLens 25M : filtrage collaboratif, contenu, "
    "embeddings, modèle hybride et learning to rank, servis par une API.",
)
ui.kpis(
    [
        ("Utilisateurs", ui.fmt_int(overview["n_users"])),
        ("Films au catalogue", ui.fmt_int(overview["n_movies_catalog"])),
        ("Notes", ui.fmt_int(overview["n_ratings"])),
        ("Films recommandables", ui.fmt_int(overview["n_movies_in_model"])),
        ("Modèles entraînés le", health["trained_at"][:10]),
    ]
)

st.subheader("Rechercher un film")
left, right = st.columns([3, 1])
query = left.text_input(
    "Titre", placeholder="Matrix, Toy Story, Amélie…", label_visibility="collapsed"
)
genre = right.selectbox(
    "Genre", ["Tous les genres", *ui.guard(client.genres)], label_visibility="collapsed"
)
if query or genre != "Tous les genres":
    found = ui.guard(
        client.movies, query or None, None if genre == "Tous les genres" else genre, 12
    )
    st.caption(f"{ui.fmt_int(found['total'])} film(s) trouvé(s)")
    if found["items"]:
        ui.movie_grid([ui.movie_card(movie) for movie in found["items"]])
    else:
        st.info("Aucun film ne correspond à cette recherche.")

st.subheader("Les plus populaires")
popular = ui.guard(client.movies, None, None, 12)
ui.movie_grid([ui.movie_card(m, rank=i) for i, m in enumerate(popular["items"], start=1)])

st.subheader("Recommandés pour un utilisateur")
users = ui.guard(client.sample_users, 30)
choice = st.selectbox(
    "Utilisateur d'exemple",
    users,
    index=len(users) // 2,
    format_func=lambda u: f"Utilisateur {u['user_id']} — {u['n_ratings']} notes",
)
recommended = ui.guard(client.recommendations, choice["user_id"], 12)
ui.movie_grid(
    [
        ui.movie_card(
            item["movie"],
            rank=item["rank"],
            why=item["because_you_liked"][0]
            and f"Parce que vous avez aimé {item['because_you_liked'][0]}"
            if item["because_you_liked"]
            else None,
        )
        for item in recommended["items"]
    ]
)
st.markdown(
    f"<p class='note'>Stratégie : {ui.MODEL_LABELS['xgb_ranker']} · calculé en "
    f"{recommended['elapsed_ms']:.0f} ms par l'API ({client.base_url}).</p>",
    unsafe_allow_html=True,
)
