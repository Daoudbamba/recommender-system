"""Recommandations personnalisees : utilisateur connu, ou nouveau visiteur (cold start)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import ui  # noqa: E402

client = ui.setup_page("Personalized Recommendations")
ui.hero(
    "Personalized Recommendations",
    "Choisissez un utilisateur pour voir son historique, ses genres préférés et ce que le système "
    "lui propose, avec la raison de chaque recommandation.",
)
STRATEGIES = {
    "ranker": "Ranking XGBoost (meilleur modèle)",
    "hybrid": "Hybride (RRF)",
    "popularity": "Popularité (non personnalisé)",
}
known_tab, visitor_tab = st.tabs(["Utilisateur existant", "Nouveau visiteur"])

with known_tab:
    users = ui.guard(client.sample_users, 40)
    left, middle, right = st.columns([2, 1, 1])
    choice = left.selectbox(
        "Utilisateur d'exemple",
        users,
        index=len(users) // 2,
        format_func=lambda u: f"Utilisateur {u['user_id']} — {u['n_ratings']} notes",
    )
    typed = middle.number_input("…ou un identifiant précis", min_value=0, value=0, step=1)
    user_id = int(typed) if typed else choice["user_id"]
    top_n = right.radio("Taille de la liste", [10, 20], horizontal=True)
    left, right = st.columns([2, 2])
    strategy = left.selectbox("Modèle", list(STRATEGIES), format_func=STRATEGIES.get)
    diversity = right.slider(
        "Diversité de la liste", 0.3, 1.0, 1.0, 0.1,
        help="1 = pertinence seule. Plus bas, les films trop semblables entre eux sont écartés.",
    )  # fmt: skip

    recommended = ui.guard(client.recommendations, user_id, top_n, strategy, diversity)
    if not recommended["known_user"]:
        st.warning(
            f"L'utilisateur {user_id} est inconnu : aucune personnalisation possible, "
            "l'API renvoie les films les plus appréciés."
        )
    else:
        history = ui.guard(client.history, user_id, 15)
        ui.kpis(
            [
                ("Films notés", ui.fmt_int(history["n_ratings"])),
                ("Note moyenne", f"{history['mean_rating']:.2f}"),
                ("Films aimés (note ≥ 4)", ui.fmt_int(history["n_liked"])),
                ("Temps de calcul", f"{recommended['elapsed_ms']:.0f} ms"),
            ]
        )
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Films les mieux notés par cet utilisateur")
            table = pd.DataFrame(
                [
                    {
                        "Film": entry["movie"]["title"],
                        "Genres": ", ".join(entry["movie"]["genres"]),
                        "Note": entry["rating"],
                    }
                    for entry in history["items"]
                ]
            )
            st.dataframe(table, hide_index=True, use_container_width=True, height=330)
        with right:
            st.subheader("Genres préférés")
            genres = pd.DataFrame(history["favourite_genres"])
            if len(genres):
                fig = px.bar(genres.iloc[::-1], x="share", y="genre", orientation="h")
                fig.update_traces(
                    marker_color=ui.PALETTE[0], hovertemplate="%{y} : %{x:.0%}<extra></extra>"
                )
                fig.update_layout(
                    xaxis_tickformat=".0%", xaxis_title="Part des films aimés", yaxis_title=""
                )
                ui.show(fig, 330)
            else:
                st.info("Cet utilisateur n'a encore aimé aucun film.")

    st.subheader(f"Top {top_n} recommandé")
    ui.movie_grid(
        [
            ui.movie_card(
                item["movie"],
                rank=item["rank"],
                badge=f"score {item['score']:.2f}",
                why=item["reasons"][-1] if item["reasons"] else None,
            )
            for item in recommended["items"]
        ]
    )
    if (
        recommended["known_user"]
        and recommended["items"]
        and recommended["items"][0]["contributions"]
    ):
        st.subheader("Pourquoi ces films ? Part de chaque modèle dans le score")
        rows = [
            {
                "Film": f"{item['rank']}. {item['movie']['title']}",
                "Modèle": ui.MODEL_LABELS.get(name, name),
                "Part": share,
            }
            for item in recommended["items"][:10]
            for name, share in item["contributions"].items()
        ]
        fig = px.bar(pd.DataFrame(rows), x="Part", y="Film", color="Modèle", orientation="h")
        fig.update_layout(
            xaxis_tickformat=".0%",
            yaxis={"autorange": "reversed"},
            barmode="stack",
            xaxis_title="Contribution au score de l'hybride",
            yaxis_title="",
        )
        ui.show(fig, 420)
        st.markdown(
            "<p class='note'>Les contributions décrivent le score de l'hybride, qui génère les "
            "candidats ; le modèle de ranking les réordonne ensuite.</p>",
            unsafe_allow_html=True,
        )

with visitor_tab:
    st.markdown(
        "Aucun compte, aucun historique : notez quelques films et l'API calcule des "
        "recommandations à la volée, sans réentraîner de modèle."
    )
    if "visitor" not in st.session_state:
        st.session_state.visitor = {}
    search = st.text_input("Chercher un film à noter", placeholder="Inception, Amélie, Alien…")
    if search:
        found = ui.guard(client.movies, search, None, 8)["items"]
        for movie in found:
            cols = st.columns([4, 2, 1])
            cols[0].markdown(f"**{movie['title']}** — {', '.join(movie['genres'])}")
            rating = cols[1].select_slider(
                "Note", [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=4.5,
                key=f"rate_{movie['movie_id']}", label_visibility="collapsed",
            )  # fmt: skip
            if cols[2].button("Ajouter", key=f"add_{movie['movie_id']}"):
                st.session_state.visitor[movie["movie_id"]] = (movie["title"], float(rating))
    if st.session_state.visitor:
        st.markdown(
            "".join(
                f"<span class='pill'>{title} · {value:g}</span>"
                for title, value in st.session_state.visitor.values()
            ),
            unsafe_allow_html=True,
        )
        if st.button("Tout effacer"):
            st.session_state.visitor = {}
            st.rerun()
    ratings = {movie: value for movie, (_, value) in st.session_state.visitor.items()}
    result = ui.guard(client.cold_start, ratings, 12)
    st.caption(
        f"{result['n_ratings_used']} note(s) prise(s) en compte · poids de la popularité : "
        f"{result['popularity_weight']:.0%} · {result['elapsed_ms']:.0f} ms"
    )
    ui.movie_grid([ui.movie_card(i["movie"], rank=i["rank"]) for i in result["items"]])
