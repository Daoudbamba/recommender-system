"""Theme sombre et composants visuels du dashboard.

Aucune affiche de film n'est disponible dans MovieLens : les films sont
presentes par des cartes typographiques, colorees selon leur genre principal.
"""

from __future__ import annotations

import html
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.api_client import ApiClient, ApiError

SURFACE = "#141414"
PANEL = "#1f1f1f"
TEXT = "#f0efec"
MUTED = "#a3a29b"
ACCENT = "#e50914"
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
GENRE_HUES = {
    "Action": 8, "Adventure": 28, "Animation": 48, "Children": 58, "Comedy": 42, "Crime": 0,
    "Documentary": 160, "Drama": 215, "Fantasy": 275, "Film-Noir": 230, "Horror": 350,
    "IMAX": 190, "Musical": 320, "Mystery": 255, "Romance": 335, "Sci-Fi": 200,
    "Thriller": 12, "War": 95, "Western": 30,
}  # fmt: skip
MODEL_LABELS = {
    "popularity": "Popularité",
    "bias_baseline": "Biais (moyennes)",
    "item_knn": "Item-kNN",
    "user_knn": "User-kNN",
    "als_explicit": "ALS explicite",
    "als_implicit": "ALS implicite",
    "tfidf": "Contenu TF-IDF",
    "embeddings": "Contenu embeddings",
    "hybrid_weighted": "Hybride pondéré",
    "hybrid_zscore": "Hybride centré-réduit",
    "hybrid_rrf": "Hybride (RRF)",
    "xgb_ranker": "Ranking XGBoost",
    "xgb_ranker_no_recency": "Ranking sans récence",
    "xgb_ranker_scores_only": "Ranking scores seuls",
}

# Plotly Express fixe les couleurs trace par trace : on lui donne la palette du projet.
px.defaults.color_discrete_sequence = PALETTE

CSS = f"""
<style>
.stApp {{ background: {SURFACE}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background: #0b0b0b; border-right: 1px solid #262626; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
.hero {{ padding: 28px 32px; border-radius: 14px; margin-bottom: 18px;
        background: linear-gradient(120deg, #2a0a0d 0%, #141414 55%, #10202e 100%);
        border: 1px solid #2a2a2a; }}
.hero h1 {{ margin: 0 0 6px 0; font-size: 2.1rem; }}
.hero p {{ margin: 0; color: {MUTED}; font-size: 1.02rem; max-width: 780px; }}
.brand {{ color: {ACCENT}; font-weight: 800; letter-spacing: 0.04em; }}
.kpi {{ background: {PANEL}; border: 1px solid #2a2a2a; border-radius: 12px; padding: 14px 16px; }}
.kpi .value {{ font-size: 1.55rem; font-weight: 700; color: {TEXT}; }}
.kpi .label {{ color: {MUTED}; font-size: 0.82rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; }}
.card {{ border-radius: 12px; padding: 14px; min-height: 172px; position: relative;
        border: 1px solid rgba(255,255,255,0.08); display: flex; flex-direction: column;
        justify-content: flex-end; transition: transform .15s ease, border-color .15s ease; }}
.card:hover {{ transform: translateY(-3px); border-color: rgba(255,255,255,0.35); }}
.card .rank {{ position: absolute; top: 10px; left: 12px; font-size: 1.5rem; font-weight: 800;
              color: rgba(255,255,255,0.9); }}
.card .badge {{ position: absolute; top: 12px; right: 12px; background: rgba(0,0,0,0.55);
               padding: 2px 8px; border-radius: 999px; font-size: 0.74rem; color: {TEXT}; }}
.card .title {{ font-weight: 700; font-size: 0.98rem; line-height: 1.2; color: #fff; }}
.card .meta {{ color: rgba(255,255,255,0.78); font-size: 0.76rem; margin-top: 4px; }}
.card .why {{ color: rgba(255,255,255,0.9); font-size: 0.72rem; margin-top: 6px;
             border-top: 1px solid rgba(255,255,255,0.18); padding-top: 6px; }}
.pill {{ display: inline-block; background: #2a2a2a; color: {TEXT}; border-radius: 999px;
        padding: 2px 10px; margin: 2px 4px 2px 0; font-size: 0.78rem; }}
.note {{ color: {MUTED}; font-size: 0.85rem; }}
</style>
"""


def setup_page(title: str) -> ApiClient:
    """Configuration commune a toutes les pages ; retourne le client de l'API."""
    st.set_page_config(page_title=f"{title} · Recommender", page_icon="🎬", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### <span class='brand'>RECOMMENDER</span>", unsafe_allow_html=True)
        st.caption("Recommandation de films · MovieLens 25M")
    return get_client()


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


def guard(call, *args: Any, **kwargs: Any) -> Any:
    """Appelle l'API ; en cas d'echec, affiche l'erreur et arrete proprement la page."""
    try:
        return call(*args, **kwargs)
    except ApiError as error:
        st.error(str(error))
        st.stop()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"<div class='hero'><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>",
        unsafe_allow_html=True,
    )


def kpis(items: list[tuple[str, str]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.markdown(
            f"<div class='kpi'><div class='value'>{html.escape(value)}</div>"
            f"<div class='label'>{html.escape(label)}</div></div>",
            unsafe_allow_html=True,
        )


def fmt_int(value: float) -> str:
    return f"{int(value):,}".replace(",", " ")


def _gradient(genres: list[str]) -> str:
    hue = GENRE_HUES.get(genres[0], 210) if genres else 210
    return f"linear-gradient(160deg, hsl({hue} 55% 30%) 0%, hsl({(hue + 35) % 360} 45% 13%) 100%)"


def movie_card(
    movie: dict[str, Any], rank: int | None = None, badge: str | None = None, why: str | None = None
) -> str:
    """HTML d'une carte de film (titre, annee, genres, note moyenne)."""
    genres = movie.get("genres", [])
    rating = movie.get("mean_rating")
    meta = " · ".join(
        part
        for part in (
            str(movie["year"]) if movie.get("year") else "",
            ", ".join(genres[:3]),
            f"★ {rating:.2f} ({fmt_int(movie['n_ratings'])} notes)" if rating else "",
        )
        if part
    )
    return (
        f"<div class='card' style='background:{_gradient(genres)}'>"
        + (f"<div class='rank'>{rank}</div>" if rank else "")
        + (f"<div class='badge'>{html.escape(badge)}</div>" if badge else "")
        + f"<div class='title'>{html.escape(movie['title'])}</div>"
        + f"<div class='meta'>{html.escape(meta)}</div>"
        + (f"<div class='why'>{html.escape(why)}</div>" if why else "")
        + "</div>"
    )


def movie_grid(cards: list[str]) -> None:
    st.markdown(f"<div class='grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Applique le theme sombre commun a toutes les figures Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT, "size": 13},
        height=height,
        margin={"l": 10, "r": 10, "t": 46, "b": 10},
        colorway=PALETTE,
        legend={"orientation": "h", "y": -0.18},
    )
    fig.update_xaxes(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a")
    fig.update_yaxes(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a")
    return fig


def show(fig: go.Figure, height: int = 380) -> None:
    st.plotly_chart(style_figure(fig, height), use_container_width=True)
