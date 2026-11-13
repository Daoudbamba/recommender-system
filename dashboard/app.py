"""Point d'entree du dashboard : declare la navigation entre les pages.

Lancement (depuis la racine du depot, API demarree) :
    streamlit run dashboard/app.py

L'adresse de l'API se regle avec la variable d'environnement RECSYS_API_URL
(par defaut http://localhost:8000).
"""

from __future__ import annotations

import streamlit as st

PAGES = [
    st.Page("pages/0_Home.py", title="Home", icon=":material/home:", default=True),
    st.Page(
        "pages/1_Personalized_Recommendations.py",
        title="Personalized Recommendations",
        icon=":material/person:",
    ),
    st.Page("pages/2_Similar_Movies.py", title="Similar Movies", icon=":material/movie:"),
    st.Page(
        "pages/3_Model_Performance.py", title="Model Performance", icon=":material/leaderboard:"
    ),
    st.Page("pages/4_Data_Insights.py", title="Data Insights", icon=":material/insights:"),
]

st.navigation(PAGES).run()
