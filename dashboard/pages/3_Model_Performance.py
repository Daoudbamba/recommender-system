"""Comparaison des modeles : resultats de test reellement mesures, servis par l'API."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components import ui  # noqa: E402

client = ui.setup_page("Model Performance")
ui.hero(
    "Model Performance",
    "Tous les modèles sont mesurés sur les mêmes utilisateurs, avec une séparation temporelle : "
    "on apprend sur le passé, on évalue sur le futur.",
)
metrics = ui.guard(client.metrics)
frame = pd.DataFrame(metrics["models"])
frame["Modèle"] = frame["model"].map(ui.MODEL_LABELS).fillna(frame["model"])
fold, protocol = metrics["test_fold"], metrics["protocol"]
ui.kpis(
    [
        ("Utilisateurs évalués", ui.fmt_int(fold["eval_users"])),
        ("Notes d'entraînement", ui.fmt_int(fold["train_ratings"])),
        ("Films candidats", ui.fmt_int(fold["train_items"])),
        ("Film pertinent", f"note ≥ {protocol['relevance_threshold']:g}"),
        ("Coupure du test", fold["train_end"][:10]),
    ]
)

families = {
    "collaborative": "Filtrage collaboratif",
    "content": "Contenu",
    "hybrid": "Hybride",
    "ranking": "Learning to rank",
}
selected = st.multiselect(
    "Familles de modèles", list(families), default=list(families), format_func=families.get
)
frame = frame[frame["source"].isin(selected)]
if frame.empty:
    st.info("Sélectionnez au moins une famille de modèles.")
    st.stop()

left, right = st.columns([1, 1])
k = left.radio("K", protocol["ks"], index=1, horizontal=True)
names = {"ndcg": "NDCG", "precision": "Precision", "recall": "Recall", "map": "MAP", "f1": "F1",
         "coverage": "Couverture du catalogue"}  # fmt: skip
metric = right.selectbox("Métrique", list(names), format_func=names.get)
column = f"{metric}@{k}"
ordered = frame.sort_values(column)
fig = px.bar(ordered, x=column, y="Modèle", orientation="h", color="source",
             category_orders={"source": list(families)}, labels={"source": "Famille"})  # fmt: skip
fig.for_each_trace(lambda trace: trace.update(name=families.get(trace.name, trace.name)))
fig.update_layout(
    xaxis_title=f"{names[metric]}@{k} sur le test",
    yaxis_title="",
    yaxis={"categoryorder": "array", "categoryarray": ordered["Modèle"].tolist()},
)
ui.show(fig, 520)

left, right = st.columns(2)
with left:
    st.subheader("Recall selon K")
    long = frame.melt(
        id_vars="Modèle", value_vars=[f"recall@{value}" for value in protocol["ks"]],
        var_name="K", value_name="Recall",
    )  # fmt: skip
    long["K"] = long["K"].str.replace("recall@", "").astype(int)
    fig = px.line(long, x="K", y="Recall", color="Modèle", markers=True)
    fig.update_xaxes(tickvals=protocol["ks"])
    ui.show(fig, 430)
with right:
    st.subheader("Prédiction de note (plus bas = mieux)")
    rated = frame.dropna(subset=["rmse"]) if "rmse" in frame else frame.iloc[0:0]
    if rated.empty:
        st.info("Aucun des modèles sélectionnés ne prédit de note.")
    else:
        long = rated.melt(id_vars="Modèle", value_vars=["rmse", "mae"], var_name="Métrique",
                          value_name="Erreur")  # fmt: skip
        long["Métrique"] = long["Métrique"].str.upper()
        fig = px.bar(long, x="Modèle", y="Erreur", color="Métrique", barmode="group")
        fig.update_layout(xaxis_title="")
        ui.show(fig, 430)

st.subheader("Tableau complet")
columns = [
    "Modèle",
    *[f"{m}@{k}" for m in ("precision", "recall", "f1", "map", "ndcg", "coverage")],
]
extra = [c for c in ("rmse", "mae", "ms_per_user", "fit_seconds") if c in frame]
table = frame[columns + extra].sort_values(f"ndcg@{k}", ascending=False)
st.dataframe(
    table.rename(columns={"ms_per_user": "ms / utilisateur", "fit_seconds": "entraînement (s)"}),
    hide_index=True, use_container_width=True,
    column_config={c: st.column_config.NumberColumn(format="%.4f") for c in columns[1:]},
)  # fmt: skip
st.markdown(
    "<p class='note'>Un film recommandé mais jamais noté par l'utilisateur compte comme une "
    "erreur : les valeurs absolues sont des bornes basses, ce sont les écarts entre modèles "
    "qui s'interprètent. Les scores proviennent des fichiers <code>reports/*_results.json</code> "
    "écrits par les expériences.</p>",
    unsafe_allow_html=True,
)
