# Démonstration pas à pas (environ 4 minutes)

Script pour présenter le projet en direct ou enregistrer la vidéo de démonstration.
Chaque étape indique ce qu'on montre et ce qu'on dit ; les chiffres cités sont ceux des
rapports générés (`reports/*_report.md`).

## Avant de commencer

1. Le bundle de production existe (`artifacts/production/`, sinon `python scripts/train_production.py`).
2. Lancer la stack et attendre que l'API soit prête (environ une minute, le temps de charger les modèles) :

```bash
docker compose up -d
```

```bash
docker compose ps
```

L'API doit être `healthy`. Ouvrir trois onglets : <http://localhost:8501> (dashboard),
<http://localhost:8000/docs> (API), <http://localhost:5000> (MLflow).

Pour MLflow, importer une fois l'historique des expériences dans le serveur Docker :

```bash
MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/import_results_to_mlflow.py
```

## 1. Le problème (20 s) — page Home

- Montrer les chiffres : 162 541 utilisateurs, 62 423 films, 25 millions de notes.
- À dire : « Un utilisateur ne verra qu'une infime partie du catalogue. Le système doit choisir
  pour lui les dix films à mettre en avant, sans recommander les mêmes succès à tout le monde. »
- Taper « Inception » dans la recherche pour montrer que le catalogue est bien réel.

## 2. Une recommandation expliquée (60 s) — page Personalized Recommendations

- Choisir l'utilisateur proposé par défaut. Montrer son historique et ses genres préférés.
- Montrer le top 10 : chaque carte dit **pourquoi** le film est proposé (« Parce que vous avez
  aimé… »).
- Descendre jusqu'au graphique « Part de chaque modèle » : le score combine filtrage
  collaboratif (goûts d'utilisateurs proches), item-kNN et contenu (tags).
- Passer le modèle sur « Popularité » puis revenir au ranking : la liste devient personnelle.
- Baisser le curseur de diversité à 0,5 : les films trop semblables entre eux sont écartés.

## 3. Le cold start (40 s) — onglet Nouveau visiteur

- Sans aucune note : la liste est celle des films les plus appréciés.
- Ajouter *Interstellar*, *Inception*, *The Matrix*, *Blade Runner 2049* notés 5.
- La liste devient science-fiction et thrillers (*The Dark Knight*, *Shutter Island*,
  *Memento*…), et le poids de la popularité affiché baisse : « plus on en sait sur
  l'utilisateur, moins on s'appuie sur la popularité. »

## 4. Films similaires (30 s) — page Similar Movies

- *Matrix* avec le moteur « Embeddings » : les suites, puis *Animatrix*.
- Changer pour « TF-IDF » puis « Collaboratif » : trois notions différentes de ressemblance
  (le texte, les tags communs, les mêmes spectateurs).

## 5. La rigueur de l'évaluation (50 s) — page Model Performance

- Montrer la coupure temporelle : on apprend sur le passé, on évalue sur le futur.
- Classer par NDCG@10 : le ranking XGBoost (0,208) devant l'hybride (0,153), l'ALS implicite
  (0,138) et la popularité (0,079).
- Choisir « Couverture du catalogue » : le classement change, l'item-kNN recommande une plus
  grande part du catalogue. « Il n'y a pas de meilleur modèle dans l'absolu, il y a des
  compromis mesurés. »
- Afficher RMSE : l'ALS explicite prédit le mieux les notes mais classe le moins bien.
  « Bien prédire une note n'est pas bien recommander. »

## 6. L'industrialisation (40 s)

- **API** (<http://localhost:8000/docs>) : ouvrir `GET /recommendations/{user_id}`, l'exécuter
  avec `294`, montrer la réponse JSON et ses explications. Exécuter avec un identifiant
  inconnu : réponse populaire, pas d'erreur.
- **MLflow** (<http://localhost:5000>) : ouvrir l'expérience `recsys-ranking`, comparer les
  configurations, montrer les paramètres et le fichier de résultats joint.
- **GitHub** : l'onglet Actions avec la CI au vert (lint, 212 tests, construction de l'image).

## Pour conclure (10 s)

« Données vérifiées, six familles de modèles comparées sans fuite, un modèle de ranking qui
améliore le NDCG@10 de 36 % sur l'hybride, servi par une API et un dashboard, suivi dans
MLflow, testé et conteneurisé. »

## Arrêter

```bash
docker compose down
```
