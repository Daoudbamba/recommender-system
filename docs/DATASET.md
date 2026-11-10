# Jeu de données : MovieLens 25M

## Provenance

| | |
|---|---|
| Éditeur | GroupLens Research, Université du Minnesota |
| Page officielle | <https://grouplens.org/datasets/movielens/25m/> |
| Archive | <https://files.grouplens.org/datasets/movielens/ml-25m.zip> (262 Mo) |
| Somme MD5 | `6b51fb2759a8657d3bfcbfc42b592ada` (publiée par GroupLens, vérifiée par `scripts/download_data.py`) |
| Date de génération | 21 novembre 2019 |
| Période couverte | 9 janvier 1995 – 21 novembre 2019 |

D'après le `README.txt` fourni dans l'archive : les utilisateurs ont été tirés au hasard,
chacun a noté au moins 20 films, et aucune information démographique n'est incluse.

## Pourquoi la version 25M

- La version 1M ne contient ni `tags.csv` ni `links.csv`, nécessaires au filtrage par contenu
  et aux embeddings.
- `ml-latest-small` évolue dans le temps et GroupLens la déconseille pour publier des résultats ;
  la 25M est un benchmark figé, donc reproductible.
- Elle reste exploitable sur un portable : environ 500 Mo en mémoire avec des types compacts.

## Fichiers

Colonnes vérifiées sur les fichiers téléchargés (voir `notebooks/01_data_exploration.ipynb`).

| Fichier | Lignes | Colonnes | Usage dans le projet |
|---|---|---|---|
| `ratings.csv` | 25 000 095 | `userId`, `movieId`, `rating`, `timestamp` | Interactions (tous les modèles) |
| `movies.csv` | 62 423 | `movieId`, `title`, `genres` | Catalogue, genres, année extraite du titre |
| `tags.csv` | 1 093 360 | `userId`, `movieId`, `tag`, `timestamp` | Texte libre pour TF-IDF et embeddings |
| `links.csv` | 62 423 | `movieId`, `imdbId`, `tmdbId` | Identifiants externes |
| `genome-scores.csv` | non chargé | — | Non utilisé : scores non datés, incompatibles avec une évaluation temporelle sans fuite |
| `genome-tags.csv` | non chargé | — | Idem |

Le dataset ne contient **aucune description de film**. Le projet n'en invente pas : le contenu
textuel se limite au titre, aux genres et aux tags.

## Conditions d'utilisation

Résumé de la section *Usage License* du `README.txt` de l'archive, qui fait foi :

- usage autorisé à des fins de recherche ;
- ne pas laisser entendre que l'Université du Minnesota ou GroupLens cautionne le projet ;
- citer le jeu de données dans toute publication qui l'utilise ;
- **ne pas redistribuer les données** sans autorisation ;
- pas d'usage commercial sans accord préalable de GroupLens.

Conséquences dans ce dépôt :

- `data/` est exclu de Git ; chacun télécharge l'archive depuis la source officielle ;
- le projet est une démonstration non commerciale ;
- les images Docker des sprints suivants ne devront pas embarquer les données.

## Citation

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
> ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
> <https://doi.org/10.1145/2827872>
