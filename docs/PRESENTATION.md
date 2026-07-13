# Présentation de soutenance — POC Puls-Events

## Slide 1 - Titre

- Assistant intelligent de recommandation d'événements pour Puls-Events
- POC local avec API FastAPI, FAISS, Mistral et LangChain

## Slide 2 - Contexte et probleme

- Puls-Events souhaite valider la faisabilité d'un chatbot de recommandation culturelle
- Les événements OpenAgenda sont nombreux et hétérogènes
- Une recherche par mots-cles est limitee
- Besoin d'une recherche semantique et d'une reponse naturelle

## Slide 3 - Objectif du projet

- Démontrer un parcours complet, de la collecte à la réponse API
- Collecter les événements utiles
- Construire un index vectoriel
- Repondre a une question utilisateur via un endpoint API
- Presenter une demo locale reproductible

## Slide 4 - Donnees utilisees

- Source : OpenAgenda
- Zone cible : Ile-de-France
- Normalisation en JSONL
- Metadonnees conservees : titre, dates, lieu, description, agendas sources

## Slide 5 - Pipeline de preparation

- Collecte OpenAgenda
- Nettoyage / normalisation
- Construction du champ `document`
- Decoupage en chunks
- Vectorisation Mistral

## Slide 6 - Architecture technique

- `fetch_openagenda_events.py`
- `build_faiss_index.py`
- `src/rag_oc/rag_service.py`
- `src/rag_oc/api.py`
- FAISS pour le retrieval
- Mistral pour embeddings + generation

## Slide 7 - Fonctionnement du moteur RAG

- Question utilisateur
- Embedding de la question
- Recherche des chunks proches dans FAISS
- Reconstruction du contexte
- Generation de la reponse finale

## Slide 8 - Choix techniques

- `IndexFlatIP` retenu : recherche exacte et sans paramétrage, adaptée à ~65 000 chunks
- `IndexIVFFlat` : plus adapté à une volumétrie élevée, mais nécessite entraînement et réglage (`nlist`, `nprobe`), avec un recall non garanti
- HNSW : très faible latence à grande échelle, mais plus de mémoire et davantage de paramètres à régler
- Mistral `mistral-embed` pour les embeddings et `mistral-small-latest` pour la génération
- LangChain pour le templating de prompt et FastAPI pour la documentation automatique

## Slide 9 - Fraîcheur des données

- Le corpus contient environ 89 % d'événements terminés.
- Le moteur cherche dans un pool élargi (×20, minimum 100) avant de retirer les événements passés.
- La reconstruction régulière de l'index reste nécessaire pour améliorer la pertinence.

## Slide 10 - API et démo locale

- FastAPI : `/health`, `/ask`, `/rebuild` et Swagger `/docs`.
- Dockerfile et Docker Compose permettent de démarrer l'API localement.
- Trois scénarios réalistes sont déjà proposés dans Swagger.

## Slide 11 - Résultats et évaluation

- 76 tests passent ; couverture de code : 71 %.
- Évaluation RAGAS static sur 10 cas annotés : context precision 1,00 ; answer relevancy 0,85 ; faithfulness 0,70.
- La fidélité de la réponse est l'axe d'amélioration prioritaire.

## Slide 12 - Conclusion et perspectives

- POC fonctionnel : collecte, indexation, retrieval, génération, API et conteneurisation.
- Priorités : renforcer l'ancrage des réponses, ajouter des filtres métier et un reranking.
- Le POC constitue une base pour une expérimentation produit plus large.
