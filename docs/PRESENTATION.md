# Proposition de presentation soutenance

## Slide 1 - Titre

- Systeme RAG OpenAgenda pour la recommandation d'evenements
- POC local avec API FastAPI, FAISS et Mistral

## Slide 2 - Contexte et probleme

- Les evenements OpenAgenda sont nombreux et heterogenes
- Une recherche par mots-cles est limitee
- Besoin d'une recherche semantique et d'une reponse naturelle

## Slide 3 - Objectif du projet

- Collecter les evenements utiles
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

## Slide 8 - API et conteneurisation

- FastAPI avec `/ask`, `/rebuild`, `/health`
- Swagger via `/docs` avec scenarios de test preconfigures
- Dockerfile base sur `uv` + `pyproject.toml`
- Docker Compose pour la demo rapide

## Slide 9 - Choix techniques

- FAISS pour la vitesse et la simplicite
- Mistral `mistral-embed` pour les embeddings
- FastAPI pour la documentation automatique
- Separation nette entre logique metier et interface API

## Slide 10 - Resultats observes

- Recherche semantique fonctionnelle
- Reponses naturelles produites a partir du contexte
- Filtrage des evenements deja passes
- Rebuild d'index a la demande

## Slide 11 - Evaluation

- Tests unitaires et API
- Verification fonctionnelle locale
- Pistes Ragas pour evaluation automatique
- Observation qualitative sur les scenarios de demo exposes aussi dans Swagger

## Slide 12 - Scenarios de demo

- "Je cherche un atelier a Paris ce week-end"
- "Quels evenements famille sont proposes prochainement ?"
- "Je veux une exposition gratuite en Ile-de-France"

## Slide 13 - Limites

- Dependance a l'API Mistral pour embeddings et generation
- Temps de rebuild de l'index
- Qualite variable selon les donnees source et le retrieval
- Pas d'interface frontend metier dediee dans cette version finale

## Slide 14 - Perspectives

- Reranking des resultats
- Evaluation automatique plus poussee avec Ragas
- Filtres metier explicites par date / lieu / categorie
- Deploiement cloud et protection de l'endpoint `/rebuild`

## Slide 15 - Conclusion

- POC RAG local operationnel
- Recherche semantique + API demonstrable
- Base propre pour extension produit ou industrialisation
