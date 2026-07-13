# Collecte OpenAgenda et API RAG

Ce depot contient un pipeline RAG simple autour d'OpenAgenda, FAISS et Mistral :

- `Dockerfile` et `docker-compose.yml` pour executer l'API localement dans un conteneur ;
- l'image Docker installe les dependances via `uv` a partir de `pyproject.toml` et `uv.lock` ;
- `scripts/fetch_openagenda_events.py` collecte et normalise les evenements OpenAgenda ;
- `scripts/build_faiss_index.py` est le point d'entree CLI pour construire l'index FAISS ;
- `src/rag_oc/build_faiss_index.py` porte la logique d'indexation reutilisee par l'API ;
- `src/rag_oc/rag_service.py` centralise la logique metier RAG ;
- `scripts/chat_openagenda.py` permet de tester le RAG en CLI ;
- `scripts/evaluate_rag.py` sert de base pour l'evaluation automatique ;
- `scripts/api_smoke_test.py` permet de verifier rapidement le contrat HTTP ;
- `src/rag_oc/api.py` expose le RAG en API REST FastAPI.

## Objectif

Le projet sert a :

- collecter des evenements OpenAgenda en `Île-de-France` ;
- produire un dataset `JSONL` adapte au retrieval ;
- transformer les textes en embeddings Mistral ;
- indexer ces embeddings dans FAISS ;
- repondre a une question utilisateur avec un flux RAG ;
- exposer ce systeme localement via HTTP.

## Architecture

```text
OpenAgenda API
    -> fetch_openagenda_events.py
    -> data/openagenda/*.jsonl(.gz)
    -> scripts/build_faiss_index.py
    -> data/faiss/openagenda.index
    -> data/faiss/openagenda_metadata.pkl
    -> src/rag_oc/build_faiss_index.py
    -> src/rag_oc/rag_service.py
    -> chat_openagenda.py / src/rag_oc/api.py
```

## Structure

```text
.
├── Dockerfile
├── docker-compose.yml
├── main.py
├── pyproject.toml
├── README.md
├── src/
│   └── rag_oc/
│       ├── __init__.py
│       ├── api.py
│       ├── build_faiss_index.py
│       └── rag_service.py
├── docs/
│   ├── MISTRAL.md
│   ├── PRESENTATION.md
│   └── project_presentation.html
├── scripts/
│   ├── api_smoke_test.py
│   ├── build_faiss_index.py
│   ├── chat_openagenda.py
│   ├── evaluate_rag.py
│   ├── fetch_openagenda_events.py
│   └── __init__.py
└── tests/
    ├── rag_eval_sample.jsonl
    ├── test_api.py
    ├── test_build_faiss_index.py
    ├── test_evaluate_rag.py
    ├── test_openagenda.py
    └── test_rag_service.py
```

## Pre-requis

- Python `>= 3.13`
- `uv`
- une cle API OpenAgenda
- une cle API Mistral

Les cles peuvent etre passees :

- via les options CLI ;
- via des variables d'environnement ;
- via un fichier `.env`.

## 1. Collecte OpenAgenda

Le script [scripts/fetch_openagenda_events.py](/home/zmxw1768/Documents/rag_oc/scripts/fetch_openagenda_events.py:1) :

- appelle l'API OpenAgenda ;
- filtre les evenements par region ;
- applique une fenetre de dates ;
- normalise les champs utiles pour le RAG ;
- ecrit un `jsonl` ou `jsonl.gz`.

Commande simple :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --api-key "VOTRE_CLE_OPENAGENDA"
```

Exemple avec options :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --api-key "VOTRE_CLE_OPENAGENDA" \
  --source-mode agendas \
  --workers 12 \
  --output data/openagenda/full.jsonl.gz
```

### Regle temporelle de collecte

Le script borne toujours `date_from` a `maintenant - 365 jours`.

Important :

- cela ne veut pas dire "evenements futurs uniquement" ;
- cela veut dire "evenements presents dans les 365 derniers jours" ;
- un evenement de juillet 2025 peut donc encore etre dans le dataset en juillet 2026 si le run a ete fait dans cette fenetre.

Le filtrage des evenements deja termines se fait ensuite dans le moteur RAG au moment de la recherche.

### Format de sortie

Chaque ligne contient notamment :

- `id`
- `event_uid`
- `title`
- `description`
- `long_description`
- `event_types`
- `date_summary`
- `timings`
- `first_timing`
- `last_timing`
- `location`
- `source_agendas`
- `document`

Le champ `document` est le texte principal utilise pour le chunking et la vectorisation.

Le script genere aussi un manifeste `*.manifest.json` avec :

- la date du run ;
- la fenetre de dates ;
- le mode de collecte ;
- le nombre d'agendas scannes ;
- le nombre d'evenements ecrits.

## 2. Construction de l'index FAISS

Le script [scripts/build_faiss_index.py](/home/zmxw1768/Documents/rag_oc/scripts/build_faiss_index.py:1) :

1. lit le `JSONL` OpenAgenda ;
2. reutilise les `chunks` existants s'ils sont presents ;
3. sinon decoupe `document` en chunks ;
4. calcule un embedding Mistral pour chaque chunk ;
5. construit un index FAISS ;
6. ecrit les metadonnees associees.

Commande standard :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/full.jsonl.gz \
  --index-output data/faiss/openagenda.index \
  --metadata-output data/faiss/openagenda_metadata.pkl
```

Exemple plus prudent si les embeddings sont longs ou limites :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/full.jsonl.gz \
  --batch-size 16 \
  --request-pause-seconds 1 \
  --max-retries 8 \
  --retry-base-seconds 3
```

### Fichiers produits

- `data/faiss/openagenda.index` : index vectoriel FAISS ;
- `data/faiss/openagenda_metadata.pkl` : metadonnees alignees sur chaque vecteur.

### Ce que contient la metadata d'un chunk

Chaque embedding est associe a des infos comme :

- `id`
- `event_uid`
- `title`
- `date_summary`
- `first_timing`
- `last_timing`
- `location`
- `source_agendas`
- `document`
- `chunk_id`
- `chunk_index`
- `chunk_text`

### Type d'index

Par defaut :

- les vecteurs sont normalises ;
- l'index utilise `IndexFlatIP`.

`IndexFlatIP` est retenu pour le volume actuel (~65 000 chunks) : la recherche est exacte, ne demande ni entrainement ni parametre de rappel, et reste rapide a cette echelle.

L'alternative `IndexIVFFlat` peut etre utilisee pour une volumetrie plus elevee, mais elle demande un entrainement et le reglage de `nlist` et `nprobe` ; la recherche devient approchée. HNSW est une autre option pour une latence faible a grande echelle, au prix d'une consommation memoire et de reglages supplementaires.

On peut demander `ivfflat` avec :

```bash
--index-type ivfflat
```

## 3. Moteur RAG

La logique metier est dans [rag_service.py](/home/zmxw1768/Documents/rag_oc/src/rag_oc/rag_service.py:1).

Le flux de `RagService.ask(...)` est :

1. la question utilisateur est transformee en embedding avec `mistral-embed` ;
2. ce vecteur est compare aux vecteurs stockes dans FAISS ;
3. un **pool elargi** de candidats est recupere (x20 le `top_k`, minimum 100) ;
4. les evenements deja termines sont filtres par defaut via `last_timing` ;
5. les `top_k` meilleurs parmi les evenements restants sont conserves ;
6. un contexte texte est reconstruit a partir des chunks retenus ;
7. ce contexte et la question sont envoyes au modele de chat Mistral ;
8. la reponse finale est generee.

### Pourquoi un pool de recherche elargi ?

Le dataset contient ~89 % d'evenements passes. Avec un pool trop petit
(l'ancien x4), la quasi-totalite des resultats FAISS etait filtree par date,
ne laissant que des correspondances tres faibles. Le pool elargi (x20, min 100)
garantit qu'assez d'evenements futurs subsistent apres le filtrage temporel
pour fournir des recommandations pertinentes.

Important :

- FAISS fait la recherche semantique ;
- Mistral embeddings produit les vecteurs ;
- Mistral chat redige la reponse finale.

Le prompt est template avec `langchain-core` quand la dependance est disponible.
Dans l'etat actuel du projet, LangChain n'orchestre pas tout le RAG ; il sert surtout au templating du prompt.

## 4. Test en CLI

Le script [scripts/chat_openagenda.py](/home/zmxw1768/Documents/rag_oc/scripts/chat_openagenda.py:1) est un wrapper terminal autour de `RagService`.

One-shot :

```bash
uv run python scripts/chat_openagenda.py \
  --index-path data/faiss/openagenda.index \
  --metadata-path data/faiss/openagenda_metadata.pkl \
  --question "Je cherche un atelier a Paris ce week-end"
```

Mode interactif :

```bash
uv run python scripts/chat_openagenda.py \
  --index-path data/faiss/openagenda.index \
  --metadata-path data/faiss/openagenda_metadata.pkl
```

Options utiles :

- `--top-k`
- `--max-context-items`
- `--chat-model`
- `--temperature`

Si on modifie la structure des metadonnees ou les regles temporelles, il faut reconstruire l'index et `openagenda_metadata.pkl`.

## 5. API REST FastAPI

L'API dans [api.py](/home/zmxw1768/Documents/rag_oc/src/rag_oc/api.py:1) reutilise le meme moteur RAG que la CLI.

Lancement :

```bash
uv run uvicorn rag_oc.api:app --reload
```

Documentation Swagger :

```text
http://127.0.0.1:8000/docs
```

L'interface Swagger propose maintenant un selecteur `Examples` avec plusieurs scenarios de test preconfigures sur `POST /ask` et `POST /rebuild`.

Endpoints exposes :

- `GET /health`
- `POST /ask`
- `POST /rebuild`

### Scenarios `POST /ask`

```json
{
  "question": "Je cherche une sortie culturelle a Paris ce week-end",
  "top_k": 5,
  "max_context_items": 4,
  "temperature": 0.2
}
```

Scenarios disponibles dans Swagger :

- `atelier_paris_weekend`
- `sortie_famille`
- `expo_gratuite`

La reponse contient :

- `answer`
- `sources`
- `context`

### Scenarios `POST /rebuild`

```json
{
  "input_path": "data/openagenda/full.jsonl.gz",
  "index_output": "data/faiss/openagenda.index",
  "metadata_output": "data/faiss/openagenda_metadata.pkl",
  "index_type": "flat"
}
```

Scenarios disponibles dans Swagger :

- `rebuild_standard`
- `rebuild_ivf`

Le endpoint relance la construction de l'index puis recharge le service en memoire si besoin.

## 6. Docker

Le depot peut etre execute localement dans un conteneur Docker.

Build :

```bash
docker build -t rag-oc-api .
```

L'image recupere les binaires `uv` et `uvx` via `COPY --from`, copie `pyproject.toml` et `uv.lock`, puis execute `uv sync --frozen --no-dev`.

Run :

```bash
docker run --rm -p 8000:8000 \
  -e MISTRAL_API_KEY="$MISTRAL_API_KEY" \
  rag-oc-api
```

Avec Docker Compose :

```bash
docker compose up --build
```

Le `docker-compose.yml` monte `./data` dans le conteneur pour reutiliser les fichiers d'index et de donnees deja produits.

Sans index present dans `data/faiss/`, l'API demarrera mais `/ask` ne pourra pas charger le moteur RAG.

Test HTTP rapide apres lancement :

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Je cherche un atelier a Paris ce week-end","top_k":5,"max_context_items":4,"temperature":0.2}'
```

## 7. Demo et soutenance

Pour une demo fluide :

1. verifier que `data/faiss/openagenda.index` et `data/faiss/openagenda_metadata.pkl` existent
2. lancer l'API via Docker ou `uvicorn`
3. ouvrir `http://127.0.0.1:8000/docs`
4. tester 2 ou 3 scenarios d'usage realistes via le selecteur `Examples` de Swagger

Scenarios conseilles :

- `Je cherche un atelier a Paris ce week-end`
- `Je veux une exposition gratuite en Ile-de-France`
- `Quels evenements famille sont proposes prochainement ?`

Le support de soutenance final compte 12 slides : [PRESENTATION.md](/home/zmxw1768/Documents/rag_oc/docs/PRESENTATION.md:1) et [RAG_OpenAgenda_presentation.pptx](/home/zmxw1768/Documents/rag_oc/docs/RAG_OpenAgenda_presentation.pptx).

## 8. Evaluation avec RAGAS

Le script [scripts/evaluate_rag.py](/home/zmxw1768/Documents/rag_oc/scripts/evaluate_rag.py:1) mesure la qualite du RAG avec [RAGAS](https://docs.ragas.io/) sur trois metriques. Il utilise les modeles Mistral deja employes par le pipeline (`mistral-small-latest` et `mistral-embed`) : une cle `MISTRAL_API_KEY` est donc requise dans les deux modes.

- **answer_relevancy** : la reponse est-elle pertinente par rapport a la question ?
- **faithfulness** : la reponse est-elle fidele au contexte fourni (pas d'hallucination) ?
- **context_precision** : les chunks recuperes sont-ils pertinents par rapport a la reference ?

### Installation des dependances d'evaluation

```bash
uv sync --extra eval
```

### Mode static (donnees pre-calculees)

Lit un JSONL avec `question`, `answer`, `ground_truth` et `contexts` :

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
uv run python scripts/evaluate_rag.py --input tests/rag_eval_sample.jsonl --mode static
```

Par defaut, l'evaluation lance une seule metrique a la fois et espace les appels
Mistral de cinq secondes pour respecter les limites d'API. En cas de `429`, elle
effectue jusqu'a cinq nouvelles tentatives et laisse jusqu'a dix minutes a chaque
metrique. Vous pouvez ajuster ce rythme, par exemple `--requests-per-second 0.5`
(une requete toutes les deux secondes), si le quota le permet.

### Mode live (interrogation du RAG reel)

Lit un JSONL avec `question` et `ground_truth`, puis interroge le RAG (FAISS + Mistral) pour obtenir la reponse et les contextes :

```bash
uv run python scripts/evaluate_rag.py --input tests/rag_eval_sample.jsonl --mode live
```

Ce mode necessite une cle Mistral valide (via `.env` ou `MISTRAL_API_KEY`).

### Sauvegarder les scores

```bash
SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
uv run python scripts/evaluate_rag.py --input tests/rag_eval_sample.jsonl --mode static --output scores.json
```

Un jeu de test annote d'exemple est fourni dans [tests/rag_eval_sample.jsonl](/home/zmxw1768/Documents/rag_oc/tests/rag_eval_sample.jsonl:1).

### Resultats observes

Sur le jeu de test static de 10 cas, la campagne mesuree produit les scores suivants :

| Metrique | Score | Interpretation |
|---|---:|---|
| `context_precision` | 1,0000 | Les contextes recuperes sont pertinents pour les references de ce jeu de test. |
| `answer_relevancy` | 0,8536 | Les reponses sont globalement adaptees aux questions. |
| `faithfulness` | 0,7000 | La fidelite au contexte est le principal axe d'amelioration : certaines reponses extrapolent au-dela des chunks. |

Ces mesures sont indicatives : le jeu de test est volontairement petit et annote. Elles doivent etre rejouees apres toute evolution de l'index, du prompt ou du modele.

## 9. Tests

Tests unitaires avec couverture de code :

```bash
uv run pytest
```

La couverture est configuree dans `pyproject.toml` et s'affiche automatiquement.
Un rapport HTML est genere dans `htmlcov/`.

Test fonctionnel HTTP :

```bash
uv run python scripts/api_smoke_test.py
```

Les tests couvrent :

- la fenetre temporelle de collecte ;
- la normalisation OpenAgenda ;
- le chunking et le chargement JSONL/GZ ;
- la construction de l'index FAISS (Flat et IVFFlat) ;
- le formatage du contexte et des lieux ;
- le filtrage des evenements passes ;
- la lecture de la cle API depuis `.env` ;
- la normalisation des dates ISO 8601 ;
- la generation de reponse (mock du client Mistral) ;
- le contrat HTTP de l'API ;
- la presence des scenarios OpenAPI exposes dans Swagger ;
- la validation du format du jeu de test annote pour l'evaluation (modes static et live).

## Sources

- OpenAgenda Authentification : https://developers.openagenda.com/authentification/
- OpenAgenda Lecture evenements : https://developers.openagenda.com/evenements/lecture/
- OpenAgenda Recherche agendas : https://developers.openagenda.com/agendas/recherche/
- Mistral Embeddings : https://docs.mistral.ai/studio-api/knowledge-rag/embeddings/
- Mistral RAG quickstart : https://docs.mistral.ai/studio-api/knowledge-rag/rag_quickstart
