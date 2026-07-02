# Collecte OpenAgenda pour futur RAG

Ce depot contient deux scripts simples :

- `scripts/fetch_openagenda_events.py` pour collecter les evenements OpenAgenda en Ile-de-France ;
- `scripts/build_faiss_index.py` pour transformer le `JSONL` en index FAISS avec les embeddings Mistral.

## Objectif

Le script :

- parcourt les agendas OpenAgenda publiés ;
- récupère les événements localisés en `Île-de-France` ;
- applique une fenêtre de dates configurable ;
- force une borne basse à `aujourd'hui - 365 jours` pour exclure les événements trop anciens ;
- permet de filtrer les mots-cles d'evenement ;
- exclut par defaut les evenements ayant le mot-cle `concert` ;
- produit un fichier `JSONL` normalisé, plus pratique qu'un CSV pour un pipeline RAG.
- utilise un format compact, centre sur le RAG, pour eviter des fichiers de plusieurs centaines de Mo.

## Pourquoi du JSONL

Le fichier de sortie contient une ligne JSON par événement.

Ce format est adapté au RAG parce qu'il permet de conserver à la fois :

- un texte prêt à indexer dans `document` ;
- des métadonnées structurées comme `title`, `timings`, `location`, `event_types`, `source_agendas`.

Un futur script de vectorisation pourra donc :

1. lire chaque ligne ;
2. utiliser `document` comme contenu principal ;
3. garder les métadonnées pour le filtrage, la citation des sources et le reranking.

Le script ne conserve pas tous les champs OpenAgenda bruts. Il garde seulement les donnees utiles au retrieval et a la restitution, ce qui reduit fortement la taille du dataset.

## Structure

```text
.
├── main.py
├── pyproject.toml
├── README.md
├── scripts/
│   ├── build_faiss_index.py
│   └── fetch_openagenda_events.py
└── tests/
    └── test_openagenda.py
```

## Pré-requis

- Python `>= 3.13`
- `uv`
- une clé API publique OpenAgenda

La documentation officielle OpenAgenda confirme que la lecture API se fait avec une clé publique passée dans l'en-tête `key`. Le script tente d'abord la lecture transverse via `/v2/events`, puis bascule automatiquement sur `/v2/agendas/{agendaUID}/events` si cette route n'est pas disponible pour ton compte.

Sources :

- https://developers.openagenda.com/authentification/
- https://developers.openagenda.com/evenements/lecture/
- https://developers.openagenda.com/agendas/recherche/

## Lancement

La configuration principale est directement dans [scripts/fetch_openagenda_events.py](/home/zmxw1768/Documents/rag_oc/scripts/fetch_openagenda_events.py:14), dans le bloc `DEFAULT_CONFIG`.

Par defaut :

- `region` vaut `Ile-de-France` ;
- `excluded_event_types` vaut `["concert"]` ;
- `included_event_types` est vide, donc on ne limite pas la collecte a un seul type.

Exemple simple :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --api-key "VOTRE_CLE_OPENAGENDA"
```

Exemple avec filtres :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --api-key "VOTRE_CLE_OPENAGENDA" \
  --event-type concert \
  --event-type jazz \
  --date-from "2026-01-01T00:00:00+00:00" \
  --date-to "2026-12-31T23:59:59+00:00" \
  --output data/openagenda/concerts_idf.jsonl
```

## Options utiles

```bash
uv run python scripts/fetch_openagenda_events.py --help
```

Principales options :

- `--api-key` : clé publique OpenAgenda ;
- `--event-type` : inclut seulement certains mots-cles via `keyword[]` ;
- `--exclude-event-type` : exclut localement certains mots-cles, par defaut `concert` ;
- `--search` : filtre texte complémentaire ;
- `--date-from` : début de période ;
- `--date-to` : fin de période ;
- `--region` : région ciblée, par défaut `Île-de-France` ;
- `--official-only` : limite la collecte aux agendas officiels ;
- `--max-agendas` : borne de sécurité pour les gros runs ;
- `--pause-seconds` : pause entre deux agendas si nécessaire ;
- `--source-mode` : `auto`, `transverse` ou `agendas` ;
- `--workers` : nombre de requêtes agendas parallèles en mode `agendas` ;
- `--output` : chemin du fichier JSONL ;
- `--manifest-output` : chemin du manifeste JSON.

Si `--output` finit par `.gz`, le script ecrit directement un `JSONL` compresse.

## Strategie de collecte

Le script propose trois modes :

- `auto` : tente `/v2/events`, puis fallback sur la collecte agenda par agenda ;
- `transverse` : force `/v2/events` ;
- `agendas` : force `/v2/agendas/{agendaUID}/events`.

La route transverse est la plus interessante en performance, mais la documentation OpenAgenda precise qu'elle est experimentale et qu'il faut parfois contacter le support pour y avoir acces.

Quand le script tombe en mode `agendas`, tu peux accelerer la collecte avec `--workers`.

Exemple :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --output data/openagenda/test.jsonl \
  --source-mode agendas \
  --official-only \
  --workers 12
```

## Règle sur la date

Même si `--date-from` est fourni, le script n'accepte jamais une date de début antérieure à `maintenant - 365 jours`.

Exemple :

- date du jour du run : `2026-07-02`
- `--date-from 2024-01-01T00:00:00+00:00`
- date réellement utilisée : `2025-07-02` plus l'heure courante UTC du lancement

Cette règle garantit que le dataset ne contient pas d'événements vieux de plus d'un an.

## Que mettre dans `event-type`

OpenAgenda n'expose pas ici une taxonomie fermee universelle du type "event_type" strict. Dans ce script, `event-type` correspond aux `keywords` retournes par les evenements OpenAgenda.

Concretement, tu peux mettre des valeurs comme :

- `exposition`
- `conference`
- `atelier`
- `festival`
- `theatre`
- `cinema`
- `visite`
- `patrimoine`
- `jeunesse`

Point important :

- ces valeurs dependent des mots-cles reellement saisis dans les agendas ;
- ce n'est pas une liste garantie globale ;
- `concert` peut apparaitre seul ou avec d'autres mots-cles comme `jazz`, `musique`, `live`.

Si tu ne veux pas les concerts, le plus robuste est ce que fait maintenant le script :

- ne pas filtrer uniquement sur un type a l'entree ;
- exclure ensuite localement tous les evenements dont les `keywords` contiennent `concert`.

## Format de sortie

Chaque ligne du fichier JSONL ressemble à ceci :

```json
{
  "id": "openagenda:987",
  "event_uid": 987,
  "title": "Concert de jazz",
  "description": "Soirée musicale",
  "long_description": "Un trio en live.",
  "event_types": ["concert", "jazz"],
  "date_summary": "2026-07-03T18:00:00.000Z -> 2026-07-03T20:00:00.000Z",
  "occurrences_count": 1,
  "timings": [
    {
      "begin": "2026-07-03T18:00:00.000Z",
      "end": "2026-07-03T20:00:00.000Z"
    }
  ],
  "location": {
    "name": "Salle des fêtes",
    "city": "Paris",
    "department": "Paris",
    "region": "Île-de-France"
  },
  "source_agendas": [
    {
      "uid": 123,
      "title": "Agenda Culture",
      "slug": "agenda-culture"
    }
  ],
  "document": "Titre: Concert de jazz\nRésumé: Soirée musicale\nDescription: Un trio en live.\nTypes: concert, jazz\nDates: 2026-07-03T18:00:00.000Z -> 2026-07-03T20:00:00.000Z\nLieu: Salle des fêtes, Paris, Paris, Île-de-France"
}
```

Pour reduire encore la taille disque, tu peux sortir directement en gzip :

```bash
uv run python scripts/fetch_openagenda_events.py \
  --output data/openagenda/test.jsonl.gz
```

Le script génère aussi un manifeste `*.manifest.json` avec :

- la date de génération ;
- la fenêtre de dates utilisée ;
- les filtres appliqués ;
- le nombre d'agendas parcourus ;
- le nombre d'événements écrits.

## Conversion vers FAISS avec Mistral

Installe d'abord les dependances necessaires :

```bash
uv sync
```

Puis construis l'index :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/ile_de_france_events.jsonl \
  --index-output data/faiss/openagenda.index \
  --metadata-output data/faiss/openagenda_metadata.pkl
```

Le script lit la cle Mistral via `MISTRAL_API_KEY` ou `--api-key`.

Exemple avec fichier compresse :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/test.jsonl.gz \
  --index-output data/faiss/openagenda.index \
  --metadata-output data/faiss/openagenda_metadata.pkl
```

Le resultat :

- `data/faiss/openagenda.index` contient les vecteurs ;
- `data/faiss/openagenda_metadata.pkl` contient les metadonnees a renvoyer avec les resultats.

Le script suit le flux recommande par la documentation Mistral pour un RAG "from scratch" :

- embeddings via `client.embeddings.create(...)`
- modele `mistral-embed`
- stockage des vecteurs dans FAISS

Dans le RAG, le flux devient :

1. collecter les evenements avec `fetch_openagenda_events.py` ;
2. transformer le champ `document` en embeddings Mistral ;
3. indexer dans FAISS ;
4. au moment d'une question, embedder la question ;
5. chercher les voisins les plus proches dans FAISS ;
6. reconstruire le contexte avec les metadonnees.

Sources officielles Mistral :

- Embeddings : https://docs.mistral.ai/studio-api/knowledge-rag/embeddings/
- RAG quickstart : https://docs.mistral.ai/studio-api/knowledge-rag/rag_quickstart

## Tests

Les tests couvrent pour l’instant :

- le bornage de la fenêtre de dates à 365 jours ;
- la normalisation d’un événement vers un enregistrement orienté RAG.

Lancer les tests :

```bash
uv run python -m unittest discover -s tests
```
