# Mistral dans ce projet

Ce document decrit l'usage de Mistral dans le pipeline RAG du depot.

## Role de Mistral

Mistral intervient a deux endroits distincts :

1. pour les embeddings
2. pour la generation de la reponse finale

En pratique :

- `mistral-embed` transforme les chunks et les questions en vecteurs ;
- un modele de chat Mistral transforme ensuite `question + contexte` en reponse naturelle.

## 1. Embeddings

Le script [scripts/build_faiss_index.py](/home/zmxw1768/Documents/rag_oc/scripts/build_faiss_index.py:1) :

1. lit le dataset OpenAgenda ;
2. reutilise les chunks existants ou decoupe `document` ;
3. appelle `client.embeddings.create(...)` ;
4. recupere un embedding par chunk ;
5. construit l'index FAISS ;
6. sauvegarde les metadonnees alignees avec chaque vecteur.

Le flux est donc :

```text
chunk texte -> embedding Mistral -> FAISS
```

Exemple de commande :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/full.jsonl.gz \
  --index-output data/faiss/openagenda.index \
  --metadata-output data/faiss/openagenda_metadata.pkl
```

## 2. Reponse RAG

Le moteur metier dans [rag_service.py](/home/zmxw1768/Documents/rag_oc/src/rag_oc/rag_service.py:1) fonctionne ainsi :

1. la question utilisateur est transformee en embedding avec `mistral-embed` ;
2. FAISS retrouve les chunks les plus proches ;
3. les metadonnees associees sont recuperees ;
4. les evenements deja termines sont filtres par defaut ;
5. un contexte texte est reconstruit ;
6. ce contexte est envoye a `client.chat.complete(...)`.

Le flux complet devient :

```text
question
  -> embedding Mistral
  -> recherche FAISS
  -> recuperation des chunks
  -> prompt
  -> modele de chat Mistral
  -> reponse finale
```

## Modele d'embeddings

Le projet utilise par defaut :

- `mistral-embed`

Ce modele est utilise pour :

- les chunks d'evenements pendant la phase d'indexation ;
- les questions utilisateur pendant la phase de retrieval.

Le point important est que le meme type de representation semantique est utilise des deux cotes.

## Index FAISS

Le projet n'utilise plus `IndexFlatL2`.

Actuellement :

- les vecteurs sont normalises ;
- l'index par defaut est `IndexFlatIP` ;
- `ivfflat` est aussi disponible en option.

Cela permet une recherche semantique simple et rapide sur des vecteurs comparables.

## Metadonnees sauvegardees

Le fichier `data/faiss/openagenda_metadata.pkl` contient les donnees associees a chaque vecteur, par exemple :

- `id`
- `event_uid`
- `title`
- `date_summary`
- `first_timing`
- `last_timing`
- `location`
- `source_agendas`
- `chunk_id`
- `chunk_text`
- `document`

Ces informations servent a reconstruire le contexte donne au modele.

## Gestion du rate limit

La phase d'embeddings peut etre longue.

Le script gere :

- les appels par batch ;
- les pauses volontaires entre batches ;
- les retries automatiques ;
- le backoff exponentiel sur les erreurs `429` et `5xx`.

Exemple prudent :

```bash
uv run python scripts/build_faiss_index.py \
  --input data/openagenda/full.jsonl.gz \
  --batch-size 16 \
  --request-pause-seconds 1 \
  --max-retries 8 \
  --retry-base-seconds 3
```

## Variable d'environnement

La cle Mistral peut etre lue dans :

- `MISTRAL_API_KEY`

Exemple `.env` :

```env
MISTRAL_API_KEY="ta_cle_mistral"
```

## Ce qui est disponible maintenant

Le projet ne se limite plus a l'indexation.

Il contient aujourd'hui :

- un script de construction d'index ;
- un moteur RAG reutilisable ;
- une CLI de test ;
- une API FastAPI avec `/ask` et `/rebuild`.

## Limites a garder en tete

- les appels embeddings et chat consomment de l'API Mistral ;
- le rebuild peut etre long sur un gros dataset ;
- la qualite finale depend a la fois du retrieval FAISS et du modele de chat ;
- si les metadonnees evoluent, il faut reconstruire l'index et le fichier `.pkl`.

## Sources

- Embeddings Mistral : https://docs.mistral.ai/studio-api/knowledge-rag/embeddings/
- RAG quickstart Mistral : https://docs.mistral.ai/studio-api/knowledge-rag/rag_quickstart
