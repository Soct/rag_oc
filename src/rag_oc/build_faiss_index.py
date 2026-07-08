"""Construction de l'index FAISS a partir du JSONL OpenAgenda.

Pipeline : JSONL(.gz) -> chunking du champ ``document`` -> embeddings Mistral
-> normalisation L2 -> IndexFlatIP (similarite cosinus) + metadonnees pickle.

Le fichier d'entree par defaut est ``full.jsonl.gz`` (produit par
``scripts/fetch_openagenda_events.py``).  L'ancien chemin
``ile_de_france_events.jsonl`` a ete corrige car ce fichier n'existe plus.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path


# Chemin corrige : le vrai dump est full.jsonl.gz (pas ile_de_france_events.jsonl).
DEFAULT_INPUT_PATH = Path("data/openagenda/full.jsonl.gz")
DEFAULT_INDEX_PATH = Path("data/faiss/openagenda.index")
DEFAULT_METADATA_PATH = Path("data/faiss/openagenda_metadata.pkl")
DEFAULT_MODEL = "mistral-embed"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 6
DEFAULT_RETRY_BASE_SECONDS = 2.0
DEFAULT_REQUEST_PAUSE_SECONDS = 0.25
# Taille cible d'un chunk en caracteres.  Un chevauchement (overlap) est
# applique entre chunks consecutifs pour ne pas couper le sens en plein milieu.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
ENV_FILE_PATH = Path(".env")
API_ENV_KEYS = ("MISTRAL_API_KEY",)


@dataclass(slots=True)
class ChunkRecord:
    metadata: dict
    text: str


@dataclass(slots=True)
class IndexBuildConfig:
    input_path: Path = DEFAULT_INPUT_PATH
    index_output: Path = DEFAULT_INDEX_PATH
    metadata_output: Path = DEFAULT_METADATA_PATH
    api_key: str | None = None
    model: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    text_field: str = "document"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    index_type: str = "flat"
    ivf_nlist: int = 256
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS
    request_pause_seconds: float = DEFAULT_REQUEST_PAUSE_SECONDS


def read_api_key_from_env_file(path: Path = ENV_FILE_PATH) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in API_ENV_KEYS:
            continue
        value = value.strip().strip("'").strip('"')
        if value:
            return value
    return None


def parse_args() -> argparse.Namespace:
    default_api_key = None
    for env_key in API_ENV_KEYS:
        default_api_key = os.getenv(env_key)
        if default_api_key:
            break
    if not default_api_key:
        default_api_key = read_api_key_from_env_file()

    parser = argparse.ArgumentParser(
        description="Construit un index FAISS a partir du JSONL OpenAgenda avec les embeddings Mistral."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Fichier JSONL source.")
    parser.add_argument("--index-output", type=Path, default=DEFAULT_INDEX_PATH, help="Fichier index FAISS.")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Fichier metadata associe a l'index.",
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help="Cle API Mistral. Peut aussi etre passee via MISTRAL_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modele d'embeddings Mistral. Defaut: mistral-embed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Nombre de documents envoye a Mistral par requete embeddings.",
    )
    parser.add_argument(
        "--text-field",
        default="document",
        help="Champ texte du JSONL a vectoriser. Defaut: document.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Taille cible d'un chunk si aucun chunk n'est deja present dans les donnees.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Recouvrement entre chunks consecutifs.",
    )
    parser.add_argument(
        "--index-type",
        choices=("flat", "ivfflat"),
        default="flat",
        help="Type d'index FAISS a construire.",
    )
    parser.add_argument(
        "--ivf-nlist",
        type=int,
        default=256,
        help="Nombre de centroides pour IndexIVFFlat.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Nombre maximum de tentatives par batch en cas de rate limit ou erreur 5xx.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=DEFAULT_RETRY_BASE_SECONDS,
        help="Base du backoff exponentiel entre deux retries.",
    )
    parser.add_argument(
        "--request-pause-seconds",
        type=float,
        default=DEFAULT_REQUEST_PAUSE_SECONDS,
        help="Pause volontaire entre deux batches pour eviter les 429.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("La cle Mistral est requise via --api-key ou MISTRAL_API_KEY.")
    if args.batch_size < 1:
        parser.error("--batch-size doit etre >= 1.")
    if args.chunk_size < 1:
        parser.error("--chunk-size doit etre >= 1.")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap doit etre >= 0.")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap doit etre strictement inferieur a --chunk-size.")
    if args.ivf_nlist < 1:
        parser.error("--ivf-nlist doit etre >= 1.")
    if args.max_retries < 0:
        parser.error("--max-retries doit etre >= 0.")
    if args.retry_base_seconds < 0:
        parser.error("--retry-base-seconds doit etre >= 0.")
    if args.request_pause_seconds < 0:
        parser.error("--request-pause-seconds doit etre >= 0.")
    return args


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_records(path: Path) -> list[dict]:
    with open_text_file(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    step = chunk_size - chunk_overlap
    start = 0
    chunks: list[tuple[str, int, int]] = []
    text_length = len(normalized)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            split_at = normalized.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))
        if end >= text_length:
            break
        start += step
    return chunks


def build_chunk_records(
    records: list[dict],
    text_field: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Transforme les records JSONL en chunks indexables.

    Si un record contient deja un champ ``chunks``, on les reutilise tels
    quels.  Sinon on decoupe le champ ``text_field`` avec ``split_text``.
    Chaque chunk embarque les metadonnees de l'evenement (titre, dates,
    lieu…) qui seront stockees a cote de l'index FAISS pour pouvoir
    reconstruire le contexte RAG.
    """
    chunk_records: list[ChunkRecord] = []

    for record in records:
        existing_chunks = record.get("chunks")
        if isinstance(existing_chunks, list) and existing_chunks:
            chunk_payloads = []
            for index, raw_chunk in enumerate(existing_chunks):
                if isinstance(raw_chunk, str):
                    text = raw_chunk.strip()
                    start = None
                    end = None
                elif isinstance(raw_chunk, dict):
                    text = str(raw_chunk.get("text", "")).strip()
                    start = raw_chunk.get("start")
                    end = raw_chunk.get("end")
                else:
                    continue
                if text:
                    chunk_payloads.append((index, text, start, end))
        else:
            text_value = str(record.get(text_field, "")).strip()
            chunk_payloads = [
                (index, chunk_text, start, end)
                for index, (chunk_text, start, end) in enumerate(
                    split_text(text_value, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                )
            ]

        for index, chunk_text, start, end in chunk_payloads:
            metadata = {
                "id": record.get("id"),
                "event_uid": record.get("event_uid"),
                "title": record.get("title"),
                "date_summary": record.get("date_summary"),
                "first_timing": record.get("first_timing"),
                "last_timing": record.get("last_timing"),
                "location": record.get("location"),
                "source_agendas": record.get("source_agendas"),
                "document": record.get(text_field),
                "chunk_id": f"{record.get('id')}:chunk:{index}",
                "chunk_index": index,
                "chunk_text": chunk_text,
                "chunk_start": start,
                "chunk_end": end,
            }
            chunk_records.append(ChunkRecord(metadata=metadata, text=chunk_text))
    return chunk_records


def batch_items(items: list[str], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def extract_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    message = str(exc)
    if "Status 429" in message:
        return 429
    if "Status 500" in message:
        return 500
    if "Status 502" in message:
        return 502
    if "Status 503" in message:
        return 503
    if "Status 504" in message:
        return 504
    return None


def fetch_batch_embeddings(
    client,
    model: str,
    batch: list[str],
    max_retries: int,
    retry_base_seconds: float,
):
    from mistralai.client.errors.sdkerror import SDKError

    attempt = 0
    while True:
        try:
            return client.embeddings.create(model=model, inputs=batch)
        except SDKError as exc:
            status_code = extract_status_code(exc)
            retryable = status_code in {429, 500, 502, 503, 504}
            if not retryable or attempt >= max_retries:
                raise
            sleep_seconds = retry_base_seconds * (2**attempt)
            print(
                f"Batch rate-limite ou erreur serveur (status={status_code}). "
                f"Retry {attempt + 1}/{max_retries} dans {sleep_seconds:.1f}s."
            )
            time.sleep(sleep_seconds)
            attempt += 1


def fetch_embeddings(
    client,
    model: str,
    documents: list[str],
    batch_size: int,
    max_retries: int,
    retry_base_seconds: float,
    request_pause_seconds: float,
):
    import numpy as np
    from tqdm import tqdm

    all_vectors: list[list[float]] = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for batch in tqdm(batch_items(documents, batch_size), total=total_batches, desc="Embeddings", unit="batch"):
        response = fetch_batch_embeddings(
            client=client,
            model=model,
            batch=batch,
            max_retries=max_retries,
            retry_base_seconds=retry_base_seconds,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        all_vectors.extend(item.embedding for item in ordered)
        if request_pause_seconds:
            time.sleep(request_pause_seconds)
    return np.asarray(all_vectors, dtype="float32")


def build_faiss_index(vectors, index_type: str, ivf_nlist: int):
    """Construit l'index FAISS a partir des vecteurs d'embeddings.

    Les vecteurs sont normalises L2 puis inseres dans un IndexFlatIP
    (produit scalaire = similarite cosinus apres normalisation).
    Une variante IVFFlat est disponible pour de plus gros volumes.
    """
    import faiss
    import numpy as np

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Les vecteurs doivent former une matrice 2D non vide.")

    normalized_vectors = np.asarray(vectors, dtype="float32").copy()
    faiss.normalize_L2(normalized_vectors)

    dimension = normalized_vectors.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dimension)
        index.add(normalized_vectors)
        return index

    nlist = min(ivf_nlist, normalized_vectors.shape[0])
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(normalized_vectors)
    index.add(normalized_vectors)
    return index


def rebuild_index(config: IndexBuildConfig) -> dict:
    """Pipeline complet : lecture JSONL -> chunking -> embeddings -> index FAISS.

    Retourne un dict resume avec les chemins et stats de construction.
    Appele par le CLI (``scripts/build_faiss_index.py``) et par l'endpoint
    ``POST /rebuild`` de l'API.
    """
    from mistralai.client import Mistral

    if not config.api_key:
        raise ValueError("La cle Mistral est requise pour reconstruire l'index.")

    records = load_records(config.input_path)
    filtered_records = [record for record in records if record.get(config.text_field)]
    chunk_records = build_chunk_records(
        filtered_records,
        text_field=config.text_field,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    documents = [chunk.text for chunk in chunk_records]
    metadata = [chunk.metadata for chunk in chunk_records]

    if not documents:
        raise ValueError(f"Aucun chunk exploitable trouve a partir du champ `{config.text_field}`.")

    client = Mistral(api_key=config.api_key)
    vectors = fetch_embeddings(
        client=client,
        model=config.model,
        documents=documents,
        batch_size=config.batch_size,
        max_retries=config.max_retries,
        retry_base_seconds=config.retry_base_seconds,
        request_pause_seconds=config.request_pause_seconds,
    )

    index = build_faiss_index(vectors=vectors, index_type=config.index_type, ivf_nlist=config.ivf_nlist)

    config.index_output.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_output.parent.mkdir(parents=True, exist_ok=True)

    import faiss

    faiss.write_index(index, str(config.index_output))
    with config.metadata_output.open("wb") as handle:
        pickle.dump(metadata, handle)

    return {
        "index_path": str(config.index_output),
        "metadata_path": str(config.metadata_output),
        "chunks_indexed": len(metadata),
        "source_records": len(filtered_records),
        "index_type": config.index_type,
        "embedding_model": config.model,
    }


def main() -> None:
    args = parse_args()
    result = rebuild_index(
        IndexBuildConfig(
            input_path=args.input,
            index_output=args.index_output,
            metadata_output=args.metadata_output,
            api_key=args.api_key,
            model=args.model,
            batch_size=args.batch_size,
            text_field=args.text_field,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            index_type=args.index_type,
            ivf_nlist=args.ivf_nlist,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            request_pause_seconds=args.request_pause_seconds,
        )
    )

    print(f"Index FAISS ecrit dans {result['index_path']}")
    print(f"Metadonnees ecrites dans {result['metadata_path']}")
    print(f"Chunks indexes: {result['chunks_indexed']}")
    print(f"Evenements sources: {result['source_records']}")
    print(f"Type d'index FAISS: {result['index_type']}")
    print(f"Modele d'embeddings: {result['embedding_model']}")


if __name__ == "__main__":
    main()
