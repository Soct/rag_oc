from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/openagenda/ile_de_france_events.jsonl")
DEFAULT_INDEX_PATH = Path("data/faiss/openagenda.index")
DEFAULT_METADATA_PATH = Path("data/faiss/openagenda_metadata.pkl")
DEFAULT_MODEL = "mistral-embed"
DEFAULT_BATCH_SIZE = 32
ENV_FILE_PATH = Path(".env")
API_ENV_KEYS = ("MISTRAL_API_KEY",)


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
    args = parser.parse_args()
    if not args.api_key:
        parser.error("La cle Mistral est requise via --api-key ou MISTRAL_API_KEY.")
    if args.batch_size < 1:
        parser.error("--batch-size doit etre >= 1.")
    return args


def open_text_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def load_records(path: Path) -> list[dict]:
    with open_text_file(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_metadata(records: list[dict], text_field: str) -> list[dict]:
    metadata: list[dict] = []
    for record in records:
        metadata.append(
            {
                "id": record.get("id"),
                "event_uid": record.get("event_uid"),
                "title": record.get("title"),
                "date_summary": record.get("date_summary"),
                "location": record.get("location"),
                "source_agendas": record.get("source_agendas"),
                "document": record.get(text_field),
            }
        )
    return metadata


def batch_items(items: list[str], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def fetch_embeddings(client, model: str, documents: list[str], batch_size: int):
    import numpy as np
    from tqdm import tqdm

    all_vectors: list[list[float]] = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for batch in tqdm(batch_items(documents, batch_size), total=total_batches, desc="Embeddings", unit="batch"):
        response = client.embeddings.create(model=model, inputs=batch)
        ordered = sorted(response.data, key=lambda item: item.index)
        all_vectors.extend(item.embedding for item in ordered)
    return np.asarray(all_vectors, dtype="float32")


def main() -> None:
    args = parse_args()

    import faiss
    from mistralai.client import Mistral

    records = load_records(args.input)
    filtered_records = [record for record in records if record.get(args.text_field)]
    documents = [record[args.text_field] for record in filtered_records]
    metadata = build_metadata(filtered_records, args.text_field)

    if not documents:
        raise ValueError(f"Aucun document exploitable trouve dans le champ `{args.text_field}`.")

    client = Mistral(api_key=args.api_key)
    vectors = fetch_embeddings(client, args.model, documents, args.batch_size)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index_output))
    with args.metadata_output.open("wb") as handle:
        pickle.dump(metadata, handle)

    print(f"Index FAISS ecrit dans {args.index_output}")
    print(f"Metadonnees ecrites dans {args.metadata_output}")
    print(f"Documents indexes: {len(metadata)}")
    print(f"Modele d'embeddings: {args.model}")


if __name__ == "__main__":
    main()
