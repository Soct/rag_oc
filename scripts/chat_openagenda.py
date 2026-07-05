from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mistralai.client import Mistral

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_oc.rag_service import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_MAX_CONTEXT_ITEMS,
    DEFAULT_METADATA_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    RagService,
    format_location,
    resolve_api_key,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chatbot RAG OpenAgenda base sur FAISS et Mistral, sans historique de conversation."
    )
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH, help="Chemin de l'index FAISS.")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Chemin du fichier de metadonnees pickle.",
    )
    parser.add_argument("--api-key", default=resolve_api_key(), help="Cle API Mistral.")
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Modele Mistral utilise pour embedder les questions.",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help="Modele Mistral utilise pour generer la reponse finale.",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Nombre de voisins FAISS a recuperer.")
    parser.add_argument(
        "--max-context-items",
        type=int,
        default=DEFAULT_MAX_CONTEXT_ITEMS,
        help="Nombre maximum de chunks injectes dans le prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature du modele de chat.",
    )
    parser.add_argument("--question", default=None, help="Question utilisateur a traiter en mode one-shot.")
    args = parser.parse_args()
    if not args.api_key:
        parser.error("La cle Mistral est requise via --api-key ou MISTRAL_API_KEY.")
    if args.top_k < 1:
        parser.error("--top-k doit etre >= 1.")
    if args.max_context_items < 1:
        parser.error("--max-context-items doit etre >= 1.")
    if args.temperature < 0:
        parser.error("--temperature doit etre >= 0.")
    return args


def print_sources(matches: list[dict], max_context_items: int) -> None:
    print("\nSources utilisees:")
    for item in matches[:max_context_items]:
        print(
            f"- {item.get('title') or 'Titre non renseigne'} | "
            f"{item.get('date_summary') or 'Date non renseignee'} | "
            f"{format_location(item.get('location'))}"
        )


def build_service(args: argparse.Namespace) -> RagService:
    client = Mistral(api_key=args.api_key)
    return RagService.from_paths(
        client=client,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        embedding_model=args.embedding_model,
        chat_model=args.chat_model,
        top_k=args.top_k,
        max_context_items=args.max_context_items,
        temperature=args.temperature,
    )


def interactive_loop(args: argparse.Namespace) -> None:
    service = build_service(args)

    if args.question:
        result = service.ask(args.question)
        print(result.answer)
        print_sources(result.matches, args.max_context_items)
        return

    print("Mode interactif OpenAgenda. Tape une question ou `quit` pour sortir.")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        result = service.ask(question)
        print(result.answer)
        print_sources(result.matches, args.max_context_items)
        print("")


def main() -> None:
    args = parse_args()
    interactive_loop(args)


if __name__ == "__main__":
    main()
