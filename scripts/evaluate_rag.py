"""Evaluation du pipeline RAG avec RAGAS.

Deux modes d'utilisation :

1. ``--mode static`` (defaut) : lit un fichier JSONL pre-rempli contenant
   question, answer, ground_truth et contexts. Utile pour rejouer une
   evaluation reproductible sans appeler le RAG ni Mistral.

2. ``--mode live`` : lit un fichier JSONL contenant au minimum question et
   ground_truth, puis interroge le RAG reel (FAISS + Mistral) pour obtenir
   answer et contexts.  Necessite une cle Mistral valide.

Dependances (installees via ``uv sync --extra eval``) :
  ragas, datasets, langchain-community, langchain-mistralai

Exemple d'utilisation :

    uv run python scripts/evaluate_rag.py --input tests/rag_eval_sample.jsonl --mode static
    uv run python scripts/evaluate_rag.py --input tests/rag_eval_sample.jsonl --mode live
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Champs requis pour le mode static (evaluation sur donnees pre-calculees).
STATIC_REQUIRED_FIELDS = ("question", "answer", "ground_truth", "contexts")
# Champs requis pour le mode live (le RAG fournit answer et contexts).
LIVE_REQUIRED_FIELDS = ("question", "ground_truth")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation RAG avec RAGAS.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Fichier JSONL d'evaluation.",
    )
    parser.add_argument(
        "--mode",
        choices=("static", "live"),
        default="static",
        help="static = donnees pre-remplies, live = interroge le RAG reel.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier JSON optionnel pour sauvegarder les scores.",
    )
    return parser.parse_args()


def load_rows(path: Path, *, required_fields: tuple[str, ...] = STATIC_REQUIRED_FIELDS) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, line_number=line_number, required_fields=required_fields)
            rows.append(row)
    if not rows:
        raise ValueError(f"Aucune ligne exploitable dans {path}.")
    return rows


def validate_row(
    row: dict,
    *,
    line_number: int,
    required_fields: tuple[str, ...] = STATIC_REQUIRED_FIELDS,
) -> None:
    missing = [field for field in required_fields if field not in row]
    if missing:
        raise ValueError(f"Ligne {line_number}: champs manquants: {', '.join(missing)}.")

    for field in ("question", "ground_truth"):
        if field not in row:
            continue
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Ligne {line_number}: `{field}` doit etre une chaine non vide.")

    if "contexts" in required_fields and "contexts" in row:
        contexts = row["contexts"]
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"Ligne {line_number}: `contexts` doit etre une liste non vide.")
        if not all(isinstance(item, str) and item.strip() for item in contexts):
            raise ValueError(f"Ligne {line_number}: chaque element de `contexts` doit etre une chaine non vide.")


def enrich_with_rag(rows: list[dict]) -> list[dict]:
    """Interroge le RAG reel pour remplir answer et contexts."""
    from mistralai.client import Mistral

    from rag_oc.rag_service import RagService, resolve_api_key

    api_key = resolve_api_key()
    if not api_key:
        raise ValueError("Cle Mistral introuvable. Requis pour le mode live.")

    client = Mistral(api_key=api_key)
    service = RagService.from_paths(client=client, include_past_events=True)

    enriched: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        question = row["question"]
        print(f"  [{idx}/{len(rows)}] {question[:60]}...")
        result = service.ask(question)
        enriched.append({
            "question": question,
            "answer": result.answer,
            "ground_truth": row["ground_truth"],
            "contexts": [
                m.get("chunk_text") or m.get("document") or ""
                for m in result.matches
                if (m.get("chunk_text") or m.get("document"))
            ] or ["Aucun contexte recupere."],
        })
    return enriched


def run_evaluation(rows: list[dict], output_path: Path | None) -> None:
    """Lance l'evaluation RAGAS et affiche les scores."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = Dataset.from_list(rows)
    # RAGAS utilise un LLM pour evaluer.  Par defaut il utilise OpenAI ;
    # on peut le surcharger avec langchain-mistralai si MISTRAL_API_KEY
    # est disponible.  Si aucun LLM n'est configure, ragas leve une erreur
    # explicite.
    result = evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_precision])

    print("\n--- Scores RAGAS ---")
    for metric_name, score in result.items():
        print(f"  {metric_name}: {score:.4f}")

    if output_path is not None:
        scores = {k: round(v, 4) for k, v in result.items()}
        output_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nScores sauvegardes dans {output_path}")


def main() -> None:
    try:
        from datasets import Dataset  # noqa: F401
        from ragas import evaluate  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "RAGAS n'est pas installe. Lance `uv sync --extra eval` pour installer les dependances d'evaluation."
        ) from exc

    args = parse_args()

    if args.mode == "live":
        print("Mode live : interrogation du RAG reel pour chaque question...")
        try:
            rows = load_rows(args.input, required_fields=LIVE_REQUIRED_FIELDS)
            rows = enrich_with_rag(rows)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        print("Mode static : lecture des donnees pre-calculees...")
        try:
            rows = load_rows(args.input, required_fields=STATIC_REQUIRED_FIELDS)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    run_evaluation(rows, args.output)


if __name__ == "__main__":
    main()
