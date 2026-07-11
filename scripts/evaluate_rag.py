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
import math
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Nombre maximal de metriques evaluees en parallele (defaut : 1).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=0.2,
        help="Debit maximal des appels Mistral pendant l'evaluation (defaut : 0.2).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Delai maximal par metrique RAGAS en secondes (defaut : 600).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Nombre maximal de nouvelles tentatives apres une erreur API (defaut : 5).",
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers doit etre superieur ou egal a 1.")
    if args.requests_per_second <= 0:
        parser.error("--requests-per-second doit etre strictement positif.")
    if args.timeout < 1:
        parser.error("--timeout doit etre superieur ou egal a 1.")
    if args.max_retries < 0:
        parser.error("--max-retries doit etre superieur ou egal a 0.")
    return args


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


def build_ragas_dependencies(*, requests_per_second: float):
    """Configure RAGAS avec les modeles Mistral deja utilises par le projet.

    RAGAS ne doit pas choisir son fournisseur par defaut (OpenAI). En plus de
    demander une cle qui n'est pas celle du projet, ce comportement peut
    instancier un adaptateur incompatible avec la version de LangChain.
    """
    from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_core.outputs import LLMResult
    from ragas.embeddings.base import LangchainEmbeddingsWrapper
    from ragas.llms.base import LangchainLLMWrapper

    from rag_oc.rag_service import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, resolve_api_key

    api_key = resolve_api_key()
    if not api_key:
        raise ValueError(
            "Cle Mistral introuvable. RAGAS utilise les modeles Mistral du projet ; "
            "definissez MISTRAL_API_KEY (ou ajoutez-la dans .env)."
        )

    class SequentialLangchainLLMWrapper(LangchainLLMWrapper):
        """Evite le regroupement defectueux de sorties dans langchain-mistralai.

        ``answer_relevancy`` demande plusieurs generations pour chaque ligne.
        Avec les versions installees, ``ChatMistralAI.agenerate_prompt`` echoue
        lorsqu'il tente de combiner plusieurs usages de tokens. Les appels sont
        donc effectues un par un, sans modifier la metrique ni son strictness.
        """

        async def agenerate_text(
            self,
            prompt,
            n: int = 1,
            temperature: float | None = 0.01,
            stop=None,
            callbacks=None,
        ):
            if n == 1:
                return await super().agenerate_text(
                    prompt,
                    n=1,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                )

            results = []
            for _ in range(n):
                results.append(
                    await super().agenerate_text(
                        prompt,
                        n=1,
                        temperature=temperature,
                        stop=stop,
                        callbacks=callbacks,
                    )
                )
            return LLMResult(generations=[[result.generations[0][0] for result in results]])

    llm = SequentialLangchainLLMWrapper(
        ChatMistralAI(
            api_key=api_key,
            model=DEFAULT_CHAT_MODEL,
            temperature=0,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=requests_per_second,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            ),
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(api_key=api_key, model=DEFAULT_EMBEDDING_MODEL)
    )
    return llm, embeddings


def mean_scores(scores: list[dict]) -> dict[str, float]:
    """Calcule les moyennes RAGAS v0.3 sans dependre d'une API interne."""
    if not scores:
        raise ValueError("RAGAS n'a retourne aucun score.")

    averages: dict[str, float] = {}
    for metric_name in scores[0]:
        values = [
            float(row[metric_name])
            for row in scores
            if row.get(metric_name) is not None and not math.isnan(float(row[metric_name]))
        ]
        averages[metric_name] = sum(values) / len(values) if values else float("nan")
    return averages


def run_evaluation(
    rows: list[dict],
    output_path: Path | None,
    *,
    max_workers: int,
    requests_per_second: float,
    timeout: int,
    max_retries: int,
) -> None:
    """Lance l'evaluation RAGAS et affiche les scores."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    from ragas.run_config import RunConfig

    dataset = Dataset.from_list(rows)
    llm, embeddings = build_ragas_dependencies(requests_per_second=requests_per_second)
    result = evaluate(
        dataset,
        metrics=[answer_relevancy, faithfulness, context_precision],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            max_workers=max_workers,
            timeout=timeout,
            max_retries=max_retries,
            max_wait=120,
        ),
        raise_exceptions=True,
    )
    scores = mean_scores(result.scores)

    print("\n--- Scores RAGAS ---")
    for metric_name, score in scores.items():
        print(f"  {metric_name}: {score:.4f}")

    if output_path is not None:
        rounded_scores = {k: round(v, 4) for k, v in scores.items()}
        output_path.write_text(json.dumps(rounded_scores, indent=2, ensure_ascii=False), encoding="utf-8")
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

    run_evaluation(
        rows,
        args.output,
        max_workers=args.max_workers,
        requests_per_second=args.requests_per_second,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
