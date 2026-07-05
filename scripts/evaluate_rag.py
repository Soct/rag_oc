from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_FIELDS = ("question", "answer", "ground_truth", "contexts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation RAG avec Ragas a partir d'un fichier JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Fichier JSONL avec `question`, `answer`, `ground_truth` et `contexts`.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row, line_number=line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"Aucune ligne exploitable dans {path}.")
    return rows


def validate_row(row: dict, *, line_number: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Ligne {line_number}: champs manquants: {', '.join(missing)}.")

    for field in ("question", "answer", "ground_truth"):
        value = row[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Ligne {line_number}: `{field}` doit etre une chaine non vide.")

    contexts = row["contexts"]
    if not isinstance(contexts, list) or not contexts:
        raise ValueError(f"Ligne {line_number}: `contexts` doit etre une liste non vide.")
    if not all(isinstance(item, str) and item.strip() for item in contexts):
        raise ValueError(f"Ligne {line_number}: chaque element de `contexts` doit etre une chaine non vide.")


def main() -> None:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise SystemExit(
            "Ragas n'est pas installe. Ajoute les dependances d'evaluation avant d'executer scripts/evaluate_rag.py."
        ) from exc

    args = parse_args()
    try:
        rows = load_rows(args.input)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[answer_relevancy, faithfulness, context_precision])
    print(result)


if __name__ == "__main__":
    main()
