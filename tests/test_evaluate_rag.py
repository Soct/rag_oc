from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.evaluate_rag as evaluate_rag
from scripts.evaluate_rag import (
    LIVE_REQUIRED_FIELDS,
    STATIC_REQUIRED_FIELDS,
    load_rows,
    mean_scores,
    validate_row,
)


class EvaluateRagTests(unittest.TestCase):
    def write_jsonl(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "sample.jsonl"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_rows_accepts_valid_annotated_dataset(self) -> None:
        path = self.write_jsonl(
            '{"question":"Question 1","answer":"Reponse 1","ground_truth":"Reference 1","contexts":["Contexte 1"]}\n'
        )

        rows = load_rows(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["question"], "Question 1")

    def test_load_rows_rejects_missing_contexts(self) -> None:
        path = self.write_jsonl(
            '{"question":"Question 1","answer":"Reponse 1","ground_truth":"Reference 1"}\n'
        )

        with self.assertRaisesRegex(ValueError, "champs manquants"):
            load_rows(path)

    def test_load_rows_rejects_empty_context_list(self) -> None:
        path = self.write_jsonl(
            '{"question":"Question 1","answer":"Reponse 1","ground_truth":"Reference 1","contexts":[]}\n'
        )

        with self.assertRaisesRegex(ValueError, "`contexts` doit etre une liste non vide"):
            load_rows(path)

    def test_load_rows_live_mode_accepts_without_answer(self) -> None:
        path = self.write_jsonl(
            '{"question":"Question 1","ground_truth":"Reference 1"}\n'
        )
        rows = load_rows(path, required_fields=LIVE_REQUIRED_FIELDS)
        self.assertEqual(len(rows), 1)

    def test_validate_row_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "`question`"):
            validate_row(
                {"question": "", "answer": "a", "ground_truth": "g", "contexts": ["c"]},
                line_number=1,
            )

    def test_validate_row_rejects_empty_ground_truth(self) -> None:
        with self.assertRaisesRegex(ValueError, "`ground_truth`"):
            validate_row(
                {"question": "q", "answer": "a", "ground_truth": "  ", "contexts": ["c"]},
                line_number=1,
            )

    def test_load_rows_rejects_empty_file(self) -> None:
        path = self.write_jsonl("")
        with self.assertRaisesRegex(ValueError, "Aucune ligne"):
            load_rows(path)

    def test_mean_scores_calculates_metric_averages(self) -> None:
        scores = mean_scores(
            [
                {"faithfulness": 1.0, "answer_relevancy": 0.5},
                {"faithfulness": 0.5, "answer_relevancy": 1.0},
            ]
        )

        self.assertEqual(scores, {"faithfulness": 0.75, "answer_relevancy": 0.75})

    def test_mean_scores_ignores_missing_or_nan_values(self) -> None:
        scores = mean_scores(
            [
                {"faithfulness": float("nan")},
                {"faithfulness": 0.5},
            ]
        )

        self.assertEqual(scores["faithfulness"], 0.5)

    def test_parse_args_uses_safe_rate_limit_defaults(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["evaluate_rag.py", "--input", "sample.jsonl"],
        ):
            args = evaluate_rag.parse_args()

        self.assertEqual(args.max_workers, 1)
        self.assertEqual(args.requests_per_second, 0.2)
        self.assertEqual(args.timeout, 600)
        self.assertEqual(args.max_retries, 5)

    def test_build_ragas_dependencies_uses_sequential_llm_wrapper(self) -> None:
        class FakeChatMistralAI:
            def __init__(self, **kwargs) -> None:
                self.temperature = kwargs["temperature"]
                self.calls = 0

            async def agenerate_prompt(self, **kwargs):
                from langchain_core.outputs import Generation, LLMResult

                self.calls += 1
                return LLMResult(generations=[[Generation(text="ok")]])

        class FakeEmbeddings:
            def __init__(self, **kwargs) -> None:
                pass

        with (
            patch("rag_oc.rag_service.resolve_api_key", return_value="test-key"),
            patch("langchain_mistralai.ChatMistralAI", FakeChatMistralAI),
            patch("langchain_mistralai.MistralAIEmbeddings", FakeEmbeddings),
        ):
            llm, _ = evaluate_rag.build_ragas_dependencies(requests_per_second=0.2)

        self.assertEqual(type(llm).__name__, "SequentialLangchainLLMWrapper")
        result = asyncio.run(llm.agenerate_text(object(), n=3))
        self.assertEqual(llm.langchain_llm.calls, 3)
        self.assertEqual(len(result.generations[0]), 3)


if __name__ == "__main__":
    unittest.main()
