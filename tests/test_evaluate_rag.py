from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag import load_rows, validate_row, STATIC_REQUIRED_FIELDS, LIVE_REQUIRED_FIELDS


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


if __name__ == "__main__":
    unittest.main()
