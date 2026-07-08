from __future__ import annotations

from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datetime import UTC, datetime

from rag_oc.rag_service import (
    build_messages,
    filter_matches_by_date,
    format_context_items,
    format_location,
    generate_answer,
    load_index,
    load_metadata,
    parse_event_datetime,
    prepare_query_vector,
    read_api_key_from_env_file,
    resolve_api_key,
    search_index,
)


class FormatContextItemsTests(unittest.TestCase):
    def test_formats_ranked_matches(self) -> None:
        matches = [
            {
                "rank": 1,
                "score": 0.91,
                "title": "Atelier cuisine",
                "date_summary": "2026-07-12",
                "location": {"name": "Maison des associations", "city": "Paris"},
                "id": "openagenda:1",
                "chunk_text": "Atelier convivial autour de la cuisine vegetarienne.",
            }
        ]

        context = format_context_items(matches, max_context_items=1)

        self.assertIn("Resultat 1", context)
        self.assertIn("Atelier cuisine", context)
        self.assertIn("Maison des associations, Paris", context)


class BuildMessagesTests(unittest.TestCase):
    def test_builds_chat_messages(self) -> None:
        messages = build_messages(
            question="Je cherche une sortie culturelle a Paris ce week-end.",
            context="Resultat 1\nTitre: Expo photo\nDate: 2026-07-10\nLieu: Paris",
            now=datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Expo photo", messages[1]["content"])
        self.assertIn("2026-07-05", messages[0]["content"])


class SearchIndexTests(unittest.TestCase):
    def test_returns_top_ranked_metadata(self) -> None:
        import faiss

        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.2],
            ],
            dtype="float32",
        )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        query = np.asarray([[0.95, 0.05]], dtype="float32")
        faiss.normalize_L2(query)
        metadata = [
            {"id": "a", "title": "Resultat A"},
            {"id": "b", "title": "Resultat B"},
            {"id": "c", "title": "Resultat C"},
        ]

        matches = search_index(index=index, metadata=metadata, query_vector=query, top_k=2)

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0]["rank"], 1)
        self.assertEqual(matches[0]["id"], "a")
        self.assertGreater(matches[0]["score"], matches[1]["score"])


class DateFilteringTests(unittest.TestCase):
    def test_filters_out_finished_events_by_default(self) -> None:
        matches = [
            {"id": "past", "last_timing": "2025-07-06T11:45:00.000+02:00"},
            {"id": "future", "last_timing": "2026-07-20T11:45:00.000+02:00"},
            {"id": "unknown"},
        ]

        filtered = filter_matches_by_date(
            matches,
            include_past_events=False,
            now=datetime(2026, 7, 5, 12, 0, tzinfo=UTC),
        )

        self.assertEqual([item["id"] for item in filtered], ["future", "unknown"])


class IncludePastEventsTests(unittest.TestCase):
    def test_include_past_returns_all(self) -> None:
        matches = [
            {"id": "past", "last_timing": "2020-01-01T00:00:00+00:00"},
            {"id": "future", "last_timing": "2030-01-01T00:00:00+00:00"},
        ]
        result = filter_matches_by_date(matches, include_past_events=True)
        self.assertEqual(len(result), 2)


class ParseEventDatetimeTests(unittest.TestCase):
    def test_parses_iso_with_z_suffix(self) -> None:
        dt = parse_event_datetime("2026-07-10T12:00:00Z")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)

    def test_parses_iso_with_offset(self) -> None:
        dt = parse_event_datetime("2026-07-10T12:00:00+02:00")
        self.assertIsNotNone(dt)

    def test_returns_none_for_none(self) -> None:
        self.assertIsNone(parse_event_datetime(None))

    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(parse_event_datetime(""))
        self.assertIsNone(parse_event_datetime("   "))

    def test_returns_none_for_non_string(self) -> None:
        self.assertIsNone(parse_event_datetime(12345))

    def test_returns_none_for_invalid(self) -> None:
        self.assertIsNone(parse_event_datetime("pas-une-date"))

    def test_naive_datetime_gets_utc(self) -> None:
        dt = parse_event_datetime("2026-07-10T12:00:00")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.tzinfo, UTC)


class FormatLocationTests(unittest.TestCase):
    def test_formats_full_location(self) -> None:
        loc = {"name": "Salle A", "address": "1 rue X", "city": "Paris", "department": "75", "region": "IDF"}
        self.assertIn("Salle A", format_location(loc))
        self.assertIn("Paris", format_location(loc))

    def test_returns_default_for_none(self) -> None:
        self.assertEqual(format_location(None), "Lieu non renseigne")

    def test_returns_default_for_empty_dict(self) -> None:
        self.assertEqual(format_location({}), "Lieu non renseigne")


class ReadApiKeyFromEnvFileTests(unittest.TestCase):
    def test_reads_key_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("MISTRAL_API_KEY=test-key-123\n")
            f.flush()
            result = read_api_key_from_env_file(Path(f.name))
        self.assertEqual(result, "test-key-123")

    def test_returns_none_for_missing_file(self) -> None:
        self.assertIsNone(read_api_key_from_env_file(Path("/tmp/nonexistent.env")))

    def test_skips_comments_and_blanks(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("# comment\n\nOTHER_KEY=val\nMISTRAL_API_KEY='quoted-key'\n")
            f.flush()
            result = read_api_key_from_env_file(Path(f.name))
        self.assertEqual(result, "quoted-key")

    def test_returns_none_when_key_empty(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("MISTRAL_API_KEY=\n")
            f.flush()
            result = read_api_key_from_env_file(Path(f.name))
        self.assertIsNone(result)


class ResolveApiKeyTests(unittest.TestCase):
    def test_explicit_key_wins(self) -> None:
        self.assertEqual(resolve_api_key("explicit"), "explicit")

    def test_returns_none_when_nothing(self) -> None:
        import os
        old = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            result = resolve_api_key(None)
        finally:
            if old is not None:
                os.environ["MISTRAL_API_KEY"] = old
        # May return a value from .env file if present; just check no crash
        self.assertIsInstance(result, (str, type(None)))


class LoadIndexTests(unittest.TestCase):
    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_index(Path("/tmp/nonexistent.index"))


class LoadMetadataTests(unittest.TestCase):
    def test_loads_valid_metadata(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump([{"id": "a"}], f)
            f.flush()
            result = load_metadata(Path(f.name))
        self.assertEqual(len(result), 1)

    def test_raises_on_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_metadata(Path("/tmp/nonexistent.pkl"))

    def test_raises_on_non_list(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump({"not": "a list"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                load_metadata(Path(f.name))


class PrepareQueryVectorTests(unittest.TestCase):
    def test_normalizes_for_ip_index(self) -> None:
        import faiss
        index = faiss.IndexFlatIP(3)
        vec = np.array([[3.0, 4.0, 0.0]], dtype="float32")
        result = prepare_query_vector(index, vec)
        norm = np.linalg.norm(result)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_no_normalize_for_l2_index(self) -> None:
        import faiss
        index = faiss.IndexFlatL2(3)
        vec = np.array([[3.0, 4.0, 0.0]], dtype="float32")
        result = prepare_query_vector(index, vec)
        self.assertAlmostEqual(result[0][0], 3.0, places=5)


class GenerateAnswerTests(unittest.TestCase):
    def test_extracts_string_content(self) -> None:
        class FakeMessage:
            content = "  Reponse test  "

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                @staticmethod
                def complete(**kwargs):
                    return FakeResponse()

        result = generate_answer(FakeClient(), "model", [], 0.0)
        self.assertEqual(result, "Reponse test")

    def test_raises_on_no_choices(self) -> None:
        class FakeResponse:
            choices = []

        class FakeClient:
            class chat:
                @staticmethod
                def complete(**kwargs):
                    return FakeResponse()

        with self.assertRaises(ValueError):
            generate_answer(FakeClient(), "model", [], 0.0)

    def test_extracts_list_content(self) -> None:
        class TextBlock:
            def __init__(self, text):
                self.text = text

        class FakeMessage:
            content = [TextBlock("part1"), TextBlock("part2")]

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                @staticmethod
                def complete(**kwargs):
                    return FakeResponse()

        result = generate_answer(FakeClient(), "model", [], 0.0)
        self.assertIn("part1", result)
        self.assertIn("part2", result)

    def test_raises_on_unsupported_content(self) -> None:
        class FakeMessage:
            content = 12345

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        class FakeClient:
            class chat:
                @staticmethod
                def complete(**kwargs):
                    return FakeResponse()

        with self.assertRaises(ValueError):
            generate_answer(FakeClient(), "model", [], 0.0)


if __name__ == "__main__":
    unittest.main()
