from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datetime import UTC, datetime

from rag_oc.rag_service import build_messages, filter_matches_by_date, format_context_items, search_index


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


if __name__ == "__main__":
    unittest.main()
