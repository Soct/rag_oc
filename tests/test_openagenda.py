from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from datetime import UTC, datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "fetch_openagenda_events.py"
SPEC = importlib.util.spec_from_file_location("fetch_openagenda_events", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

AgendaRef = MODULE.AgendaRef
compute_date_window = MODULE.compute_date_window
normalize_event = MODULE.normalize_event
should_keep_event = MODULE.should_keep_event
extract_source_agendas = MODULE.extract_source_agendas


class ComputeDateWindowTests(unittest.TestCase):
    def test_clamps_date_from_to_last_365_days(self) -> None:
        now = datetime(2026, 7, 2, 12, 0, tzinfo=UTC)
        window = compute_date_window(
            date_from="2024-01-01T00:00:00+00:00",
            date_to="2026-07-10T00:00:00+00:00",
            now=now,
        )

        self.assertEqual(window.start, datetime(2025, 7, 2, 12, 0, tzinfo=UTC))
        self.assertEqual(window.end, datetime(2026, 7, 10, 0, 0, tzinfo=UTC))


class NormalizeEventTests(unittest.TestCase):
    def test_builds_rag_ready_record(self) -> None:
        agenda = AgendaRef(uid=123, title="Agenda Culture", slug="agenda-culture")
        event = {
            "uid": 987,
            "slug": "concert-jazz",
            "title": "Concert de jazz",
            "description": "Soirée musicale",
            "longDescription": "Un trio en live.",
            "conditions": "Entrée libre",
            "keywords": {"fr": ["concert", "jazz"]},
            "timings": [
                {
                    "begin": "2026-07-03T18:00:00.000Z",
                    "end": "2026-07-03T20:00:00.000Z",
                }
            ],
            "attendanceMode": 1,
            "onlineAccessLink": None,
            "registration": [],
            "accessibility": {"vi": True},
            "status": 1,
            "state": 2,
            "createdAt": "2026-06-20T10:00:00.000Z",
            "updatedAt": "2026-06-21T10:00:00.000Z",
            "locationUid": 456,
            "location": {
                "name": "Salle des fêtes",
                "address": "1 rue Exemple",
                "adminLevel4": "Paris",
                "adminLevel2": "Paris",
                "adminLevel1": "Île-de-France",
                "postalCode": "75001",
                "latitude": 48.86,
                "longitude": 2.34,
            },
        }

        record = normalize_event(event, agenda)

        self.assertEqual(record["id"], "openagenda:987")
        self.assertEqual(record["event_types"], ["concert", "jazz"])
        self.assertEqual(record["location"]["region"], "Île-de-France")
        self.assertIn("Concert de jazz", record["document"])
        self.assertEqual(record["source_agendas"][0]["uid"], 123)
        self.assertEqual(record["occurrences_count"], 1)
        self.assertNotIn("accessibility", record)
        self.assertNotIn("registration", record)


class EventFilteringTests(unittest.TestCase):
    def test_excludes_concert_keyword(self) -> None:
        record = {"event_types": ["concert", "jazz"]}
        self.assertFalse(should_keep_event(record, ["concert"]))

    def test_keeps_non_concert_event(self) -> None:
        record = {"event_types": ["exposition", "musee"]}
        self.assertTrue(should_keep_event(record, ["concert"]))


class TransverseNormalizationTests(unittest.TestCase):
    def test_extracts_agendas_from_transverse_event(self) -> None:
        event = {
            "agendas": [
                {"uid": 1, "title": "Agenda A", "slug": "agenda-a"},
                {"uid": 2, "title": "Agenda B", "slug": "agenda-b"},
            ]
        }
        agendas = extract_source_agendas(event, None)
        self.assertEqual(len(agendas), 2)
        self.assertEqual(agendas[0]["uid"], 1)


if __name__ == "__main__":
    unittest.main()
