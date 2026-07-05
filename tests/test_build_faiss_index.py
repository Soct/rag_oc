from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "build_faiss_index.py"
SPEC = importlib.util.spec_from_file_location("build_faiss_index", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_chunk_records = MODULE.build_chunk_records
build_faiss_index = MODULE.build_faiss_index
split_text = MODULE.split_text


class SplitTextTests(unittest.TestCase):
    def test_splits_long_text_with_overlap(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"

        chunks = split_text(text, chunk_size=18, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk_text for chunk_text, _, _ in chunks))
        self.assertLess(chunks[0][1], chunks[1][1])


class BuildChunkRecordsTests(unittest.TestCase):
    def test_uses_existing_chunks_when_present(self) -> None:
        records = [
            {
                "id": "openagenda:1",
                "event_uid": 1,
                "title": "Atelier",
                "date_summary": "2026-07-05",
                "location": {"city": "Paris"},
                "source_agendas": [{"uid": 10}],
                "document": "document complet",
                "chunks": [
                    {"text": "premier chunk", "start": 0, "end": 13},
                    {"text": "second chunk", "start": 14, "end": 26},
                ],
            }
        ]

        chunk_records = build_chunk_records(records, text_field="document", chunk_size=50, chunk_overlap=10)

        self.assertEqual(len(chunk_records), 2)
        self.assertEqual(chunk_records[0].metadata["chunk_id"], "openagenda:1:chunk:0")
        self.assertEqual(chunk_records[1].text, "second chunk")
        self.assertEqual(chunk_records[0].metadata["chunk_start"], 0)

    def test_builds_chunks_from_document_when_missing(self) -> None:
        records = [
            {
                "id": "openagenda:2",
                "event_uid": 2,
                "title": "Conference",
                "date_summary": "2026-07-05",
                "location": {"city": "Paris"},
                "source_agendas": [{"uid": 20}],
                "document": " ".join(f"mot{i}" for i in range(40)),
            }
        ]

        chunk_records = build_chunk_records(records, text_field="document", chunk_size=40, chunk_overlap=8)

        self.assertGreater(len(chunk_records), 1)
        self.assertTrue(all(item.metadata["chunk_text"] for item in chunk_records))
        self.assertEqual(chunk_records[0].metadata["id"], "openagenda:2")


class BuildFaissIndexTests(unittest.TestCase):
    def test_builds_inner_product_index_with_normalized_vectors(self) -> None:
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ],
            dtype="float32",
        )
        query = np.asarray([[0.9, 0.1, 0.0]], dtype="float32")

        index = build_faiss_index(vectors, index_type="flat", ivf_nlist=4)

        import faiss

        faiss.normalize_L2(query)
        distances, indices = index.search(query, 2)

        self.assertEqual(index.ntotal, 3)
        self.assertEqual(int(indices[0][0]), 0)
        self.assertGreater(float(distances[0][0]), float(distances[0][1]))


if __name__ == "__main__":
    unittest.main()
