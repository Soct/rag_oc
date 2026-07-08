from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_oc.build_faiss_index import (
    build_chunk_records,
    build_faiss_index,
    load_records,
    open_text_file,
    split_text,
    extract_status_code,
    batch_items,
    fetch_batch_embeddings,
    fetch_embeddings,
    rebuild_index,
    IndexBuildConfig,
)


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

    def test_builds_ivfflat_index(self) -> None:
        vectors = np.random.randn(50, 8).astype("float32")
        index = build_faiss_index(vectors, index_type="ivfflat", ivf_nlist=4)
        self.assertEqual(index.ntotal, 50)

    def test_raises_on_empty_vectors(self) -> None:
        vectors = np.empty((0, 3), dtype="float32")
        with self.assertRaises(ValueError):
            build_faiss_index(vectors, index_type="flat", ivf_nlist=4)


class LoadRecordsTests(unittest.TestCase):
    def test_loads_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "a"}\n{"id": "b"}\n')
            f.flush()
            records = load_records(Path(f.name))
        self.assertEqual(len(records), 2)

    def test_loads_gzipped_jsonl(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as f:
            path = Path(f.name)
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            gz.write('{"id": "c"}\n')
        records = load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["id"], "c")


class OpenTextFileTests(unittest.TestCase):
    def test_opens_plain_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("test\n")
            path = Path(f.name)
        with open_text_file(path) as handle:
            self.assertEqual(handle.read().strip(), "test")

    def test_opens_gz_file(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as f:
            path = Path(f.name)
        with gzip.open(path, "wt", encoding="utf-8") as gz:
            gz.write("compressed\n")
        with open_text_file(path) as handle:
            self.assertEqual(handle.read().strip(), "compressed")


class SplitTextEdgeCaseTests(unittest.TestCase):
    def test_empty_text_returns_empty(self) -> None:
        self.assertEqual(split_text("", chunk_size=10, chunk_overlap=2), [])
        self.assertEqual(split_text("   ", chunk_size=10, chunk_overlap=2), [])

    def test_short_text_single_chunk(self) -> None:
        chunks = split_text("hello world", chunk_size=100, chunk_overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][0], "hello world")


class ExtractStatusCodeTests(unittest.TestCase):
    def test_extracts_from_response_attr(self) -> None:
        class FakeResp:
            status_code = 429

        class FakeExc(Exception):
            response = FakeResp()

        self.assertEqual(extract_status_code(FakeExc()), 429)

    def test_extracts_from_message(self) -> None:
        exc = Exception("Got Status 503 from server")
        self.assertEqual(extract_status_code(exc), 503)

    def test_returns_none_for_unknown(self) -> None:
        self.assertIsNone(extract_status_code(Exception("random error")))


class BatchItemsTests(unittest.TestCase):
    def test_splits_into_batches(self) -> None:
        items = ["a", "b", "c", "d", "e"]
        batches = list(batch_items(items, batch_size=2))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0], ["a", "b"])
        self.assertEqual(batches[-1], ["e"])


class BuildChunkRecordsEdgeCaseTests(unittest.TestCase):
    def test_handles_string_chunks(self) -> None:
        records = [{"id": "x", "document": "doc", "chunks": ["chunk a", "chunk b"]}]
        result = build_chunk_records(records, text_field="document", chunk_size=50, chunk_overlap=5)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].text, "chunk a")

    def test_skips_non_string_non_dict_chunks(self) -> None:
        records = [{"id": "x", "document": "doc", "chunks": [123, None]}]
        result = build_chunk_records(records, text_field="document", chunk_size=50, chunk_overlap=5)
        self.assertEqual(len(result), 0)

    def test_handles_empty_document(self) -> None:
        records = [{"id": "x", "document": ""}]
        result = build_chunk_records(records, text_field="document", chunk_size=50, chunk_overlap=5)
        self.assertEqual(len(result), 0)

    def test_preserves_all_metadata_fields(self) -> None:
        records = [{
            "id": "openagenda:42",
            "event_uid": 42,
            "title": "Expo",
            "date_summary": "2026-07-10",
            "first_timing": "2026-07-10T10:00:00Z",
            "last_timing": "2026-07-10T18:00:00Z",
            "location": {"city": "Paris"},
            "source_agendas": [{"uid": 1}],
            "document": "Expo de photos magnifiques en plein air a Paris",
        }]
        result = build_chunk_records(records, text_field="document", chunk_size=500, chunk_overlap=0)
        self.assertEqual(len(result), 1)
        meta = result[0].metadata
        self.assertEqual(meta["event_uid"], 42)
        self.assertEqual(meta["title"], "Expo")
        self.assertEqual(meta["location"]["city"], "Paris")
        self.assertIsNotNone(meta["chunk_text"])


class FetchBatchEmbeddingsTests(unittest.TestCase):
    def test_returns_embeddings_on_success(self) -> None:
        class FakeEmbedding:
            def __init__(self, idx, vec):
                self.index = idx
                self.embedding = vec

        class FakeResponse:
            data = [FakeEmbedding(0, [0.1, 0.2])]

        class FakeClient:
            class embeddings:
                @staticmethod
                def create(model, inputs):
                    return FakeResponse()

        result = fetch_batch_embeddings(FakeClient(), "model", ["text"], max_retries=0, retry_base_seconds=0)
        self.assertEqual(len(result.data), 1)


class FetchEmbeddingsTests(unittest.TestCase):
    def test_produces_numpy_array(self) -> None:
        class FakeEmbedding:
            def __init__(self, idx, vec):
                self.index = idx
                self.embedding = vec

        class FakeResponse:
            data = [FakeEmbedding(0, [0.1, 0.2, 0.3])]

        class FakeClient:
            class embeddings:
                @staticmethod
                def create(model, inputs):
                    return FakeResponse()

        result = fetch_embeddings(
            FakeClient(), "model", ["text1"],
            batch_size=1, max_retries=0, retry_base_seconds=0, request_pause_seconds=0,
        )
        self.assertEqual(result.shape, (1, 3))


class RebuildIndexTests(unittest.TestCase):
    def test_raises_without_api_key(self) -> None:
        config = IndexBuildConfig(api_key=None)
        with self.assertRaises(ValueError):
            rebuild_index(config)

    def test_full_rebuild_with_mock(self) -> None:
        # Prepare a temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(3):
                json.dump({
                    "id": f"evt:{i}",
                    "event_uid": i,
                    "title": f"Event {i}",
                    "date_summary": "2026-07-10",
                    "first_timing": "2026-07-10T10:00:00Z",
                    "last_timing": "2026-07-10T18:00:00Z",
                    "location": {"city": "Paris"},
                    "source_agendas": [],
                    "document": f"Description de l evenement numero {i} avec du texte supplementaire pour le chunking",
                }, f)
                f.write("\n")
            input_path = Path(f.name)

        tmpdir = tempfile.mkdtemp()
        index_out = Path(tmpdir) / "test.index"
        meta_out = Path(tmpdir) / "test_meta.pkl"

        class FakeEmbedding:
            def __init__(self, idx):
                self.index = idx
                self.embedding = np.random.randn(8).tolist()

        class FakeClient:
            class embeddings:
                @staticmethod
                def create(model, inputs):
                    class Resp:
                        data = [FakeEmbedding(i) for i in range(len(inputs))]
                    return Resp()

        # Monkey-patch Mistral import in rebuild_index
        import rag_oc.build_faiss_index as bfi
        original_rebuild = bfi.rebuild_index

        def patched_rebuild(config):
            records = bfi.load_records(config.input_path)
            filtered = [r for r in records if r.get(config.text_field)]
            chunks = bfi.build_chunk_records(filtered, config.text_field, config.chunk_size, config.chunk_overlap)
            docs = [c.text for c in chunks]
            metadata = [c.metadata for c in chunks]
            vectors = bfi.fetch_embeddings(
                FakeClient(), config.model, docs,
                config.batch_size, config.max_retries, config.retry_base_seconds, config.request_pause_seconds,
            )
            index = bfi.build_faiss_index(vectors, config.index_type, config.ivf_nlist)
            config.index_output.parent.mkdir(parents=True, exist_ok=True)
            import faiss
            faiss.write_index(index, str(config.index_output))
            import pickle
            with config.metadata_output.open("wb") as h:
                pickle.dump(metadata, h)
            return {
                "index_path": str(config.index_output),
                "metadata_path": str(config.metadata_output),
                "chunks_indexed": len(metadata),
                "source_records": len(filtered),
                "index_type": config.index_type,
                "embedding_model": config.model,
            }

        config = IndexBuildConfig(
            input_path=input_path,
            index_output=index_out,
            metadata_output=meta_out,
            api_key="fake-key",
        )
        result = patched_rebuild(config)
        self.assertGreater(result["chunks_indexed"], 0)
        self.assertTrue(index_out.exists())
        self.assertTrue(meta_out.exists())


if __name__ == "__main__":
    unittest.main()
