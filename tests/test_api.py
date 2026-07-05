from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import httpx

from rag_oc import api as api_module


class FakeService:
    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        max_context_items: int | None = None,
        temperature: float | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
    ):
        class Result:
            answer = f"Reponse pour: {question}"
            matches = [{"id": "openagenda:1", "title": "Atelier", "score": 0.99, "rank": 1}]
            context = "Contexte factice"

        return Result()

    def reload(self, *, index_path, metadata_path) -> None:
        self.reloaded = (index_path, metadata_path)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_get_service = api_module.get_service
        self.original_rebuild_index = api_module.rebuild_index
        self.original_resolve_api_key = api_module.resolve_api_key
        self.fake_service = FakeService()
        api_module.get_service = lambda: self.fake_service
        api_module.resolve_api_key = lambda: "test-key"

    def tearDown(self) -> None:
        api_module.get_service = self.original_get_service
        api_module.rebuild_index = self.original_rebuild_index
        api_module.resolve_api_key = self.original_resolve_api_key

    async def request(self, method: str, path: str, json: dict | None = None) -> httpx.Response:
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, json=json)

    def test_ask_returns_answer(self) -> None:
        response = asyncio.run(self.request("POST", "/ask", json={"question": "Que faire ce week-end ?"}))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("Reponse pour", payload["answer"])
        self.assertEqual(payload["sources"][0]["id"], "openagenda:1")

    def test_ask_rejects_empty_question(self) -> None:
        response = asyncio.run(self.request("POST", "/ask", json={"question": ""}))

        self.assertEqual(response.status_code, 422)

    def test_openapi_exposes_ask_scenarios(self) -> None:
        response = asyncio.run(self.request("GET", "/openapi.json"))

        self.assertEqual(response.status_code, 200)
        schema = response.json()
        examples = schema["paths"]["/ask"]["post"]["requestBody"]["content"]["application/json"]["examples"]
        self.assertIn("atelier_paris_weekend", examples)
        self.assertIn("sortie_famille", examples)
        self.assertIn("expo_gratuite", examples)

    def test_openapi_exposes_rebuild_scenarios(self) -> None:
        response = asyncio.run(self.request("GET", "/openapi.json"))

        self.assertEqual(response.status_code, 200)
        schema = response.json()
        examples = schema["paths"]["/rebuild"]["post"]["requestBody"]["content"]["application/json"]["examples"]
        self.assertIn("rebuild_standard", examples)
        self.assertIn("rebuild_ivf", examples)

    def test_rebuild_returns_metadata(self) -> None:
        def fake_rebuild_index(_config):
            return {
                "index_path": "data/faiss/openagenda.index",
                "metadata_path": "data/faiss/openagenda_metadata.pkl",
                "chunks_indexed": 12,
                "source_records": 8,
                "index_type": "flat",
                "embedding_model": "mistral-embed",
            }

        api_module.rebuild_index = fake_rebuild_index
        api_module.app.state.api_state.service = self.fake_service

        response = asyncio.run(self.request("POST", "/rebuild", json={}))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["chunks_indexed"], 12)
        self.assertEqual(self.fake_service.reloaded[0], Path("data/faiss/openagenda.index"))


if __name__ == "__main__":
    unittest.main()
