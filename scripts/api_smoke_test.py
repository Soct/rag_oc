from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rag_oc import api


class FakeService:
    def ask(self, question: str, **_kwargs):
        class Result:
            answer = f"Reponse factice pour: {question}"
            matches = [{"id": "openagenda:demo", "title": "Atelier demo", "score": 0.95, "rank": 1}]
            context = "Contexte de demonstration"

        return Result()


def main() -> None:
    original_get_service = api.get_service
    api.get_service = lambda: FakeService()

    try:
        transport = httpx.ASGITransport(app=api.app)
        async def run_requests():
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                health = await client.get("/health")
                print("GET /health", health.status_code, health.json())

                ask = await client.post("/ask", json={"question": "Je cherche un atelier creatif a Paris."})
                print("POST /ask", ask.status_code)
                print(ask.json())

        asyncio.run(run_requests())
    finally:
        api.get_service = original_get_service


if __name__ == "__main__":
    main()
