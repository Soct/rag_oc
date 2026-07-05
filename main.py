def main():
    print("Collecte: `uv run python scripts/fetch_openagenda_events.py --help`.")
    print("Indexation: `uv run python scripts/build_faiss_index.py --help`.")
    print("Chatbot RAG: `uv run python scripts/chat_openagenda.py --help`.")
    print("Evaluation: `uv run python scripts/evaluate_rag.py --help`.")
    print("Smoke test API: `uv run python scripts/api_smoke_test.py`.")
    print("API REST: `uv run uvicorn rag_oc.api:app --reload`.")


if __name__ == "__main__":
    main()
