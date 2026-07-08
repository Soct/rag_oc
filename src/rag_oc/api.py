"""API REST FastAPI pour le systeme RAG OpenAgenda.

Endpoints :
- GET  /health  : healthcheck
- POST /ask     : poser une question au moteur RAG
- POST /rebuild : reconstruire l'index FAISS a la demande

Le service RAG est initialise paresseusement (lazy) au premier appel
a ``/ask`` et reutilise ensuite.  Apres un ``/rebuild``, l'index en
memoire est recharge automatiquement.
"""
from __future__ import annotations

from pathlib import Path

from typing import Annotated

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_oc.rag_service import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INDEX_PATH,
    DEFAULT_MAX_CONTEXT_ITEMS,
    DEFAULT_METADATA_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    RagService,
    resolve_api_key,
)
from rag_oc.build_faiss_index import IndexBuildConfig, rebuild_index

APP_DESCRIPTION = (
    "API REST locale pour interroger le systeme RAG OpenAgenda et reconstruire l'index FAISS a la demande."
)

ASK_REQUEST_EXAMPLES = {
    "atelier_paris_weekend": {
        "summary": "Atelier a Paris ce week-end",
        "description": "Scenario de demo centre sur une recherche d'activite creative a court terme.",
        "value": {
            "question": "Je cherche un atelier creatif a Paris ce week-end",
            "top_k": 5,
            "max_context_items": 4,
            "temperature": 0.2,
        },
    },
    "sortie_famille": {
        "summary": "Sortie famille prochaine",
        "description": "Scenario oriente famille avec un contexte un peu plus large.",
        "value": {
            "question": "Quels evenements famille sont proposes prochainement en Ile-de-France ?",
            "top_k": 6,
            "max_context_items": 5,
            "temperature": 0.1,
        },
    },
    "expo_gratuite": {
        "summary": "Exposition gratuite",
        "description": "Scenario utile pour verifier la recherche semantique sur des contraintes simples.",
        "value": {
            "question": "Je veux une exposition gratuite en Ile-de-France",
            "top_k": 5,
            "max_context_items": 4,
            "temperature": 0.0,
        },
    },
}

REBUILD_REQUEST_EXAMPLES = {
    "rebuild_standard": {
        "summary": "Rebuild standard",
        "description": "Reconstruit l'index FAISS plat avec les chemins par defaut.",
        "value": {
            "input_path": "data/openagenda/full.jsonl.gz",
            "index_output": "data/faiss/openagenda.index",
            "metadata_output": "data/faiss/openagenda_metadata.pkl",
            "index_type": "flat",
        },
    },
    "rebuild_ivf": {
        "summary": "Rebuild IVF-Flat",
        "description": "Scenario de test pour un index IVF avec parametres plus prudents.",
        "value": {
            "input_path": "data/openagenda/full.jsonl.gz",
            "index_output": "data/faiss/openagenda_ivf.index",
            "metadata_output": "data/faiss/openagenda_ivf_metadata.pkl",
            "index_type": "ivfflat",
            "ivf_nlist": 128,
            "batch_size": 16,
            "request_pause_seconds": 1.0,
        },
    },
}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question utilisateur a poser au systeme RAG.")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)
    max_context_items: int = Field(default=DEFAULT_MAX_CONTEXT_ITEMS, ge=1, le=20)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    chat_model: str = Field(default=DEFAULT_CHAT_MODEL)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    context: str


class RebuildRequest(BaseModel):
    input_path: str = Field(default=str(Path("data/openagenda/full.jsonl.gz")))
    index_output: str = Field(default=str(DEFAULT_INDEX_PATH))
    metadata_output: str = Field(default=str(DEFAULT_METADATA_PATH))
    model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    batch_size: int = Field(default=32, ge=1)
    text_field: str = Field(default="document")
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=120, ge=0)
    index_type: str = Field(default="flat")
    ivf_nlist: int = Field(default=256, ge=1)
    max_retries: int = Field(default=6, ge=0)
    retry_base_seconds: float = Field(default=2.0, ge=0.0)
    request_pause_seconds: float = Field(default=0.25, ge=0.0)


class RebuildResponse(BaseModel):
    index_path: str
    metadata_path: str
    chunks_indexed: int
    source_records: int
    index_type: str
    embedding_model: str


class ApiState:
    def __init__(self) -> None:
        self.service: RagService | None = None


app = FastAPI(
    title="OpenAgenda RAG API",
    description=APP_DESCRIPTION,
    version="0.1.0",
    openapi_tags=[
        {
            "name": "demo",
            "description": "Endpoints documentes avec des scenarios prets a rejouer dans Swagger.",
        }
    ],
)
app.state.api_state = ApiState()


def get_service() -> RagService:
    """Initialisation paresseuse du service RAG.

    Le service est cree au premier appel puis mis en cache dans
    ``app.state``.  Cela evite de charger l'index FAISS au demarrage
    si aucune requete n'est faite.
    """
    state: ApiState = app.state.api_state
    if state.service is None:
        api_key = resolve_api_key()
        if not api_key:
            raise HTTPException(status_code=500, detail="Cle Mistral introuvable.")
        from mistralai.client import Mistral

        client = Mistral(api_key=api_key)
        try:
            state.service = RagService.from_paths(client=client)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return state.service


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question au systeme RAG",
    description="Le formulaire Swagger propose plusieurs scenarios de test preconfigures via le selecteur `Examples`.",
    tags=["demo"],
)
def ask(
    request: Annotated[
        AskRequest,
        Body(openapi_examples=ASK_REQUEST_EXAMPLES),
    ],
) -> AskResponse:
    try:
        result = get_service().ask(
            request.question,
            top_k=request.top_k,
            max_context_items=request.max_context_items,
            temperature=request.temperature,
            chat_model=request.chat_model,
            embedding_model=request.embedding_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur interne RAG: {exc}") from exc

    return AskResponse(answer=result.answer, sources=result.matches, context=result.context)


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire l'index FAISS",
    description="Le formulaire Swagger propose des scenarios standards pour rejouer une reconstruction d'index.",
    tags=["demo"],
)
def rebuild(
    request: Annotated[
        RebuildRequest,
        Body(openapi_examples=REBUILD_REQUEST_EXAMPLES),
    ],
) -> RebuildResponse:
    if request.chunk_overlap >= request.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap doit etre strictement inferieur a chunk_size.")
    if request.index_type not in {"flat", "ivfflat"}:
        raise HTTPException(status_code=400, detail="index_type doit valoir `flat` ou `ivfflat`.")

    api_key = resolve_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="Cle Mistral introuvable.")

    try:
        result = rebuild_index(
            IndexBuildConfig(
                input_path=Path(request.input_path),
                index_output=Path(request.index_output),
                metadata_output=Path(request.metadata_output),
                api_key=api_key,
                model=request.model,
                batch_size=request.batch_size,
                text_field=request.text_field,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                index_type=request.index_type,
                ivf_nlist=request.ivf_nlist,
                max_retries=request.max_retries,
                retry_base_seconds=request.retry_base_seconds,
                request_pause_seconds=request.request_pause_seconds,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erreur pendant la reconstruction: {exc}") from exc

    state: ApiState = app.state.api_state
    if state.service is not None:
        state.service.reload(index_path=Path(result["index_path"]), metadata_path=Path(result["metadata_path"]))

    return RebuildResponse(**result)
