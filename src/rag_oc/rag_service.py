from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_INDEX_PATH = Path("data/faiss/openagenda.index")
DEFAULT_METADATA_PATH = Path("data/faiss/openagenda_metadata.pkl")
DEFAULT_EMBEDDING_MODEL = "mistral-embed"
DEFAULT_CHAT_MODEL = "mistral-small-latest"
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_ITEMS = 4
DEFAULT_TEMPERATURE = 0.2
ENV_FILE_PATH = Path(".env")
API_ENV_KEYS = ("MISTRAL_API_KEY",)
ROLE_MAPPING = {
    "human": "user",
    "ai": "assistant",
}


def read_api_key_from_env_file(path: Path = ENV_FILE_PATH) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in API_ENV_KEYS:
            continue
        value = value.strip().strip("'").strip('"')
        if value:
            return value
    return None


def resolve_api_key(explicit_api_key: str | None = None) -> str | None:
    if explicit_api_key:
        return explicit_api_key
    for env_key in API_ENV_KEYS:
        env_value = os.getenv(env_key)
        if env_value:
            return env_value
    return read_api_key_from_env_file(ENV_FILE_PATH)


def load_index(index_path: Path):
    import faiss

    if not index_path.exists():
        raise FileNotFoundError(f"Index introuvable: {index_path}")
    return faiss.read_index(str(index_path))


def load_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadonnees introuvables: {metadata_path}")
    with metadata_path.open("rb") as handle:
        metadata = pickle.load(handle)
    if not isinstance(metadata, list):
        raise ValueError("Le fichier de metadonnees doit contenir une liste.")
    return metadata


def parse_event_datetime(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def extract_query_vector(client, model: str, question: str):
    import numpy as np

    response = client.embeddings.create(model=model, inputs=[question])
    if not response.data:
        raise ValueError("Aucun embedding recu pour la question.")
    return np.asarray([response.data[0].embedding], dtype="float32")


def prepare_query_vector(index, vector):
    import faiss
    import numpy as np

    query_vector = np.asarray(vector, dtype="float32").copy()
    if getattr(index, "metric_type", None) == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(query_vector)
    return query_vector


def search_index(index, metadata: list[dict], query_vector, top_k: int) -> list[dict]:
    distances, indices = index.search(query_vector, top_k)
    matches: list[dict] = []
    for rank, (score, metadata_index) in enumerate(zip(distances[0], indices[0]), start=1):
        if metadata_index < 0 or metadata_index >= len(metadata):
            continue
        item = dict(metadata[metadata_index])
        item["rank"] = rank
        item["score"] = float(score)
        matches.append(item)
    return matches


def filter_matches_by_date(
    matches: list[dict],
    *,
    include_past_events: bool,
    now: datetime | None = None,
) -> list[dict]:
    if include_past_events:
        return matches

    reference_now = (now or datetime.now(UTC)).astimezone(UTC)
    filtered: list[dict] = []
    for item in matches:
        last_timing = parse_event_datetime(item.get("last_timing"))
        if last_timing is None or last_timing >= reference_now:
            filtered.append(item)
    return filtered


def format_location(location: dict | None) -> str:
    if not location:
        return "Lieu non renseigne"
    parts = [
        location.get("name"),
        location.get("address"),
        location.get("city"),
        location.get("department"),
        location.get("region"),
    ]
    values = [str(value).strip() for value in parts if value]
    return ", ".join(values) if values else "Lieu non renseigne"


def format_context_items(matches: list[dict], max_context_items: int) -> str:
    lines: list[str] = []
    for item in matches[:max_context_items]:
        lines.append(f"Resultat {item['rank']} (score={item['score']:.4f})")
        lines.append(f"Titre: {item.get('title') or 'Titre non renseigne'}")
        lines.append(f"Date: {item.get('date_summary') or 'Date non renseignee'}")
        lines.append(f"Lieu: {format_location(item.get('location'))}")
        lines.append(f"Source: {item.get('id') or 'id inconnu'}")
        chunk_text = item.get("chunk_text") or item.get("document") or ""
        lines.append(f"Contenu: {chunk_text}")
        lines.append("")
    return "\n".join(lines).strip()


def build_messages(question: str, context: str, now: datetime | None = None):
    reference_now = (now or datetime.now(UTC)).astimezone(UTC)
    current_date = reference_now.strftime("%Y-%m-%d")
    system_prompt = (
        "Tu es un assistant de recommandation d'evenements. "
        "Tu reponds uniquement a partir du contexte fourni. "
        "Si le contexte est insuffisant, dis-le clairement. "
        "Propose des recommandations concretes, cite les titres, dates et lieux quand ils sont disponibles. "
        "N'invente jamais d'evenement absent du contexte. "
        f"La date courante est {current_date}. "
        "Interprete les expressions temporelles relatives comme `aujourd'hui`, `demain`, `ce week-end`, "
        "`la semaine prochaine` ou `ce mois` par rapport a cette date courante. "
        "Quand plusieurs evenements sont pertinents, privilegie ceux dont la date correspond le mieux "
        "a la demande temporelle de l'utilisateur. "
        "Si le contexte ne contient pas d'evenement dans la bonne periode, dis-le clairement."
    )
    user_prompt = (
        f"Question utilisateur:\n{question}\n\n"
        f"Contexte d'evenements:\n{context}\n\n"
        "Redige une reponse naturelle en francais. "
        "Si plusieurs options sont pertinentes, recommande les plus adaptees a la demande."
    )

    try:
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Question utilisateur:\n{question}\n\nContexte d'evenements:\n{context}\n\n{instruction}"),
        ]
    )
    rendered = prompt.format_messages(
        question=question,
        context=context,
        instruction=(
            "Redige une reponse naturelle en francais. "
            "Si plusieurs options sont pertinentes, recommande les plus adaptees a la demande."
        ),
    )
    return [
        {"role": ROLE_MAPPING.get(message.type, message.type), "content": message.content}
        for message in rendered
    ]


def generate_answer(client, model: str, messages, temperature: float) -> str:
    response = client.chat.complete(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if not response.choices:
        raise ValueError("Le modele de chat n'a retourne aucun choix.")
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        fragments = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                fragments.append(text)
        return "\n".join(fragments).strip()
    raise ValueError("Format de reponse du modele non pris en charge.")


@dataclass(slots=True)
class RagAnswer:
    answer: str
    matches: list[dict]
    context: str


class RagService:
    def __init__(
        self,
        *,
        client,
        index,
        metadata: list[dict],
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        max_context_items: int = DEFAULT_MAX_CONTEXT_ITEMS,
        temperature: float = DEFAULT_TEMPERATURE,
        include_past_events: bool = False,
    ) -> None:
        self.client = client
        self.index = index
        self.metadata = metadata
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.top_k = top_k
        self.max_context_items = max_context_items
        self.temperature = temperature
        self.include_past_events = include_past_events

    @classmethod
    def from_paths(
        cls,
        *,
        client,
        index_path: Path = DEFAULT_INDEX_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chat_model: str = DEFAULT_CHAT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        max_context_items: int = DEFAULT_MAX_CONTEXT_ITEMS,
        temperature: float = DEFAULT_TEMPERATURE,
        include_past_events: bool = False,
    ) -> "RagService":
        return cls(
            client=client,
            index=load_index(index_path),
            metadata=load_metadata(metadata_path),
            embedding_model=embedding_model,
            chat_model=chat_model,
            top_k=top_k,
            max_context_items=max_context_items,
            temperature=temperature,
            include_past_events=include_past_events,
        )

    def reload(self, *, index_path: Path = DEFAULT_INDEX_PATH, metadata_path: Path = DEFAULT_METADATA_PATH) -> None:
        self.index = load_index(index_path)
        self.metadata = load_metadata(metadata_path)

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        max_context_items: int | None = None,
        temperature: float | None = None,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        include_past_events: bool | None = None,
    ) -> RagAnswer:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("La question ne peut pas etre vide.")

        query_vector = extract_query_vector(
            client=self.client,
            model=embedding_model or self.embedding_model,
            question=clean_question,
        )
        prepared_query = prepare_query_vector(index=self.index, vector=query_vector)
        raw_matches = search_index(
            index=self.index,
            metadata=self.metadata,
            query_vector=prepared_query,
            top_k=max((top_k or self.top_k) * 4, top_k or self.top_k),
        )
        matches = filter_matches_by_date(
            raw_matches,
            include_past_events=self.include_past_events if include_past_events is None else include_past_events,
        )[: top_k or self.top_k]
        context = format_context_items(matches, max_context_items=max_context_items or self.max_context_items)
        messages = build_messages(question=clean_question, context=context)
        answer = generate_answer(
            client=self.client,
            model=chat_model or self.chat_model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return RagAnswer(answer=answer, matches=matches, context=context)
