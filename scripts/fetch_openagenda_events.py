from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import gzip
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for environments without tqdm
    tqdm = None


API_BASE_URL = "https://api.openagenda.com/v2"
ILE_DE_FRANCE = "Île-de-France"
DEFAULT_OUTPUT_PATH = Path("data/openagenda/ile_de_france_events.jsonl")
EVENT_INCLUDE_FIELDS = [
    "uid",
    "slug",
    "title",
    "description",
    "longDescription",
    "conditions",
    "keywords",
    "timings",
    "attendanceMode",
    "onlineAccessLink",
    "registration",
    "accessibility",
    "status",
    "state",
    "createdAt",
    "updatedAt",
    "image",
    "locationUid",
    "location.name",
    "location.address",
    "location.adminLevel4",
    "location.adminLevel2",
    "location.adminLevel1",
    "location.postalCode",
    "location.latitude",
    "location.longitude",
]
DEFAULT_CONFIG = {
    "region": ILE_DE_FRANCE,
    "date_from": None,
    "date_to": None,
    "search": None,
    "included_event_types": [],
    "excluded_event_types": ["concert"],
    "official_only": False,
    "max_agendas": None,
    "pause_seconds": 0.0,
    "source_mode": "auto",
    "workers": 8,
}
ENV_FILE_PATH = Path(".env")
API_ENV_KEYS = ("OPENAGENDA_API_KEY", "OPEN_AGENDA_API_KEY")


class OpenAgendaError(RuntimeError):
    """Raised when the OpenAgenda API returns an unexpected error."""


class NullProgress:
    def __init__(self, iterable: Any) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix(self, **_: Any) -> None:
        return None

    def close(self) -> None:
        return None

    def update(self, _: int = 1) -> None:
        return None


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


@dataclass(frozen=True)
class AgendaRef:
    uid: int
    title: str
    slug: str | None


@dataclass(frozen=True)
class DateWindow:
    start: datetime
    end: datetime | None


def parse_args() -> argparse.Namespace:
    default_api_key = None
    for env_key in API_ENV_KEYS:
        default_api_key = os.getenv(env_key)
        if default_api_key:
            break
    if not default_api_key:
        default_api_key = read_api_key_from_env_file()

    parser = argparse.ArgumentParser(
        description=(
            "Recupere les evenements OpenAgenda situes en Ile-de-France, "
            "normalises en JSONL pour un futur pipeline RAG."
        )
    )
    parser.add_argument(
        "--api-key",
        default=default_api_key,
        help="Cle publique OpenAgenda. Peut aussi etre passee via OPENAGENDA_API_KEY.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Fichier JSONL de sortie. Defaut: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Chemin optionnel du manifeste JSON. Defaut: meme dossier que --output.",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        default=None,
        help=(
            "Filtre portable par type d'evenement via keyword[]. "
            "Option repetable: --event-type concert --event-type exposition."
        ),
    )
    parser.add_argument(
        "--exclude-event-type",
        action="append",
        default=None,
        help=(
            "Exclut localement les evenements contenant ce keyword. "
            "Option repetable: --exclude-event-type concert."
        ),
    )
    parser.add_argument("--search", default=None, help="Recherche texte complementaire cote OpenAgenda.")
    parser.add_argument(
        "--date-from",
        default=DEFAULT_CONFIG["date_from"],
        help=(
            "Debut de fenetre ISO 8601. Si anterieur a 365 jours, "
            "il sera automatiquement ramene a aujourd'hui - 365 jours."
        ),
    )
    parser.add_argument(
        "--date-to",
        default=DEFAULT_CONFIG["date_to"],
        help="Fin de fenetre ISO 8601. Facultatif.",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_CONFIG["region"],
        help=f"Region a filtrer. Defaut: {DEFAULT_CONFIG['region']}",
    )
    parser.add_argument("--official-only", action="store_true", help="Ne parcourt que les agendas officiels.")
    parser.add_argument(
        "--max-agendas",
        type=int,
        default=DEFAULT_CONFIG["max_agendas"],
        help="Limite de securite pour borner le nombre d'agendas parcourus.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=DEFAULT_CONFIG["pause_seconds"],
        help="Pause optionnelle entre deux appels d'agenda.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("auto", "transverse", "agendas"),
        default=DEFAULT_CONFIG["source_mode"],
        help=(
            "Strategie de collecte. `auto` tente /v2/events puis bascule sur "
            "/v2/agendas/{uid}/events si besoin."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_CONFIG["workers"],
        help="Nombre de requetes agendas executees en parallele en mode `agendas`.",
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("La cle OpenAgenda est requise via --api-key ou OPENAGENDA_API_KEY.")
    if args.workers < 1:
        parser.error("--workers doit etre >= 1.")
    if args.event_type is None:
        args.event_type = list(DEFAULT_CONFIG["included_event_types"])
    if args.exclude_event_type is None:
        args.exclude_event_type = list(DEFAULT_CONFIG["excluded_event_types"])
    if args.search is None:
        args.search = DEFAULT_CONFIG["search"]
    return args


def parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def compute_date_window(
    date_from: str | None,
    date_to: str | None,
    now: datetime | None = None,
) -> DateWindow:
    reference_now = (now or datetime.now(UTC)).astimezone(UTC)
    min_start = reference_now - timedelta(days=365)
    start = parse_iso_datetime(date_from) if date_from else min_start
    if start < min_start:
        start = min_start

    end = parse_iso_datetime(date_to) if date_to else None
    if end is not None and end < start:
        raise ValueError("--date-to doit etre posterieur ou egal a --date-from.")

    return DateWindow(start=start, end=end)


def to_api_datetime(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def flatten_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in ("fr", "en", "de", "es", "it"):
            text = value.get(key)
            if isinstance(text, str) and text.strip():
                return text.strip()
        for text in value.values():
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def flatten_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, dict):
        items: list[str] = []
        for candidate in value.values():
            if isinstance(candidate, list):
                items.extend(
                    item.strip()
                    for item in candidate
                    if isinstance(item, str) and item.strip()
                )
        return items
    return []


def build_document(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record.get("title"):
        parts.append(f"Titre: {record['title']}")
    if record.get("description"):
        parts.append(f"Resume: {record['description']}")
    if record.get("long_description"):
        parts.append(f"Description: {record['long_description']}")
    if record.get("event_types"):
        parts.append("Types: " + ", ".join(record["event_types"]))
    if record.get("date_summary"):
        parts.append(f"Dates: {record['date_summary']}")
    location = record.get("location") or {}
    location_parts = [
        location.get("name"),
        location.get("address"),
        location.get("city"),
        location.get("department"),
        location.get("region"),
    ]
    location_text = ", ".join(part for part in location_parts if part)
    if location_text:
        parts.append(f"Lieu: {location_text}")
    if record.get("conditions"):
        parts.append(f"Conditions: {record['conditions']}")
    if record.get("online_access_link"):
        parts.append(f"Acces en ligne: {record['online_access_link']}")
    return "\n".join(parts)


def format_date_summary(timings: list[dict[str, Any]]) -> str | None:
    if not timings:
        return None
    begins = [slot.get("begin") for slot in timings if slot.get("begin")]
    ends = [slot.get("end") for slot in timings if slot.get("end")]
    if not begins:
        return None
    first_begin = begins[0]
    last_end = ends[-1] if ends else begins[-1]
    if first_begin == last_end:
        return first_begin
    return f"{first_begin} -> {last_end}"


def extract_timings_for_storage(timings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stored_timings: list[dict[str, Any]] = []
    for slot in timings:
        begin = slot.get("begin")
        end = slot.get("end")
        if begin or end:
            stored_timings.append({"begin": begin, "end": end})
    return stored_timings


def extract_source_agendas(event: dict[str, Any], agenda: AgendaRef | None) -> list[dict[str, Any]]:
    if agenda is not None:
        return [{"uid": agenda.uid, "title": agenda.title, "slug": agenda.slug}]

    source_agendas: list[dict[str, Any]] = []
    candidates = event.get("agendas")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            uid = candidate.get("uid")
            if uid is None:
                continue
            source_agendas.append(
                {
                    "uid": uid,
                    "title": candidate.get("title"),
                    "slug": candidate.get("slug"),
                }
            )
    return source_agendas


def normalize_event(event: dict[str, Any], agenda: AgendaRef | None = None) -> dict[str, Any]:
    location = event.get("location") or {}
    timings = event.get("timings") or []
    first_timing = timings[0]["begin"] if timings else None
    last_timing = timings[-1]["end"] if timings else None
    event_types = flatten_keywords(event.get("keywords"))
    record = {
        "id": f"openagenda:{event['uid']}",
        "event_uid": event["uid"],
        "slug": event.get("slug"),
        "title": flatten_text(event.get("title")),
        "description": flatten_text(event.get("description")),
        "long_description": flatten_text(event.get("longDescription")),
        "conditions": flatten_text(event.get("conditions")),
        "event_types": event_types,
        "date_summary": format_date_summary(timings),
        "timings": extract_timings_for_storage(timings),
        "occurrences_count": len(timings),
        "first_timing": first_timing,
        "last_timing": last_timing,
        "location": {
            "name": location.get("name"),
            "address": location.get("address"),
            "city": location.get("adminLevel4") or location.get("city"),
            "department": location.get("adminLevel2"),
            "region": location.get("adminLevel1"),
        },
        "source_agendas": extract_source_agendas(event, agenda),
        "raw_source": "openagenda",
    }
    record["document"] = build_document(record)
    return record


def should_keep_event(record: dict[str, Any], excluded_event_types: list[str]) -> bool:
    if not excluded_event_types:
        return True

    excluded = {item.strip().casefold() for item in excluded_event_types if item.strip()}
    event_types = {
        item.strip().casefold()
        for item in record.get("event_types", [])
        if isinstance(item, str) and item.strip()
    }
    return event_types.isdisjoint(excluded)


class OpenAgendaClient:
    def __init__(self, api_key: str, timeout: int = 30) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(params, doseq=True)
        request = Request(
            f"{API_BASE_URL}{path}?{query}",
            headers={"key": self.api_key, "Accept": "application/json"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenAgendaError(f"HTTP {exc.code} sur {path}: {body}") from exc
        except URLError as exc:
            raise OpenAgendaError(f"Erreur reseau OpenAgenda: {exc.reason}") from exc

    def iter_agendas(
        self,
        official_only: bool = False,
        max_agendas: int | None = None,
    ):
        after: list[Any] | None = None
        yielded = 0

        while True:
            params: dict[str, Any] = {
                "size": 100,
                "if[]": ["uid", "title", "slug"],
            }
            if official_only:
                params["official"] = 1
            if after:
                params["after[]"] = after

            payload = self.get_json("/agendas", params)
            for agenda in payload.get("agendas", []):
                yield AgendaRef(
                    uid=agenda["uid"],
                    title=agenda.get("title", "").strip(),
                    slug=agenda.get("slug"),
                )
                yielded += 1
                if max_agendas is not None and yielded >= max_agendas:
                    return

            after = payload.get("after")
            if not after:
                return

    def iter_events_for_agenda(
        self,
        agenda: AgendaRef,
        region: str,
        window: DateWindow,
        event_types: list[str],
        search: str | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        after: list[Any] | None = None

        while True:
            params: dict[str, Any] = {
                "adminLevel1": region,
                "timings[gte]": to_api_datetime(window.start),
                "size": 300,
                "monolingual": "fr",
                "sort": "timings.asc",
                "state": 2,
                "if[]": EVENT_INCLUDE_FIELDS,
            }
            if window.end is not None:
                params["timings[lte]"] = to_api_datetime(window.end)
            if event_types:
                params["keyword[]"] = event_types
            if search:
                params["search"] = search
                params["threshold"] = "auto"
            if after:
                params["after[]"] = after

            payload = self.get_json(f"/agendas/{agenda.uid}/events", params)
            events.extend(payload.get("events", []))
            after = payload.get("after")
            if not after:
                return events

    def iter_transverse_events(
        self,
        region: str,
        window: DateWindow,
        event_types: list[str],
        search: str | None,
    ):
        after: list[Any] | None = None

        while True:
            params: dict[str, Any] = {
                "adminLevel1": region,
                "timings[gte]": to_api_datetime(window.start),
                "size": 300,
                "monolingual": "fr",
                "sort": "timings.asc",
                "if[]": EVENT_INCLUDE_FIELDS,
                "relative[]": ["current", "upcoming", "passed"],
            }
            if window.end is not None:
                params["timings[lte]"] = to_api_datetime(window.end)
            if event_types:
                params["keyword[]"] = event_types
            if search:
                params["search"] = search
                params["threshold"] = "auto"
            if after:
                params["after[]"] = after

            payload = self.get_json("/events", params)
            yield payload
            after = payload.get("after")
            if not after:
                return


def merge_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for record in records:
        event_uid = record["event_uid"]
        if event_uid not in merged:
            merged[event_uid] = record
            continue

        existing = merged[event_uid]
        known_agenda_uids = {agenda["uid"] for agenda in existing.get("source_agendas", [])}
        for agenda in record.get("source_agendas", []):
            if agenda["uid"] not in known_agenda_uids:
                existing["source_agendas"].append(agenda)
        existing_types = set(existing.get("event_types", []))
        for event_type in record.get("event_types", []):
            if event_type not in existing_types:
                existing["event_types"].append(event_type)
                existing_types.add(event_type)
    return sorted(merged.values(), key=lambda item: (item["first_timing"] or "", item["event_uid"]))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        handle = gzip.open(path, "wt", encoding="utf-8")
    else:
        handle = path.open("w", encoding="utf-8")
    with handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_manifest(
    path: Path,
    output_path: Path,
    region: str,
    window: DateWindow,
    agendas_count: int,
    records: list[dict[str, Any]],
    event_types: list[str],
    search: str | None,
    source_mode: str,
) -> None:
    timings = [record["first_timing"] for record in records if record.get("first_timing")]
    manifest = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "openagenda",
        "region": region,
        "date_window": {
            "start": to_api_datetime(window.start),
            "end": to_api_datetime(window.end) if window.end else None,
        },
        "filters": {
            "event_types": event_types,
            "search": search,
        },
        "collection_mode": source_mode,
        "agendas_scanned": agendas_count,
        "records_written": len(records),
        "output_file": str(output_path),
        "first_event_timing": min(timings) if timings else None,
        "last_event_timing": max(timings) if timings else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect_via_agendas(
    client: OpenAgendaClient,
    args: argparse.Namespace,
    window: DateWindow,
) -> tuple[list[dict[str, Any]], int, str]:
    normalized_records: list[dict[str, Any]] = []
    agendas = list(
        client.iter_agendas(
            official_only=args.official_only,
            max_agendas=args.max_agendas,
        )
    )
    progress = (
        tqdm(total=len(agendas), desc="Agendas", unit="agenda")
        if tqdm is not None
        else NullProgress([])
    )
    kept_events = 0
    agendas_scanned = len(agendas)

    def fetch_agenda_events(agenda: AgendaRef) -> tuple[AgendaRef, list[dict[str, Any]]]:
        events = client.iter_events_for_agenda(
            agenda=agenda,
            region=args.region,
            window=window,
            event_types=args.event_type,
            search=args.search,
        )
        return agenda, events

    if args.workers == 1:
        for agenda in agendas:
            _, events = fetch_agenda_events(agenda)
            for event in events:
                record = normalize_event(event, agenda)
                if should_keep_event(record, args.exclude_event_type):
                    normalized_records.append(record)
                    kept_events += 1
            progress.update(1)
            progress.set_postfix(events=kept_events)
            if args.pause_seconds:
                time.sleep(args.pause_seconds)
        progress.close()
        return normalized_records, agendas_scanned, "agendas"

    max_pending = max(args.workers * 2, args.workers)
    agenda_iter = iter(agendas)
    pending: dict[Future, AgendaRef] = {}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(pending) < max_pending:
            try:
                agenda = next(agenda_iter)
            except StopIteration:
                break
            pending[executor.submit(fetch_agenda_events, agenda)] = agenda
            if args.pause_seconds:
                time.sleep(args.pause_seconds)

        while pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                agenda = pending.pop(future)
                _, events = future.result()
                for event in events:
                    record = normalize_event(event, agenda)
                    if should_keep_event(record, args.exclude_event_type):
                        normalized_records.append(record)
                        kept_events += 1
                progress.update(1)
                progress.set_postfix(events=kept_events)

                try:
                    next_agenda = next(agenda_iter)
                except StopIteration:
                    continue
                pending[executor.submit(fetch_agenda_events, next_agenda)] = next_agenda
                if args.pause_seconds:
                    time.sleep(args.pause_seconds)
    progress.close()
    return normalized_records, agendas_scanned, "agendas"


def collect_via_transverse(
    client: OpenAgendaClient,
    args: argparse.Namespace,
    window: DateWindow,
) -> tuple[list[dict[str, Any]], int, str]:
    normalized_records: list[dict[str, Any]] = []
    progress = tqdm(desc="Events", unit="event") if tqdm is not None else NullProgress([])
    kept_events = 0

    for payload in client.iter_transverse_events(
        region=args.region,
        window=window,
        event_types=args.event_type,
        search=args.search,
    ):
        events = payload.get("events", [])
        progress.update(len(events))
        for event in events:
            record = normalize_event(event)
            if should_keep_event(record, args.exclude_event_type):
                normalized_records.append(record)
                kept_events += 1
        progress.set_postfix(events=kept_events)
    progress.close()
    return normalized_records, 0, "transverse"


def run_collection(args: argparse.Namespace) -> tuple[Path, Path, int]:
    client = OpenAgendaClient(api_key=args.api_key)
    window = compute_date_window(args.date_from, args.date_to)
    if args.source_mode == "agendas":
        normalized_records, agendas_scanned, used_mode = collect_via_agendas(client, args, window)
    elif args.source_mode == "transverse":
        normalized_records, agendas_scanned, used_mode = collect_via_transverse(client, args, window)
    else:
        try:
            normalized_records, agendas_scanned, used_mode = collect_via_transverse(client, args, window)
        except OpenAgendaError as exc:
            print(f"Route transverse indisponible ({exc}). Fallback sur les agendas.")
            normalized_records, agendas_scanned, used_mode = collect_via_agendas(client, args, window)

    merged_records = merge_records(normalized_records)
    output_path = args.output
    manifest_path = args.manifest_output or output_path.with_suffix(".manifest.json")
    write_jsonl(output_path, merged_records)
    write_manifest(
        manifest_path,
        output_path,
        args.region,
        window,
        agendas_scanned,
        merged_records,
        args.event_type,
        args.search,
        used_mode,
    )
    return output_path, manifest_path, len(merged_records)


def main() -> None:
    args = parse_args()
    output_path, manifest_path, count = run_collection(args)
    print(f"{count} evenements ecrits dans {output_path}")
    print(f"Manifeste ecrit dans {manifest_path}")


if __name__ == "__main__":
    main()
    
