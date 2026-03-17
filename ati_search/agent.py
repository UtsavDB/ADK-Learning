from __future__ import annotations

import html
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google.adk.agents import Agent

AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=AGENT_DIR / ".env")
load_dotenv(dotenv_path=AGENT_DIR.parent / ".env")

API_URL = "https://aristocrat-genai.fluidtopics.net/api/khub/clustered-search"
DEFAULT_PAGE = 1
PER_PAGE = 10
REQUEST_TIMEOUT_SECONDS = 20
DEFAULT_MODEL = os.getenv("ATI_SEARCH_MODEL", "gemini-2.0-flash")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = WHITESPACE_RE.sub(" ", value).strip()
    return text or None


def _strip_html_tags(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def _metadata_items(metadata: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            items.append((str(key), value))
        return items
    if not isinstance(metadata, list):
        return items

    for item in metadata:
        if not isinstance(item, dict):
            continue
        key = item.get("key") or item.get("name") or item.get("id")
        if not key:
            continue
        value = (
            item.get("value")
            or item.get("values")
            or item.get("label")
            or item.get("labels")
        )
        items.append((str(key), value))
    return items


def _normalize_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [_normalize_metadata_value(item) for item in value]
        joined = ", ".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        for key in ("value", "label", "name", "title", "id"):
            normalized = _normalize_metadata_value(value.get(key))
            if normalized:
                return normalized
        return None
    return _clean_text(value)


def _get_metadata_value(metadata: Any, *keys: str) -> str | None:
    key_lookup = {key.lower() for key in keys}
    for key, value in _metadata_items(metadata):
        if key.lower() in key_lookup:
            normalized = _normalize_metadata_value(value)
            if normalized:
                return normalized
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate:
        return None

    normalized = candidate.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _build_search_url(page: int = DEFAULT_PAGE, per_page: int = PER_PAGE) -> str:
    return f"{API_URL}?page={page}&per_page={per_page}"


def _build_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Ft-Calling-App": "visualqa-orchestrator",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _build_payload(query: str, content_locale: str = "en-US") -> dict[str, str]:
    return {
        "query": query,
        "contentLocale": content_locale,
    }


def _pick_first(*values: Any) -> str | None:
    for value in values:
        cleaned = _clean_text(value)
        if cleaned:
            return cleaned
    return None


def _normalize_entry(entry: Any, query: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Entry is not an object.")

    topic = entry.get("topic")
    if topic is not None and not isinstance(topic, dict):
        raise ValueError("Entry topic is not an object.")
    topic = topic or {}
    metadata = topic.get("metadata") or entry.get("metadata") or []

    excerpt_html = _pick_first(entry.get("excerpt"), topic.get("excerpt"))
    title = _pick_first(entry.get("title"), topic.get("title"))
    breadcrumb = _pick_first(
        entry.get("breadcrumb"),
        topic.get("breadcrumb"),
        _get_metadata_value(metadata, "breadcrumb", "ft:breadcrumb"),
    )

    return {
        "query": query,
        "title": title,
        "breadcrumb": breadcrumb,
        "cleaned_excerpt": _strip_html_tags(excerpt_html),
        "raw_html_excerpt": excerpt_html,
        "product": _get_metadata_value(metadata, "product", "ft:product"),
        "version": _get_metadata_value(metadata, "version", "ft:version"),
        "document_type": _get_metadata_value(
            metadata, "documentType", "document_type", "ft:documentType", "type"
        ),
        "audience": _get_metadata_value(metadata, "audience", "ft:audience"),
        "readerUrl": _pick_first(entry.get("readerUrl"), topic.get("readerUrl")),
        "contentUrl": _pick_first(entry.get("contentUrl"), topic.get("contentUrl")),
        "topicUrl": _pick_first(entry.get("topicUrl"), topic.get("topicUrl")),
        "lastEditionDate": _pick_first(
            entry.get("lastEditionDate"),
            topic.get("lastEditionDate"),
            _get_metadata_value(metadata, "lastEditionDate", "ft:lastEditionDate"),
        ),
        "lastPublication": _pick_first(
            entry.get("lastPublication"),
            topic.get("lastPublication"),
            _get_metadata_value(metadata, "lastPublication", "ft:lastPublication"),
        ),
        "publication_title": _pick_first(
            entry.get("publicationTitle"),
            topic.get("publicationTitle"),
            _get_metadata_value(metadata, "publicationTitle", "ft:publicationTitle"),
        ),
        "source_name": _pick_first(
            entry.get("sourceName"),
            topic.get("sourceName"),
            _get_metadata_value(metadata, "sourceName", "source", "ft:sourceName"),
        ),
        "base_id": _get_metadata_value(metadata, "ft:baseId", "baseId"),
        "cluster_id": _get_metadata_value(metadata, "ft:clusterId", "clusterId"),
        "last_tech_change_timestamp": _get_metadata_value(
            metadata, "ft:lastTechChangeTimestamp", "lastTechChangeTimestamp"
        ),
    }


def _sort_timestamp(entry: dict[str, Any]) -> tuple[int, datetime]:
    for key in (
        "last_tech_change_timestamp",
        "lastPublication",
        "lastEditionDate",
    ):
        parsed = _parse_datetime(entry.get(key))
        if parsed is not None:
            return (1, parsed)
    return (0, datetime.min)


def _dedupe_key(entry: dict[str, Any]) -> str:
    base_id = entry.get("base_id")
    if base_id:
        return f"base:{base_id}"
    cluster_id = entry.get("cluster_id")
    if cluster_id:
        return f"cluster:{cluster_id}"
    title = entry.get("title") or ""
    reader_url = entry.get("readerUrl") or ""
    return f"title_url:{title}|{reader_url}"


def _flatten_clustered_response(response_data: Any, query: str) -> dict[str, Any]:
    if not isinstance(response_data, dict):
        raise ValueError("API response root is not an object.")

    results = response_data.get("results")
    if results is None:
        raise ValueError("API response does not include a 'results' field.")
    if not isinstance(results, list):
        raise ValueError("API response 'results' field is not a list.")

    flattened_entries: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        entries = result.get("entries") or []
        if not isinstance(entries, list):
            continue
        for entry in entries:
            try:
                flattened_entries.append(_normalize_entry(entry, query))
            except ValueError:
                continue

    deduped_entries: dict[str, dict[str, Any]] = {}
    for entry in flattened_entries:
        key = _dedupe_key(entry)
        current = deduped_entries.get(key)
        if current is None or _sort_timestamp(entry) > _sort_timestamp(current):
            deduped_entries[key] = entry

    normalized_entries = sorted(
        deduped_entries.values(),
        key=_sort_timestamp,
        reverse=True,
    )

    page = response_data.get("page", DEFAULT_PAGE)
    is_last_page = response_data.get("isLastPage")
    if is_last_page is None:
        total_pages = response_data.get("totalPages")
        if isinstance(total_pages, int) and isinstance(page, int):
            is_last_page = page >= total_pages

    return {
        "query": query,
        "total_results_count": response_data.get("totalResultsCount")
        or response_data.get("totalResults")
        or len(flattened_entries),
        "total_clusters_count": response_data.get("totalClustersCount")
        or response_data.get("totalClusters")
        or len(results),
        "current_page": page if isinstance(page, int) else DEFAULT_PAGE,
        "is_last_page": bool(is_last_page) if is_last_page is not None else None,
        "top_normalized_entries": normalized_entries,
    }


def avid_search(query: str) -> dict[str, Any]:
    """Search Aristocrat documentation for a free-form text query."""
    cleaned_query = _clean_text(query)
    if not cleaned_query:
        return {
            "status": "error",
            "error_type": "validation_error",
            "message": "A non-empty search query is required.",
        }

    token = os.getenv("ATI_SEARCH_BEARER_TOKEN")
    if not token:
        return {
            "status": "error",
            "error_type": "missing_configuration",
            "message": (
                "Missing ATI_SEARCH_BEARER_TOKEN. Add it to your environment or .env file."
            ),
        }

    try:
        response = requests.post(
            _build_search_url(page=DEFAULT_PAGE, per_page=PER_PAGE),
            headers=_build_headers(token),
            json=_build_payload(cleaned_query),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return {
            "status": "error",
            "error_type": "timeout",
            "message": "The ATI search request timed out.",
        }
    except requests.RequestException as exc:
        return {
            "status": "error",
            "error_type": "request_error",
            "message": f"ATI search request failed: {exc}",
        }

    if response.status_code != 200:
        return {
            "status": "error",
            "error_type": "http_error",
            "status_code": response.status_code,
            "message": "ATI search returned a non-200 response.",
            "response_text": _clean_text(response.text),
        }

    try:
        raw_response = response.json()
    except ValueError:
        return {
            "status": "error",
            "error_type": "invalid_json",
            "message": "ATI search returned invalid JSON.",
            "response_text": _clean_text(response.text),
        }

    try:
        summary = _flatten_clustered_response(raw_response, cleaned_query)
    except ValueError as exc:
        return {
            "status": "error",
            "error_type": "unexpected_response_structure",
            "message": str(exc),
            "raw_response": raw_response,
        }

    return {
        "status": "success",
        "raw_response": raw_response,
        "summary": summary,
    }


root_agent = Agent(
    name="ATI Search",
    model=DEFAULT_MODEL,
    description="Searches internal Aristocrat documentation and summarizes the results.",
    instruction=(
        "You are ATI Search. "
        "Use avid_search whenever the user wants to search Aristocrat documentation. "
        "Summarize results in plain English. "
        "Prefer the latest relevant documentation when duplicates exist. "
        "Mention product and version when relevant. "
        "Include reader links when available. "
        "If no useful results are found, say so clearly. "
        "Keep the final answer plain text and do not dump raw JSON unless the user explicitly asks for it."
    ),
    tools=[avid_search],
)
