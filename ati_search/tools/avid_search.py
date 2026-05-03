from __future__ import annotations
"""
This module provides the `avid_search` function, which enables searching Aristocrat documentation using a free-form text query. It interacts with the Aristocrat Knowledge Hub API to retrieve clustered search results, normalizes and deduplicates the entries, and returns a structured summary of the most relevant documentation topics. The module also includes utility functions for building API requests, handling errors, and processing the API response into a user-friendly format.
"""

from typing import Any

import requests

from ati_search.env import get_env_value
from ati_search.tool_utils import (
    DEFAULT_PAGE,
    PER_PAGE,
    REQUEST_TIMEOUT_SECONDS,
    clean_text,
    dedupe_key,
    get_metadata_value,
    pick_first,
    sort_timestamp,
    strip_html_tags,
)

API_URL = "https://aristocrat-genai.fluidtopics.net/api/khub/clustered-search"


def build_search_url(page: int = DEFAULT_PAGE, per_page: int = PER_PAGE) -> str:
    return f"{API_URL}?page={page}&per_page={per_page}"


def build_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Ft-Calling-App": "visualqa-orchestrator",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def build_payload(query: str, content_locale: str = "en-US") -> dict[str, str]:
    return {
        "query": query,
        "contentLocale": content_locale,
    }


def error_response(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    response = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    response.update(extra)
    return response


def normalize_entry(entry: Any, query: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("Entry is not an object.")

    topic = entry.get("topic")
    if topic is not None and not isinstance(topic, dict):
        raise ValueError("Entry topic is not an object.")
    topic = topic or {}
    metadata = topic.get("metadata") or entry.get("metadata") or []

    excerpt_html = pick_first(entry.get("excerpt"), topic.get("excerpt"))
    title = pick_first(entry.get("title"), topic.get("title"))
    breadcrumb = pick_first(
        entry.get("breadcrumb"),
        topic.get("breadcrumb"),
        get_metadata_value(metadata, "breadcrumb", "ft:breadcrumb"),
    )

    return {
        "query": query,
        "title": title,
        "breadcrumb": breadcrumb,
        "cleaned_excerpt": strip_html_tags(excerpt_html),
        "raw_html_excerpt": excerpt_html,
        "product": get_metadata_value(metadata, "product", "ft:product"),
        "version": get_metadata_value(metadata, "version", "ft:version"),
        "document_type": get_metadata_value(
            metadata, "documentType", "document_type", "ft:documentType", "type"
        ),
        "audience": get_metadata_value(metadata, "audience", "ft:audience"),
        "readerUrl": pick_first(entry.get("readerUrl"), topic.get("readerUrl")),
        "contentUrl": pick_first(entry.get("contentUrl"), topic.get("contentUrl")),
        "topicUrl": pick_first(entry.get("topicUrl"), topic.get("topicUrl")),
        "lastEditionDate": pick_first(
            entry.get("lastEditionDate"),
            topic.get("lastEditionDate"),
            get_metadata_value(metadata, "lastEditionDate", "ft:lastEditionDate"),
        ),
        "lastPublication": pick_first(
            entry.get("lastPublication"),
            topic.get("lastPublication"),
            get_metadata_value(metadata, "lastPublication", "ft:lastPublication"),
        ),
        "publication_title": pick_first(
            entry.get("publicationTitle"),
            topic.get("publicationTitle"),
            get_metadata_value(metadata, "publicationTitle", "ft:publicationTitle"),
        ),
        "source_name": pick_first(
            entry.get("sourceName"),
            topic.get("sourceName"),
            get_metadata_value(metadata, "sourceName", "source", "ft:sourceName"),
        ),
        "base_id": get_metadata_value(metadata, "ft:baseId", "baseId"),
        "cluster_id": get_metadata_value(metadata, "ft:clusterId", "clusterId"),
        "last_tech_change_timestamp": get_metadata_value(
            metadata, "ft:lastTechChangeTimestamp", "lastTechChangeTimestamp"
        ),
    }


def flatten_clustered_response(response_data: Any, query: str) -> dict[str, Any]:
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
                flattened_entries.append(normalize_entry(entry, query))
            except ValueError:
                continue

    deduped_entries: dict[str, dict[str, Any]] = {}
    for entry in flattened_entries:
        key = dedupe_key(entry)
        current = deduped_entries.get(key)
        if current is None or sort_timestamp(entry) > sort_timestamp(current):
            deduped_entries[key] = entry

    normalized_entries = sorted(
        deduped_entries.values(),
        key=sort_timestamp,
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
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return error_response("validation_error", "A non-empty search query is required.")

    token = get_env_value("ATI_SEARCH_BEARER_TOKEN")
    if not token:
        return error_response(
            "missing_configuration",
            "Missing ATI_SEARCH_BEARER_TOKEN. Add it to your environment or .env file.",
        )

    try:
        response = requests.post(
            build_search_url(page=DEFAULT_PAGE, per_page=PER_PAGE),
            headers=build_headers(token),
            json=build_payload(cleaned_query),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return error_response("timeout", "The ATI search request timed out.")
    except requests.RequestException as exc:
        return error_response("request_error", f"ATI search request failed: {exc}")

    if response.status_code != 200:
        return error_response(
            "http_error",
            "ATI search returned a non-200 response.",
            status_code=response.status_code,
            response_text=clean_text(response.text),
        )

    try:
        raw_response = response.json()
    except ValueError:
        return error_response(
            "invalid_json",
            "ATI search returned invalid JSON.",
            response_text=clean_text(response.text),
        )

    try:
        summary = flatten_clustered_response(raw_response, cleaned_query)
    except ValueError as exc:
        return error_response(
            "unexpected_response_structure",
            str(exc),
            raw_response=raw_response,
        )

    return {
        "status": "success",
        "raw_response": raw_response,
        "summary": summary,
    }
