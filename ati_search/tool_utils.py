from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_PAGE = 1
PER_PAGE = 10
REQUEST_TIMEOUT_SECONDS = 20

DEFAULT_TFS_API_VERSION = "4.1"
DEFAULT_TFS_RESULT_LIMIT = 10
TFS_TIMEOUT_SECONDS = 20

HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = WHITESPACE_RE.sub(" ", value).strip()
    return text or None


def strip_html_tags(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def metadata_items(metadata: Any) -> list[tuple[str, Any]]:
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


def normalize_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        parts = [normalize_metadata_value(item) for item in value]
        joined = ", ".join(part for part in parts if part)
        return joined or None
    if isinstance(value, dict):
        for key in ("value", "label", "name", "title", "id"):
            normalized = normalize_metadata_value(value.get(key))
            if normalized:
                return normalized
        return None
    return clean_text(value)


def get_metadata_value(metadata: Any, *keys: str) -> str | None:
    key_lookup = {key.lower() for key in keys}
    for key, value in metadata_items(metadata):
        if key.lower() in key_lookup:
            normalized = normalize_metadata_value(value)
            if normalized:
                return normalized
    return None


def parse_datetime(value: str | None) -> datetime | None:
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


def pick_first(*values: Any) -> str | None:
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            return cleaned
    return None


def build_git_item_url(tfs_url: str, project: str, repo_name: str, path: str | None) -> str:
    base = f"{tfs_url}/{project}/_git/{quote(repo_name, safe='')}"
    if not path:
        return base
    return f"{base}?path={quote(path, safe='/')}"


def dedupe_key(entry: dict[str, Any]) -> str:
    base_id = entry.get("base_id")
    if base_id:
        return f"base:{base_id}"
    cluster_id = entry.get("cluster_id")
    if cluster_id:
        return f"cluster:{cluster_id}"
    title = entry.get("title") or ""
    reader_url = entry.get("readerUrl") or ""
    return f"title_url:{title}|{reader_url}"


def sort_timestamp(entry: dict[str, Any]) -> tuple[int, datetime]:
    for key in (
        "last_tech_change_timestamp",
        "lastPublication",
        "lastEditionDate",
    ):
        parsed = parse_datetime(entry.get(key))
        if parsed is not None:
            return (1, parsed)
    return (0, datetime.min)


def file_name_from_path(path: str | None) -> str | None:
    return Path(path).name if path else None
