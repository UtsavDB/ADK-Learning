"""Persistent SQLite-backed mapping store for ATI semantic concept expansion."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from ati_search.tool_utils import clean_text

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "ati_mapping.db"
TERM_MAP_SCHEMA = """
CREATE TABLE IF NOT EXISTS term_map (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  concept TEXT NOT NULL,
  synonyms TEXT,
  patterns TEXT,
  related_concepts TEXT,
  weight REAL DEFAULT 1.0,
  source TEXT,
  approved INTEGER DEFAULT 0,
  created_by TEXT,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT
);
CREATE TABLE IF NOT EXISTS map_change_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  term_map_id INTEGER,
  action TEXT,
  payload TEXT,
  changed_by TEXT,
  changed_at TEXT DEFAULT (datetime('now'))
);
"""


def _resolve_db_path(path: str | Path | None = None) -> Path:
    if path is None:
        return DEFAULT_DB_PATH
    return Path(path)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _connect(path: Path) -> sqlite3.Connection:
    _ensure_parent_dir(path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _normalize_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(cleaned)
    return normalized


def _loads_json_array(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return _normalize_list(value)
    try:
        loaded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return _normalize_list(loaded)


def _row_to_entry(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "concept": row["concept"],
        "synonyms": _loads_json_array(row["synonyms"]),
        "patterns": _loads_json_array(row["patterns"]),
        "related_concepts": _loads_json_array(row["related_concepts"]),
        "weight": float(row["weight"] or 0.0),
        "source": row["source"],
        "approved": bool(row["approved"]),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _payload_json(entry: dict[str, Any]) -> str:
    return json.dumps(entry, sort_keys=True)


def _matches_keyword(entry: dict[str, Any], keyword: str) -> bool:
    lowered_keyword = keyword.lower()
    if lowered_keyword in entry["concept"].lower():
        return True
    for field_name in ("synonyms", "patterns", "related_concepts"):
        for value in entry[field_name]:
            if lowered_keyword in value.lower():
                return True
    return False


def create_mapping_db(path: str | Path | None = None) -> Path:
    """Create the SQLite mapping database and required tables if they do not exist."""
    db_path = _resolve_db_path(path)
    with _connect(db_path) as connection:
        connection.executescript(TERM_MAP_SCHEMA)
        connection.commit()
    return db_path


def list_all_mappings(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return every mapping entry ordered by concept and identifier."""
    db_path = create_mapping_db(path)
    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM term_map
            ORDER BY lower(concept) ASC, id ASC
            """
        ).fetchall()
    return [_row_to_entry(row) for row in rows]


def upsert_mapping_entry(
    entry: dict[str, Any],
    approver: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Insert or update a mapping entry, requiring an approver before approved=1 is set."""
    concept = clean_text(entry.get("concept"))
    if not concept:
        raise ValueError("Mapping entry must include a non-empty 'concept'.")

    normalized_entry = {
        "concept": concept,
        "synonyms": _normalize_list(entry.get("synonyms")),
        "patterns": _normalize_list(entry.get("patterns")),
        "related_concepts": _normalize_list(entry.get("related_concepts")),
        "weight": float(entry.get("weight", 1.0) or 1.0),
        "source": clean_text(entry.get("source")),
        "created_by": clean_text(entry.get("created_by")) or "system",
    }
    db_path = create_mapping_db(path)

    with _connect(db_path) as connection:
        existing = connection.execute(
            """
            SELECT *
            FROM term_map
            WHERE lower(concept) = lower(?)
            ORDER BY id DESC
            LIMIT 1
            """,
            (concept,),
        ).fetchone()

        approved = bool(existing["approved"]) if existing else False
        if approver:
            approved = True

        if existing:
            connection.execute(
                """
                UPDATE term_map
                SET concept = ?,
                    synonyms = ?,
                    patterns = ?,
                    related_concepts = ?,
                    weight = ?,
                    source = ?,
                    approved = ?,
                    created_by = COALESCE(created_by, ?),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    normalized_entry["concept"],
                    json.dumps(normalized_entry["synonyms"]),
                    json.dumps(normalized_entry["patterns"]),
                    json.dumps(normalized_entry["related_concepts"]),
                    normalized_entry["weight"],
                    normalized_entry["source"],
                    int(approved),
                    normalized_entry["created_by"],
                    existing["id"],
                ),
            )
            term_map_id = int(existing["id"])
            action = "update"
        else:
            cursor = connection.execute(
                """
                INSERT INTO term_map (
                    concept,
                    synonyms,
                    patterns,
                    related_concepts,
                    weight,
                    source,
                    approved,
                    created_by,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    normalized_entry["concept"],
                    json.dumps(normalized_entry["synonyms"]),
                    json.dumps(normalized_entry["patterns"]),
                    json.dumps(normalized_entry["related_concepts"]),
                    normalized_entry["weight"],
                    normalized_entry["source"],
                    int(approved),
                    normalized_entry["created_by"],
                ),
            )
            term_map_id = int(cursor.lastrowid)
            action = "insert"

        payload = {
            **normalized_entry,
            "approved": approved,
            "approver": approver,
        }
        connection.execute(
            """
            INSERT INTO map_change_log (term_map_id, action, payload, changed_by)
            VALUES (?, ?, ?, ?)
            """,
            (
                term_map_id,
                action,
                _payload_json(payload),
                clean_text(approver) or normalized_entry["created_by"],
            ),
        )
        connection.commit()

        saved = connection.execute(
            """
            SELECT *
            FROM term_map
            WHERE id = ?
            """,
            (term_map_id,),
        ).fetchone()

    if saved is None:
        raise RuntimeError(f"Failed to persist mapping entry for concept '{concept}'.")
    return _row_to_entry(saved)


def seed_mapping(
    entries: Iterable[dict[str, Any]],
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Seed the mapping database with a collection of entries."""
    saved_entries: list[dict[str, Any]] = []
    for entry in entries:
        approver = clean_text(entry.get("approver")) if isinstance(entry, dict) else None
        if isinstance(entry, dict) and entry.get("approved") and not approver:
            approver = clean_text(entry.get("created_by")) or "seed"
        saved_entries.append(upsert_mapping_entry(dict(entry), approver=approver, path=path))
    return saved_entries


def query_mapping_by_keyword(
    keyword: str,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return approved mapping entries whose concept, synonyms, patterns, or related terms match."""
    normalized_keyword = clean_text(keyword)
    if not normalized_keyword:
        return []

    matches: list[dict[str, Any]] = []
    for entry in list_all_mappings(path):
        if not entry["approved"]:
            continue
        if _matches_keyword(entry, normalized_keyword):
            matches.append(entry)
    return matches


def expand_concepts(
    concepts: Iterable[str],
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Expand approved concepts into their related synonyms, patterns, and related concepts."""
    normalized_concepts = {
        concept.lower()
        for concept in _normalize_list(list(concepts))
    }
    entries = [
        entry
        for entry in list_all_mappings(path)
        if entry["approved"] and entry["concept"].lower() in normalized_concepts
    ]

    expanded_synonyms: list[str] = []
    expanded_patterns: list[str] = []
    expanded_related: list[str] = []
    concept_weights: dict[str, float] = {}
    seen_synonyms: set[str] = set()
    seen_patterns: set[str] = set()
    seen_related: set[str] = set()

    for entry in entries:
        concept_weights[entry["concept"]] = float(entry["weight"])
        for value in entry["synonyms"]:
            lowered = value.lower()
            if lowered not in seen_synonyms:
                seen_synonyms.add(lowered)
                expanded_synonyms.append(value)
        for value in entry["patterns"]:
            lowered = value.lower()
            if lowered not in seen_patterns:
                seen_patterns.add(lowered)
                expanded_patterns.append(value)
        for value in entry["related_concepts"]:
            lowered = value.lower()
            if lowered not in seen_related:
                seen_related.add(lowered)
                expanded_related.append(value)

    return {
        "concepts": entries,
        "expanded_synonyms": expanded_synonyms,
        "expanded_patterns": expanded_patterns,
        "expanded_related_concepts": expanded_related,
        "concept_weights": concept_weights,
    }
