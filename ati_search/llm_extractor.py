"""Strict JSON concept extractor for ATI semantic search with guarded provider fallback."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from ati_search.tool_utils import clean_text

LOGGER = logging.getLogger(__name__)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[a-z0-9]+")
STRICT_SYSTEM_PROMPT = """You extract ATI domain concepts from free-text search queries.
Return only valid JSON. Do not add markdown, comments, or prose.
The JSON must match this schema exactly:
{
  "concepts":[
    {
      "canonical":"player_merge",
      "synonyms":["merge player","player merge","account merge"],
      "related":["unimerge","duplicate_player_resolution"],
      "keywords":["merge","player id","account merge","duplicate player"],
      "confidence":0.9
    }
  ],
  "global_keywords":["merge","player","account"],
  "notes":"optional short note string"
}
"""
STRICT_USER_PROMPT = """Extract ATI domain concepts from this query and respond with JSON only.

Few-shot example 1
Query: "merging player"
JSON:
{
  "concepts": [
    {
      "canonical": "player_merge",
      "synonyms": ["merge player", "player merge", "account merge"],
      "related": ["unimerge", "duplicate_player_resolution"],
      "keywords": ["merge", "player id", "account merge", "duplicate player"],
      "confidence": 0.9
    }
  ],
  "global_keywords": ["merge", "player", "account"],
  "notes": "Player-merge terminology often appears as unimerge in defects."
}

Few-shot example 2
Query: "player id collision on transfer"
JSON:
{
  "concepts": [
    {
      "canonical": "player_transfer_collision",
      "synonyms": ["player transfer collision", "transfer id collision"],
      "related": ["duplicate_player_resolution", "identity_collision"],
      "keywords": ["player id collision", "transfer", "collision", "player id"],
      "confidence": 0.93
    }
  ],
  "global_keywords": ["player", "transfer", "collision"],
  "notes": "Focus on transfer-specific identity collisions."
}

Few-shot example 3
Query: "tier points not credited when promo applied"
JSON:
{
  "concepts": [
    {
      "canonical": "tier_points_credit_failure",
      "synonyms": ["tier points missing", "points not credited", "promo points failure"],
      "related": ["promotion_crediting", "loyalty_points_adjustment"],
      "keywords": ["tier points", "credited", "promo", "loyalty"],
      "confidence": 0.95
    }
  ],
  "global_keywords": ["tier points", "credited", "promo"],
  "notes": "This is a crediting failure tied to a promotion."
}

Now extract from:
Query: "{query}"
JSON:
"""


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.3


def _normalize_keyword_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
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


def _normalize_canonical(value: Any) -> str:
    cleaned = clean_text(value)
    if not cleaned:
        return "unknown_concept"
    canonical = re.sub(r"[^a-z0-9]+", "_", cleaned.lower()).strip("_")
    return canonical or "unknown_concept"


def _heuristic_canonical(query: str) -> str:
    lowered = query.lower()
    tokens = set(TOKEN_RE.findall(lowered))
    if "player" in tokens and any(token.startswith("merg") for token in tokens):
        return "player_merge"
    if {"player", "transfer", "collision"} <= tokens or (
        "player" in tokens and "transfer" in tokens and any(token.startswith("coll") for token in tokens)
    ):
        return "player_transfer_collision"
    if "tier" in tokens and "points" in tokens and (
        any(token.startswith("credit") for token in tokens) or "promo" in tokens
    ):
        return "tier_points_credit_failure"
    ordered_tokens = TOKEN_RE.findall(lowered)[:4]
    return "_".join(ordered_tokens) or "unknown_concept"


def _fallback_payload(query: str, note: str, raw_output: str = "") -> dict[str, Any]:
    keywords = _normalize_keyword_list(TOKEN_RE.findall(query.lower())[:6])
    canonical = _heuristic_canonical(query)
    return {
        "concepts": [
            {
                "canonical": canonical,
                "synonyms": [],
                "related": [],
                "keywords": keywords,
                "confidence": 0.3,
            }
        ],
        "global_keywords": keywords,
        "notes": note,
        "_audit": {
            "provider": "fallback",
            "raw_output": raw_output,
            "validation_errors": [note],
        },
    }


def _extract_json_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = JSON_OBJECT_RE.search(stripped)
    if match:
        return match.group(0)
    raise ValueError("Model response did not contain a JSON object.")


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Extractor response must be a JSON object.")

    raw_concepts = payload.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise ValueError("Extractor response must contain a non-empty 'concepts' list.")

    normalized_concepts: list[dict[str, Any]] = []
    for item in raw_concepts:
        if not isinstance(item, dict):
            continue
        normalized_concepts.append(
            {
                "canonical": _normalize_canonical(item.get("canonical")),
                "synonyms": _normalize_keyword_list(item.get("synonyms")),
                "related": _normalize_keyword_list(item.get("related")),
                "keywords": _normalize_keyword_list(item.get("keywords")),
                "confidence": _clamp_confidence(item.get("confidence")),
            }
        )

    if not normalized_concepts:
        raise ValueError("Extractor response did not contain any valid concept objects.")

    return {
        "concepts": normalized_concepts,
        "global_keywords": _normalize_keyword_list(payload.get("global_keywords")),
        "notes": clean_text(payload.get("notes")) or "",
    }


def _generate_with_adk(query: str, model: Any = None) -> tuple[str | None, dict[str, Any]]:
    audit: dict[str, Any] = {"provider": "adk", "model": None}
    try:
        if model is None:
            from shared.adk_model_provider import build_agent_model

            model = build_agent_model("ati_search", default_provider="azure")
        audit["model"] = getattr(model, "model", model if isinstance(model, str) else type(model).__name__)
    except Exception as exc:
        audit["error"] = f"ADK provider unavailable: {exc}"
        return None, audit

    if isinstance(model, str):
        audit["error"] = "ADK provider returned a model name string, not a callable LLM object."
        return None, audit

    prompt = STRICT_USER_PROMPT.format(query=query)

    candidates: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
        ("generate_content", (prompt,), {"temperature": 0}),
        ("generate_content", (), {"contents": prompt, "temperature": 0}),
        ("invoke", (prompt,), {"temperature": 0}),
        ("invoke", (prompt,), {}),
        ("call", (prompt,), {"temperature": 0}),
        ("complete", (prompt,), {"temperature": 0}),
    ]

    for method_name, args, kwargs in candidates:
        method = getattr(model, method_name, None)
        if method is None:
            continue
        try:
            response = method(*args, **kwargs)
        except TypeError:
            continue
        except Exception as exc:
            audit["error"] = f"ADK generation failed via {method_name}: {exc}"
            return None, audit
        content = _response_text(response)
        if content:
            return content, audit

    if callable(model):
        try:
            response = model(prompt)
        except Exception as exc:
            audit["error"] = f"ADK callable model failed: {exc}"
            return None, audit
        content = _response_text(response)
        if content:
            return content, audit

    audit["error"] = "ADK provider did not expose a usable generation method."
    return None, audit


def _response_text(response: Any) -> str | None:
    if response is None:
        return None
    if isinstance(response, str):
        return response
    text = clean_text(getattr(response, "text", None))
    if text:
        return text
    candidates = [
        getattr(response, "content", None),
        getattr(response, "output_text", None),
        getattr(response, "message", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str):
            return candidate
        if isinstance(candidate, list):
            parts = []
            for item in candidate:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    part = clean_text(item.get("text") or item.get("content"))
                    if part:
                        parts.append(part)
            joined = " ".join(parts).strip()
            if joined:
                return joined
    if hasattr(response, "choices"):
        choices = getattr(response, "choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = getattr(first, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
    return clean_text(response)


def _generate_with_openai(query: str, model: str | None = None) -> tuple[str | None, dict[str, Any]]:
    audit: dict[str, Any] = {"provider": "openai", "model": model or os.getenv("ATI_SEARCH_OPENAI_MODEL")}
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        audit["error"] = "OPENAI_API_KEY is not configured."
        return None, audit

    model_name = model or os.getenv("ATI_SEARCH_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI_MODEL

    try:
        from openai import OpenAI
    except Exception as exc:
        audit["error"] = f"openai package unavailable: {exc}"
        return None, audit

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": STRICT_SYSTEM_PROMPT},
                {"role": "user", "content": STRICT_USER_PROMPT.format(query=query)},
            ],
        )
    except Exception as exc:
        audit["error"] = f"OpenAI completion failed: {exc}"
        return None, audit

    content = response.choices[0].message.content if response.choices else None
    return clean_text(content), {"provider": "openai", "model": model_name}


def llm_extract_candidates(query: str, model: Any = None) -> dict[str, Any]:
    """Extract canonical concepts and keywords from a query, returning validated JSON."""
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return _fallback_payload(query="", note="Empty query passed to extractor.")

    raw_output: str | None = None
    audit: dict[str, Any] = {"provider": "fallback", "model": None}

    raw_output, audit = _generate_with_adk(cleaned_query, model=model)
    if not raw_output:
        raw_output, audit = _generate_with_openai(cleaned_query, model=model if isinstance(model, str) else None)

    if not raw_output:
        note = audit.get("error") or "No LLM provider was available."
        return _fallback_payload(cleaned_query, note=note)

    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        try:
            payload = json.loads(_extract_json_text(raw_output))
        except (ValueError, json.JSONDecodeError) as exc:
            note = f"Failed to parse extractor JSON: {exc}"
            LOGGER.warning(note)
            return _fallback_payload(cleaned_query, note=note, raw_output=raw_output)

    try:
        normalized = _normalize_payload(payload)
    except ValueError as exc:
        note = f"Failed to validate extractor payload: {exc}"
        LOGGER.warning(note)
        return _fallback_payload(cleaned_query, note=note, raw_output=raw_output)

    normalized["_audit"] = {
        "provider": audit.get("provider"),
        "model": audit.get("model"),
        "raw_output": raw_output,
        "validation_errors": [],
    }
    return normalized
