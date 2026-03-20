"""Semantic ATI search tool that expands approved concepts and reranks docs plus defects."""

from __future__ import annotations

import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from ati_search.llm_extractor import llm_extract_candidates
from ati_search.mapping import expand_concepts, query_mapping_by_keyword
from ati_search.tool_utils import clean_text
from ati_search.tools.avid_search import avid_search
from ati_search.tools.tfs_git_search import (
    DEFAULT_DEFECT_SEARCH_FIELDS,
    DEFECT_WORK_ITEM_TYPES,
    tfs_git_search,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_WEIGHTS = {
    "semantic": 0.4,
    "lexical": 0.25,
    "concept": 0.2,
    "cross": 0.15,
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
_EMBEDDER_CACHE: dict[str, Any] = {}
_CROSS_ENCODER_CACHE: dict[str, Any] = {}


def _normalized_weights(include_cross: bool) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not include_cross:
        weights["cross"] = 0.0
    total = sum(weights.values()) or 1.0
    return {name: value / total for name, value in weights.items()}


def _lowered_tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.lower()))


def _normalize_doc_id(document: dict[str, Any]) -> str:
    for key in ("doc_id", "base_id", "id", "readerUrl", "topicUrl", "url", "title"):
        candidate = clean_text(document.get(key))
        if candidate:
            return candidate
    return "unknown-doc"


def _document_text(document: dict[str, Any]) -> str:
    return " ".join(
        value
        for value in [
            clean_text(document.get("title")),
            clean_text(document.get("cleaned_excerpt")),
            clean_text(document.get("tags")),
            clean_text(document.get("source_name")),
        ]
        if value
    ).strip()


def _normalize_documentation_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or response.get("status") != "success":
        return []

    summary = response.get("summary", {})
    entries = summary.get("top_normalized_entries", []) if isinstance(summary, dict) else []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "doc_id": _normalize_doc_id(entry),
                "corpus": "documentation",
                "title": clean_text(entry.get("title")) or "Untitled documentation result",
                "cleaned_excerpt": clean_text(entry.get("cleaned_excerpt")) or "",
                "source_name": clean_text(entry.get("source_name")) or "avid_search",
                "readerUrl": clean_text(entry.get("readerUrl")),
                "topicUrl": clean_text(entry.get("topicUrl")),
                "base_id": clean_text(entry.get("base_id")),
                "tags": clean_text(entry.get("document_type")),
                "provenance": {
                    "tool": "avid_search",
                    "readerUrl": clean_text(entry.get("readerUrl")),
                    "topicUrl": clean_text(entry.get("topicUrl")),
                    "base_id": clean_text(entry.get("base_id")),
                },
            }
        )
    return normalized


def _normalize_defect_response(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict) or response.get("status") != "success":
        return []

    summary = response.get("summary", {})
    entries = summary.get("work_item_matches", []) if isinstance(summary, dict) else []
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        normalized.append(
            {
                "doc_id": _normalize_doc_id(entry),
                "corpus": "defects",
                "title": clean_text(entry.get("title")) or "Untitled defect",
                "cleaned_excerpt": clean_text(entry.get("ATI.Bug.Description"))
                or clean_text(entry.get("description"))
                or clean_text(entry.get("title"))
                or "",
                "source_name": "tfs_git_search",
                "readerUrl": clean_text(entry.get("url")),
                "topicUrl": clean_text(entry.get("url")),
                "base_id": clean_text(entry.get("id")),
                "tags": clean_text(entry.get("work_item_type")),
                "provenance": {
                    "tool": "tfs_git_search",
                    "work_item_id": clean_text(entry.get("id")),
                    "url": clean_text(entry.get("url")),
                    "work_item_type": clean_text(entry.get("work_item_type")),
                    "state": clean_text(entry.get("state")),
                },
            }
        )
    return normalized


def _normalize_search_results(docs_response: dict[str, Any], defects_response: dict[str, Any]) -> list[dict[str, Any]]:
    documents = _normalize_documentation_response(docs_response)
    defects = _normalize_defect_response(defects_response)
    return documents + defects


def _embedder(model_name: str) -> Any:
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached

    from sentence_transformers import SentenceTransformer

    # Later optimization: pre-embed docs and store vectors in SQLite or a FAISS index
    # instead of encoding every candidate on demand.
    cached = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = cached
    return cached


def _cross_encoder(model_name: str) -> Any:
    cached = _CROSS_ENCODER_CACHE.get(model_name)
    if cached is not None:
        return cached

    from sentence_transformers import CrossEncoder

    cached = CrossEncoder(model_name)
    _CROSS_ENCODER_CACHE[model_name] = cached
    return cached


def _safe_cosine_similarity(query: str, documents: list[str]) -> tuple[list[float], dict[str, Any]]:
    model_name = os.getenv("ATI_SEARCH_SENTENCE_TRANSFORMER_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        model = _embedder(model_name)
        embeddings = model.encode([query] + documents, convert_to_numpy=True, normalize_embeddings=True)
        query_vector = embeddings[0]
        scores = []
        for vector in embeddings[1:]:
            score = float(query_vector @ vector)
            scores.append(max(0.0, min((score + 1.0) / 2.0, 1.0)))
        return scores, {"model": model_name, "method": "sentence_transformers"}
    except Exception as exc:
        LOGGER.warning("Falling back from sentence-transformers embeddings: %s", exc)
        query_tokens = _lowered_tokens(query)
        scores = []
        for document in documents:
            doc_tokens = _lowered_tokens(document)
            union = query_tokens | doc_tokens
            score = len(query_tokens & doc_tokens) / len(union) if union else 0.0
            scores.append(score)
        return scores, {"model": "token_overlap_fallback", "method": "fallback", "error": str(exc)}


def _cross_encoder_scores(query: str, documents: list[str]) -> tuple[list[float], dict[str, Any]]:
    enabled = os.getenv("ATI_SEARCH_ENABLE_CROSS_ENCODER", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return [0.0 for _ in documents], {"enabled": False, "method": "disabled"}

    model_name = os.getenv("ATI_SEARCH_CROSS_ENCODER_MODEL", DEFAULT_CROSS_ENCODER_MODEL)
    try:
        model = _cross_encoder(model_name)
        raw_scores = model.predict([(query, document) for document in documents])
        normalized = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]
        # Later optimization: persist pairwise reranker features or use a lighter hosted reranker.
        return normalized, {"enabled": True, "model": model_name, "method": "cross_encoder"}
    except Exception as exc:
        LOGGER.warning("CrossEncoder scoring unavailable: %s", exc)
        return [0.0 for _ in documents], {"enabled": False, "method": "unavailable", "error": str(exc)}


def _keyword_candidates(extractor_payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    llm_keywords = list(extractor_payload.get("global_keywords", []))
    llm_concepts: list[str] = []
    for concept in extractor_payload.get("concepts", []):
        if not isinstance(concept, dict):
            continue
        canonical = clean_text(concept.get("canonical"))
        if canonical:
            llm_concepts.append(canonical)
        llm_keywords.extend(concept.get("keywords", []))
    deduped_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for keyword in llm_keywords:
        cleaned = clean_text(keyword)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen_keywords:
            continue
        seen_keywords.add(lowered)
        deduped_keywords.append(cleaned)
    return deduped_keywords, llm_concepts


def _mapping_entries_for_keywords(keywords: Iterable[str]) -> dict[str, dict[str, Any]]:
    entries_by_concept: dict[str, dict[str, Any]] = {}
    for keyword in keywords:
        for entry in query_mapping_by_keyword(keyword):
            entries_by_concept[entry["concept"]] = entry
    return entries_by_concept


def _canonical_concepts(
    extractor_payload: dict[str, Any],
    mapping_entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    concepts: dict[str, dict[str, Any]] = {}

    for concept_name, entry in mapping_entries.items():
        concepts[concept_name] = {
            "canonical": concept_name,
            "confidence": float(entry.get("weight", 1.0)),
            "source": "mapping",
            "approved": True,
        }

    for concept in extractor_payload.get("concepts", []):
        if not isinstance(concept, dict):
            continue
        canonical = clean_text(concept.get("canonical"))
        if not canonical:
            continue
        if canonical in concepts:
            concepts[canonical]["confidence"] = max(
                float(concepts[canonical]["confidence"]),
                float(concept.get("confidence", 0.0) or 0.0),
            )
            continue
        concepts[canonical] = {
            "canonical": canonical,
            "confidence": float(concept.get("confidence", 0.0) or 0.0),
            "source": "llm_suggested",
            "approved": False,
        }

    return sorted(concepts.values(), key=lambda item: (-item["confidence"], item["canonical"]))


def _lexical_signal(
    document: dict[str, Any],
    expanded: dict[str, Any],
    expanded_keywords: list[str],
) -> tuple[float, list[str], list[str]]:
    haystack = " ".join(
        value
        for value in [
            clean_text(document.get("title")),
            clean_text(document.get("cleaned_excerpt")),
            clean_text(document.get("tags")),
        ]
        if value
    ).lower()
    matched_terms: list[str] = []
    matched_reasons: list[str] = []

    for synonym in expanded.get("expanded_synonyms", []):
        if synonym.lower() in haystack:
            matched_terms.append(synonym)
            matched_reasons.append(f"Matched approved mapping synonym '{synonym}'.")

    for related in expanded.get("expanded_related_concepts", []):
        if related.lower() in haystack:
            matched_terms.append(related)
            matched_reasons.append(f"Matched approved mapping related concept '{related}'.")

    for pattern in expanded.get("expanded_patterns", []):
        try:
            if re.search(pattern, haystack, re.IGNORECASE):
                matched_terms.append(pattern)
                matched_reasons.append(f"Matched approved mapping pattern '{pattern}'.")
        except re.error:
            if pattern.lower() in haystack:
                matched_terms.append(pattern)
                matched_reasons.append(f"Matched approved mapping pattern '{pattern}'.")

    for keyword in expanded_keywords:
        if keyword.lower() in haystack:
            matched_terms.append(keyword)

    unique_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in matched_terms:
        lowered = term.lower()
        if lowered in seen_terms:
            continue
        seen_terms.add(lowered)
        unique_terms.append(term)

    lexical_space = max(
        len(expanded_keywords)
        + len(expanded.get("expanded_synonyms", []))
        + len(expanded.get("expanded_related_concepts", []))
        + len(expanded.get("expanded_patterns", [])),
        1,
    )
    score = min(1.0, len(unique_terms) / lexical_space * 3.0)
    return score, unique_terms, matched_reasons


def _concept_signal(
    matched_terms: list[str],
    matched_reasons: list[str],
    mapping_entries: dict[str, dict[str, Any]],
) -> tuple[float, list[str]]:
    if not matched_terms or not mapping_entries:
        return 0.0, matched_reasons

    weight = 0.0
    reasons = list(matched_reasons)
    lowered_terms = {term.lower() for term in matched_terms}
    for concept, entry in mapping_entries.items():
        entry_terms = {
            value.lower()
            for value in (
                entry.get("synonyms", [])
                + entry.get("patterns", [])
                + entry.get("related_concepts", [])
                + [entry.get("concept", "")]
            )
            if clean_text(value)
        }
        if lowered_terms & entry_terms:
            weight = max(weight, min(float(entry.get("weight", 1.0) or 1.0), 1.0))
            reasons.append(
                f"Approved mapping boosted canonical concept '{concept}' with weight {float(entry.get('weight', 1.0)):.2f}."
            )
    score = min(1.0, 0.3 + weight * 0.7) if weight else 0.0
    return score, reasons


def _match_method(semantic_score: float, lexical_score: float, concept_score: float) -> str:
    has_semantic = semantic_score >= 0.2
    has_lexical = lexical_score > 0.0 or concept_score > 0.0
    if has_semantic and has_lexical:
        return "both"
    if has_semantic:
        return "semantic"
    return "lexical"


def semantic_search_tool(query: str, topk: int = 10) -> dict[str, Any]:
    """Search docs and defects in parallel, expand mapped concepts, and rerank combined results."""
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return {
            "query": "",
            "canonical_concepts": [],
            "expanded_keywords": [],
            "results": [],
            "debug": {"error": "A non-empty query is required."},
        }

    extractor_payload = llm_extract_candidates(cleaned_query)
    expanded_keywords, llm_concepts = _keyword_candidates(extractor_payload)
    mapping_entries = _mapping_entries_for_keywords(expanded_keywords)
    approved_concepts = list(mapping_entries)
    expanded = expand_concepts(approved_concepts or llm_concepts)

    combined_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for keyword in (
        expanded_keywords
        + expanded.get("expanded_synonyms", [])
        + expanded.get("expanded_related_concepts", [])
        + expanded.get("expanded_patterns", [])
    ):
        cleaned = clean_text(keyword)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen_keywords:
            continue
        seen_keywords.add(lowered)
        combined_keywords.append(cleaned)

    with ThreadPoolExecutor(max_workers=2) as executor:
        docs_future = executor.submit(avid_search, cleaned_query)
        defects_future = executor.submit(
            tfs_git_search,
            cleaned_query,
            include_git_matches=False,
            include_work_items=True,
            top=max(topk, 10),
            work_item_type=DEFECT_WORK_ITEM_TYPES,
            work_item_search_fields=DEFAULT_DEFECT_SEARCH_FIELDS,
        )
        docs_response = docs_future.result()
        defects_response = defects_future.result()

    documents = _normalize_search_results(docs_response, defects_response)
    if not documents:
        return {
            "query": cleaned_query,
            "canonical_concepts": _canonical_concepts(extractor_payload, mapping_entries),
            "expanded_keywords": combined_keywords,
            "results": [],
            "debug": {
                "llm": extractor_payload.get("_audit", {}),
                "documents_seen": 0,
                "embedding": {"method": "not_run"},
            },
        }

    doc_texts = [_document_text(document) for document in documents]
    semantic_scores, embedding_debug = _safe_cosine_similarity(cleaned_query, doc_texts)
    cross_scores, cross_debug = _cross_encoder_scores(cleaned_query, doc_texts)
    weights = _normalized_weights(include_cross=any(score > 0.0 for score in cross_scores))

    scored_results: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        lexical_score, matched_terms, matched_reasons = _lexical_signal(document, expanded, combined_keywords)
        concept_score, reasons = _concept_signal(matched_terms, matched_reasons, mapping_entries)
        semantic_score = semantic_scores[index]
        cross_score = cross_scores[index]
        final_score = (
            semantic_score * weights["semantic"]
            + lexical_score * weights["lexical"]
            + concept_score * weights["concept"]
            + cross_score * weights["cross"]
        )
        why_matched = reasons or ["Matched by semantic similarity."]
        result = {
            "doc_id": document["doc_id"],
            "corpus": document["corpus"],
            "title": document["title"],
            "score": round(final_score, 6),
            "match_excerpt": document["cleaned_excerpt"],
            "match_method": _match_method(semantic_score, lexical_score, concept_score),
            "why_matched": " ".join(why_matched),
            "provenance": document["provenance"],
        }
        scored_results.append(result)
        debug_rows.append(
            {
                "doc_id": document["doc_id"],
                "semantic_score": semantic_score,
                "lexical_score": lexical_score,
                "concept_match_weight": concept_score,
                "cross_score": cross_score,
                "matched_terms": matched_terms,
            }
        )

    scored_results.sort(key=lambda item: (-item["score"], item["corpus"], item["doc_id"]))
    debug_rows.sort(key=lambda item: next(
        (
            index
            for index, result in enumerate(scored_results)
            if result["doc_id"] == item["doc_id"]
        ),
        0,
    ))

    return {
        "query": cleaned_query,
        "canonical_concepts": _canonical_concepts(extractor_payload, mapping_entries),
        "expanded_keywords": combined_keywords,
        "results": scored_results[: max(1, topk)],
        "debug": {
            "llm": extractor_payload.get("_audit", {}),
            "llm_normalized": {
                "concepts": extractor_payload.get("concepts", []),
                "global_keywords": extractor_payload.get("global_keywords", []),
                "notes": extractor_payload.get("notes", ""),
            },
            "weights": weights,
            "embedding": embedding_debug,
            "cross_encoder": cross_debug,
            "scores": debug_rows[: max(1, topk)],
        },
    }
