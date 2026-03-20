"""Tests for the additive ATI semantic search extension."""

from __future__ import annotations

from ati_search.mapping import seed_mapping
from ati_search.tools.semantic_tool import semantic_search_tool


def test_semantic_search_tool_surfaces_unimerge_defect_via_mapping(tmp_path, monkeypatch) -> None:
    """A free-text merge query should resolve to player_merge and surface unimerge defects."""
    db_path = tmp_path / "ati_mapping.db"
    monkeypatch.setenv("ATI_SEARCH_ENABLE_CROSS_ENCODER", "0")

    seed_mapping(
        [
            {
                "concept": "player_merge",
                "synonyms": ["merge player", "player merge", "account merge"],
                "patterns": ["merge", "merging player", "duplicate player"],
                "related_concepts": ["unimerge", "duplicate_player_resolution"],
                "weight": 1.0,
                "source": "pytest",
                "created_by": "pytest",
                "approved": True,
                "approver": "pytest-reviewer",
            }
        ],
        path=db_path,
    )

    monkeypatch.setattr(
        "ati_search.tools.semantic_tool.query_mapping_by_keyword",
        lambda keyword: __import__("ati_search.mapping", fromlist=["query_mapping_by_keyword"]).query_mapping_by_keyword(
            keyword, path=db_path
        ),
    )
    monkeypatch.setattr(
        "ati_search.tools.semantic_tool.expand_concepts",
        lambda concepts: __import__("ati_search.mapping", fromlist=["expand_concepts"]).expand_concepts(
            concepts, path=db_path
        ),
    )
    monkeypatch.setattr(
        "ati_search.tools.semantic_tool.avid_search",
        lambda query: {
            "status": "success",
            "summary": {
                "top_normalized_entries": [
                    {
                        "title": "Player merge guide",
                        "cleaned_excerpt": "This document explains how to merge player records",
                        "source_name": "ATI Docs",
                        "readerUrl": "https://docs.example/player-merge",
                        "topicUrl": "https://docs.example/player-merge/topic",
                        "base_id": "doc-1",
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(
        "ati_search.tools.semantic_tool.tfs_git_search",
        lambda query, **_: {
            "status": "success",
            "summary": {
                "work_item_matches": [
                    {
                        "id": "BUG-7",
                        "title": "Account consolidation issue",
                        "ATI.Bug.Description": "unimerge was triggered",
                        "work_item_type": "Defect",
                        "state": "Active",
                        "url": "https://tfs.example/BUG-7",
                    }
                ]
            },
        },
    )
    monkeypatch.setattr(
        "ati_search.tools.semantic_tool._safe_cosine_similarity",
        lambda query, documents: ([0.82, 0.41], {"model": "test-double", "method": "stub"}),
    )
    monkeypatch.setattr(
        "ati_search.tools.semantic_tool._cross_encoder_scores",
        lambda query, documents: ([0.0, 0.0], {"enabled": False, "method": "stub"}),
    )

    result = semantic_search_tool("merging player")

    canonical_concepts = {item["canonical"] for item in result["canonical_concepts"]}
    assert "player_merge" in canonical_concepts

    result_titles = {item["title"] for item in result["results"]}
    assert "Player merge guide" in result_titles
    assert "Account consolidation issue" in result_titles

    defect_result = next(item for item in result["results"] if item["corpus"] == "defects")
    assert "unimerge" in defect_result["match_excerpt"].lower()
    assert "related concept" in defect_result["why_matched"].lower()
