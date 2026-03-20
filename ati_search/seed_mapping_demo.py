"""Create and seed the ATI mapping database with a minimal approved demo ontology."""

from __future__ import annotations

from ati_search.mapping import create_mapping_db, seed_mapping


def main() -> None:
    """Create the default mapping DB and seed a demo player-merge concept."""
    db_path = create_mapping_db()
    seed_mapping(
        [
            {
                "concept": "player_merge",
                "synonyms": ["merge player", "player merge", "account merge"],
                "patterns": ["merge", "merging player", "duplicate player"],
                "related_concepts": ["unimerge", "duplicate_player_resolution"],
                "weight": 1.0,
                "source": "demo_seed",
                "created_by": "seed_mapping_demo",
                "approved": True,
                "approver": "seed_mapping_demo",
            }
        ]
    )
    print(f"Seeded mapping database at {db_path}")


if __name__ == "__main__":
    main()
