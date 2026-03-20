from __future__ import annotations

import argparse
import json
import os
from typing import Any

from ati_search.tools.tfs_git_search import get_tfs_config
from ati_search.tools.tfs_git_search import list_repositories
from ati_search.tools.tfs_git_search import tfs_git_search


# Edit these defaults if you prefer "Run Python File" in an IDE over CLI args.
DEFAULT_QUERY = "SprintReport"
DEFAULT_REPO_NAME: str | None = None
DEFAULT_INCLUDE_WORK_ITEMS = True
DEFAULT_TOP = 5
DEFAULT_WORK_ITEM_TYPE = "Defect"
DEFAULT_WORK_ITEM_SEARCH_FIELDS = ["System.Title", "ATI.Bug.Description"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug entrypoint for ati_search.tools.tfs_git_search",
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Search text passed to tfs_git_search().",
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help="Optional repository name filter.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="Maximum number of results to request.",
    )
    parser.add_argument(
        "--no-work-items",
        action="store_true",
        help="Disable related work item search.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved TFS configuration before running the search.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full tool response including raw payloads.",
    )
    parser.add_argument("--tfs-url", help="Override TFS_URL for this run.")
    parser.add_argument("--project", help="Override PROJECT for this run.")
    parser.add_argument("--pat", help="Override PAT for this run.")
    parser.add_argument("--api-version", help="Override API_VERSION for this run.")
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="List repositories for the resolved TFS project and exit.",
    )
    return parser


def run_debug_search(
    query: str = DEFAULT_QUERY,
    repo_name: str | None = DEFAULT_REPO_NAME,
    include_work_items: bool = DEFAULT_INCLUDE_WORK_ITEMS,
    top: int = DEFAULT_TOP,
) -> dict[str, Any]:
    return tfs_git_search(
        query=query,
        repo_name=repo_name,
        include_work_items=include_work_items,
        include_git_matches=False,
        top=top,
        work_item_type=DEFAULT_WORK_ITEM_TYPE,
        work_item_search_fields=DEFAULT_WORK_ITEM_SEARCH_FIELDS,
    )


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "TFS_URL": args.tfs_url,
        "PROJECT": args.project,
        "PAT": args.pat,
        "API_VERSION": args.api_version,
    }
    for key, value in overrides.items():
        if value:
            os.environ[key] = value

    if args.print_config:
        config, error = get_tfs_config()
        masked = None
        if config is not None:
            masked = dict(config)
            pat = masked.get("pat")
            if pat:
                masked["pat"] = f"{pat[:4]}...{pat[-4:]}"
        print(
            json.dumps(
                {"config": masked, "config_error": error},
                indent=2,
                sort_keys=True,
            )
        )
    config, error = get_tfs_config()
    if args.list_repos:
        if config is None:
            print(json.dumps({"config_error": error}, indent=2, sort_keys=True))
            return 1
        repositories, repo_error = list_repositories(config)
        print(
            json.dumps(
                {
                    "project": config["project"],
                    "repositories": [repo.get("name") for repo in repositories or []],
                    "repository_error": repo_error if repositories is None else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_debug_search(
        query=args.query,
        repo_name=args.repo_name,
        include_work_items=not args.no_work_items,
        top=args.top,
    )
    if args.full or result.get("status") != "success":
        payload = result if args.full else result.get("summary", result)
    else:
        summary = result.get("summary", {})
        work_item_matches = summary.get("work_item_matches", [])
        payload = {
            "work_item_matches": [
                {
                    "title": item.get("title"),
                    "ATI.Bug.Description": item.get("ATI.Bug.Description"),
                }
                for item in work_item_matches
                if isinstance(item, dict) and item.get("title")
            ]
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
