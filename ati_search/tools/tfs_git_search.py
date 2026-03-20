from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

from ati_search.env import DOTENV_VALUES, get_env_value
from ati_search.tool_utils import (
    DEFAULT_TFS_API_VERSION,
    DEFAULT_TFS_RESULT_LIMIT,
    TFS_TIMEOUT_SECONDS,
    build_git_item_url,
    clean_text,
    file_name_from_path,
    pick_first,
    strip_html_tags,
)

DEFAULT_DEFECT_SEARCH_FIELDS = ["System.Title", "System.Tags", "ATI.Bug.Description"]
DEFECT_WORK_ITEM_TYPES = ["Defect", "Bug"]
DEFECT_KEYWORD_RE = re.compile(r"\b(defect|defects|bug|bugs)\b", re.IGNORECASE)
WORK_ITEM_TERM_RE = re.compile(r"\b(work\s*items?|tickets?|issues?)\b", re.IGNORECASE)
CODE_SEARCH_TERM_RE = re.compile(
    r"\b(repo|repos|repository|repositories|git|code|file|files|path|paths|branch|source)\b",
    re.IGNORECASE,
)
LEADING_SEARCH_VERB_RE = re.compile(
    r"^(?:can you|could you|would you|please|kindly|find|search(?: for)?|look up|lookup|show(?: me)?|list|get|fetch)\b[\s,:-]*",
    re.IGNORECASE,
)
TRAILING_PUNCTUATION_RE = re.compile(r"^[\s?.!,:;]+|[\s?.!,:;]+$")


def tfs_auth(config: dict[str, str]) -> HTTPBasicAuth:
    return HTTPBasicAuth("", config["pat"])


def tfs_error_response(error_type: str, message: str, **extra: Any) -> dict[str, Any]:
    response = {
        "status": "error",
        "error_type": error_type,
        "message": message,
    }
    response.update(extra)
    return response


def normalize_work_item_types(work_item_type: str | Sequence[str] | None) -> list[str]:
    if work_item_type is None:
        return []
    if isinstance(work_item_type, str):
        values = [work_item_type]
    else:
        values = list(work_item_type)

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


def infer_search_options(
    query: str,
    *,
    repo_name: str | None,
    include_git_matches: bool,
    work_item_type: str | Sequence[str] | None,
    work_item_search_fields: list[str] | None,
) -> tuple[str, bool, str | list[str] | None, list[str] | None, list[str]]:
    effective_query = clean_text(query) or ""
    while True:
        stripped_query = LEADING_SEARCH_VERB_RE.sub("", effective_query, count=1)
        if stripped_query == effective_query:
            break
        effective_query = stripped_query
    effective_query = TRAILING_PUNCTUATION_RE.sub("", effective_query)

    notes: list[str] = []
    has_defect_intent = bool(DEFECT_KEYWORD_RE.search(effective_query))
    has_code_intent = bool(CODE_SEARCH_TERM_RE.search(effective_query))

    if has_defect_intent:
        stripped_query = DEFECT_KEYWORD_RE.sub(" ", effective_query)
        stripped_query = WORK_ITEM_TERM_RE.sub(" ", stripped_query)
        stripped_query = clean_text(stripped_query) or ""
        if stripped_query:
            effective_query = stripped_query

        if work_item_search_fields is None:
            work_item_search_fields = DEFAULT_DEFECT_SEARCH_FIELDS.copy()

        if work_item_type is None:
            work_item_type = DEFECT_WORK_ITEM_TYPES.copy()
            notes.append(
                "Interpreted the request as a defect search and matched work item types Defect or Bug."
            )

        if include_git_matches and not repo_name and not has_code_intent:
            include_git_matches = False
            notes.append(
                "Skipped Git and repository path search because the request looked like a work item defect search."
            )

    normalized_query = clean_text(effective_query) or clean_text(query) or ""
    original_query = clean_text(query) or ""
    if normalized_query and normalized_query.lower() != original_query.lower():
        notes.append(f"Normalized the search text from '{original_query}' to '{normalized_query}'.")

    return normalized_query, include_git_matches, work_item_type, work_item_search_fields, notes


def get_tfs_config(
    env_values: dict[str, str] | None = None,
) -> tuple[dict[str, str] | None, str | None]:
    source = env_values if env_values is not None else DOTENV_VALUES
    tfs_url = get_env_value("TFS_URL", source)
    project = get_env_value("PROJECT", source)
    pat = get_env_value("PAT", source)
    api_version = get_env_value("API_VERSION", source) or DEFAULT_TFS_API_VERSION
    default_repo = get_env_value("TFS_DEFAULT_REPO", source)

    missing = [
        name
        for name, value in (("TFS_URL", tfs_url), ("PROJECT", project), ("PAT", pat))
        if not value
    ]
    if missing:
        return None, (
            "Missing TFS configuration: "
            + ", ".join(missing)
            + ". Configure them in ati_search/.env, shared/.env, repo .env, or the ScrumMaster2 .env fallback."
        )

    return {
        "tfs_url": tfs_url.rstrip("/"),
        "project": project,
        "pat": pat,
        "api_version": api_version,
        "default_repo": default_repo or "",
    }, None


def tfs_request(
    method: str,
    url: str,
    config: dict[str, str],
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: int = TFS_TIMEOUT_SECONDS,
) -> requests.Response:
    return requests.request(
        method=method,
        url=url,
        params=params,
        json=json_body,
        auth=tfs_auth(config),
        timeout=timeout,
    )


def project_url(config: dict[str, str], suffix: str) -> str:
    return f"{config['tfs_url']}/{config['project']}{suffix}"


def collection_url(config: dict[str, str], suffix: str) -> str:
    return f"{config['tfs_url']}{suffix}"


def field_reference_name(field_name: str) -> str | None:
    normalized = clean_text(field_name)
    if not normalized:
        return None
    if "." in normalized and " " not in normalized:
        return normalized
    return None


def resolve_work_item_field_references(
    config: dict[str, str],
    search_fields: list[str] | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if not search_fields:
        return ["System.Title", "System.Tags"], [], {}

    resolved: list[str] = []
    unresolved: list[str] = []
    raw: dict[str, Any] = {}
    pending = [field for field in search_fields if clean_text(field)]
    direct = [field for field in pending if field_reference_name(field)]
    lookup = [field for field in pending if field not in direct]

    resolved.extend(field_reference_name(field) for field in direct if field_reference_name(field))
    if not lookup:
        return resolved, unresolved, raw

    fields_url = collection_url(config, "/_apis/wit/fields")
    try:
        response = tfs_request(
            "GET",
            fields_url,
            config,
            params={"api-version": config["api_version"]},
        )
    except requests.Timeout:
        return resolved, lookup, raw
    except requests.RequestException:
        return resolved, lookup, raw

    if response.status_code != 200:
        raw["wit_fields"] = {
            "status_code": response.status_code,
            "response_text": clean_text(response.text),
        }
        return resolved, lookup, raw

    try:
        payload = response.json()
    except ValueError:
        return resolved, lookup, raw

    raw["wit_fields"] = payload
    available_fields = payload.get("value", [])
    by_name: dict[str, str] = {}
    by_reference: dict[str, str] = {}
    for item in available_fields:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name"))
        reference_name = clean_text(item.get("referenceName"))
        if name and reference_name:
            by_name[name.lower()] = reference_name
            by_reference[reference_name.lower()] = reference_name

    for field in lookup:
        normalized = clean_text(field)
        if not normalized:
            continue
        match = by_reference.get(normalized.lower()) or by_name.get(normalized.lower())
        if match:
            resolved.append(match)
        else:
            unresolved.append(normalized)

    return resolved, unresolved, raw


def normalize_repo(repo: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    name = pick_first(repo.get("name"), repo.get("repositoryName")) or "unknown"
    remote_url = pick_first(repo.get("remoteUrl"), repo.get("webUrl"), repo.get("url"))
    return {
        "id": pick_first(repo.get("id")),
        "name": name,
        "default_branch": pick_first(repo.get("defaultBranch")),
        "remote_url": remote_url,
        "web_url": pick_first(
            repo.get("webUrl"),
            remote_url,
            build_git_item_url(config["tfs_url"], config["project"], name, None),
        ),
    }


def list_repositories(
    config: dict[str, str],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    url = project_url(config, "/_apis/git/repositories")
    try:
        response = tfs_request(
            "GET",
            url,
            config,
            params={"api-version": config["api_version"]},
        )
    except requests.Timeout:
        return None, tfs_error_response(
            "timeout",
            "Timed out while listing TFS Git repositories.",
        )
    except requests.RequestException as exc:
        return None, tfs_error_response(
            "request_error",
            f"Failed to list TFS Git repositories: {exc}",
        )

    if response.status_code != 200:
        return None, tfs_error_response(
            "http_error",
            "TFS repository listing returned a non-200 response.",
            status_code=response.status_code,
            response_text=clean_text(response.text),
        )

    try:
        payload = response.json()
    except ValueError:
        return None, tfs_error_response(
            "invalid_json",
            "TFS repository listing returned invalid JSON.",
        )

    repos = payload.get("value")
    if not isinstance(repos, list):
        return None, tfs_error_response(
            "unexpected_response_structure",
            "TFS repository listing response does not include a 'value' array.",
        )

    normalized = [normalize_repo(repo, config) for repo in repos if isinstance(repo, dict)]
    return normalized, {"repositories": payload}


def resolve_repositories(
    repositories: list[dict[str, Any]],
    repo_name: str | None,
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    if not repo_name:
        return repositories, None

    target = repo_name.strip().lower()
    selected = [repo for repo in repositories if repo.get("name", "").lower() == target]
    if selected:
        return selected, None

    return None, tfs_error_response(
        "unknown_repository",
        f"Repository '{repo_name}' was not found in the configured TFS project.",
        available_repositories=sorted(repo.get("name", "") for repo in repositories),
    )


def extract_code_search_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_results = payload.get("results")
    if isinstance(raw_results, dict):
        values = raw_results.get("values")
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
        return []
    if isinstance(raw_results, list):
        return [item for item in raw_results if isinstance(item, dict)]
    if isinstance(payload.get("value"), list):
        return [item for item in payload["value"] if isinstance(item, dict)]
    return []


def normalize_code_search_match(
    item: dict[str, Any],
    config: dict[str, str],
    repositories_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    repository_info = item.get("repository") if isinstance(item.get("repository"), dict) else {}
    repo_name = pick_first(
        item.get("repositoryName"),
        repository_info.get("name"),
        item.get("repoName"),
    ) or "unknown"
    repo = repositories_by_name.get(repo_name.lower(), {})
    path = pick_first(
        item.get("path"),
        item.get("filePath"),
        item.get("fileName"),
    )
    branch = pick_first(
        item.get("branchName"),
        item.get("version"),
        repo.get("default_branch"),
    )
    snippet = pick_first(
        item.get("content"),
        item.get("matches"),
        item.get("hitHighlightedSummary"),
    )
    return {
        "match_type": "code_search",
        "repository": repo_name,
        "path": path,
        "file_name": file_name_from_path(path),
        "branch": branch,
        "url": pick_first(
            item.get("webUrl"),
            item.get("url"),
            build_git_item_url(config["tfs_url"], config["project"], repo_name, path),
        ),
        "snippet": strip_html_tags(snippet),
    }


def search_code_api(
    query: str,
    repositories: list[dict[str, Any]],
    config: dict[str, str],
    top: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    url = collection_url(config, "/_apis/search/codesearchresults")
    payload: dict[str, Any] = {
        "searchText": query,
        "$skip": 0,
        "$top": top,
        "includeFacets": False,
        "filters": {"Project": [config["project"]]},
    }
    if repositories:
        payload["filters"]["Repository"] = [repo["name"] for repo in repositories]

    notes: list[str] = []
    raw: dict[str, Any] = {}
    try:
        response = tfs_request(
            "POST",
            url,
            config,
            params={"api-version": "4.1-preview.1"},
            json_body=payload,
        )
    except requests.Timeout:
        notes.append("TFS code search timed out. Falling back to repository path matching.")
        return [], notes, raw
    except requests.RequestException as exc:
        notes.append(
            f"TFS code search request failed ({exc}). Falling back to repository path matching."
        )
        return [], notes, raw

    if response.status_code != 200:
        if response.status_code in {400, 404, 405, 500, 503}:
            notes.append(
                "TFS code-content search is unavailable on this server. Using repository path matching instead."
            )
            raw["code_search"] = {
                "status_code": response.status_code,
                "response_text": clean_text(response.text),
            }
            return [], notes, raw
        raise RuntimeError(
            f"TFS code search returned HTTP {response.status_code}: "
            f"{clean_text(response.text) or 'no response body'}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        notes.append(
            f"TFS code search returned invalid JSON ({exc}). Falling back to repository path matching."
        )
        return [], notes, raw

    raw["code_search"] = data
    repositories_by_name = {repo["name"].lower(): repo for repo in repositories}
    matches = [
        normalize_code_search_match(item, config, repositories_by_name)
        for item in extract_code_search_entries(data)
    ]
    return matches[:top], notes, raw


def normalize_path_match(
    item: dict[str, Any],
    repo: dict[str, Any],
    config: dict[str, str],
) -> dict[str, Any]:
    path = pick_first(item.get("path"), item.get("gitObjectPath"))
    return {
        "match_type": "path_match",
        "repository": repo["name"],
        "path": path,
        "file_name": file_name_from_path(path),
        "branch": repo.get("default_branch"),
        "url": build_git_item_url(config["tfs_url"], config["project"], repo["name"], path),
        "snippet": None,
    }


def branch_name_from_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    prefix = "refs/heads/"
    return ref[len(prefix) :] if ref.startswith(prefix) else ref


def search_repository_paths(
    query: str,
    repositories: list[dict[str, Any]],
    config: dict[str, str],
    top: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    lowered_query = query.lower()
    matches: list[dict[str, Any]] = []
    raw_items: list[dict[str, Any]] = []
    notes = ["Repository fallback only searched repo and file paths. Code-content search was unavailable."]

    for repo in repositories:
        if len(matches) >= top:
            break
        repo_id = repo.get("id")
        if not repo_id:
            continue

        url = project_url(config, f"/_apis/git/repositories/{repo_id}/items")
        params = {
            "scopePath": "/",
            "recursionLevel": "Full",
            "includeContentMetadata": "true",
            "latestProcessedChange": "true",
            "api-version": config["api_version"],
        }
        branch_name = branch_name_from_ref(repo.get("default_branch"))
        if branch_name:
            params["versionDescriptor.versionType"] = "branch"
            params["versionDescriptor.version"] = branch_name
        try:
            response = tfs_request(
                "GET",
                url,
                config,
                params=params,
                timeout=max(TFS_TIMEOUT_SECONDS, 60),
            )
        except requests.Timeout:
            notes.append(f"Timed out while scanning repository '{repo['name']}'.")
            continue
        except requests.RequestException as exc:
            notes.append(f"Failed to scan repository '{repo['name']}': {exc}")
            continue

        if response.status_code != 200:
            notes.append(f"Repository '{repo['name']}' path scan returned HTTP {response.status_code}.")
            continue

        try:
            payload = response.json()
        except ValueError:
            notes.append(f"Repository '{repo['name']}' path scan returned invalid JSON.")
            continue

        raw_items.append({"repository": repo["name"], "payload": payload})
        for item in payload.get("value", []):
            if not isinstance(item, dict) or item.get("isFolder"):
                continue
            path = pick_first(item.get("path"), item.get("gitObjectPath"))
            if not path or lowered_query not in path.lower():
                continue
            matches.append(normalize_path_match(item, repo, config))
            if len(matches) >= top:
                break

    return matches, notes, {"path_search": raw_items}


def work_item_query_text(
    query: str,
    top: int,
    config: dict[str, str],
    *,
    search_fields: list[str] | None = None,
    work_item_type: str | Sequence[str] | None = None,
) -> str:
    escaped = query.replace("'", "''")
    clauses = []
    for field_name in search_fields or ["System.Title", "System.Tags"]:
        clauses.append(f"[{field_name}] CONTAINS '{escaped}'")
    filters = [f"[System.TeamProject] = '{config['project']}'"]
    work_item_types = normalize_work_item_types(work_item_type)
    if len(work_item_types) == 1:
        escaped_work_item_type = work_item_types[0].replace("'", "''")
        filters.append(f"[System.WorkItemType] = '{escaped_work_item_type}'")
    elif work_item_types:
        escaped_work_item_types = ", ".join(
            "'" + item.replace("'", "''") + "'" for item in work_item_types
        )
        filters.append(f"[System.WorkItemType] IN ({escaped_work_item_types})")
    return f"""
        SELECT [System.Id]
        FROM WorkItems
        WHERE {" AND ".join(filters)}
        AND (
            {" OR ".join(clauses)}
        )
        ORDER BY [System.ChangedDate] DESC
    """


def normalize_work_item(item: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    fields = item.get("fields", {}) if isinstance(item.get("fields"), dict) else {}
    work_item_id = pick_first(item.get("id"))
    return {
        "id": work_item_id,
        "title": pick_first(fields.get("System.Title")),
        "ATI.Bug.Description": strip_html_tags(fields.get("ATI.Bug.Description")),
        "state": pick_first(fields.get("System.State")),
        "work_item_type": pick_first(fields.get("System.WorkItemType")),
        "assigned_to": pick_first(fields.get("System.AssignedTo")),
        "changed_date": pick_first(fields.get("System.ChangedDate")),
        "url": (
            f"{config['tfs_url']}/{config['project']}/_workitems/edit/{work_item_id}"
            if work_item_id
            else None
        ),
    }


def search_work_items(
    query: str,
    config: dict[str, str],
    top: int,
    *,
    work_item_type: str | Sequence[str] | None = None,
    search_fields: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if top <= 0:
        return [], [], {}

    wiql_url = project_url(config, "/_apis/wit/wiql")
    notes: list[str] = []
    raw: dict[str, Any] = {}
    resolved_search_fields, unresolved_fields, field_raw = resolve_work_item_field_references(
        config,
        search_fields,
    )
    raw.update(field_raw)
    if unresolved_fields:
        notes.append(
            "Could not resolve work item search fields: "
            + ", ".join(unresolved_fields)
            + "."
        )
    if not resolved_search_fields:
        return [], notes, raw

    try:
        wiql_response = tfs_request(
            "POST",
            wiql_url,
            config,
            params={"api-version": config["api_version"]},
            json_body={
                "query": work_item_query_text(
                    query,
                    top,
                    config,
                    search_fields=resolved_search_fields,
                    work_item_type=work_item_type,
                )
            },
        )
    except requests.Timeout:
        notes.append("TFS work item search timed out.")
        return [], notes, raw
    except requests.RequestException as exc:
        notes.append(f"TFS work item search request failed: {exc}")
        return [], notes, raw

    if wiql_response.status_code != 200:
        notes.append(f"TFS work item search returned HTTP {wiql_response.status_code}.")
        raw["wiql"] = {
            "status_code": wiql_response.status_code,
            "response_text": clean_text(wiql_response.text),
        }
        return [], notes, raw

    try:
        wiql_payload = wiql_response.json()
    except ValueError:
        notes.append("TFS work item search returned invalid JSON.")
        return [], notes, raw

    raw["wiql"] = wiql_payload
    work_items = wiql_payload.get("workItems", [])
    ids = [str(item.get("id")) for item in work_items if isinstance(item, dict) and item.get("id")]
    if not ids:
        return [], notes, raw

    fields = ",".join(
        [
            "System.Id",
            "System.Title",
            "ATI.Bug.Description",
            "System.State",
            "System.WorkItemType",
            "System.AssignedTo",
            "System.ChangedDate",
        ]
    )
    details_url = collection_url(config, "/_apis/wit/workitems")
    try:
        details_response = tfs_request(
            "GET",
            details_url,
            config,
            params={
                "ids": ",".join(ids[:top]),
                "fields": fields,
                "api-version": config["api_version"],
            },
        )
    except requests.Timeout:
        notes.append("Fetching TFS work item details timed out.")
        return [], notes, raw
    except requests.RequestException as exc:
        notes.append(f"Fetching TFS work item details failed: {exc}")
        return [], notes, raw

    if details_response.status_code != 200:
        notes.append(f"TFS work item details returned HTTP {details_response.status_code}.")
        raw["work_items"] = {
            "status_code": details_response.status_code,
            "response_text": clean_text(details_response.text),
        }
        return [], notes, raw

    try:
        details_payload = details_response.json()
    except ValueError:
        notes.append("TFS work item details returned invalid JSON.")
        return [], notes, raw

    raw["work_items"] = details_payload
    normalized = [
        normalize_work_item(item, config)
        for item in details_payload.get("value", [])
        if isinstance(item, dict)
    ]
    return normalized[:top], notes, raw


def tfs_git_search(
    query: str,
    repo_name: str | None = None,
    include_work_items: bool = True,
    include_git_matches: bool = True,
    top: int = DEFAULT_TFS_RESULT_LIMIT,
    work_item_type: str | Sequence[str] | None = None,
    work_item_search_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Search TFS/Azure DevOps Git repositories and optionally related work items."""
    cleaned_query = clean_text(query)
    if not cleaned_query:
        return tfs_error_response("validation_error", "A non-empty search query is required.")

    try:
        normalized_top = max(1, min(int(top), 50))
    except (TypeError, ValueError):
        return tfs_error_response("validation_error", "'top' must be an integer between 1 and 50.")

    (
        effective_query,
        effective_include_git_matches,
        effective_work_item_type,
        effective_work_item_search_fields,
        inferred_notes,
    ) = infer_search_options(
        cleaned_query,
        repo_name=repo_name,
        include_git_matches=include_git_matches,
        work_item_type=work_item_type,
        work_item_search_fields=work_item_search_fields,
    )

    config, config_error = get_tfs_config()
    if config is None:
        return tfs_error_response(
            "missing_configuration",
            config_error or "Missing TFS configuration.",
        )

    selected_repositories: list[dict[str, Any]] = []
    if effective_include_git_matches:
        repositories, repo_list_error = list_repositories(config)
        if repositories is None:
            return repo_list_error or tfs_error_response(
                "request_error",
                "Failed to list TFS repositories.",
            )

        selected_repositories, repo_selection_error = resolve_repositories(repositories, repo_name)
        if selected_repositories is None:
            return repo_selection_error or tfs_error_response(
                "unknown_repository",
                "Repository not found.",
            )

    notes: list[str] = list(inferred_notes)
    raw: dict[str, Any] = {}
    git_matches: list[dict[str, Any]] = []

    if effective_include_git_matches:
        try:
            git_matches, code_search_notes, code_search_raw = search_code_api(
                effective_query,
                selected_repositories,
                config,
                normalized_top,
            )
        except RuntimeError as exc:
            return tfs_error_response("http_error", str(exc))

        notes.extend(code_search_notes)
        raw.update(code_search_raw)

        if not git_matches:
            fallback_matches, fallback_notes, fallback_raw = search_repository_paths(
                effective_query,
                selected_repositories,
                config,
                normalized_top,
            )
            notes.extend(fallback_notes)
            raw.update(fallback_raw)
            git_matches = fallback_matches

    work_item_limit = max(0, normalized_top - len(git_matches))
    work_item_matches: list[dict[str, Any]] = []
    if include_work_items and work_item_limit > 0:
        work_item_matches, work_item_notes, work_item_raw = search_work_items(
            effective_query,
            config,
            work_item_limit,
            work_item_type=effective_work_item_type,
            search_fields=effective_work_item_search_fields,
        )
        notes.extend(work_item_notes)
        raw.update(work_item_raw)

    return {
        "status": "success",
        "summary": {
            "requested_query": cleaned_query,
            "query": effective_query,
            "repositories_scanned": [repo["name"] for repo in selected_repositories],
            "git_matches": git_matches[:normalized_top],
            "work_item_matches": work_item_matches[
                : max(0, normalized_top - len(git_matches[:normalized_top]))
            ],
            "notes": notes,
        },
        "raw": raw,
    }
