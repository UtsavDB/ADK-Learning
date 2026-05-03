"""Synchronous Loyalty PlayerInfo API client and ADK-style PlayerInfo agent."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import quote

import requests


ENVIRONMENTS: Dict[str, Dict[str, str]] = {
    "dev": {
        "base_url": "https://dev-api.example.com/",
        "token_url": "https://dev-auth.example.com/oauth2/token",
        "client_id": "your-dev-client-id",
        "client_secret": "your-dev-client-secret",
        "scope": "your-dev-scope",
    },
    "qa": {
        "base_url": "https://qa-api.example.com/",
        "token_url": "https://qa-auth.example.com/oauth2/token",
        "client_id": "your-qa-client-id",
        "client_secret": "your-qa-client-secret",
        "scope": "your-qa-scope",
    },
    "prod": {
        "base_url": "https://prod-api.example.com/",
        "token_url": "https://prod-auth.example.com/oauth2/token",
        "client_id": "your-prod-client-id",
        "client_secret": "your-prod-client-secret",
        "scope": "your-prod-scope",
    },
}


class EnvironmentConfigError(ValueError):
    pass


class TokenFetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ApiCallResult:
    environment: str
    http_method: str
    relative_path: str
    request_query: Optional[Dict[str, str]]
    status_code: int
    is_success: bool
    json_body: Optional[Any]
    raw_body: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": self.environment,
            "http_method": self.http_method,
            "relative_path": self.relative_path,
            "request_query": self.request_query,
            "status_code": self.status_code,
            "is_success": self.is_success,
            "json_body": self.json_body,
            "raw_body": self.raw_body,
        }


@dataclass(frozen=True)
class _TokenCacheEntry:
    access_token: str
    expires_at: float


class OAuthTokenService:
    def __init__(
        self,
        environments: Dict[str, Dict[str, str]],
        session: Optional[requests.Session] = None,
        refresh_window_seconds: int = 120,
    ) -> None:
        self._environments = dict(environments)
        self._session = session or requests.Session()
        self._refresh_window_seconds = max(0, int(refresh_window_seconds))
        self._cache: Dict[str, _TokenCacheEntry] = {}
        self._lock = threading.RLock()

    def _normalize_environment(self, environment: str) -> str:
        normalized = environment.strip().lower() if isinstance(environment, str) else ""
        if not normalized:
            raise EnvironmentConfigError("Environment is required.")
        return normalized

    def get_config(self, environment: str) -> Dict[str, str]:
        normalized = self._normalize_environment(environment)
        if normalized not in self._environments:
            raise EnvironmentConfigError(f"Environment '{environment}' is not configured.")
        config = self._environments[normalized]
        required_non_empty = ("base_url", "token_url", "client_id", "client_secret")
        for key in required_non_empty:
            value = config.get(key, "")
            if not isinstance(value, str) or not value.strip():
                raise EnvironmentConfigError(
                    f"Environment '{environment}' is missing required config '{key}'."
                )
        if "scope" not in config:
            raise EnvironmentConfigError(
                f"Environment '{environment}' is missing required config 'scope'."
            )
        return {
            "base_url": config["base_url"].strip(),
            "token_url": config["token_url"].strip(),
            "client_id": config["client_id"].strip(),
            "client_secret": config["client_secret"].strip(),
            "scope": config.get("scope", ""),
        }

    def _is_token_valid(self, entry: _TokenCacheEntry, now: float) -> bool:
        return now < (entry.expires_at - self._refresh_window_seconds)

    def get_access_token(self, environment: str) -> str:
        normalized = self._normalize_environment(environment)
        now = time.time()
        with self._lock:
            cached = self._cache.get(normalized)
            if cached and self._is_token_valid(cached, now):
                return cached.access_token

            config = self.get_config(normalized)
            payload = {
                "grant_type": "client_credentials",
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "scope": config.get("scope", ""),
            }
            try:
                response = self._session.post(
                    config["token_url"],
                    data=payload,
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
            except requests.RequestException as exc:
                raise TokenFetchError(
                    f"OAuth token retrieval failed for environment '{normalized}': {exc}"
                ) from exc

            raw_body = response.text
            if response.status_code < 200 or response.status_code >= 300:
                raise TokenFetchError(
                    f"OAuth token retrieval failed for environment '{normalized}'. "
                    f"HTTP {response.status_code}. Body: {raw_body}"
                )

            try:
                token_response = response.json()
            except ValueError as exc:
                raise TokenFetchError(
                    f"OAuth token response for environment '{normalized}' is not valid JSON."
                ) from exc

            access_token = token_response.get("access_token")
            if not isinstance(access_token, str) or not access_token.strip():
                raise TokenFetchError(
                    f"Token response for environment '{normalized}' does not contain access_token."
                )

            expires_in_raw = token_response.get("expires_in", 3600)
            try:
                expires_in = int(expires_in_raw)
            except (TypeError, ValueError):
                expires_in = 3600

            entry = _TokenCacheEntry(
                access_token=access_token.strip(),
                expires_at=time.time() + max(1, expires_in),
            )
            self._cache[normalized] = entry
            return entry.access_token

    def clear_cache(self, environment: Optional[str] = None) -> None:
        with self._lock:
            if environment is None:
                self._cache.clear()
                return
            normalized = self._normalize_environment(environment)
            self._cache.pop(normalized, None)


class LoyaltyApiClient:
    def __init__(
        self,
        oauth_service: OAuthTokenService,
        default_api_version: str = "1",
        request_timeout: int = 30,
    ) -> None:
        self._oauth_service = oauth_service
        self._default_api_version = default_api_version
        self._request_timeout = int(request_timeout)
        self._session = requests.Session()

    def _build_versioned_path(self, api_version: Optional[str], suffix: str) -> str:
        version = (
            api_version.strip()
            if isinstance(api_version, str) and api_version.strip()
            else self._default_api_version
        )
        return f"api/v{version}/{suffix.lstrip('/')}"

    def _clean_query(self, source: Mapping[str, Optional[Any]]) -> Optional[Dict[str, str]]:
        cleaned: Dict[str, str] = {}
        for key, value in source.items():
            if value is None:
                continue
            if isinstance(value, str):
                if not value.strip():
                    continue
                cleaned[key] = value
                continue
            cleaned[key] = str(value)
        return cleaned or None

    def _build_search_clause(self, parts: List[Tuple[str, Optional[str]]]) -> Optional[str]:
        tokens = [
            f"{key.lower()}={value}"
            for key, value in parts
            if isinstance(value, str) and value.strip()
        ]
        return ",".join(tokens) if tokens else None

    def _send_request(
        self,
        environment: str,
        method: str,
        relative_path: str,
        query: Optional[Mapping[str, Optional[Any]]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> ApiCallResult:
        config = self._oauth_service.get_config(environment)
        token = self._oauth_service.get_access_token(environment)
        cleaned_query = self._clean_query(query or {})
        normalized_method = method.upper()
        full_url = f"{config['base_url'].rstrip('/')}/{relative_path.lstrip('/')}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if normalized_method in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"

        try:
            response = self._session.request(
                method=normalized_method,
                url=full_url,
                params=cleaned_query,
                json=body,
                headers=headers,
                timeout=self._request_timeout,
            )
        except requests.RequestException as exc:
            return ApiCallResult(
                environment=self._oauth_service._normalize_environment(environment),
                http_method=normalized_method,
                relative_path=relative_path,
                request_query=cleaned_query,
                status_code=0,
                is_success=False,
                json_body=None,
                raw_body=str(exc),
            )

        raw_body = response.text
        parsed_json: Optional[Any] = None
        if raw_body.strip():
            try:
                parsed_json = response.json()
            except ValueError:
                parsed_json = None

        return ApiCallResult(
            environment=self._oauth_service._normalize_environment(environment),
            http_method=normalized_method,
            relative_path=relative_path,
            request_query=cleaned_query,
            status_code=response.status_code,
            is_success=200 <= response.status_code < 300,
            json_body=parsed_json,
            raw_body=raw_body,
        )

    def get_player(
        self,
        environment: str,
        player_id: str,
        api_version: str = "1",
        select: Optional[str] = None,
        search: Optional[str] = None,
    ) -> ApiCallResult:
        path = self._build_versioned_path(api_version, f"playerinfo/{quote(player_id, safe='')}")
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
            query={"$select": select, "$search": search},
        )

    def search_players(
        self,
        environment: str,
        search: str,
        api_version: str = "1",
        top: Optional[int] = None,
        skip: Optional[int] = None,
        select: Optional[str] = None,
    ) -> ApiCallResult:
        path = self._build_versioned_path(api_version, "playerinfo/query")
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
            query={"$search": search, "$top": top, "$skip": skip, "$select": select},
        )

    def get_player_balance(
        self,
        environment: str,
        player_id: str,
        api_version: str = "1",
        search: Optional[str] = None,
        top: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version, f"playerinfo/{quote(player_id, safe='')}/balance"
        )
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
            query={"$search": search, "$top": top, "$skip": skip},
        )

    def get_player_club_info(
        self,
        environment: str,
        player_id: str,
        api_version: str = "1",
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version, f"playerinfo/{quote(player_id, safe='')}/clubinfo"
        )
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
        )

    def get_player_groups(
        self,
        environment: str,
        player_id: str,
        api_version: str = "1",
        property_code: Optional[str] = None,
        end_date: Optional[str] = None,
        top: Optional[int] = None,
        skip: Optional[int] = None,
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version, f"playerinfo/{quote(player_id, safe='')}/group"
        )
        search_clause = self._build_search_clause(
            [("propertycode", property_code), ("enddate", end_date)]
        )
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
            query={"$search": search_clause, "$top": top, "$skip": skip},
        )

    def add_player_to_group(
        self,
        environment: str,
        player_id: str,
        group_id: int,
        body: Dict[str, Any],
        api_version: str = "1",
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version,
            f"playerinfo/{quote(player_id, safe='')}/group/{group_id}",
        )
        return self._send_request(
            environment=environment,
            method="POST",
            relative_path=path,
            body=body,
        )

    def create_player(
        self,
        environment: str,
        body: Dict[str, Any],
        api_version: str = "1",
    ) -> ApiCallResult:
        path = self._build_versioned_path(api_version, "playerinfo")
        return self._send_request(
            environment=environment,
            method="POST",
            relative_path=path,
            body=body,
        )

    def update_player(
        self,
        environment: str,
        player_id: str,
        body: Dict[str, Any],
        api_version: str = "1",
    ) -> ApiCallResult:
        path = self._build_versioned_path(api_version, f"playerinfo/{quote(player_id, safe='')}")
        return self._send_request(
            environment=environment,
            method="PATCH",
            relative_path=path,
            body=body,
        )

    def validate_player_pin(
        self,
        environment: str,
        player_id: str,
        body: Dict[str, Any],
        api_version: str = "1",
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version,
            f"playerinfo/{quote(player_id, safe='')}/validatePIN",
        )
        return self._send_request(
            environment=environment,
            method="PUT",
            relative_path=path,
            body=body,
        )

    def get_player_eligibility(
        self,
        environment: str,
        player_id: str,
        corp_prop: str,
        api_version: str = "1",
        search: Optional[str] = None,
    ) -> ApiCallResult:
        path = self._build_versioned_path(
            api_version,
            f"playerinfo/{quote(player_id, safe='')}/{quote(corp_prop, safe='')}/eligibility",
        )
        return self._send_request(
            environment=environment,
            method="GET",
            relative_path=path,
            query={"$search": search},
        )

    def call_loyalty_api(
        self,
        environment: str,
        method: str,
        relative_path: str,
        api_version: str = "1",
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> ApiCallResult:
        normalized_method = method.upper()
        if normalized_method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("method must be one of GET, POST, PUT, PATCH, DELETE.")
        normalized_path = relative_path.lstrip("/")
        prefix = f"api/v{api_version}/"
        if not normalized_path.lower().startswith(prefix.lower()):
            normalized_path = f"{prefix}{normalized_path}"
        return self._send_request(
            environment=environment,
            method=normalized_method,
            relative_path=normalized_path,
            query=query,
            body=body,
        )


def _environment_enum() -> List[str]:
    return sorted(ENVIRONMENTS.keys())


def _base_input_schema(required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "environment": {
                "type": "string",
                "enum": _environment_enum(),
            },
            "apiVersion": {
                "type": "string",
                "default": "1",
            },
        },
        "required": list(required),
        "additionalProperties": False,
    }


def _extend_schema(
    base_schema: Dict[str, Any],
    extra_properties: Dict[str, Any],
    extra_required: Optional[List[str]] = None,
) -> Dict[str, Any]:
    schema = {
        "type": base_schema["type"],
        "properties": dict(base_schema["properties"]),
        "required": list(base_schema["required"]),
        "additionalProperties": base_schema["additionalProperties"],
    }
    schema["properties"].update(extra_properties)
    if extra_required:
        schema["required"].extend(extra_required)
    return schema


def adk_tools() -> List[Dict[str, Any]]:
    base_schema = _base_input_schema(["environment"])
    body_schema: Dict[str, Any] = {"type": "object", "additionalProperties": True}
    return [
        {
            "name": "get_player",
            "description": "Get a player by universal player ID from ATI PlayerInfo loyalty API.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "select": {"type": "string", "description": "Optional OData $select list."},
                    "search": {"type": "string", "description": "Optional OData $search value."},
                },
                ["playerId"],
            ),
        },
        {
            "name": "search_players",
            "description": "Search players via GET /api/v{version}/playerinfo/query using OData-style search criteria.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "search": {"type": "string", "description": "Raw OData $search expression."},
                    "top": {"type": "integer"},
                    "skip": {"type": "integer"},
                    "select": {"type": "string"},
                },
                ["search"],
            ),
        },
        {
            "name": "get_player_groups",
            "description": "Get groups for a player. Maps to GET /playerinfo/{playerId}/group.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "propertyCode": {"type": "string"},
                    "endDate": {
                        "type": "string",
                        "description": "Expected date string used in the API's $search.",
                    },
                    "top": {"type": "integer"},
                    "skip": {"type": "integer"},
                },
                ["playerId"],
            ),
        },
        {
            "name": "add_player_to_group",
            "description": "Add a player to a group. Maps to POST /playerinfo/{playerId}/group/{groupId}. Pass the Patron payload in body.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "groupId": {"type": "integer"},
                    "body": dict(body_schema),
                },
                ["playerId", "groupId", "body"],
            ),
        },
        {
            "name": "get_player_balance",
            "description": "Get player balance. Maps to GET /playerinfo/{playerId}/balance.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "search": {"type": "string"},
                    "top": {"type": "integer"},
                    "skip": {"type": "integer"},
                },
                ["playerId"],
            ),
        },
        {
            "name": "get_player_club_info",
            "description": "Get player club info. Maps to GET /playerinfo/{playerId}/clubinfo.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                },
                ["playerId"],
            ),
        },
        {
            "name": "create_player",
            "description": "Create a player. Maps to POST /playerinfo. Pass the PlayerInfoList payload in body.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "body": dict(body_schema),
                },
                ["body"],
            ),
        },
        {
            "name": "update_player",
            "description": "Update player information. Maps to PATCH /playerinfo/{playerId}. Pass the PlayerInfoList payload in body.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "body": dict(body_schema),
                },
                ["playerId", "body"],
            ),
        },
        {
            "name": "validate_player_pin",
            "description": "Validate a player's PIN. Maps to PUT /playerinfo/{playerId}/validatePIN. Pass the PlayerPin payload in body.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "body": dict(body_schema),
                },
                ["playerId", "body"],
            ),
        },
        {
            "name": "get_player_eligibility",
            "description": "Get player eligibility by corp prop. Maps to GET /playerinfo/{playerId}/{corpProp}/eligibility.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "playerId": {"type": "string"},
                    "corpProp": {"type": "string"},
                    "search": {"type": "string"},
                },
                ["playerId", "corpProp"],
            ),
        },
        {
            "name": "call_loyalty_api",
            "description": "Generic escape-hatch tool for routes not yet wrapped. relativePath may be full api/v{version}/... or just the suffix after it.",
            "input_schema": _extend_schema(
                base_schema,
                {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    },
                    "relativePath": {"type": "string"},
                    "query": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "body": dict(body_schema),
                },
                ["method", "relativePath"],
            ),
        },
    ]


def _require_mapping(input_data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = input_data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' is required and must be an object.")
    return value


def _require_string(input_data: Dict[str, Any], key: str) -> str:
    value = input_data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' is required and must be a non-empty string.")
    return value


def _optional_string(input_data: Dict[str, Any], key: str) -> Optional[str]:
    value = input_data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string when provided.")
    return value


def _optional_int(input_data: Dict[str, Any], key: str) -> Optional[int]:
    value = input_data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{key}' must be an integer when provided.")
    return value


def _require_int(input_data: Dict[str, Any], key: str) -> int:
    value = input_data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"'{key}' is required and must be an integer.")
    return value


def _api_version(input_data: Dict[str, Any]) -> str:
    value = input_data.get("apiVersion", "1")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'apiVersion' must be a non-empty string when provided.")
    return value


def tool_handlers(client: LoyaltyApiClient) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
    def get_player_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.get_player(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            api_version=_api_version(input_data),
            select=_optional_string(input_data, "select"),
            search=_optional_string(input_data, "search"),
        )
        return result.to_dict()

    def search_players_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.search_players(
            environment=_require_string(input_data, "environment"),
            search=_require_string(input_data, "search"),
            api_version=_api_version(input_data),
            top=_optional_int(input_data, "top"),
            skip=_optional_int(input_data, "skip"),
            select=_optional_string(input_data, "select"),
        )
        return result.to_dict()

    def get_player_groups_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.get_player_groups(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            api_version=_api_version(input_data),
            property_code=_optional_string(input_data, "propertyCode"),
            end_date=_optional_string(input_data, "endDate"),
            top=_optional_int(input_data, "top"),
            skip=_optional_int(input_data, "skip"),
        )
        return result.to_dict()

    def add_player_to_group_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.add_player_to_group(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            group_id=_require_int(input_data, "groupId"),
            body=_require_mapping(input_data, "body"),
            api_version=_api_version(input_data),
        )
        return result.to_dict()

    def get_player_balance_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.get_player_balance(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            api_version=_api_version(input_data),
            search=_optional_string(input_data, "search"),
            top=_optional_int(input_data, "top"),
            skip=_optional_int(input_data, "skip"),
        )
        return result.to_dict()

    def get_player_club_info_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.get_player_club_info(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            api_version=_api_version(input_data),
        )
        return result.to_dict()

    def create_player_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.create_player(
            environment=_require_string(input_data, "environment"),
            body=_require_mapping(input_data, "body"),
            api_version=_api_version(input_data),
        )
        return result.to_dict()

    def update_player_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.update_player(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            body=_require_mapping(input_data, "body"),
            api_version=_api_version(input_data),
        )
        return result.to_dict()

    def validate_player_pin_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.validate_player_pin(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            body=_require_mapping(input_data, "body"),
            api_version=_api_version(input_data),
        )
        return result.to_dict()

    def get_player_eligibility_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        result = client.get_player_eligibility(
            environment=_require_string(input_data, "environment"),
            player_id=_require_string(input_data, "playerId"),
            corp_prop=_require_string(input_data, "corpProp"),
            api_version=_api_version(input_data),
            search=_optional_string(input_data, "search"),
        )
        return result.to_dict()

    def call_loyalty_api_handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        query_value = input_data.get("query")
        if query_value is not None and not isinstance(query_value, dict):
            raise ValueError("'query' must be an object when provided.")
        body_value = input_data.get("body")
        if body_value is not None and not isinstance(body_value, dict):
            raise ValueError("'body' must be an object when provided.")
        result = client.call_loyalty_api(
            environment=_require_string(input_data, "environment"),
            method=_require_string(input_data, "method"),
            relative_path=_require_string(input_data, "relativePath"),
            api_version=_api_version(input_data),
            query=query_value,
            body=body_value,
        )
        return result.to_dict()

    return {
        "get_player": get_player_handler,
        "search_players": search_players_handler,
        "get_player_groups": get_player_groups_handler,
        "add_player_to_group": add_player_to_group_handler,
        "get_player_balance": get_player_balance_handler,
        "get_player_club_info": get_player_club_info_handler,
        "create_player": create_player_handler,
        "update_player": update_player_handler,
        "validate_player_pin": validate_player_pin_handler,
        "get_player_eligibility": get_player_eligibility_handler,
        "call_loyalty_api": call_loyalty_api_handler,
    }


class PlayerInfoAgent:
    name = "PlayerInfo"

    def __init__(self, client: LoyaltyApiClient) -> None:
        self._client = client

    @staticmethod
    def adk_tools() -> List[Dict[str, Any]]:
        return adk_tools()

    def handlers(self) -> Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]]:
        return tool_handlers(self._client)

    def register(self, runtime: Any) -> None:
        runtime.register_tools(self.adk_tools())
        runtime.register_handlers(self.handlers())


if __name__ == "__main__":
    oauth_service = OAuthTokenService(ENVIRONMENTS)
    client = LoyaltyApiClient(oauth_service)
    result = client.get_player("qa", "12345")
    print(result.status_code)
    print(result.json_body)
