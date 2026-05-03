from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Optional

from google.adk.agents import Agent

from playerinfo_adk import ENVIRONMENTS, LoyaltyApiClient, OAuthTokenService
from shared import build_agent_generation_config, build_agent_model

_CLIENT_LOCK = Lock()
_CLIENT: Optional[LoyaltyApiClient] = None


def _client() -> LoyaltyApiClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = LoyaltyApiClient(OAuthTokenService(ENVIRONMENTS))
    return _CLIENT


def get_player(
    environment: str,
    playerId: str,
    apiVersion: str = "1",
    select: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    return _client().get_player(environment, playerId, apiVersion, select, search).to_dict()


def search_players(
    environment: str,
    search: str,
    apiVersion: str = "1",
    top: Optional[int] = None,
    skip: Optional[int] = None,
    select: Optional[str] = None,
) -> Dict[str, Any]:
    return _client().search_players(environment, search, apiVersion, top, skip, select).to_dict()


def get_player_groups(
    environment: str,
    playerId: str,
    apiVersion: str = "1",
    propertyCode: Optional[str] = None,
    endDate: Optional[str] = None,
    top: Optional[int] = None,
    skip: Optional[int] = None,
) -> Dict[str, Any]:
    return _client().get_player_groups(
        environment=environment,
        player_id=playerId,
        api_version=apiVersion,
        property_code=propertyCode,
        end_date=endDate,
        top=top,
        skip=skip,
    ).to_dict()


def add_player_to_group(
    environment: str,
    playerId: str,
    groupId: int,
    body: Dict[str, Any],
    apiVersion: str = "1",
) -> Dict[str, Any]:
    return _client().add_player_to_group(environment, playerId, groupId, body, apiVersion).to_dict()


def get_player_balance(
    environment: str,
    playerId: str,
    apiVersion: str = "1",
    search: Optional[str] = None,
    top: Optional[int] = None,
    skip: Optional[int] = None,
) -> Dict[str, Any]:
    return _client().get_player_balance(
        environment=environment,
        player_id=playerId,
        api_version=apiVersion,
        search=search,
        top=top,
        skip=skip,
    ).to_dict()


def get_player_club_info(
    environment: str,
    playerId: str,
    apiVersion: str = "1",
) -> Dict[str, Any]:
    return _client().get_player_club_info(environment, playerId, apiVersion).to_dict()


def create_player(
    environment: str,
    body: Dict[str, Any],
    apiVersion: str = "1",
) -> Dict[str, Any]:
    return _client().create_player(environment, body, apiVersion).to_dict()


def update_player(
    environment: str,
    playerId: str,
    body: Dict[str, Any],
    apiVersion: str = "1",
) -> Dict[str, Any]:
    return _client().update_player(environment, playerId, body, apiVersion).to_dict()


def validate_player_pin(
    environment: str,
    playerId: str,
    body: Dict[str, Any],
    apiVersion: str = "1",
) -> Dict[str, Any]:
    return _client().validate_player_pin(environment, playerId, body, apiVersion).to_dict()


def get_player_eligibility(
    environment: str,
    playerId: str,
    corpProp: str,
    apiVersion: str = "1",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    return _client().get_player_eligibility(
        environment=environment,
        player_id=playerId,
        corp_prop=corpProp,
        api_version=apiVersion,
        search=search,
    ).to_dict()


def call_loyalty_api(
    environment: str,
    method: str,
    relativePath: str,
    apiVersion: str = "1",
    query: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _client().call_loyalty_api(
        environment=environment,
        method=method,
        relative_path=relativePath,
        api_version=apiVersion,
        query=query,
        body=body,
    ).to_dict()


AGENT_MODEL = build_agent_model("playerinfo", default_provider="azure")
AGENT_GENERATION_CONFIG = build_agent_generation_config("playerinfo")

root_agent = Agent(
    name="playerinfo",
    model=AGENT_MODEL,
    generate_content_config=AGENT_GENERATION_CONFIG,
    description="Calls the Loyalty PlayerInfo API for player, group, balance, club info, eligibility, and generic route operations.",
    instruction=(
        "You are PlayerInfo. "
        "Use the matching tool for the user task instead of inventing API payloads when a dedicated tool already exists. "
        "Use get_player for a player lookup by playerId. "
        "Use search_players for OData-style player search. "
        "Use get_player_groups, add_player_to_group, get_player_balance, get_player_club_info, "
        "create_player, update_player, validate_player_pin, and get_player_eligibility for those exact operations. "
        "Use call_loyalty_api only when the requested route is not covered by a dedicated tool. "
        "Summarize API results clearly and include status_code when relevant. "
        "If the API call fails, explain the failure using the returned raw_body or json_body instead of pretending it succeeded."
    ),
    tools=[
        get_player,
        search_players,
        get_player_groups,
        add_player_to_group,
        get_player_balance,
        get_player_club_info,
        create_player,
        update_player,
        validate_player_pin,
        get_player_eligibility,
        call_loyalty_api,
    ],
)
