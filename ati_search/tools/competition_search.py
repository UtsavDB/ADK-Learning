from __future__ import annotations

"""ADK-native competitor CMS search tool backed by Gemini Google Search."""

from google.adk.agents import Agent
from google.adk.tools import AgentTool, google_search
from pydantic import BaseModel, Field, field_validator

from ati_search.env import get_env_value
from shared.adk_model_provider import DEFAULT_GOOGLE_MODEL
from shared import build_agent_generation_config


class CompetitionSearchRequest(BaseModel):
    """Input schema for public competitor CMS research."""

    query: str = Field(
        ...,
        description=(
            "The feature, workflow, topic, or keywords to look up across Konami, "
            "IGT, and Light & Wonder casino management systems."
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must be a non-empty string.")
        return cleaned


def _competition_search_model() -> str:
    return (
        get_env_value("ATI_SEARCH_COMPETITION_GOOGLE_MODEL")
        or get_env_value("ATI_SEARCH_GOOGLE_MODEL")
        or get_env_value("ATI_SEARCH_MODEL")
        or DEFAULT_GOOGLE_MODEL
    )


competition_search_agent = Agent(
    name="competition_search",
    model=_competition_search_model(),
    generate_content_config=build_agent_generation_config("ati_search"),
    description=(
        "Searches public competitor CMS information for Konami, IGT, and Light & "
        "Wonder using Google Search."
    ),
    instruction=(
        "You are a focused public-domain research agent for casino management systems. "
        "Use Google Search to find relevant public information for all three vendors: "
        "Konami Gaming, IGT, and Light & Wonder. "
        "Treat the user's request as a topic that may require keyword expansion rather "
        "than exact text matching. Break the request into likely feature keywords, "
        "module names, and product synonyms before searching. "
        "Search these public sites first: konamigaming.com for Konami, igt.com for IGT, "
        "and gaming.lnw.com, lightandwonder.com, or lnw.com for Light & Wonder. "
        "Use likely product names when helpful, including SYNKROS for Konami, "
        "IGT ADVANTAGE or ADVANTAGE X for IGT, and ACSC, CMP, or SDS for "
        "Light & Wonder when those appear relevant. "
        "Prefer official product pages, brochures, PDFs, help pages, press releases, "
        "and case studies. If official material is sparse, you may use reputable public "
        "industry sources, but say clearly when a conclusion is inferred from related "
        "keywords rather than an exact match. "
        "Only use public information. Do not claim access to private portals, customer "
        "sites, or non-public documentation. "
        "Return plain text with these sections in this order: "
        "Normalized query, Search keywords used, Konami, IGT, Light & Wonder, Overall take. "
        "For each vendor section, include: likely matching product or module names, useful "
        "publicly available details, and 1-3 source URLs. "
        "If a vendor has no clear public match, say so explicitly instead of guessing."
    ),
    input_schema=CompetitionSearchRequest,
    tools=[google_search],
)

competition_search = AgentTool(
    agent=competition_search_agent,
    skip_summarization=True,
)

__all__ = [
    "CompetitionSearchRequest",
    "competition_search",
    "competition_search_agent",
]
