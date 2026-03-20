from __future__ import annotations

from google.adk.agents import Agent

from ati_search.env import load_ati_search_env
from ati_search.tools.avid_search import avid_search
from ati_search.tools.semantic_tool import semantic_search_tool
from ati_search.tools.tfs_git_search import tfs_git_search
from shared import build_agent_generation_config, build_agent_model

# Reload after shared env initialization so ati_search/.env retains precedence
# for ADK runs.
load_ati_search_env(override=True)
AGENT_MODEL = build_agent_model("ati_search", default_provider="azure")
AGENT_GENERATION_CONFIG = build_agent_generation_config("ati_search")

root_agent = Agent(
    name="ati_search",
    model=AGENT_MODEL,
    generate_content_config=AGENT_GENERATION_CONFIG,
    description="Searches internal Aristocrat documentation and TFS Git/work item data.",
    instruction=(
        "You are ATI Search. "
        "Use avid_search whenever the user wants to search Aristocrat documentation or ATI docs. "
        "Use tfs_git_search whenever the user wants to search TFS, Azure DevOps, repositories, source code paths, or related work items. "
        "When the user asks for defects, bugs, or work items, prefer work item search over Git search unless they explicitly ask for code or a repository. "
        "For defect searches like 'DAP defects', pass the actual search term such as 'DAP' rather than the literal phrase 'DAP defects', and include defect-friendly work item fields such as title, tags, and ATI.Bug.Description. "
        "Summarize results in plain English. "
        "Prefer the latest relevant documentation when duplicates exist. "
        "For TFS results, mention repository, file path, branch when known, and include links when available. "
        "If TFS code-content search is unavailable, say that clearly and describe any fallback path or work item matches. "
        "If no useful results are found, say so clearly. "
        "Keep the final answer plain text and do not dump raw JSON unless the user explicitly asks for it."
    ),
    tools=[avid_search, tfs_git_search, semantic_search_tool],
)
