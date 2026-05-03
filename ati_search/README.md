# ATI Search ADK Demo

This folder contains an ADK Python demo agent named `ATI Search`.

It demonstrates an ADK agent that can:

- call the internal Aristocrat Fluid Topics clustered search API
- search TFS/Azure DevOps repositories and related work items
- infer defect-focused intent from natural language and adjust work-item search behavior
- fall back from unavailable TFS code-content search to repository path matching
- search public competitor sources (Konami, IGT, and Light & Wonder) with keyword expansion
- normalize nested API responses into a cleaner summary shape
- deduplicate equivalent ATI topics across versions and prefer the newest one
- answer end users in plain text instead of dumping raw JSON

ATI docs pagination is intentionally not implemented yet. The ATI docs API call always uses page `1`.

## Files

- `agent.py`: ADK agent definition and tool registration
- `env.py`: dotenv loading for ATI/TFS-specific configuration
- `tools/avid_search.py`: Aristocrat documentation search tool
- `tools/competition_search.py`: public competitor CMS search tool powered by ADK Google Search through a dedicated sub-agent
- `tools/tfs_git_search.py`: TFS/Azure DevOps Git and work-item search tool
- `requirements.txt`: minimal Python dependencies for this demo
- `.env.example`: ATI/TFS service environment variables for this demo

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Create your env files

Create `ati_search/.env` from `ati_search/.env.example` for ATI/TFS settings.
Create `shared/.env` from `shared/.env.example` for LLM provider settings.

ATI/TFS config is loaded in this order:

1. `ati_search/.env`
2. `shared/.env`
3. repo-root `.env`
4. `C:\W\GIT\Experiments\ScrumMaster2\.env`

Earlier files win when the same key appears in multiple files.

Minimum required for the Aristocrat search API:

```env
ATI_SEARCH_BEARER_TOKEN=your_real_token
```

`ATI Search` defaults to Azure through the shared ADK model-provider library. Put these agent-scoped variables in `shared/.env`:

```env
ATI_SEARCH_PROVIDER=azure
ATI_SEARCH_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
ATI_SEARCH_AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
ATI_SEARCH_AZURE_OPENAI_API_VERSION=v1
ATI_SEARCH_AZURE_OPENAI_MODEL=your_azure_deployment_name
ATI_SEARCH_MAX_OUTPUT_TOKENS=256
```

If you want to run `ATI Search` with Google instead, set provider and model in `shared/.env`:

```env
ATI_SEARCH_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key
ATI_SEARCH_GOOGLE_MODEL=gemini-2.0-flash
```

Optional for competitor search sub-agent model override:

```env
ATI_SEARCH_COMPETITION_GOOGLE_MODEL=gemini-2.0-flash
```

Required for TFS Git and work item search:

```env
TFS_URL=http://your-tfs-server:8080/tfs/DefaultCollection
PROJECT=YourProject
PAT=your_tfs_pat_here
```

Optional for TFS:

```env
API_VERSION=4.1
TFS_DEFAULT_REPO=optional_default_repo_name
```

Tool-level optional arguments for `tfs_git_search`:

- `repo_name`: limit Git search to one repository
- `include_work_items`: include work-item results (default `true`)
- `include_git_matches`: include Git/path matches (default `true`)
- `top`: cap total returned results (1-50, default `5`)
- `work_item_type`: filter work items by one type or multiple types
- `work_item_search_fields`: override WIQL fields used for work-item text matching

### 3. Install dependencies

```bash
pip install -r ati_search/requirements.txt
```

## Run locally with Google ADK

This demo uses the standard ADK `agent.py` + `root_agent` structure inside the `ati_search` folder.

Run the CLI from the repository root:

```bash
adk run ati_search
```

Or start the local ADK web UI:

```bash
adk web .
```

Then select `ati_search` and chat with it.

## Example prompt

```text
Search Aristocrat docs for error code
```

```text
Search competitor CMS products for jackpot setup across Konami, IGT, and Light & Wonder
```

```text
Search TFS Git for SprintReport in all repos
```

```text
Search TFS work items and Git in repo ScrumMaster2 for burndown
```

```text
Find DAP defects
```

```text
Compare competitor CMS support for multi-site progressive jackpots
```

## Notes

- The `avid_search` tool sends a `POST` request to the Aristocrat clustered-search endpoint.
- The bearer token is loaded from `ATI_SEARCH_BEARER_TOKEN`.
- Model selection is resolved by the shared `build_agent_model()` helper, which now loads `shared/.env`.
- The `competition_search` tool uses a dedicated Google Search sub-agent and expands user intent into feature/product synonyms before searching public sources.
- `competition_search` outputs plain text sections in this order: `Normalized query`, `Search keywords used`, `Konami`, `IGT`, `Light & Wonder`, `Overall take`.
- The `tfs_git_search` tool searches TFS/Azure DevOps repos across the configured project by default and can also return related work item matches.
- `tfs_git_search` first attempts a TFS code-search API call and falls back to repo/file path matching if code-content search is unavailable on the server.
- For defect intent (for example, `DAP defects`), `tfs_git_search` normalizes the term (for example, to `DAP`), biases toward work-item search, and can skip Git matching when the request is clearly work-item focused.
- `.env` loading is supported through `python-dotenv`.
- The tools return both a normalized summary and internal raw payloads.
- The agent is instructed to present results to users in plain English.
- Pagination is intentionally not implemented yet; this demo always calls ATI docs page `1`.
