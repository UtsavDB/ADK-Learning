# ATI Search ADK Demo

This folder contains an ADK Python demo agent named `ATI Search`.

It proves that an ADK agent can:

- call the internal Aristocrat Fluid Topics clustered search API
- search TFS/Azure DevOps repositories and related work items
- flatten the nested response into a cleaner summary
- deduplicate equivalent topics across versions and prefer the newest one
- answer the end user in plain text instead of raw JSON

Only page `1` is supported for the ATI docs API in this demo.

## Files

- `agent.py`: ADK agent definition and tool registration
- `env.py`: dotenv loading for ATI/TFS-specific configuration
- `tools/avid_search.py`: Aristocrat documentation search tool
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

`ATI Search` now defaults to Azure OpenAI through the shared ADK model-provider library. Put these agent-scoped variables in `shared/.env`:

```env
ATI_SEARCH_PROVIDER=azure
ATI_SEARCH_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
ATI_SEARCH_AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
ATI_SEARCH_AZURE_OPENAI_API_VERSION=v1
ATI_SEARCH_AZURE_OPENAI_MODEL=your_azure_deployment_name
```

If you want to run `ATI Search` with Google instead, also put that in `shared/.env`:

```env
ATI_SEARCH_PROVIDER=google
GOOGLE_API_KEY=your_google_api_key
ATI_SEARCH_GOOGLE_MODEL=gemini-2.0-flash
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
Search TFS Git for SprintReport in all repos
```

```text
Search TFS work items and Git in repo ScrumMaster2 for burndown
```

## Notes

- The `avid_search` tool sends a `POST` request to the Aristocrat clustered-search endpoint.
- The bearer token is loaded from `ATI_SEARCH_BEARER_TOKEN`.
- Model selection is resolved by the shared `build_agent_model()` helper, which now loads `shared/.env`.
- The `tfs_git_search` tool searches TFS/Azure DevOps repos across the configured project by default and can also return related work item matches.
- `tfs_git_search` first attempts a TFS code-search API call and falls back to repo/file path matching if code-content search is unavailable on the server.
- `.env` loading is supported through `python-dotenv`.
- The tools return both a normalized summary and internal raw payloads.
- The agent is instructed to present results to users in plain English.
- Pagination is intentionally not implemented yet; this demo always calls page `1`.
