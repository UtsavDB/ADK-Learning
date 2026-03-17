# ATI Search ADK Demo

This folder contains a minimal Google Agent Development Kit (ADK) Python demo agent named `ATI Search`.

It proves that an ADK agent can:

- call the internal Aristocrat Fluid Topics clustered search API
- flatten the nested response into a cleaner summary
- deduplicate equivalent topics across versions and prefer the newest one
- answer the end user in plain text instead of raw JSON

Only page `1` is supported in this demo.

## Files

- `agent.py`: ADK agent definition, `avid_search` tool, and helper utilities
- `requirements.txt`: minimal Python dependencies for this demo
- `.env.example`: example environment variables for this demo

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

### 2. Create your `.env`

Create a `.env` file in this folder and populate it from `.env.example`.
The agent will also fall back to a repo-root `.env` if you prefer to keep shared ADK settings there.

Minimum required for the Aristocrat search API:

```env
ATI_SEARCH_BEARER_TOKEN=your_real_token
```

You will also usually need a model key so ADK can run the Gemini-backed agent:

```env
GOOGLE_API_KEY=your_google_api_key
```

Optional:

```env
ATI_SEARCH_MODEL=gemini-2.0-flash
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

## Notes

- The `avid_search` tool sends a `POST` request to the Aristocrat clustered-search endpoint.
- The bearer token is loaded from `ATI_SEARCH_BEARER_TOKEN`.
- `.env` loading is supported through `python-dotenv`.
- The tool returns both the full raw API response and a flattened summary internally.
- The agent is instructed to present results to users in plain English.
- Pagination is intentionally not implemented yet; this demo always calls page `1`.
