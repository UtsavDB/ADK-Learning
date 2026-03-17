# ADK Learning

## Setup

### 1) Create a virtual environment

```bash
python3 -m venv .venv
```

### 2) Activate the virtual environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env
```

Set either:

- `GOOGLE_API_KEY`
- or `GEMINI_API_KEY`

Agents now use a shared provider-aware ADK model builder with agent-scoped config.
Set `<AGENT>_PROVIDER` to `azure` or `google`, then provide matching scoped settings.

Examples:

- `MULTI_AGENTS_PROVIDER=azure`
- `MULTI_AGENTS_AZURE_OPENAI_ENDPOINT=...`
- `MULTI_AGENTS_AZURE_OPENAI_API_KEY=...`
- `MULTI_AGENTS_AZURE_OPENAI_API_VERSION=v1`
- `MULTI_AGENTS_AZURE_OPENAI_MODEL=your_azure_deployment_name`
- `ATI_SEARCH_PROVIDER=azure`
- `ATI_SEARCH_AZURE_OPENAI_ENDPOINT=...`
- `ATI_SEARCH_AZURE_OPENAI_API_KEY=...`
- `ATI_SEARCH_AZURE_OPENAI_API_VERSION=v1`
- `ATI_SEARCH_AZURE_OPENAI_MODEL=your_azure_deployment_name`

`ATI Search` now defaults to Azure if `ATI_SEARCH_PROVIDER` is omitted.
`multi_agents` also defaults to Azure.
If Azure Foundry gives you a URL ending in `/openai/v1/`, you can use that endpoint directly and set the scoped API version to `v1`.
If you want a Google-backed agent instead, set `<AGENT>_PROVIDER=google` and provide `<AGENT>_GOOGLE_MODEL`.

### 5) Run the project

```bash
python main.py
```
