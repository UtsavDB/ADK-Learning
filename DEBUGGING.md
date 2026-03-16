# ADK + Ollama Debugging Playbook

Use this whenever an agent appears in ADK Web but does not respond.

## 1) Start clean

```bash
cd /Users/utsavverma/git/ADK-Learning
source .venv/bin/activate
pkill -f "adk web" || true
```

## 2) Run ADK Web with logs

```bash
uv run adk web --port 8011 > /tmp/adk_web.log 2>&1
```

Keep this terminal open.

## 3) In another terminal, confirm app discovery

```bash
curl -s http://127.0.0.1:8011/list-apps
```

If your app is missing, fix package/folder/import issues first.

## 4) Create a session (required before `/run`)

Replace `<app_name>` with your folder name (example: `multi_agents`).

```bash
curl -s -X POST \
  http://127.0.0.1:8011/apps/<app_name>/users/u1/sessions/s1 \
  -H "Content-Type: application/json" \
  -d '{}'
```

## 5) Call `/run` directly (bypass UI)

```bash
curl -s -X POST http://127.0.0.1:8011/run \
  -H "Content-Type: application/json" \
  -d '{"appName":"<app_name>","userId":"u1","sessionId":"s1","newMessage":{"role":"user","parts":[{"text":"hello"}]}}'
```

## 6) Read backend errors immediately

```bash
tail -n 200 /tmp/adk_web.log
```

This is where root cause usually appears (ImportError, ValidationError, etc.).

## 7) Validate app import directly

```bash
uv run python - <<'PY'
import importlib
importlib.import_module("<app_name>")
print("import ok")
PY
```

If this fails, ADK Web will fail too.

## 8) Validate `agent.py` and `root_agent`

```bash
uv run python - <<'PY'
import importlib.util
from pathlib import Path

p = Path("<app_name>/agent.py").resolve()
s = importlib.util.spec_from_file_location("tmp_agent", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
print("root_agent exists:", hasattr(m, "root_agent"))
PY
```

## 9) Validate Ollama availability

```bash
ollama list
```

Confirm your model exists (for example `gemma3:1b`).

## 10) Common failure signatures

- `ImportError: LiteLLM support requires ...`  
  Install extensions dependency:
  ```bash
  uv sync
  ```
  Ensure `google-adk[extensions]` is present in deps.

- `Invalid app name 'x-y': must be a valid identifier`  
  Rename app folder to use letters/digits/underscores only (example: `multi_agents`).

- `{"detail":"Session not found: s1"}`  
  You skipped session creation step.

## 11) Regression check before claiming fixed

```bash
uv run python -m unittest
```

Then rerun steps 4 and 5 to confirm API path works end-to-end.
