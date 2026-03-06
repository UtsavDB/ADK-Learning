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

### 5) Run the project

```bash
python main.py
```
