from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT_DIR = AGENT_DIR.parent
SHARED_DIR = REPO_ROOT_DIR / "shared"
SCRUMMASTER_ENV_PATH = Path(r"C:\W\GIT\Experiments\ScrumMaster2\.env")


def dotenv_paths() -> list[Path]:
    return [
        AGENT_DIR / ".env",
        SHARED_DIR / ".env",
        REPO_ROOT_DIR / ".env",
        SCRUMMASTER_ENV_PATH,
    ]


def load_ati_search_env() -> None:
    for path in dotenv_paths():
        if path.exists():
            load_dotenv(dotenv_path=path, override=False)


def read_dotenv_layers(paths: list[Path] | None = None) -> dict[str, str]:
    merged: dict[str, str] = {}
    candidate_paths = paths or dotenv_paths()
    for path in reversed(candidate_paths):
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value is not None:
                merged[key] = value
    return merged


load_ati_search_env()
DOTENV_VALUES = read_dotenv_layers()


def get_env_value(name: str, env_values: dict[str, str] | None = None) -> str | None:
    value = os.getenv(name)
    if value not in (None, ""):
        return value
    source = env_values if env_values is not None else DOTENV_VALUES
    fallback = source.get(name)
    if fallback in (None, ""):
        return None
    return fallback
