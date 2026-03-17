from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values, load_dotenv
from google.adk.models.lite_llm import LiteLlm

DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash"
_AGENT_NAME_RE = re.compile(r"[^A-Z0-9]+")
_SHARED_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DIR = _SHARED_DIR.parent


def _load_shared_env() -> None:
    repo_env = _REPO_ROOT_DIR / ".env"
    shared_env = _SHARED_DIR / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=False)
    if shared_env.exists():
        load_dotenv(dotenv_path=shared_env, override=True)


_load_shared_env()


def _agent_env_prefix(agent_name: str) -> str:
    normalized = _AGENT_NAME_RE.sub("_", agent_name.strip().upper()).strip("_")
    if not normalized:
        raise ValueError("agent_name must include at least one letter or digit.")
    return normalized


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _dotenv_paths(agent_prefix: str) -> list[Path]:
    agent_dir = _REPO_ROOT_DIR / agent_prefix.lower()
    return [
        agent_dir / ".env",
        _SHARED_DIR / ".env",
        _REPO_ROOT_DIR / ".env",
    ]


def _dotenv_values_for(agent_prefix: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    for path in reversed(_dotenv_paths(agent_prefix)):
        if not path.exists():
            continue
        for key, value in dotenv_values(path).items():
            if value not in (None, ""):
                merged[key] = value
    return merged


def _first_setting(agent_prefix: str, *names: str) -> str | None:
    env_value = _first_env(*names)
    if env_value:
        return env_value

    dotenv_values_map = _dotenv_values_for(agent_prefix)
    for name in names:
        value = dotenv_values_map.get(name)
        if value:
            return value
    return None


def _normalize_provider(provider: str | None, default_provider: str) -> str:
    selected = (provider or default_provider).strip().lower()
    if selected not in {"azure", "google"}:
        raise RuntimeError(
            f"Unsupported provider '{selected}'. Supported providers are: azure, google."
        )
    return selected


def _normalize_azure_endpoint(raw_endpoint: str) -> tuple[str, bool]:
    endpoint = raw_endpoint.strip().rstrip("/")
    parsed = urlparse(endpoint)
    path = (parsed.path or "").rstrip("/")
    uses_v1_route = path.endswith("/openai/v1")

    if uses_v1_route:
        normalized_path = path[: -len("/openai/v1")] or ""
        endpoint = parsed._replace(
            path=normalized_path,
            params="",
            query="",
            fragment="",
        ).geturl().rstrip("/")

    return endpoint, uses_v1_route


def _missing_vars_error(
    agent_prefix: str,
    provider: str,
    missing_vars: list[str],
) -> RuntimeError:
    missing = ", ".join(missing_vars)
    return RuntimeError(
        f"{agent_prefix} provider '{provider}' is not fully configured. "
        f"Missing environment variables: {missing}."
    )


def _build_google_model(agent_prefix: str) -> str:
    model = _first_setting(
        agent_prefix,
        f"{agent_prefix}_GOOGLE_MODEL",
        f"{agent_prefix}_MODEL",
    )
    return model or DEFAULT_GOOGLE_MODEL


def _build_azure_model(agent_prefix: str) -> LiteLlm:
    endpoint = _first_setting(
        agent_prefix,
        f"{agent_prefix}_AZURE_OPENAI_ENDPOINT",
        f"{agent_prefix}_AZURE_OPENAI_BASE_URL",
        f"{agent_prefix}_AZURE_API_BASE",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_BASE_URL",
        "AZURE_API_BASE",
    )
    api_key = _first_setting(
        agent_prefix,
        f"{agent_prefix}_AZURE_OPENAI_API_KEY",
        f"{agent_prefix}_AZURE_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_KEY",
    )
    api_version = _first_setting(
        agent_prefix,
        f"{agent_prefix}_AZURE_OPENAI_API_VERSION",
        f"{agent_prefix}_AZURE_API_VERSION",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_API_VERSION",
    )
    deployment = _first_setting(
        agent_prefix,
        f"{agent_prefix}_AZURE_OPENAI_MODEL",
        f"{agent_prefix}_AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_MODEL",
        "AZURE_OPENAI_DEPLOYMENT",
    )

    missing_vars: list[str] = []
    if not endpoint:
        missing_vars.append(f"{agent_prefix}_AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing_vars.append(f"{agent_prefix}_AZURE_OPENAI_API_KEY")
    if not deployment:
        missing_vars.append(f"{agent_prefix}_AZURE_OPENAI_MODEL")
    if missing_vars:
        raise _missing_vars_error(agent_prefix, "azure", missing_vars)

    normalized_endpoint, uses_v1_route = _normalize_azure_endpoint(endpoint)

    if uses_v1_route:
        base_url = f"{normalized_endpoint}/openai/v1/"
        return LiteLlm(
            model=deployment.removeprefix("azure/"),
            api_key=api_key,
            base_url=base_url,
            custom_llm_provider="openai",
        )

    if not api_version:
        raise _missing_vars_error(
            agent_prefix,
            "azure",
            [f"{agent_prefix}_AZURE_OPENAI_API_VERSION"],
        )

    model_name = deployment if deployment.startswith("azure/") else f"azure/{deployment}"
    return LiteLlm(
        model=model_name,
        api_base=normalized_endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def build_agent_model(agent_name: str, default_provider: str) -> object:
    agent_prefix = _agent_env_prefix(agent_name)
    provider = _normalize_provider(
        _first_setting(agent_prefix, f"{agent_prefix}_PROVIDER"),
        default_provider=default_provider,
    )

    if provider == "google":
        return _build_google_model(agent_prefix)
    if provider == "azure":
        return _build_azure_model(agent_prefix)

    raise AssertionError(f"Unhandled provider: {provider}")
