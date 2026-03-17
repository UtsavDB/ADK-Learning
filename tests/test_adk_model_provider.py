import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.adk_model_provider import (
    build_agent_generation_config,
    build_agent_model,
)


class FakeLiteLlm:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs


class AdkModelProviderTest(unittest.TestCase):
    def test_google_provider_returns_scoped_model(self) -> None:
        env = {
            "ATI_SEARCH_PROVIDER": "google",
            "ATI_SEARCH_GOOGLE_MODEL": "gemini-2.5-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            model = build_agent_model("ati_search", default_provider="azure")

        self.assertEqual(model, "gemini-2.5-flash")

    def test_google_provider_uses_legacy_model_name_as_fallback(self) -> None:
        env = {
            "ATI_SEARCH_PROVIDER": "google",
            "ATI_SEARCH_MODEL": "gemini-legacy",
        }
        with patch.dict(os.environ, env, clear=True):
            model = build_agent_model("ati_search", default_provider="azure")

        self.assertEqual(model, "gemini-legacy")

    def test_generation_config_uses_scoped_max_output_tokens(self) -> None:
        env = {
            "ATI_SEARCH_MAX_OUTPUT_TOKENS": "128",
        }
        with patch.dict(os.environ, env, clear=True):
            config = build_agent_generation_config("ati_search")

        self.assertIsNotNone(config)
        self.assertEqual(config.max_output_tokens, 128)

    def test_generation_config_returns_none_when_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = build_agent_generation_config("ati_search")

        self.assertIsNone(config)

    def test_generation_config_rejects_non_integer_values(self) -> None:
        env = {
            "ATI_SEARCH_MAX_OUTPUT_TOKENS": "abc",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as context:
                build_agent_generation_config("ati_search")

        self.assertIn("ATI_SEARCH_MAX_OUTPUT_TOKENS must be an integer", str(context.exception))

    def test_generation_config_rejects_non_positive_values(self) -> None:
        env = {
            "ATI_SEARCH_MAX_OUTPUT_TOKENS": "0",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(RuntimeError) as context:
                build_agent_generation_config("ati_search")

        self.assertIn(
            "ATI_SEARCH_MAX_OUTPUT_TOKENS must be greater than zero",
            str(context.exception),
        )

    def test_azure_v1_endpoint_uses_openai_compatible_litellm(self) -> None:
        env = {
            "ATI_SEARCH_AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/openai/v1/",
            "ATI_SEARCH_AZURE_OPENAI_API_KEY": "azure-key",
            "ATI_SEARCH_AZURE_OPENAI_MODEL": "ati-deployment",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "shared.adk_model_provider.LiteLlm",
            FakeLiteLlm,
        ):
            model = build_agent_model("ati_search", default_provider="azure")

        self.assertEqual(model.model, "ati-deployment")
        self.assertEqual(
            model.kwargs,
            {
                "api_key": "azure-key",
                "base_url": "https://example.openai.azure.com/openai/v1/",
                "custom_llm_provider": "openai",
            },
        )

    def test_azure_legacy_endpoint_passes_explicit_azure_kwargs(self) -> None:
        env = {
            "MULTI_AGENTS_AZURE_OPENAI_ENDPOINT": "https://legacy.openai.azure.com/",
            "MULTI_AGENTS_AZURE_OPENAI_API_KEY": "azure-key",
            "MULTI_AGENTS_AZURE_OPENAI_API_VERSION": "2024-10-21",
            "MULTI_AGENTS_AZURE_OPENAI_MODEL": "travel-deployment",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "shared.adk_model_provider.LiteLlm",
            FakeLiteLlm,
        ):
            model = build_agent_model("multi_agents", default_provider="azure")

        self.assertEqual(model.model, "azure/travel-deployment")
        self.assertEqual(
            model.kwargs,
            {
                "api_base": "https://legacy.openai.azure.com",
                "api_key": "azure-key",
                "api_version": "2024-10-21",
            },
        )

    def test_missing_selected_provider_vars_fails_clearly(self) -> None:
        env = {
            "ATI_SEARCH_PROVIDER": "azure",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_dir = root / "shared"
            shared_dir.mkdir()
            with patch.dict(os.environ, env, clear=True), patch(
                "shared.adk_model_provider._REPO_ROOT_DIR",
                root,
            ), patch(
                "shared.adk_model_provider._SHARED_DIR",
                shared_dir,
            ):
                with self.assertRaises(RuntimeError) as context:
                    build_agent_model("ati_search", default_provider="azure")

        message = str(context.exception)
        self.assertIn("ATI_SEARCH provider 'azure'", message)
        self.assertIn("ATI_SEARCH_AZURE_OPENAI_ENDPOINT", message)
        self.assertIn("ATI_SEARCH_AZURE_OPENAI_API_KEY", message)
        self.assertIn("ATI_SEARCH_AZURE_OPENAI_MODEL", message)

    def test_agent_scoped_azure_settings_do_not_bleed_between_agents(self) -> None:
        env = {
            "AZURE_API_BASE": "unchanged-base",
            "AZURE_API_VERSION": "unchanged-version",
            "ATI_SEARCH_AZURE_OPENAI_ENDPOINT": "https://ati.example.com/",
            "ATI_SEARCH_AZURE_OPENAI_API_KEY": "ati-key",
            "ATI_SEARCH_AZURE_OPENAI_API_VERSION": "2024-01-01",
            "ATI_SEARCH_AZURE_OPENAI_MODEL": "ati-deployment",
            "MULTI_AGENTS_AZURE_OPENAI_ENDPOINT": "https://multi.example.com/",
            "MULTI_AGENTS_AZURE_OPENAI_API_KEY": "multi-key",
            "MULTI_AGENTS_AZURE_OPENAI_API_VERSION": "2024-02-02",
            "MULTI_AGENTS_AZURE_OPENAI_MODEL": "multi-deployment",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "shared.adk_model_provider.LiteLlm",
            FakeLiteLlm,
        ):
            ati_model = build_agent_model("ati_search", default_provider="azure")
            multi_model = build_agent_model("multi_agents", default_provider="azure")

            self.assertEqual(ati_model.kwargs["api_base"], "https://ati.example.com")
            self.assertEqual(multi_model.kwargs["api_base"], "https://multi.example.com")
            self.assertEqual(os.environ["AZURE_API_BASE"], "unchanged-base")
            self.assertEqual(os.environ["AZURE_API_VERSION"], "unchanged-version")

    def test_shared_dotenv_values_are_used_when_process_env_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shared_dir = root / "shared"
            shared_dir.mkdir()
            (shared_dir / ".env").write_text(
                "\n".join(
                    [
                        "AZURE_OPENAI_ENDPOINT=https://shared.example.com/openai/v1",
                        "AZURE_OPENAI_API_KEY=shared-key",
                        "AZURE_OPENAI_MODEL=shared-deployment",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True), patch(
                "shared.adk_model_provider._REPO_ROOT_DIR",
                root,
            ), patch(
                "shared.adk_model_provider._SHARED_DIR",
                shared_dir,
            ), patch(
                "shared.adk_model_provider.LiteLlm",
                FakeLiteLlm,
            ):
                model = build_agent_model("ati_search", default_provider="azure")

        self.assertEqual(model.model, "shared-deployment")
        self.assertEqual(
            model.kwargs,
            {
                "api_key": "shared-key",
                "base_url": "https://shared.example.com/openai/v1/",
                "custom_llm_provider": "openai",
            },
        )


if __name__ == "__main__":
    unittest.main()
