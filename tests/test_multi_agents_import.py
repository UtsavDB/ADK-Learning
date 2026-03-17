import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class AgentImportTest(unittest.TestCase):
    def _load_module(self, module_name: str, path: str):
        agent_file = Path(path).resolve()
        spec = importlib.util.spec_from_file_location(module_name, agent_file)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_multi_agents_root_agent_loads(self) -> None:
        env = {
            "MULTI_AGENTS_AZURE_OPENAI_ENDPOINT": "https://multi.example.com/openai/v1/",
            "MULTI_AGENTS_AZURE_OPENAI_API_KEY": "test-key",
            "MULTI_AGENTS_AZURE_OPENAI_API_VERSION": "v1",
            "MULTI_AGENTS_AZURE_OPENAI_MODEL": "travel-planner",
        }
        with patch.dict(os.environ, env, clear=False):
            module = self._load_module("multi_agents_agent", "multi_agents/agent.py")

        self.assertTrue(hasattr(module, "root_agent"))

    def test_ati_search_root_agent_loads(self) -> None:
        env = {
            "ATI_SEARCH_AZURE_OPENAI_ENDPOINT": "https://ati.example.com/openai/v1/",
            "ATI_SEARCH_AZURE_OPENAI_API_KEY": "test-key",
            "ATI_SEARCH_AZURE_OPENAI_API_VERSION": "v1",
            "ATI_SEARCH_AZURE_OPENAI_MODEL": "ati-search",
        }
        with patch.dict(os.environ, env, clear=False):
            module = self._load_module("ati_search_agent", "ati_search/agent.py")

        self.assertTrue(hasattr(module, "root_agent"))

    def test_agent_directories_use_valid_identifiers(self) -> None:
        invalid = []
        for path in Path(".").iterdir():
            if not path.is_dir():
                continue
            if path.name.startswith("."):
                continue
            if not (path / "agent.py").exists():
                continue
            if not path.name.isidentifier():
                invalid.append(path.name)

        self.assertEqual(
            invalid,
            [],
            f"Agent directory names must be valid Python identifiers: {invalid}",
        )


if __name__ == "__main__":
    unittest.main()
