import importlib.util
from pathlib import Path
import unittest


class MultiAgentsImportTest(unittest.TestCase):
    def test_multi_agents_root_agent_loads(self) -> None:
        agent_file = Path("multi_agents/agent.py").resolve()
        spec = importlib.util.spec_from_file_location("multi_agents_agent", agent_file)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

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
