import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

os.environ.setdefault("ATI_SEARCH_PROVIDER", "google")

from ati_search import agent
from ati_search.env import get_env_value, read_dotenv_layers
from ati_search.tools import tfs_git_search as tfs_tool


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="") -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class AtiSearchAgentTest(unittest.TestCase):
    def test_env_loading_precedence_prefers_agent_then_shared_then_root_then_external(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            external = temp_path / "external.env"
            root = temp_path / "root.env"
            shared = temp_path / "shared.env"
            local = temp_path / "local.env"

            external.write_text("TFS_URL=http://external\nPROJECT=External\nPAT=external\n", encoding="utf-8")
            root.write_text("PROJECT=Root\nPAT=root\n", encoding="utf-8")
            shared.write_text("PAT=shared\nAPI_VERSION=6.0\n", encoding="utf-8")
            local.write_text("PAT=local\nAPI_VERSION=7.0\n", encoding="utf-8")

            env_values = read_dotenv_layers([local, shared, root, external])
            with patch.dict("os.environ", {}, clear=True):
                config, error = tfs_tool.get_tfs_config(env_values)

            self.assertIsNone(error)
            self.assertIsNotNone(config)
            self.assertEqual(config["tfs_url"], "http://external")
            self.assertEqual(config["project"], "Root")
            self.assertEqual(config["pat"], "local")
            self.assertEqual(config["api_version"], "7.0")

    def test_env_lookup_prefers_process_env_over_dotenv_values(self) -> None:
        with patch.dict("os.environ", {"ATI_SEARCH_BEARER_TOKEN": "runtime"}, clear=True):
            value = get_env_value("ATI_SEARCH_BEARER_TOKEN", {"ATI_SEARCH_BEARER_TOKEN": "file"})

        self.assertEqual(value, "runtime")

    def test_missing_tfs_config_returns_error(self) -> None:
        with patch.object(
            tfs_tool,
            "get_tfs_config",
            return_value=(None, "Missing TFS configuration: TFS_URL, PROJECT, PAT."),
        ):
            result = tfs_tool.tfs_git_search("burndown")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "missing_configuration")
        self.assertIn("Missing TFS configuration", result["message"])

    def test_resolve_repositories_all_or_named(self) -> None:
        repos = [{"name": "RepoA"}, {"name": "RepoB"}]

        selected, error = tfs_tool.resolve_repositories(repos, None)
        self.assertIsNone(error)
        self.assertEqual(selected, repos)

        selected, error = tfs_tool.resolve_repositories(repos, "repob")
        self.assertIsNone(error)
        self.assertEqual(selected, [{"name": "RepoB"}])

    def test_tfs_git_search_combines_git_and_work_items(self) -> None:
        config = {
            "tfs_url": "http://tfs",
            "project": "Proj",
            "pat": "pat",
            "api_version": "4.1",
            "default_repo": "",
        }
        repos = [{"name": "RepoA", "id": "1", "default_branch": "refs/heads/main"}]
        git_matches = [
            {
                "match_type": "code_search",
                "repository": "RepoA",
                "path": "/src/SprintReport.py",
                "file_name": "SprintReport.py",
                "branch": "refs/heads/main",
                "url": "http://tfs/Proj/_git/RepoA?path=/src/SprintReport.py",
                "snippet": "SprintReport",
            }
        ]
        work_items = [
            {
                "id": "42",
                "title": "Sprint report work item",
                "state": "Active",
                "work_item_type": "Task",
                "assigned_to": None,
                "changed_date": "2026-03-17",
                "url": "http://tfs/Proj/_workitems/edit/42",
            }
        ]

        with patch.dict(
            "os.environ",
            {"TFS_URL": "http://tfs", "PROJECT": "Proj", "PAT": "pat"},
            clear=True,
        ):
            with patch.object(tfs_tool, "list_repositories", return_value=(repos, {"repositories": repos})):
                with patch.object(tfs_tool, "search_code_api", return_value=(git_matches, [], {"code_search": {"ok": True}})):
                    with patch.object(tfs_tool, "search_work_items", return_value=(work_items, [], {"work_items": {"ok": True}})):
                        result = tfs_tool.tfs_git_search("SprintReport", top=3)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["summary"]["repositories_scanned"], ["RepoA"])
        self.assertEqual(result["summary"]["git_matches"][0]["repository"], "RepoA")
        self.assertEqual(result["summary"]["work_item_matches"][0]["id"], "42")
        self.assertIn("code_search", result["raw"])
        self.assertIn("work_items", result["raw"])

    def test_repository_listing_http_error_is_reported(self) -> None:
        config = {
            "tfs_url": "http://tfs",
            "project": "Proj",
            "pat": "pat",
            "api_version": "4.1",
            "default_repo": "",
        }

        with patch.object(
            tfs_tool,
            "tfs_request",
            return_value=FakeResponse(status_code=500, text="server error"),
        ):
            repos, error = tfs_tool.list_repositories(config)

        self.assertIsNone(repos)
        self.assertEqual(error["error_type"], "http_error")
        self.assertEqual(error["status_code"], 500)

    def test_code_search_timeout_falls_back_to_path_matching(self) -> None:
        repos = [{"name": "RepoA", "id": "1", "default_branch": "refs/heads/main"}]
        path_payload = {
            "value": [
                {"path": "/src/TFS_SprintReport.py", "isFolder": False},
                {"path": "/src/Other.py", "isFolder": False},
            ]
        }

        def fake_request(method, url, config, **kwargs):
            if "codesearchresults" in url:
                raise requests.Timeout()
            return FakeResponse(status_code=200, json_data=path_payload)

        with patch.dict(
            "os.environ",
            {"TFS_URL": "http://tfs", "PROJECT": "Proj", "PAT": "pat"},
            clear=True,
        ):
            with patch.object(tfs_tool, "list_repositories", return_value=(repos, {"repositories": repos})):
                with patch.object(tfs_tool, "tfs_request", side_effect=fake_request):
                    result = tfs_tool.tfs_git_search("SprintReport", include_work_items=False, top=5)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["summary"]["git_matches"][0]["match_type"], "path_match")
        self.assertTrue(
            any("Code-content search" in note or "code search timed out" in note.lower() for note in result["summary"]["notes"])
        )

    def test_code_search_http_500_falls_back_to_path_matching(self) -> None:
        repos = [{"name": "RepoA", "id": "1", "default_branch": "refs/heads/main"}]
        path_payload = {
            "value": [
                {"path": "/src/TFS_SprintReport.py", "isFolder": False},
            ]
        }

        def fake_request(method, url, config, **kwargs):
            if "codesearchresults" in url:
                return FakeResponse(status_code=500, text='{"count":1,"value":{"Message":"An error has occurred."}}')
            return FakeResponse(status_code=200, json_data=path_payload)

        with patch.dict(
            "os.environ",
            {"TFS_URL": "http://tfs", "PROJECT": "Proj", "PAT": "pat"},
            clear=True,
        ):
            with patch.object(tfs_tool, "list_repositories", return_value=(repos, {"repositories": repos})):
                with patch.object(tfs_tool, "tfs_request", side_effect=fake_request):
                    result = tfs_tool.tfs_git_search("SprintReport", include_work_items=False, top=5)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["summary"]["git_matches"][0]["match_type"], "path_match")
        self.assertTrue(
            any("code-content search is unavailable" in note.lower() for note in result["summary"]["notes"])
        )

    def test_work_item_query_text_avoids_top_clause_for_tfs_2018_compatibility(self) -> None:
        config = {
            "tfs_url": "http://tfs",
            "project": "Proj",
            "pat": "pat",
            "api_version": "4.1",
            "default_repo": "",
        }

        query = tfs_tool.work_item_query_text("SprintReport", 5, config)

        self.assertIn("SELECT [System.Id]", query)
        self.assertNotIn("TOP 5", query)

    def test_root_agent_exposes_both_tools(self) -> None:
        self.assertTrue(hasattr(agent.root_agent, "tools"))
        self.assertEqual(len(agent.root_agent.tools), 2)


if __name__ == "__main__":
    unittest.main()
