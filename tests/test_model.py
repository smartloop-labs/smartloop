import os
import unittest

from smartloop.config.model_config import ModelConfig
from smartloop.model.model_manager import ModelManager
from smartloop.model_host.client import ModelHostError, RemoteModelClient
from smartloop.workspace.workspace import Project, Workspace


class TestModel(unittest.TestCase):
    def test_stale_project_id_is_cleared(self):
        home = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp_home"))
        os.makedirs(home, exist_ok=True)
        os.environ["SLP_HOME"] = home
        Workspace.reset()
        try:
            ws_dir = os.path.join(home, "workspace-1")
            os.makedirs(ws_dir, exist_ok=True)
            workspace = Workspace(
                id="workspace-1",
                workspace_dir=ws_dir,
                models_dir=os.path.join(ws_dir, "models"),
                projects=[],
            )
            project = Project(
                id="real-project",
                project_dir=os.path.join(ws_dir, "real-project"),
                current=True,
                name="real",
            )
            os.makedirs(project.project_dir, exist_ok=True)
            workspace.projects = [project]
            workspace.save()

            config = ModelConfig(model_name="sl-mini", project_id="ghost-project")
            workspace.add_model("sl-mini", "smartloop-ai/slp-orchestra-mini", config=config)
            workspace.save()

            cleaned = ModelManager._purge_stale_project_id(workspace, config, __import__("logging").getLogger(__name__))

            self.assertIsNone(cleaned.project_id)
            self.assertIsNone(config.project_id)
            self.assertIsNone(workspace.get_model("sl-mini").config.project_id)
        finally:
            Workspace.reset()
            if "SLP_HOME" in os.environ:
                del os.environ["SLP_HOME"]

    def test_unbound_model_host_port_is_rejected(self):
        client = RemoteModelClient("127.0.0.1:0")
        with self.assertRaises(ModelHostError):
            client._connect()


if __name__ == "__main__":
    unittest.main()