"""Downloading the shared orchestrator weights belongs to the workspace owner.

``_bootstrap_orchestrator_model`` pulls the orchestrator GGUF and records it as
downloaded in ``config.json``. Both halves are workspace-level: the weights sit
in one shared ``models_dir``, and a per-project agent never writes the workspace
record, so a child running this would fetch its own copy of the weights and
then lose the flag at save, doing it all again on the next start.

Children usually never reach it, because the pool sets
``SLP_API_PRELOAD_MODEL=0`` — but only when there is a model host to reach, so
with the host disabled or failed they inherit the supervisor's preload setting
and arrive here after all.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.main import _bootstrap_orchestrator_model
from smartloop.workspace.context import pin_project


@pytest.fixture(autouse=True)
def _clear_pin():
    yield
    pin_project(None)


@pytest.fixture
def downloader():
    with patch("smartloop.model.downloader.artifact_downloader.SLPArtifactDownloader") as cls:
        instance = cls.return_value
        instance._bootstrap_source.list_models.return_value = [{"name": "sl-mini"}]
        yield instance


class TestOnlyTheWorkspaceOwnerBootstraps:
    def test_a_pinned_child_does_not_download(self, downloader):
        pin_project("alpha")
        service = MagicMock()

        with pytest.raises(FileNotFoundError, match="does not own"):
            _bootstrap_orchestrator_model(service)

        downloader.bootstrap_download.assert_not_called()

    def test_a_pinned_child_writes_nothing_to_the_workspace(self, downloader):
        pin_project("alpha")
        service = MagicMock()

        with pytest.raises(FileNotFoundError):
            _bootstrap_orchestrator_model(service)

        service._ensure_workspace.assert_not_called()

    def test_the_workspace_owner_still_bootstraps(self, downloader):
        service = MagicMock()

        _bootstrap_orchestrator_model(service)

        downloader.bootstrap_download.assert_called_once_with("sl-mini")
        service._ensure_workspace.return_value.save.assert_called_once()

    def test_the_owner_records_the_model_as_downloaded(self, downloader):
        service = MagicMock()
        workspace = service._ensure_workspace.return_value

        _bootstrap_orchestrator_model(service)

        assert workspace.get_model.return_value.downloaded is True
