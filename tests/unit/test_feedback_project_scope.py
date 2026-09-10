"""A thumbs-up belongs to the project it was given in.

``POST /v1/feedback`` names its project in the *body*, so nothing in the path
or the query string bound it and the supervisor answered it. The store was
written under the explicit project id and so landed correctly — but the
*index* was built through ``current_project``, which on an unbound process is
whichever project the UI happens to have selected. Rating an answer in one
project therefore embedded it into another's feedback index.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.workspace.workspace import Project, Workspace


def _project(pid: str, current: bool = False) -> Project:
    return Project(id=pid, project_dir=f"/tmp/{pid}", current=current, name=pid)


@pytest.fixture
def service():
    ModelService._instance = None
    made = ModelService.__new__(ModelService)
    ModelService.__init__(made)
    workspace = Workspace(id="ws-1", workspace_dir="/tmp/ws-1")
    workspace.projects = [_project("alpha", current=True), _project("beta")]
    made._model = SimpleNamespace(workspace=workspace)
    yield made
    ModelService._instance = None


@pytest.fixture
def built():
    """Capture the project_dir every FeedbackIndexer is constructed with."""
    with patch("smartloop.api.services.model_service.FeedbackIndexer") as cls:
        cls.side_effect = lambda project_dir, logger, base_dir: SimpleNamespace(
            project_dir=project_dir,
        )
        yield cls


class TestTheIndexFollowsTheAddressedProject:
    def test_an_explicit_project_wins_over_the_selection(self, service, built):
        indexer = service._ensure_feedback_indexer(
            service._model.workspace.projects[1],
        )

        assert indexer.project_dir == "/tmp/beta"

    def test_it_is_not_the_selected_project(self, service, built):
        """The bug, stated directly: alpha is selected, beta was addressed."""
        assert service._model.workspace.current_project.id == "alpha"

        indexer = service._ensure_feedback_indexer(
            service._model.workspace.projects[1],
        )

        assert indexer.project_dir != "/tmp/alpha"

    def test_no_project_still_falls_back_to_the_binding(self, service, built):
        """Unchanged for the chat path, where the request is bound."""
        indexer = service._ensure_feedback_indexer()

        assert indexer.project_dir == "/tmp/alpha"

    def test_each_project_gets_its_own_cached_index(self, service, built):
        alpha, beta = service._model.workspace.projects

        first = service._ensure_feedback_indexer(alpha)
        second = service._ensure_feedback_indexer(beta)

        assert first is not second
        assert service._ensure_feedback_indexer(alpha) is first

    def test_no_project_at_all_is_an_error(self, service, built):
        for project in service._model.workspace.projects:
            project.current = False

        with pytest.raises(ValueError, match="No active project"):
            service._ensure_feedback_indexer()
