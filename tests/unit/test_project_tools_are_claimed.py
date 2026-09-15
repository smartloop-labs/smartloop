"""A new project's tools are installed under the claim that lets them persist.

``create_project`` runs in the supervisor — creating a project is one of the
operations that straddle a child's lifetime, so it is never proxied — and it
carefully claims the project for each of its ``save()`` calls with
``own_project``.  A delegating supervisor writes no project file without one
(``Workspace._writable_projects``).

``_setup_project_tools`` sat outside every claim.  It is not a pure in-memory
step: ``install_system_tools_server`` appends the System Tools server and saves
it itself, as do ``_migrate_servers_to_local`` and ``remove_fetch_server``.  So
under delegation a project came up with no tools registered, and only grew them
back later, when a child first loaded a model for it and re-ran the same setup
from a process that could write the file.

Claiming is safe for the same reason the two saves above it already give: no
child serves this project yet — it has never been requested — so this process
writes its first ``project.json`` itself.

What is asserted here is the claim being *active around the call*, rather than
the bytes on disk: that ``own_project`` makes a write land is already covered
by ``TestTheSupervisorDoesNotWriteWhatItDelegated`` in
``test_project_state_split.py``, and the regression this guards against is the
call sitting outside it.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.workspace.context import (
    delegate_projects, owned_project_id, pin_project,
)


@pytest.fixture(autouse=True)
def _clear_process_scope():
    """The pin and the delegation flag are process-global; never leak one."""
    yield
    pin_project(None)
    delegate_projects(False)


@pytest.fixture
def svc(tmp_path):
    """A model service whose workspace is real enough to make a directory in.

    ``create_project`` joins ``workspace_dir`` into a path and appends to
    ``projects``, so those two need real values; everything else it touches is
    recorded rather than performed.
    """
    service = MagicMock()
    service._model.workspace.workspace_dir = str(tmp_path)
    service._model.workspace.projects = []
    return service


def _claims(svc) -> list:
    """The claim in force each time ``_setup_project_tools`` was called."""
    seen: list = []
    svc._setup_project_tools.side_effect = lambda project: seen.append(owned_project_id())
    return seen


class TestANewProjectsToolsAreWritten:
    def test_the_install_runs_under_a_claim_on_the_new_project(self, svc):
        claims = _claims(svc)
        delegate_projects(True)

        ModelService.create_project(svc, name="alpha")

        created = svc._model.workspace.projects[0]
        assert claims == [created.id]

    def test_the_claim_does_not_outlive_the_install(self, svc):
        """It stays as narrow as the operation, like the saves around it."""
        _claims(svc)
        delegate_projects(True)

        ModelService.create_project(svc, name="alpha")

        assert owned_project_id() is None

    def test_the_single_server_fallback_still_installs(self, svc):
        """``SLP_SUPERVISOR=0``: the claim is redundant, the install is not.

        Unpinned and undelegated, ``_writable_projects`` already returns every
        project, so the claim changes nothing — but wrapping the call must not
        have stopped it happening.
        """
        _claims(svc)

        ModelService.create_project(svc, name="alpha")

        created = svc._model.workspace.projects[0]
        svc._setup_project_tools.assert_called_once_with(created)
