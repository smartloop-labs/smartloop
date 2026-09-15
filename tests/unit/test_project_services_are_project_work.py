"""Installing a project's MCP servers and skills belongs to that project's agent.

``_setup_project_services`` is step 6 of ``/v1/init`` (step 8 of
``/v1/bootstrap``).  It runs ``_setup_project_tools`` — which registers the
system MCP server and writes the default skills — then marks the project
``system=True`` to record that the work is done, and saves.

That save claims no project.  ``/v1/init`` names no project in its path, so
under supervision it is answered by the supervisor, where ``save()`` writes
only the workspace record (``Workspace._writable_projects``).  Both halves of
the result were therefore discarded: the registered servers, and the flag that
would have stopped it happening again.  The flag going with them is what made
it re-run on every init, spawning the same MCP stdio children over and over in
a process that does not own the project they belong to — and resolving *which*
project through the ``current`` flag, so it could pick one a child was serving.

Nothing is lost by skipping it there.  ``create_project`` installs the tools
once under its own ``own_project`` claim, and the project's agent runs
``_setup_project_tools`` for the project it is pinned to when it loads a model
(``ModelService.load_model``) — where the save actually persists.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock

import pytest

from smartloop.api.routers.init import _setup_project_services
from smartloop.workspace.context import delegate_projects, pin_project


@pytest.fixture(autouse=True)
def _clear_process_scope():
    """The pin and the delegation flag are process-global; never leak one."""
    yield
    pin_project(None)
    delegate_projects(False)


def _svc(system: bool = False):
    """A model service whose current project still needs its services.

    ``system`` is the "already set up" marker, so False is the state that
    makes the body below run at all.
    """
    project = SimpleNamespace(id="alpha", name="alpha", system=system)
    svc = MagicMock()
    svc._model.workspace.current_project = project
    return svc


async def _drain(gen):
    return [event async for event in gen]


class TestOnlyAProjectsOwnProcessSetsItUp:
    async def test_a_delegating_supervisor_installs_nothing(self):
        svc = _svc()
        delegate_projects(True)

        events = await _drain(_setup_project_services(svc))

        assert events == []
        svc._setup_project_tools.assert_not_called()

    async def test_a_delegating_supervisor_writes_nothing(self):
        """The save is the half that would have been discarded anyway."""
        svc = _svc()
        delegate_projects(True)

        await _drain(_setup_project_services(svc))

        svc._model.save.assert_not_called()
        assert svc._model.workspace.current_project.system is False

    async def test_the_single_server_fallback_still_sets_up(self):
        """``SLP_SUPERVISOR=0``: unpinned and undelegated, as it always was."""
        svc = _svc()

        events = await _drain(_setup_project_services(svc))

        svc._setup_project_tools.assert_called_once()
        svc._model.save.assert_called_once()
        assert svc._model.workspace.current_project.system is True
        assert len(events) == 2


class TestTheGuardIsNarrow:
    async def test_an_already_set_up_project_is_still_skipped(self):
        """The pre-existing early return is unchanged by the new one."""
        svc = _svc(system=True)

        events = await _drain(_setup_project_services(svc))

        assert events == []
        svc._setup_project_tools.assert_not_called()

    async def test_a_project_less_workspace_is_still_skipped(self):
        svc = _svc()
        svc._model.workspace.current_project = None

        events = await _drain(_setup_project_services(svc))

        assert events == []
        svc._setup_project_tools.assert_not_called()

    async def test_delegation_is_checked_before_the_workspace_is_read(self):
        """A supervisor must not resolve ``current_project`` to decide this.

        That attribute is the ``current`` flag — the thing the per-project
        split exists to stop consulting — so reading it here would pick out
        some project a child is serving before concluding it has no business
        touching one.  The read is made to raise, so the order is asserted
        rather than assumed.
        """
        svc = _svc()
        workspace = MagicMock()
        type(workspace).current_project = PropertyMock(
            side_effect=AssertionError("resolved the current project under delegation")
        )
        svc._model.workspace = workspace
        delegate_projects(True)

        assert await _drain(_setup_project_services(svc)) == []
