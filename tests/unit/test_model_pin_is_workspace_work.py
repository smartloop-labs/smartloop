"""Pinning a model across projects is work only the workspace owner can do.

``pin_model_for_existing_projects`` freezes what a model resolves to in every
project *before* the app-default moves, so the new default only reaches
projects created after it.  A project with no ``ProjectModel`` entry reads
through to the workspace record, so without the pin an app-default change
rewrites the past.

It writes every project's ``model_settings``, which makes it workspace-owner
work in the strict sense: a per-project agent child may write one file, and a
supervisor that has delegated its projects may write none of them
(``Workspace._writable_projects``).  In either tier the entries appended here
are dropped at ``save()`` while the caller goes on to move the app-default
believing the leak was sealed — which is the one outcome the pin exists to
prevent.  The child case was already guarded; the delegating supervisor was
not, and it is the tier the desktop app actually runs.

A delegated project pins itself instead, lazily, through
``sync_project_models`` on its next project-route read or write — where the
child that owns the file is the one doing the writing.
"""

import pytest

from smartloop.config.app_settings import AppSettings
from smartloop.workspace.context import delegate_projects, pin_project
from smartloop.workspace.workspace import Project, Workspace, WorkspaceModel

BOOTSTRAP_MODEL = AppSettings().default_model


@pytest.fixture(autouse=True)
def _clear_process_scope():
    """The pin and the delegation flag are process-global; never leak one."""
    yield
    pin_project(None)
    delegate_projects(False)


def _workspace() -> Workspace:
    """Two projects, neither holding an entry for the disabled model.

    "No entry" is the state the pin exists for: both projects currently
    resolve ``qwen2-vl-2b`` through the workspace record, so moving that
    record is what would reach back into them.
    """
    return Workspace(
        id="ws1",
        models=[
            WorkspaceModel(
                model_name=BOOTSTRAP_MODEL,
                model_id="smartloop/slp-orchestra-mini",
                downloaded=True,
                enabled=True,
            ),
            WorkspaceModel(
                model_name="qwen2-vl-2b",
                model_id="Qwen/Qwen2-VL-2B-Instruct",
                downloaded=True,
                enabled=False,
            ),
        ],
        projects=[
            Project(id="alpha", project_dir="/tmp/alpha", current=True, name="alpha"),
            Project(id="beta", project_dir="/tmp/beta", current=False, name="beta"),
        ],
    )


def _settings(ws):
    return {p.id: [m.model_name for m in p.model_settings] for p in ws.projects}


class TestOnlyTheWorkspaceOwnerPins:
    def test_a_delegating_supervisor_pins_nothing(self):
        ws = _workspace()
        delegate_projects(True)

        ws.pin_model_for_existing_projects("qwen2-vl-2b")

        assert _settings(ws) == {"alpha": [], "beta": []}

    def test_a_pinned_child_pins_nothing(self):
        ws = _workspace()
        pin_project("alpha")

        ws.pin_model_for_existing_projects("qwen2-vl-2b")

        assert _settings(ws) == {"alpha": [], "beta": []}

    def test_the_single_server_fallback_still_pins(self):
        """``SLP_SUPERVISOR=0``: unpinned and undelegated, as it always was."""
        ws = _workspace()

        ws.pin_model_for_existing_projects("qwen2-vl-2b")

        assert _settings(ws) == {"alpha": ["qwen2-vl-2b"], "beta": ["qwen2-vl-2b"]}
        assert all(
            ws.project_model(p, "qwen2-vl-2b").enabled is False for p in ws.projects
        )

    def test_the_guard_does_not_depend_on_being_asked_first(self):
        """Delegation alone is enough — the supervisor is never also pinned."""
        ws = _workspace()
        delegate_projects(True)
        pin_project(None)

        ws.pin_model_for_existing_projects("qwen2-vl-2b")

        assert _settings(ws) == {"alpha": [], "beta": []}


class TestTheAppDefaultStillMoves:
    """The guard withholds the pin, not the promotion it precedes.

    ``promote_model_to_workspace_default`` calls the pin and then flips the
    workspace record.  That record is ``config.json``, which is precisely what
    a delegating supervisor *does* own, so the promotion itself must still
    land — only the per-project half is someone else's to write.
    """

    def test_a_delegating_supervisor_still_promotes(self):
        ws = _workspace()
        delegate_projects(True)

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert ws.get_model("qwen2-vl-2b").enabled is True
        assert _settings(ws) == {"alpha": [], "beta": []}

    def test_a_pinned_child_promotes_nothing(self):
        """A child cannot write ``config.json`` at all, so it must not try."""
        ws = _workspace()
        pin_project("alpha")

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert ws.get_model("qwen2-vl-2b").enabled is False

    def test_a_delegated_project_pins_itself_on_its_next_sync(self):
        """The residual is bounded by ``sync_project_models`` running first.

        A project that has been through the project route already holds an
        entry, and that entry survives the promotion the supervisor goes on to
        make — which is the mechanism standing in for the pin under
        delegation.
        """
        ws = _workspace()
        beta = ws.projects[1]
        ws.sync_project_models(beta)
        delegate_projects(True)

        ws.promote_model_to_workspace_default("qwen2-vl-2b")

        assert ws.effective_model_enabled("qwen2-vl-2b", beta) is False
        # …while a project that never synced reads through to the new default.
        assert ws.effective_model_enabled("qwen2-vl-2b", ws.projects[0]) is True
