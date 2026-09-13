"""The project a request works on is scoped to that request, not to the process.

Before this, ``project_id`` was honoured by flipping a ``current`` flag on a
workspace-wide list. Two chats on different projects therefore raced: the
second moved the flag while the first was still resolving documents, skills and
MCP servers through it. These tests pin the behaviour that replaced it.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from smartloop.workspace.context import (
    ProjectBoundExecutor,
    ProjectContext,
    active_project_id,
    bind,
    pin_project,
    pinned_project_id,
    use_project,
)
from smartloop.workspace.types import Document
from smartloop.workspace.workspace import Project, Workspace


def _project(pid: str, current: bool = False) -> Project:
    return Project(id=pid, project_dir=f"/tmp/{pid}", current=current, name=pid)


@pytest.fixture
def workspace() -> Workspace:
    ws = Workspace(id="ws-1", workspace_dir="/tmp/ws-1")
    ws.projects = [_project("alpha", current=True), _project("beta"), _project("gamma")]
    return ws


@pytest.fixture(autouse=True)
def _clear_pin():
    """A process pin is global; never let one leak into the next test."""
    yield
    pin_project(None)


class TestTheBindingIsScoped:
    def test_nothing_is_bound_by_default(self):
        assert active_project_id() is None

    def test_a_binding_lasts_only_for_its_block(self):
        with use_project("beta"):
            assert active_project_id() == "beta"
        assert active_project_id() is None

    def test_bindings_nest(self):
        with use_project("beta"):
            with use_project("gamma"):
                assert active_project_id() == "gamma"
            assert active_project_id() == "beta"

    def test_binding_to_nothing_is_a_no_op(self):
        """A request that named no project must keep falling through."""
        with use_project("beta"):
            with use_project(None):
                assert active_project_id() == "beta"


class TestWhichProjectResolves:
    def test_the_bound_project_wins_over_the_selected_one(self, workspace):
        assert workspace.current_project.id == "alpha"  # the UI's selection
        with use_project("beta"):
            assert workspace.current_project.id == "beta"

    def test_an_unbound_reader_still_gets_the_selection(self, workspace):
        """The CLI, the supervisor and clients that send no project_id."""
        assert workspace.current_project.id == "alpha"

    def test_a_binding_to_an_unknown_project_falls_back_rather_than_failing(
        self, workspace, caplog
    ):
        with use_project("not-a-project"):
            assert workspace.current_project.id == "alpha"
        assert "not in this workspace" in caplog.text

    def test_a_pinned_process_resolves_its_own_project(self, workspace):
        """A per-project agent child fixes its project once, at spawn."""
        pin_project("gamma")
        assert pinned_project_id() == "gamma"
        assert workspace.current_project.id == "gamma"

    def test_a_request_binding_still_wins_over_the_pin(self, workspace):
        pin_project("gamma")
        with use_project("beta"):
            assert workspace.current_project.id == "beta"


class TestConcurrentRequestsDoNotRaceEachOther:
    """The bug this replaced: request B moved the pointer out from under A."""

    def test_two_threads_hold_different_projects_at_once(self, workspace):
        seen: dict[str, list] = {"beta": [], "gamma": []}
        both_bound = threading.Barrier(2, timeout=5)

        def worker(pid: str) -> None:
            with use_project(pid):
                # Both bindings are live before either reads, so a shared
                # pointer would have one thread answering for the other.
                both_bound.wait()
                for _ in range(20):
                    seen[pid].append(workspace.current_project.id)
                    time.sleep(0.001)

        threads = [threading.Thread(target=worker, args=(p,)) for p in ("beta", "gamma")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert seen["beta"] == ["beta"] * 20
        assert seen["gamma"] == ["gamma"] * 20

    def test_a_second_request_does_not_disturb_the_first(self, workspace):
        with use_project("beta"):
            done = threading.Event()

            def other() -> None:
                with use_project("gamma"):
                    assert workspace.current_project.id == "gamma"
                done.set()

            t = threading.Thread(target=other)
            t.start()
            done.wait(timeout=5)
            t.join(timeout=5)

            assert workspace.current_project.id == "beta"


class TestCarryingTheBindingIntoThreads:
    """An executor hop does not copy the context; the chat path takes several."""

    def test_bind_carries_the_project_into_a_worker(self, workspace):
        result: list = []

        def read() -> None:
            result.append(workspace.current_project.id)

        with use_project("beta"):
            t = threading.Thread(target=bind(read))
            t.start()
            t.join(timeout=5)

        assert result == ["beta"]

    def test_an_unbound_hop_would_have_read_the_selection(self, workspace):
        """Shows what bind() is for — the same hop without it."""
        result: list = []

        def read() -> None:
            result.append(workspace.current_project.id)

        with use_project("beta"):
            t = threading.Thread(target=read)
            t.start()
            t.join(timeout=5)

        assert result == ["alpha"]

    def test_bind_leaves_the_worker_thread_clean(self, workspace):
        after: list = []

        def read() -> None:
            pass

        with use_project("beta"):
            wrapped = bind(read)

        def run() -> None:
            wrapped()
            after.append(active_project_id())

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=5)
        assert after == [None]

    def test_a_pool_carries_each_submitter_s_own_project(self, workspace):
        """One pool, two projects in flight — the orchestrator's fan-out."""
        pool = ProjectBoundExecutor(max_workers=4)
        try:
            with use_project("beta"):
                beta_futures = [pool.submit(lambda: workspace.current_project.id)
                                for _ in range(8)]
            with use_project("gamma"):
                gamma_futures = [pool.submit(lambda: workspace.current_project.id)
                                 for _ in range(8)]
            assert [f.result(timeout=10) for f in beta_futures] == ["beta"] * 8
            assert [f.result(timeout=10) for f in gamma_futures] == ["gamma"] * 8
        finally:
            pool.shutdown(wait=True)

    def test_the_same_binding_can_be_live_in_many_workers_at_once(self, workspace):
        """Copying rather than entering a Context — the fan-out is parallel."""
        pool = ProjectBoundExecutor(max_workers=4)
        started = threading.Barrier(4, timeout=10)
        try:
            def hold() -> str:
                started.wait()  # all four inside the binding simultaneously
                return workspace.current_project.id

            with use_project("beta"):
                futures = [pool.submit(hold) for _ in range(4)]
            assert [f.result(timeout=15) for f in futures] == ["beta"] * 4
        finally:
            pool.shutdown(wait=True)


class TestAddingADocumentBindsRatherThanFlipping:
    """The last place that still moved the selection to do its work.

    ``add_document`` / ``add_document_async`` used to set ``current`` on the
    addressed project and clear it on every other one, purely so that
    ``GenericModel.add_document`` — which resolves through
    ``Workspace.current_project`` — would write into the right place.  In a
    per-project child that write cannot persist (``current`` belongs to the
    workspace record), and under the single-server fallback it *did* persist:
    uploading a document silently moved the project the UI was looking at.
    """

    @pytest.fixture
    def service(self, workspace, monkeypatch):
        from smartloop.api.services.model_service import ModelService

        monkeypatch.setattr(ModelService, "_instance", None, raising=False)
        made = ModelService.__new__(ModelService)
        ModelService.__init__(made)

        seen: dict = {}

        class _SlpModel:
            def add_document(self, source, cwd=None, generate_filename=False):
                # Exactly how the real one resolves its target.
                project = workspace.current_project
                seen["project_id"] = project.id if project else None
                return [Document(id="d1", path=f"{project.project_dir}/d1.md")]

        made._model = SimpleNamespace(
            workspace=workspace,
            slp_model=_SlpModel(),
            save=lambda: seen.setdefault("saved", True),
            add_source=lambda source, progress_callback=None: _SlpModel().add_document(source),
        )
        made._initialize_workspace = lambda: None
        made.seen = seen
        yield made
        ModelService._instance = None

    def test_the_addressed_project_receives_the_document(self, service, workspace):
        """Even though a different project is the one marked ``current``."""
        assert workspace.current_project.id == "alpha"

        service.add_document_async("/tmp/f.md", project_id="beta")

        assert service.seen["project_id"] == "beta"

    def test_the_selection_does_not_move(self, service, workspace):
        service.add_document_async("/tmp/f.md", project_id="beta")

        assert [p.id for p in workspace.projects if p.current] == ["alpha"]

    def test_the_binding_does_not_outlive_the_call(self, service):
        service.add_document_async("/tmp/f.md", project_id="beta")

        assert active_project_id() is None

    def test_it_works_the_same_in_a_pinned_child(self, service, workspace):
        """A child is pinned to gamma; an explicit project_id still wins."""
        pin_project("gamma")

        service.add_document_async("/tmp/f.md", project_id="beta")

        assert service.seen["project_id"] == "beta"
        assert [p.id for p in workspace.projects if p.current] == ["alpha"]

    def test_an_unnamed_project_falls_back_to_the_selection(self, service):
        service.add_document_async("/tmp/f.md")

        assert service.seen["project_id"] == "alpha"

    def test_the_synchronous_path_binds_too(self, service, workspace):
        service.add_document("/tmp/f.md", project_id="beta")

        assert service.seen["project_id"] == "beta"
        assert [p.id for p in workspace.projects if p.current] == ["alpha"]


class TestProjectContext:
    def test_it_resolves_from_whatever_is_bound(self, workspace):
        with use_project("beta"):
            ctx = ProjectContext.resolve(workspace)
        assert ctx.project_id == "beta"
        assert ctx.project.id == "beta"
        assert ctx.project_dir == "/tmp/beta"

    def test_unbound_resolves_to_nothing(self, workspace):
        assert ProjectContext.resolve(workspace) is None

    def test_it_can_re_establish_itself_as_the_binding(self, workspace):
        ctx = ProjectContext(workspace, "gamma")
        with ctx.bound():
            assert workspace.current_project.id == "gamma"

    def test_lookup_is_late_so_a_reload_is_picked_up(self, workspace):
        ctx = ProjectContext(workspace, "beta")
        replacement = _project("beta")
        workspace.projects = [replacement]
        assert ctx.project is replacement
