"""Indexing a project's documents belongs to that project's own agent.

``ModelService.setup_tasks`` registers an ``IndexerTask``.  Given no project of
its own it falls back to the ``current`` flag (``indexer_task.py:59``), and
``_scan_documents`` reads the documents folder off disk rather than draining its
own queue — so in a supervisor that has delegated every project to a child it
picks up whatever a child just wrote and indexes it a second time.

Observed live before this guard: the same three PDFs indexed by both the
supervisor and the project's own agent, with the duplicated embedding pass
leaving too little memory for the large model to load — which reached the user
as an answer that had quietly fallen back to the default model.

Nothing is lost by skipping it there: a delegating ``save()`` keeps only the
workspace record, so the document state the task exists to persist is discarded
anyway while its manifest and vector-store writes land.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService
from smartloop.workspace.context import delegate_projects, pin_project


@pytest.fixture(autouse=True)
def _clear_process_scope():
    """The pin and the delegation flag are process-global; never leak one."""
    yield
    pin_project(None)
    delegate_projects(False)


@pytest.fixture
def tasks():
    """Stand in for every task class ``setup_tasks`` constructs.

    ``BuildTask``/``TrainTask`` are optional imports the module already treats as
    possibly-None, so they are pinned to None rather than mocked.
    """
    with patch("smartloop.api.services.model_service.IndexerTask") as indexer_cls, \
         patch("smartloop.api.services.model_service.MCPTask") as mcp_cls, \
         patch("smartloop.api.services.model_service.BuildTask", None), \
         patch("smartloop.api.services.model_service.TrainTask", None):
        yield SimpleNamespace(indexer=indexer_cls, mcp=mcp_cls)


def _svc():
    svc = MagicMock()
    svc._task_queue = MagicMock()
    return svc


def _workspace():
    # No projects: the ChromaDB migration loop at the top of setup_tasks has
    # nothing to walk, which keeps this about task registration.
    return SimpleNamespace(projects=[])


def _registered(svc):
    return [call.args[0] for call in svc._task_queue.register.call_args_list]


class TestOnlyAProjectsOwnProcessIndexesIt:
    def test_a_delegating_supervisor_registers_no_indexer(self, tasks):
        svc = _svc()
        delegate_projects(True)

        ModelService.setup_tasks(svc, _workspace())

        tasks.indexer.assert_not_called()
        assert tasks.indexer.return_value not in _registered(svc)

    def test_a_delegating_supervisor_starts_no_watcher(self, tasks):
        """The watcher is the half that would actually scan the folder."""
        svc = _svc()
        delegate_projects(True)

        ModelService.setup_tasks(svc, _workspace())

        tasks.indexer.return_value.start_watcher.assert_not_called()

    def test_a_pinned_child_still_indexes_its_own_project(self, tasks):
        """The child is the tier that owns the project and may persist it."""
        svc = _svc()
        pin_project("alpha")

        ModelService.setup_tasks(svc, _workspace())

        tasks.indexer.assert_called_once()
        assert tasks.indexer.return_value in _registered(svc)
        tasks.indexer.return_value.start_watcher.assert_called_once()

    def test_the_single_server_fallback_still_indexes(self, tasks):
        """``SLP_SUPERVISOR=0``: unpinned and undelegated, as it always was."""
        svc = _svc()

        ModelService.setup_tasks(svc, _workspace())

        tasks.indexer.assert_called_once()
        assert tasks.indexer.return_value in _registered(svc)
        tasks.indexer.return_value.start_watcher.assert_called_once()


class TestTheGuardIsNarrow:
    def test_a_delegating_supervisor_still_runs_the_mcp_task(self, tasks):
        """Only the indexer is project work it cannot do.

        MCPTask stays registered and is made inert by ``pending_mcp_servers``,
        which already returns nothing under delegation — so removing it here
        would change behaviour this commit is not about.
        """
        svc = _svc()
        delegate_projects(True)

        ModelService.setup_tasks(svc, _workspace())

        assert tasks.mcp.return_value in _registered(svc)
        tasks.mcp.return_value.start_loop.assert_called_once()

    def test_auto_tasks_still_start_under_delegation(self, tasks):
        svc = _svc()
        delegate_projects(True)

        ModelService.setup_tasks(svc, _workspace())

        svc._task_queue.start_auto_tasks.assert_called_once()
