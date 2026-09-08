"""Background indexing must not embed while the model is answering.

Embedding and generation are the same GPU.  A bge-m3 pass over a few hundred
chunks took a streaming reply from ~57 tok/s to ~3 — the user watching their
answer arrive a word at a time so the workspace index could be a minute
fresher.  ``ModelService._inference_slot`` already counted "is the model
answering someone right now"; this is the reader it was built for.
"""

import os
import tempfile
import threading
from unittest.mock import MagicMock, patch

from smartloop.tasks.indexer_task import IndexerTask


def _task(project_dir):
    """An IndexerTask wired to just enough state to run one scan decision."""
    task = IndexerTask.__new__(IndexerTask)
    task.logger = MagicMock()
    task.workspace = MagicMock()
    task.workspace.workspace_dir = project_dir
    task._workspace_lock = threading.Lock()
    task._workspace_indexer = MagicMock()
    task._workspace_indexer.index.return_value = {
        "indexed": 0, "skipped": 0, "removed": 0,
    }
    task._workspace_indexer_dir = project_dir
    task._last_workspace_scan = 0.0
    task._WORKSPACE_SCAN_INTERVAL = 30.0
    task._WORKSPACE_QUIET_SECONDS = 5.0
    task._last_workspace_scan_at = None
    task._last_workspace_stats = None

    project = MagicMock()
    project.project_dir = project_dir
    task._project = lambda: project
    return task


class TestAPassWaitsForAQuietModel:
    def test_a_pass_is_deferred_while_the_model_generates(self):
        with tempfile.TemporaryDirectory() as td:
            task = _task(td)
            with patch("smartloop.tasks.indexer_task.device_idle_seconds",
                       return_value=0.0):
                task._scan_workspace_files()
            task._workspace_indexer.index.assert_not_called()

    def test_a_pass_is_deferred_inside_the_gaps_between_calls_in_one_turn(self):
        # One turn is many sequential calls — classify, skill match, tool
        # resolution, judge, then the answer.  A bare "is it generating right
        # now" check would slip a pass into a gap and run through the reply.
        with tempfile.TemporaryDirectory() as td:
            task = _task(td)
            with patch("smartloop.tasks.indexer_task.device_idle_seconds",
                       return_value=1.5):
                task._scan_workspace_files()
            task._workspace_indexer.index.assert_not_called()

    def test_a_pass_runs_once_the_model_has_been_quiet(self):
        with tempfile.TemporaryDirectory() as td:
            task = _task(td)
            with patch("smartloop.tasks.indexer_task.device_idle_seconds",
                       return_value=42.0):
                task._scan_workspace_files()
            task._workspace_indexer.index.assert_called_once_with(td)

    def test_a_model_that_never_ran_does_not_block_the_first_pass(self):
        # idle_seconds() is inf before anything has generated; a fresh install
        # has to index without waiting for a turn that may never come.
        with tempfile.TemporaryDirectory() as td:
            task = _task(td)
            with patch("smartloop.tasks.indexer_task.device_idle_seconds",
                       return_value=float("inf")):
                task._scan_workspace_files()
            task._workspace_indexer.index.assert_called_once_with(td)

    def test_a_deferral_retries_within_a_quiet_period_not_a_full_interval(self):
        # The dispatcher stamps the clock when it hands the pass over, so
        # leaving it there would turn every deferral into a 30s wait and a busy
        # stretch would starve the index instead of postponing it.
        with tempfile.TemporaryDirectory() as td:
            task = _task(td)
            with patch("smartloop.tasks.indexer_task.device_idle_seconds",
                       return_value=0.0):
                task._scan_workspace_files()

            import time
            due_in = (
                task._last_workspace_scan
                + task._WORKSPACE_SCAN_INTERVAL
                - time.time()
            )
            assert 0 < due_in <= task._WORKSPACE_QUIET_SECONDS + 1


class TestTheIdleReadingComesFromTheRightProcess:
    def test_a_child_asks_the_host_that_owns_the_embedder(self):
        # A child polling its own idle time would always read "quiet" and
        # answer for the wrong device — both the embedding and the generation
        # happen in the host.
        from smartloop.db import embeddings

        remote = MagicMock()
        remote.idle_seconds.return_value = 0.0
        with patch.object(embeddings, "_remote_embedder", return_value=remote):
            assert embeddings.device_idle_seconds() == 0.0
        remote.idle_seconds.assert_called_once()

    def test_an_unreachable_host_reads_as_idle(self):
        # Unknowable means idle: an indexer that never runs is worse than one
        # that overlaps.
        from smartloop.db import embeddings

        remote = MagicMock()
        remote.idle_seconds.side_effect = OSError("socket is gone")
        with patch.object(embeddings, "_remote_embedder", return_value=remote):
            assert embeddings.device_idle_seconds() == float("inf")

    def test_the_host_itself_reads_its_own_service(self):
        from smartloop.db import embeddings
        from smartloop.api.services.model_service import ModelService

        service = MagicMock()
        service.idle_seconds.return_value = 3.0
        with patch.object(embeddings, "_remote_embedder", return_value=None), \
                patch.object(ModelService, "_instance", service):
            assert embeddings.device_idle_seconds() == 3.0
