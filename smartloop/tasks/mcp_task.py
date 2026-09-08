"""
MCP Task — background provisioning for MCP servers.

Registering an MCP server used to do all of its work on the request: launch the
process (or connect), enumerate tools, then generate description/tags/summary
with the LLM.  That is seconds to tens of seconds of subprocess, network and
inference latency in front of a UI click, and an LLM call on the registration
path competes with whatever the user is chatting to.

So registration persists the connection and returns, and this task does the
rest out of band: a queued server is discovered and enriched, and its
``status`` moves ``pending`` → ``ready``/``failed`` with ``last_error`` set,
which is what clients poll instead of blocking.  A server left ``pending`` by
an interrupted run (app quit, crash) is adopted on the first loop tick.

That is the whole job now.  This task also used to run a periodic content
crawl, vectorizing each document-type server's pages and issues on a schedule;
that is gone — a connection's content is read by calling its tools when a
question needs them, so nothing indexes at startup.

The work itself belongs to :class:`~smartloop.api.services.model_service.ModelService`
(it owns the workspace, the LLM and the connections); this task owns only the
threading, and is handed the callables at setup.
"""

import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

from smartloop.tasks.base_task import BaseTask, TaskResult, TaskStatus

# How long the loop blocks on the provisioning queue before looking around.
_QUEUE_POLL_SECONDS = 1.0


class MCPTask(BaseTask):
    """Background MCP server provisioning."""

    def __init__(
        self,
        logger,
        provisioner: Optional[Callable[[str, str], dict]] = None,
        pending_provider: Optional[Callable[[], list]] = None,
        **kwargs,
    ):
        super().__init__(name="mcp", logger=logger, auto=True, **kwargs)
        self._provisioner = provisioner
        self._pending_provider = pending_provider

        self._loop_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._inflight_lock = threading.Lock()
        self._inflight: set = set()
        self._pending_seeded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, project_id: str, server_id: str) -> None:
        """Queue a server for background provisioning."""
        self._queue.put((project_id, server_id))
        self.logger.info(
            f"[MCP] Server {server_id} queued for provisioning "
            f"(project {project_id})"
        )
        self.start_loop()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def mcp_status(self) -> dict:
        """Snapshot of MCP background work, surfaced via /health."""
        with self._inflight_lock:
            provisioning = len(self._inflight)
        return {"queued": self.queue_size, "provisioning": provisioning}

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def start_loop(self) -> None:
        if self._loop_thread and self._loop_thread.is_alive():
            return
        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=max(2, min(4, (os.cpu_count() or 2))),
            thread_name_prefix="mcp-worker",
        )
        self._loop_thread = threading.Thread(
            target=self._event_loop, name="mcp-loop", daemon=True,
        )
        self._loop_thread.start()
        self.logger.info("[MCP] Event loop started")

    # Alias matching IndexerTask, so setup_tasks reads the same for both.
    def start_watcher(self) -> None:
        self.start_loop()

    def _event_loop(self) -> None:
        self._set_status(TaskStatus.PENDING, "Waiting for MCP work")

        while not self._stop_event.is_set():
            try:
                self._seed_pending_once()

                project_id, server_id = self._queue.get(timeout=_QUEUE_POLL_SECONDS)
                with self._inflight_lock:
                    if server_id in self._inflight:
                        continue
                    self._inflight.add(server_id)
                self._set_status(TaskStatus.RUNNING, "Provisioning MCP server")
                self._executor.submit(self._run_provision, project_id, server_id)
            except queue.Empty:
                with self._inflight_lock:
                    idle = not self._inflight
                if idle:
                    self._set_status(TaskStatus.PENDING, "Waiting for MCP work")
            except Exception as e:
                self.logger.error(f"[MCP] Event loop error: {e}")

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._set_status(TaskStatus.COMPLETED, "Event loop stopped")

    def _seed_pending_once(self) -> None:
        """Queue servers left ``pending`` by a previous run, once per loop start.

        A registration that was interrupted (app quit, crash) leaves a server
        persisted but never discovered.  Nothing else would ever pick it up, so
        the first loop iteration adopts them.
        """
        if self._pending_seeded or self._pending_provider is None:
            return
        self._pending_seeded = True
        try:
            pending = self._pending_provider() or []
        except Exception as e:
            self.logger.warning(f"[MCP] Could not list pending servers: {e}")
            return
        for project_id, server_id in pending:
            self._queue.put((project_id, server_id))
        if pending:
            self.logger.info(
                f"[MCP] Resuming provisioning for {len(pending)} pending server(s)"
            )

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _run_provision(self, project_id: str, server_id: str) -> None:
        try:
            if self._provisioner is None:
                return
            self._provisioner(project_id, server_id)
        except Exception as e:
            self.logger.error(f"[MCP] Provisioning {server_id} failed: {e}")
        finally:
            with self._inflight_lock:
                self._inflight.discard(server_id)

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        # Auto-started at setup: the loop itself is the work (it owns the
        # provisioning queue).
        return True

    def execute(self, **kwargs) -> TaskResult:
        """Batch entry point for ``TaskQueue.submit()``: start the loop."""
        self.start_loop()
        return TaskResult(success=True, data={"queued": self.queue_size})

    def stop(self) -> None:
        super().stop()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
