"""
MCP Task — background provisioning and content refresh for MCP servers.

Registering an MCP server used to do all of its work on the request: launch the
process (or connect), enumerate tools, generate description/tags/summary with
the LLM, then embed the tool vectors.  That is seconds to tens of seconds of
subprocess, network and inference latency in front of a UI click, and an LLM
call on the registration path competes with whatever the user is chatting to.

So registration now persists the connection and returns, and this task does the
rest out of band:

* **Provisioning** — a queued server is discovered, enriched, tool-indexed and
  crawled.  Its ``status`` moves ``pending`` → ``ready``/``failed`` with
  ``last_error`` set, which is what clients poll instead of blocking.
* **Refresh cycle** — every ``crawl_interval`` seconds, document-type servers
  (Notion, Linear, Trello, …) are re-crawled so their vectorized content keeps
  up with the source.  The first cycle runs at startup, so a session begins
  with fresh content rather than whatever was indexed days ago.

The work itself belongs to :class:`~smartloop.api.services.model_service.ModelService`
(it owns the workspace, the LLM and the indexers); this task owns only the
threading and the schedule, and is handed the two callables at setup.
"""

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Tuple

from smartloop.tasks.base_task import BaseTask, TaskResult, TaskStatus

# How long the loop blocks on the provisioning queue before checking whether a
# refresh cycle is due.  Also the latency of an explicitly requested crawl.
_QUEUE_POLL_SECONDS = 1.0


class MCPTask(BaseTask):
    """Background MCP server provisioning and periodic content crawl."""

    # Default refresh cadence; overridden from AppSettings at construction.
    DEFAULT_CRAWL_INTERVAL = 1800.0

    def __init__(
        self,
        logger,
        provisioner: Optional[Callable[[str, str], dict]] = None,
        crawler: Optional[Callable[..., dict]] = None,
        crawl_interval: Optional[float] = None,
        pending_provider: Optional[Callable[[], list]] = None,
        **kwargs,
    ):
        super().__init__(name="mcp", logger=logger, auto=True, **kwargs)
        self._provisioner = provisioner
        self._crawler = crawler
        self._pending_provider = pending_provider
        self._crawl_interval = float(
            crawl_interval if crawl_interval is not None else self.DEFAULT_CRAWL_INTERVAL
        )
        if self._crawl_interval <= 0:
            self.logger.info(
                "[MCP] Periodic content refresh disabled "
                "(slp_mcp_crawl_interval_s=0); crawls run on request only"
            )

        self._loop_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._inflight_lock = threading.Lock()
        self._inflight: set = set()

        # A crawl is requested up front, so the first loop tick refreshes the
        # content indexed in a previous session — that is the "start cycle".
        # After that the interval decides, unless the interval is 0 (periodic
        # refresh disabled: crawls then only happen on request).
        self._crawl_requested: bool = True
        self._last_crawl_cycle: float = 0.0
        self._crawl_inflight: bool = False
        self._last_cycle_at: Optional[float] = None
        self._last_cycle_stats: Optional[dict] = None
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

    def request_crawl(self) -> None:
        """Ask the loop to run a content crawl on its next tick (~1s)."""
        self._crawl_requested = True
        self.start_loop()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def mcp_status(self) -> dict:
        """Snapshot of MCP background work, surfaced via /health."""
        with self._inflight_lock:
            provisioning = len(self._inflight)
        return {
            "queued": self.queue_size,
            "provisioning": provisioning,
            "crawling": self._crawl_inflight,
            "last_crawl_at": self._last_cycle_at,
            "last_crawl_stats": self._last_cycle_stats,
            "crawl_interval_seconds": self._crawl_interval,
        }

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

                now = time.monotonic()
                if (
                    not self._crawl_inflight
                    and self._executor is not None
                    and self._crawler is not None
                    and self._crawl_due(now)
                ):
                    self._crawl_requested = False
                    self._last_crawl_cycle = now
                    self._crawl_inflight = True
                    self._executor.submit(self._run_crawl_cycle)

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
                if idle and not self._crawl_inflight:
                    self._set_status(TaskStatus.PENDING, "Waiting for MCP work")
            except Exception as e:
                self.logger.error(f"[MCP] Event loop error: {e}")

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self._set_status(TaskStatus.COMPLETED, "Event loop stopped")

    def _crawl_due(self, now: float) -> bool:
        """True when a content crawl should start on this tick.

        An explicit request always runs.  Otherwise the interval decides, and
        an interval of 0 means the periodic refresh is switched off entirely —
        never that it is due on every tick.
        """
        if self._crawl_requested:
            return True
        if self._crawl_interval <= 0:
            return False
        return (now - self._last_crawl_cycle) >= self._crawl_interval

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

    def _run_crawl_cycle(self) -> None:
        try:
            stats = self._crawler() or {}
            self._last_cycle_stats = stats
            self._last_cycle_at = time.time()
            if stats.get("servers"):
                self.logger.info(f"[MCP] Content crawl cycle: {stats}")
        except Exception as e:
            self.logger.warning(f"[MCP] Content crawl cycle failed: {e}")
        finally:
            self._crawl_inflight = False

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def should_run(self) -> bool:
        # Auto-started at setup: the loop itself is the work (it owns both the
        # provisioning queue and the refresh cycle).
        return True

    def execute(self, **kwargs) -> TaskResult:
        """Batch entry point for ``TaskQueue.submit()``: start the loop."""
        self.start_loop()
        return TaskResult(success=True, data={"queued": self.queue_size})

    def stop(self) -> None:
        super().stop()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
