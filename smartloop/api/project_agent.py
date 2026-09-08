"""One agent process, serving exactly one project.

``python -m smartloop.api.project_agent --project <id>``

The project is fixed here, at spawn, and never varies for the life of the
process — which is the whole point.  A shared server had to decide, per
request, which project it was looking at, and did so by flipping a flag that
any other request could move.  A child has nothing to flip: it pins its project
(:func:`~smartloop.workspace.context.pin_project`) before the app is built, and
every ``current_project`` read in the process resolves to that one.

It runs the same FastAPI app as the single-process server — same routers, same
contract — on an ephemeral loopback port that the supervisor proxies to.  What
it owns is everything project-shaped: its own task queue, vector index, MCP
stdio children, sessions and orchestration.  What it does not own is the
device: generation and embedding go to the model host over
``SLP_MODEL_HOST`` when one is configured.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)


def _bind_ephemeral_port(host: str = "127.0.0.1") -> tuple[socket.socket, int]:
    """Reserve a loopback port and hand back the socket still holding it.

    Asking the OS for a free port and then closing the socket before uvicorn
    binds leaves a window in which something else can take it.  Passing the
    bound socket straight to uvicorn closes that window entirely.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    return sock, sock.getsockname()[1]


def _resolve_project_dir(project_id: str) -> Optional[str]:
    """Where this project's state lives, without loading the whole workspace."""
    from smartloop.workspace import Workspace

    workspace = Workspace.load()
    if workspace is None:
        return None
    project = next((p for p in workspace.projects if p.id == project_id), None)
    return project.project_dir if project else None


#: This project's own log, beside the project it describes.
AGENT_LOG_FILE = "agent.log"


def _install_project_logging(project_dir: str, log_level: str):
    """Send this agent's logs to its own project directory.

    One shared ``server.log`` was right when one process served every project.
    With a process per project it would interleave every project's
    orchestration, DAG, planner and MCP trace into a single file, and working
    out which lines belong to the project you are actually debugging becomes
    the hard part. A file per project makes that question disappear.

    A console handler is added on top when the agent was spawned from a
    terminal — a child inherits the supervisor's stdout, so ``make run-debug``
    still shows the trace live instead of only writing it to a file nobody is
    tailing.
    """
    from smartloop.utils.logging_setup import (
        make_log_queue, setup_root_logging, start_listener,
    )

    level = (log_level or "INFO").upper()
    try:
        queue = make_log_queue()
        setup_root_logging(queue, level="DEBUG")
        # A frozen build re-execs through main.py, which clamps "smartloop" to
        # WARNING to keep the interactive shell quiet.  Lift it when DEBUG was
        # actually asked for, exactly as the single-process server does.
        if level == "DEBUG":
            logging.getLogger("smartloop").setLevel(logging.DEBUG)
        return start_listener(
            queue,
            home_dir=project_dir,
            file_level=level,
            console_level=level,
            log_path=os.path.join(project_dir, AGENT_LOG_FILE),
        )
    except OSError as exc:
        # A log file we cannot open is not a reason to refuse to serve the
        # project; the child keeps whatever logging it already had.
        logger.warning("[project-agent] could not open the project log: %s", exc)
        return None


def _default_log_level() -> str:
    """The level this agent should log at, from ``LOG_LEVEL`` like everything else.

    Without this a child would sit at INFO while the supervisor ran at DEBUG —
    and since orchestration happens in the child, ``LOG_LEVEL=DEBUG`` would
    turn on verbose logging in the one process that has nothing verbose to say.
    """
    from smartloop.config import AppSettings

    return (AppSettings().log_level or "INFO").lower()


def run(project_id: str, host: str = "127.0.0.1", port: int = 0,
        log_level: Optional[str] = None) -> None:
    """Serve *project_id* until the process is stopped."""
    import uvicorn

    log_level = (log_level or _default_log_level()).lower()

    from smartloop.supervisor.registry import AgentRecord, _write_project_files
    from smartloop.workspace.context import pin_project

    # Before anything imports the app: every current_project read in this
    # process — routers, orchestrator, background tasks — resolves through
    # this from here on.
    pin_project(project_id)
    os.environ["SLP_PROJECT_ID"] = project_id

    from smartloop.api.main import app

    sock: Optional[socket.socket] = None
    if port == 0:
        sock, port = _bind_ephemeral_port(host)

    project_dir = _resolve_project_dir(project_id)
    if project_dir:
        _write_project_files(
            project_dir, AgentRecord(project_id, os.getpid(), port),
        )
        _install_project_logging(project_dir, log_level)

    logger.info(
        "[project-agent] project=%s pid=%d port=%d", project_id, os.getpid(), port,
    )

    config = uvicorn.Config(app, host=host, port=port, log_level=log_level,
                            log_config=None)
    server = uvicorn.Server(config)
    watch = _watch_the_supervisor(server)
    try:
        if sock is not None:
            server.run(sockets=[sock])
        else:
            server.run()
    finally:
        if watch is not None:
            watch.stop()
        if project_dir:
            from smartloop.supervisor.registry import _clear_project_files

            _clear_project_files(project_dir)


#: How long an orphaned agent is given to shut down cleanly before it is taken
#: down the blunt way.  An orphan is often orphaned *because* something in it
#: is wedged — a hung MCP call, a request blocked on a model host that died
#: with the supervisor — and the lifespan shutdown that stops a healthy process
#: can then block forever, which would leave exactly the process this is here
#: to remove.
ORPHAN_SHUTDOWN_GRACE_SECONDS = 20.0


def _watch_the_supervisor(server):
    """Exit if the supervisor that spawned this agent goes away.

    A child is spawned detached, so nothing in the OS links the two: before
    this, a supervisor that was killed rather than stopped left its children
    serving projects that no longer had a router in front of them, holding
    their task queues, MCP stdio children and model-host connections open
    until the machine rebooted.

    Asking uvicorn to exit rather than exiting here: the lifespan shutdown is
    what closes the SQLite handles, drains the task queue and terminates this
    project's MCP grandchildren, and skipping it would trade one orphan for
    several.  The hard exit is only the backstop for when that does not
    return.
    """
    from smartloop.config import AppSettings
    from smartloop.supervisor.parent_watch import start_parent_watch

    def shut_down() -> None:
        server.should_exit = True
        # Daemon, so the backstop never becomes the delay: when the graceful
        # shutdown does return, the process exits at once and takes the
        # pending timer with it instead of waiting out the grace period and
        # then hard-exiting a process that had already finished.
        backstop = threading.Timer(ORPHAN_SHUTDOWN_GRACE_SECONDS, _hard_exit)
        backstop.daemon = True
        backstop.start()

    try:
        home_dir = AppSettings().home_dir
    except Exception:
        home_dir = None
    return start_parent_watch(shut_down, home_dir=home_dir)


def _hard_exit() -> None:
    """Last resort: leave without running interpreter shutdown.

    ``os._exit`` because the graceful path has already been given its chance
    and did not return — a ``sys.exit`` from this timer thread would only
    raise in the timer thread and leave the process exactly where it was.
    """
    logger.error(
        "[project-agent] orphan shutdown did not finish in %.0fs; exiting hard",
        ORPHAN_SHUTDOWN_GRACE_SECONDS,
    )
    os._exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="smartloop.api.project_agent",
        description="Serve a single project",
    )
    parser.add_argument("--project", help="project id (default: $SLP_PROJECT_ID)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0,
                        help="0 (default) asks the OS for a free port")
    parser.add_argument("--log-level", default=None,
                        help="default: $LOG_LEVEL, else info")
    args = parser.parse_args()

    project_id = args.project or os.environ.get("SLP_PROJECT_ID")
    if not project_id:
        parser.error("no project given (pass --project or set SLP_PROJECT_ID)")

    log_level = args.log_level or _default_log_level()
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(project_id, args.host, args.port, log_level)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
