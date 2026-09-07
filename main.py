#!/usr/bin/env python3
"""
Smartloop - AI Agent harness for extracting information and creating new content.

The CLI is intentionally minimal: it manages the agent lifecycle
(start/stop/status/restart) only. All other functionality is exposed through
the internal API server that the agent runs.
"""

import argparse
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import certifi
from rich.console import Console

from version import __version__
from smartloop.config import AppSettings
from smartloop.constants import SLP_PRIMARY
from smartloop.server import (
    get_status,
    is_server_running,
    start_server,
    stop_server,
)

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("smartloop").setLevel(logging.WARNING)

settings = AppSettings()
console = Console()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _format_bytes(value) -> str:
    """Human-readable RSS, or a dash when the process could not be read."""
    if not value:
        return "-"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit != "GB" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _print_status(status: dict, host: str, port: int) -> None:
    """Report one process, or the whole tree when running supervised."""
    console.print(f"[{SLP_PRIMARY}]Agent running at http://{host}:{port}[/{SLP_PRIMARY}]")
    console.print(f"  PID: {status.get('pid') or 'unknown'}")
    console.print(f"  Model loaded: {status.get('model_loaded', False)}")
    console.print(f"  Model name: {status.get('model_name', 'none')}")

    if not status.get("supervised"):
        return

    model_host = status.get("model_host") or {}
    console.print()
    console.print(f"[{SLP_PRIMARY}]Model host[/{SLP_PRIMARY}]")
    if model_host.get("address"):
        console.print(f"  {model_host['address']} (pid {model_host.get('pid', '?')})")
    else:
        console.print("  [dim]not running — agents load their own models[/dim]")

    agents = status.get("agents") or []
    console.print()
    console.print(f"[{SLP_PRIMARY}]Project agents[/{SLP_PRIMARY}]")
    if not agents:
        console.print("  [dim]none running — started on first request[/dim]")
        return
    console.print(f"  {'project':<28}{'pid':>8}{'port':>7}{'rss':>10}{'idle':>8}")
    for agent in agents:
        name = agent.get("name") or agent.get("project_id") or "?"
        idle = agent.get("idle_seconds")
        console.print(
            f"  {name[:27]:<28}{agent.get('pid', '?'):>8}{agent.get('port', '?'):>7}"
            f"{_format_bytes(agent.get('rss_bytes')):>10}"
            f"{(f'{idle:.0f}s' if idle is not None else '-'):>8}"
        )


def cmd_agent(args, host: str, port: int) -> None:
    sub = args.agent_command
    # Supervised is the default; --no-supervised falls back to one shared
    # process for the whole workspace.
    supervised = not getattr(args, "no_supervised", False)
    if sub == "start":
        if is_server_running(host, port):
            console.print(f"[{SLP_PRIMARY}]Agent already running at http://{host}:{port}[/{SLP_PRIMARY}]")
        else:
            home_dir = Path(settings.home_dir)
            home_dir.mkdir(parents=True, exist_ok=True)
            start_server(host, port, service=not getattr(args, "no_service", False),
                         supervised=supervised)
    elif sub == "stop":
        stop_server()
    elif sub == "status":
        status = get_status(host, port)
        if status["running"]:
            _print_status(status, host, port)
        else:
            console.print("[dim]Agent not running[/dim]")
    elif sub == "restart":
        stop_server()
        time.sleep(1)
        home_dir = Path(settings.home_dir)
        home_dir.mkdir(parents=True, exist_ok=True)
        start_server(host, port, service=not getattr(args, "no_service", False),
                     supervised=supervised)
    else:
        console.print("[yellow]Usage: slp agent {start|stop|status|restart}[/yellow]")


def main():
    multiprocessing.freeze_support()

    # In a frozen (PyInstaller) build, sys.executable is the `slp` binary
    # rather than a Python interpreter. Internal stdio MCP servers are spawned
    # as `<sys.executable> -m <module>`, so the binary must honour `-m <module>`
    # the way `python -m` would.
    if len(sys.argv) >= 3 and sys.argv[1] == "-m":
        import runpy
        module = sys.argv[2]
        sys.argv = [sys.argv[0], *sys.argv[3:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    parser = argparse.ArgumentParser(
        description="AI Agent harness for extracting information and creating new content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Agent management commands
    agent_parser = subparsers.add_parser("agent", help="Agent management commands (start/stop/status/restart)")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_command", help="Agent commands")
    start_agent_parser = agent_subparsers.add_parser("start", help="Start the agent in background")
    start_agent_parser.add_argument("--no-service", action="store_true", help="Disable auto-restart on crash")
    start_agent_parser.add_argument("--host", help="API server host (default: $SLP_API_HOST or 127.0.0.1)")
    start_agent_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")
    start_agent_parser.add_argument("--no-supervised", action="store_true", help="Serve every project from one shared process (default: one process per project)")
    agent_subparsers.add_parser("stop", help="Stop the background agent")
    agent_subparsers.add_parser("status", help="Show agent status")
    restart_agent_parser = agent_subparsers.add_parser("restart", help="Restart the agent")
    restart_agent_parser.add_argument("--no-service", action="store_true", help="Disable auto-restart on crash")
    restart_agent_parser.add_argument("--host", help="API server host (default: $SLP_API_HOST or 127.0.0.1)")
    restart_agent_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")
    restart_agent_parser.add_argument("--no-supervised", action="store_true", help="Serve every project from one shared process (default: one process per project)")

    args = parser.parse_args()

    # Resolve configuration
    host = getattr(args, "host", None) or os.environ.get("SLP_API_HOST", "127.0.0.1")
    port = getattr(args, "port", None) or settings.api_port

    # Dispatch
    if args.command == "agent":
        cmd_agent(args, host, port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
