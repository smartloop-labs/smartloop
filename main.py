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

def cmd_agent(args, host: str, port: int) -> None:
    sub = args.agent_command
    if sub == "start":
        if is_server_running(host, port):
            console.print(f"[{SLP_PRIMARY}]Agent already running at http://{host}:{port}[/{SLP_PRIMARY}]")
        else:
            home_dir = Path(settings.home_dir)
            home_dir.mkdir(parents=True, exist_ok=True)
            start_server(host, port, service=not getattr(args, "no_service", False))
    elif sub == "stop":
        stop_server()
    elif sub == "status":
        status = get_status(host, port)
        if status["running"]:
            console.print(f"[{SLP_PRIMARY}]Agent running at http://{host}:{port}[/{SLP_PRIMARY}]")
            console.print(f"  PID: {status.get('pid') or 'unknown'}")
            console.print(f"  Model loaded: {status.get('model_loaded', False)}")
            console.print(f"  Model name: {status.get('model_name', 'none')}")
        else:
            console.print("[dim]Agent not running[/dim]")
    elif sub == "restart":
        stop_server()
        time.sleep(1)
        home_dir = Path(settings.home_dir)
        home_dir.mkdir(parents=True, exist_ok=True)
        start_server(host, port, service=not getattr(args, "no_service", False))
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
    agent_subparsers.add_parser("stop", help="Stop the background agent")
    agent_subparsers.add_parser("status", help="Show agent status")
    restart_agent_parser = agent_subparsers.add_parser("restart", help="Restart the agent")
    restart_agent_parser.add_argument("--no-service", action="store_true", help="Disable auto-restart on crash")
    restart_agent_parser.add_argument("--host", help="API server host (default: $SLP_API_HOST or 127.0.0.1)")
    restart_agent_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")

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
