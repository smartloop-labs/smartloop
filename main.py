#!/usr/bin/env python3
"""
Smartloop - AI Agent harness for extracting information and creating new content.
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import certifi
import requests
from prettytable import PrettyTable
from prompt_toolkit import prompt
from requests.exceptions import RequestException
from rich.console import Console

from version import __version__
from smartloop.auth import credential_store
from smartloop.config import AppSettings
from smartloop.constants import SLP_PRIMARY
from smartloop.server import (
    get_status,
    is_server_running,
    read_pid_file,
    start_server,
    stop_server,
)
from smartloop.utils import ConsoleLogger
from smartloop.utils.log_utils import print_logo

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("smartloop").setLevel(logging.WARNING)

settings = AppSettings()
console = Console()
logger = ConsoleLogger(console)

_SERVER_START_TIMEOUT = int(os.environ.get("SLP_SERVER_START_TIMEOUT", "180"))

# ANSI escape helpers for the block-style progress bar
_PINK = "\033[38;5;205m"
_DIM = "\033[0;2m"
_NC = "\033[0m"
_BAR_WIDTH = 40
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def _format_bytes(n: int) -> str:
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    return f"{n / 1024:.0f} KB"


def _render_progress(filename: str, downloaded: int, total: int) -> None:
    if total <= 0:
        return
    ratio = min(downloaded / total, 1.0)
    pct = ratio * 100
    filled = int(ratio * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled
    bar = f"{_PINK}{'█' * filled}{_DIM}{'░' * empty}{_NC}"
    label = f"{_DIM}Downloading {filename}  {_format_bytes(downloaded)} / {_format_bytes(total)}{_NC}"
    sys.stdout.write(f"{_HIDE_CURSOR}\r\033[K{label}\n\r\033[K{bar} {_PINK}{pct:3.0f}%{_NC}\033[1A\r")
    sys.stdout.flush()


def _clear_progress() -> None:
    sys.stdout.write(f"\r\033[K\n\r\033[K\033[1A\r{_SHOW_CURSOR}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _configure_server_logging(log_file: Path) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _print_server_log_tail(log_file: Path, lines: int = 15) -> None:
    try:
        tail = log_file.read_text(errors="replace").splitlines()[-lines:]
    except OSError:
        return
    if tail:
        console.print("[dim]--- server.log (tail) ---[/dim]")
        for line in tail:
            console.print(f"[dim]{line}[/dim]")


def _require_server(host: str, port: int) -> bool:
    if port and is_server_running(host, port):
        return True

    print_logo(version=__version__, console=console)

    try:
        home_dir = Path(settings.home_dir)
        log_file = home_dir / "server.log"
        os.makedirs(home_dir, exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as lf:
            if getattr(sys, "frozen", False):
                cmd = [sys.executable, "agent", "start"]
            else:
                cmd = [sys.executable, "main.py", "agent", "start"]
            cmd += ["--host", str(host), "--port", str(port)]
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, start_new_session=True, env=env)

        with console.status("Please wait while agent is heating up...", spinner="dots") as status:
            for i in range(_SERVER_START_TIMEOUT):
                time.sleep(1)
                if is_server_running(host, port):
                    return True
                if proc.poll() is not None:
                    logger.error(f"Agent process exited (code {proc.returncode}) before becoming ready.")
                    _print_server_log_tail(log_file)
                    return False
                if i % 5 == 4:
                    status.update("Still waiting for agent...")

        logger.error(f"Agent failed to start within {_SERVER_START_TIMEOUT}s. Check {log_file} for details.")
        _print_server_log_tail(log_file)
        return False
    except Exception as e:
        logger.error(f"Failed to start agent: {e}")
        return False


def _consume_sse_stream(resp: requests.Response) -> None:
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        console.print(f"[red]{detail}[/red]")
        return

    showing_progress = False
    current_event_type: str | None = None
    current_filename: str | None = None

    for raw in resp.iter_lines():
        if not raw:
            current_event_type = None
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw

        if line.startswith("event:"):
            current_event_type = line[6:].strip()
            continue
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue

        event_type = current_event_type or ""
        status = data.get("status", "")
        msg = data.get("message", "")

        if "downloaded" in data and "total" in data:
            total = data["total"]
            downloaded = data["downloaded"]
            filename = data.get("filename", "model")
            if total:
                if showing_progress and filename != current_filename:
                    _clear_progress()
                    console.print(f"[cyan][+] Downloaded {current_filename}[/cyan]")
                showing_progress = True
                current_filename = filename
                _render_progress(filename, downloaded, total)
        elif event_type == "complete" or status == "completed":
            if showing_progress:
                _clear_progress()
                console.print(f"[cyan][+] Downloaded {current_filename}[/cyan]")
                showing_progress = False
                current_filename = None
            console.print(f"[cyan][+] {msg}[/cyan]")
        elif status == "project_created":
            project = data.get("project", {})
            console.print(
                f"[{SLP_PRIMARY}][+] Project created: "
                f"{project.get('name', '')} (id={project.get('id', '')})[/{SLP_PRIMARY}]"
            )
        elif event_type == "error" or status == "error":
            if showing_progress:
                _clear_progress()
                showing_progress = False
                current_filename = None
            console.print(f"[red]{msg}[/red]")
        else:
            if msg and not msg.lower().startswith("downloading"):
                if showing_progress:
                    _clear_progress()
                    showing_progress = False
                    current_filename = None
                console.print(f"[dim]{msg}[/dim]")

        current_event_type = None

    if showing_progress:
        _clear_progress()
        if current_filename:
            console.print(f"[cyan][+] Downloaded {current_filename}[/cyan]")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args, host: str, port: int, model_name: str, project_dir: str) -> None:
    if not _require_server(host, port):
        return

    from tui.chat import SLPChat

    try:
        resume_id = getattr(args, "resume", None)
        app = SLPChat(
            server_url=_base_url(host, port),
            model_name=model_name or "",
            session_id=resume_id or "",
            project_dir=project_dir,
        )
        app.run()
    finally:
        try:
            requests.post(f"{_base_url(host, port)}/v1/models/unload", timeout=30)
            console.print("\n[bold blue]Bye![/bold blue]")
            console.print(f"Resume conversation with:\n[dim]slp --resume={app.session_id}[/dim]")
        except Exception:
            pass


def cmd_status(host: str, port: int) -> None:
    if not is_server_running(host, port):
        console.print("[dim]Server not running[/dim]")
        return

    table = PrettyTable()
    table.field_names = ["Property", "Value"]
    table.align["Property"] = "l"
    table.align["Value"] = "l"

    try:
        health = requests.get(f"{_base_url(host, port)}/health", timeout=5).json()
        table.add_row(["Server", f"http://{host}:{port}"])
        pid = read_pid_file()
        if pid:
            table.add_row(["PID", pid])
        table.add_row(["Model loaded", health.get("model_loaded", False)])
        if health.get("model_name"):
            table.add_row(["Model", health["model_name"]])
        if health.get("quantization"):
            table.add_row(["Quantization", health["quantization"]])
        if health.get("n_ctx"):
            table.add_row(["Context window", health["n_ctx"]])
        if health.get("n_gpu_layers") is not None:
            table.add_row(["GPU layers", health["n_gpu_layers"]])
        if health.get("flash_attn") is not None:
            table.add_row(["Flash attention", health["flash_attn"]])
        model_bytes = health.get("model_size_bytes", 0)
        if model_bytes:
            size_gb = model_bytes / (1024 ** 3)
            table.add_row(["Model size", f"{size_gb:.1f} GB" if size_gb >= 1 else f"{model_bytes / (1024 ** 2):.0f} MB"])
        if health.get("memory_percent") is not None:
            table.add_row(["Memory usage", f"{health['memory_percent']}%"])
        try:
            import torch
            if torch.cuda.is_available():
                table.add_row(["GPU", torch.cuda.get_device_name(0)])
                table.add_row(["GPU memory", f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.1f} GB"])
            elif torch.backends.mps.is_available():
                table.add_row(["GPU", "Apple Silicon (MPS)"])
            else:
                table.add_row(["GPU", "None (CPU only)"])
        except Exception:
            pass
    except RequestException:
        pass

    try:
        data = requests.get(f"{_base_url(host, port)}/v1/projects", timeout=10).json()
        current = next((p for p in data.get("projects", []) if p.get("current")), None)
        if current:
            table.add_row(["Active project", f"{current.get('name')} (id={current.get('id')})"])
            table.add_row(["Project model", current.get("model_name") or "default"])
    except RequestException:
        pass

    console.print(table)


def cmd_agent(args, host: str, port: int) -> None:
    sub = args.agent_command
    if sub == "start":
        if is_server_running(host, port):
            console.print(f"[{SLP_PRIMARY}]Agent already running at http://{host}:{port}[/{SLP_PRIMARY}]")
        else:
            home_dir = Path(settings.home_dir)
            home_dir.mkdir(parents=True, exist_ok=True)
            _configure_server_logging(home_dir / "server.log")
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
        _configure_server_logging(home_dir / "server.log")
        start_server(host, port, service=not getattr(args, "no_service", False))


def cmd_token(args) -> None:
    sub = args.token_command
    if sub == "set":
        t = getattr(args, "token_value", None)
        if not t:
            console.print(
                "[cyan]Paste your developer token below.[/cyan]\n"
                "[dim]Get one at[/dim] [bold]https://app.smartloop.ai/developer[/bold]"
            )
            t = prompt("Token: ")
        if t.strip():
            credential_store.set_token(t.strip())
            masked = t[:8] + "..." + t[-3:] if len(t) > 14 else "***"
            console.print(f"[green]Token saved:[/green] {masked}")
        else:
            console.print("[yellow]No token provided.[/yellow]")
    elif sub == "clear":
        credential_store.clear_token()
        credential_store.clear_access_token()
        credential_store.clear_refresh_token()
        console.print("[green]All tokens removed.[/green]")
    else:
        t = credential_store.get_token()
        if t:
            masked = t[:8] + "..." + t[-3:] if len(t) > 14 else "***"
            console.print(f"[green]Token:[/green] {masked}")
        else:
            console.print(
                "[dim]No token set. Use [bold]slp token set[/bold] to add one.[/dim]\n"
                "[dim]Get your token at[/dim] [bold]https://app.smartloop.ai/developer[/bold]"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    parser.add_argument("--resume", metavar="CONVERSATION_ID", default=None,
                        help="Resume a previous conversation by session ID")
    parser.add_argument("-C", "--path", default=None, metavar="PATH",
                        help="Working directory to use (defaults to ~/smartloop or ~/smartloop/<project>)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start interactive chat session")
    run_parser.add_argument("--model", "-m", help="Model name (default: $SLP_BASE_MODEL_NAME or gemma3-1b)")
    run_parser.add_argument("--project-name", help="Project name (default: $SLP_PROJECT_NAME)")
    run_parser.add_argument("--host", help="API server host (default: $API_HOST or 127.0.0.1)")
    run_parser.add_argument("--port", "-p", type=int, help="API server port (default: $SLP_API_PORT or 38540)")
    run_parser.add_argument("--resume", metavar="CONVERSATION_ID", default=None,
                           help="Resume a previous conversation by session ID")

    # Status command
    subparsers.add_parser("status", help="Show project status")

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

    # Token management commands
    token_parser = subparsers.add_parser("token", help="Manage your developer token")
    token_subparsers = token_parser.add_subparsers(dest="token_command", help="Token commands")
    token_set_parser = token_subparsers.add_parser("set", help="Set your developer token")
    token_set_parser.add_argument("token_value", nargs="?", default=None, help="Developer token (prompted if omitted)")
    token_subparsers.add_parser("clear", help="Remove stored token")

    args = parser.parse_args()

    # Persist -t token to keychain so it's available to all consumers
    cli_token = getattr(args, "developer_token", None)
    if cli_token:
        credential_store.set_token(cli_token)

    developer_token = cli_token or os.environ.get("SLP_DEVELOPER_TOKEN")

    # Resolve configuration
    model_name = getattr(args, "model", None) or os.environ.get("SLP_BASE_MODEL_NAME")
    project_name = getattr(args, "project_name", None) or os.environ.get("SLP_PROJECT_NAME")
    host = getattr(args, "host", None) or os.environ.get("SLP_API_HOST", "127.0.0.1")
    port = getattr(args, "port", None) or settings.api_port

    # Project working directory
    if args.path:
        target = os.path.abspath(args.path)
        if not os.path.isdir(target):
            parser.error(f"path '{args.path}' is not a valid directory")
        project_dir = target
    else:
        base = os.path.expanduser("~/smartloop")
        if project_name:
            slug = project_name.lower().replace(" ", "_")
            project_dir = os.path.join(base, slug)
        else:
            project_dir = base

    # Dispatch
    if args.command == "agent":
        cmd_agent(args, host, port)
    elif args.command == "status":
        cmd_status(host, port)
    elif args.command == "token":
        cmd_token(args)
    else:
        cmd_run(args, host, port, model_name, project_dir)


if __name__ == "__main__":
    main()
