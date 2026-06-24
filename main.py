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
import shlex
import tempfile
import webbrowser
           
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

# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_run(args, host: str, port: int, model_name: str, project_dir: str) -> None:
    if not _require_server(host, port):
        return

    from tui.chat import SLPChat

    resume_id = getattr(args, "resume", None)
    app = SLPChat(
        server_url=_base_url(host, port),
        model_name=model_name or "",
        session_id=resume_id or "",
        project_dir=project_dir,
    )
    app.run()
    console.print("\n[bold blue]Bye![/bold blue]")
    console.print(f"Resume conversation with:\n[dim]slp --resume={app.session_id}[/dim]")


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


def _resolve_current_project(host: str, port: int) -> dict | None:
    """Return the agent's currently active project, or None."""
    try:
        data = requests.get(f"{_base_url(host, port)}/v1/projects", timeout=10).json()
    except RequestException:
        return None
    projects = data.get("projects", [])
    return next((p for p in projects if p.get("current")), None)


def _frontmatter_name(content: str) -> str | None:
    """Return the `name:` value from a SKILL.md YAML frontmatter, if present."""
    if not content.startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    for line in parts[1].strip().splitlines():
        if line.startswith("name:"):
            return line[len("name:"):].strip() or None
    return None


def _read_skill_content(args) -> str:
    """Resolve skill content for `skill add` from (in priority order):
    inline text arg, --file, piped stdin, or by opening $EDITOR.

    The editor/stdin paths make pasting large skills painless, which the
    inline TUI form cannot handle well.
    """
    if getattr(args, "file", None):
        return Path(args.file).read_text(encoding="utf-8")
    if getattr(args, "text", None):
        return args.text
    if not sys.stdin.isatty():
        return sys.stdin.read()

    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
    if not editor:
        editor = "notepad" if os.name == "nt" else "vi"

    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False, encoding="utf-8") as tf:
        tf.write("# Write your skill below, then save and exit.\n")
        tmp_path = tf.name
    try:
        console.print(f"[dim]Opening {editor} to enter skill content...[/dim]")
        subprocess.call([*shlex.split(editor), tmp_path])
        content = Path(tmp_path).read_text(encoding="utf-8")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Drop the helper comment line if the user left it untouched.
    lines = content.splitlines()
    if lines and lines[0].startswith("# Write your skill below"):
        lines = lines[1:]
    return "\n".join(lines)


def cmd_skill(args, host: str, port: int) -> None:
    """Manage project skills from the CLI (companion to the desktop app).

    Skills target the agent's currently active project, the same one the
    desktop app and TUI operate on.
    """
    if not _require_server(host, port):
        return

    base = _base_url(host, port)
    project = _resolve_current_project(host, port)
    if not project:
        console.print(
            "[yellow]No active project.[/yellow] "
            "Create or switch to a project in the app first."
        )
        return
    project_id = project.get("id")
    project_name = project.get("name", project_id)

    sub = getattr(args, "skill_command", None)

    if sub == "list":
        try:
            resp = requests.get(f"{base}/v1/projects/{project_id}/skills", timeout=10)
            resp.raise_for_status()
            skills = resp.json().get("skills", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if not skills:
            console.print(f"[dim]No skills set for project '{project_name}'.[/dim]")
            return
        table = PrettyTable()
        table.field_names = ["#", "Name", "Description"]
        table.align = "l"
        for i, s in enumerate(skills, 1):
            desc = s.get("description") or s.get("content", "")
            table.add_row([i, s.get("name", ""), desc])
        console.print(table)

    elif sub == "add":
        content = _read_skill_content(args)
        if not content.strip():
            console.print("[yellow]No skill content provided.[/yellow]")
            return
        try:
            existing = requests.get(
                f"{base}/v1/projects/{project_id}/skills", timeout=10
            ).json().get("skills", [])
        except RequestException:
            existing = []
        # Respect a `name:` in the SKILL.md frontmatter; only fall back to an
        # auto-generated name when neither --name nor frontmatter provides one.
        name = getattr(args, "name", None)
        if not name and not _frontmatter_name(content):
            name = f"skill_{len(existing) + 1}"
        payload = {"content": content}
        if name:
            payload["name"] = name
        try:
            resp = requests.post(
                f"{base}/v1/projects/{project_id}/skills",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
        except RequestException as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")  # type: ignore[union-attr]
            except Exception:
                pass
            console.print(f"[red]Failed to add skill{': ' + detail if detail else ''}[/red]")
            return
        added_name = resp.json().get("name", name) or name
        console.print(f"[green]Skill '{added_name}' added to project '{project_name}'.[/green]")

    elif sub == "remove":
        try:
            index = int(args.index)
        except (TypeError, ValueError):
            console.print("[yellow]Usage: slp skill remove <#>  (see 'slp skill list')[/yellow]")
            return
        try:
            skills = requests.get(
                f"{base}/v1/projects/{project_id}/skills", timeout=10
            ).json().get("skills", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if index < 1 or index > len(skills):
            console.print(f"[yellow]Invalid index {index}. Project has {len(skills)} skill(s).[/yellow]")
            return
        skill = skills[index - 1]
        try:
            resp = requests.delete(
                f"{base}/v1/projects/{project_id}/skills/{skill.get('id')}", timeout=30
            )
            resp.raise_for_status()
        except RequestException as e:
            console.print(f"[red]Failed to remove skill: {e}[/red]")
            return
        console.print(f"[green]Removed skill '{skill.get('name', index)}'.[/green]")

    else:
        console.print("[yellow]Usage: slp skill <add|list|remove>[/yellow]")


def cmd_mcp(args, host: str, port: int) -> None:
    """Manage MCP servers from the CLI (companion to the desktop app).

    Servers are registered against the agent's currently active project. A
    full http(s) URL is treated as a remote server; anything else is a local
    (stdio) command such as `uvx <pkg>` or `npx <pkg>`.
    """
    if not _require_server(host, port):
        return

    from urllib.parse import urlparse

    base = _base_url(host, port)
    project = _resolve_current_project(host, port)
    if not project:
        console.print(
            "[yellow]No active project.[/yellow] "
            "Create or switch to a project in the app first."
        )
        return
    project_id = project.get("id")
    project_name = project.get("name", project_id)

    sub = getattr(args, "mcp_command", None)

    if sub == "list":
        try:
            resp = requests.get(f"{base}/v1/projects/{project_id}/mcp", timeout=10)
            resp.raise_for_status()
            servers = resp.json().get("servers", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if not servers:
            console.print(f"[dim]No MCP servers registered for project '{project_name}'.[/dim]")
            return
        table = PrettyTable()
        table.field_names = ["#", "Name", "Type", "Tools", "Enabled"]
        table.align = "l"
        table.align["Tools"] = "r"
        for i, s in enumerate(servers, 1):
            tools = s.get("tools", [])
            table.add_row([
                i,
                s.get("name", ""),
                s.get("server_type", "local"),
                len(tools),
                "yes" if s.get("enabled") else "no",
            ])
        console.print(table)

    elif sub == "add":
        tokens = args.target or []
        force_local = getattr(args, "local", False)
        force_remote = getattr(args, "remote", False)
        if not tokens:
            console.print("[yellow]Usage: slp mcp add <url> | <command> [args...][/yellow]")
            return

        first = tokens[0]
        parsed = urlparse(first)
        is_url = parsed.scheme in ("http", "https") and bool(parsed.netloc)

        if force_remote or (is_url and not force_local):
            if not is_url:
                console.print("[red]--remote requires a full http(s) URL.[/red]")
                return
            payload: dict = {"server_type": "remote", "server_url": first}
        else:
            payload = {"server_type": "local", "command": first}
            if tokens[1:]:
                payload["args"] = tokens[1:]

        try:
            resp = requests.post(
                f"{base}/v1/projects/{project_id}/mcp/register",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except RequestException as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")  # type: ignore[union-attr]
            except Exception:
                pass
            console.print(f"[red]Failed to register MCP server{': ' + detail if detail else ''}[/red]")
            return

        if data.get("auth_type") == "oauth":
            auth_url = data.get("auth_url")
            if not auth_url:
                console.print("[red]OAuth required but no authorization URL was returned.[/red]")
                return
            webbrowser.open(auth_url)
            console.print(
                "[cyan]Opening browser for authorization.[/cyan] "
                "Complete authorization to finish registering the server.\n"
                f"[dim]If the browser did not open, visit:[/dim] {auth_url}"
            )
            return

        server = data.get("server") or {}
        tool_count = len(server.get("tools", []))
        console.print(
            f"[green]MCP server registered: {server.get('name', first)} "
            f"({tool_count} tools) on project '{project_name}'.[/green]"
        )

    elif sub == "remove":
        try:
            index = int(args.index)
        except (TypeError, ValueError):
            console.print("[yellow]Usage: slp mcp remove <#>  (see 'slp mcp list')[/yellow]")
            return
        try:
            servers = requests.get(
                f"{base}/v1/projects/{project_id}/mcp", timeout=10
            ).json().get("servers", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if index < 1 or index > len(servers):
            console.print(f"[yellow]Invalid index {index}. Project has {len(servers)} server(s).[/yellow]")
            return
        server = servers[index - 1]
        try:
            resp = requests.delete(
                f"{base}/v1/projects/{project_id}/mcp/{server.get('id')}", timeout=30
            )
            resp.raise_for_status()
        except RequestException as e:
            console.print(f"[red]Failed to remove MCP server: {e}[/red]")
            return
        console.print(f"[green]Removed MCP server '{server.get('name', index)}'.[/green]")

    else:
        console.print("[yellow]Usage: slp mcp <add|list|remove>[/yellow]")


def cmd_project(args, host: str, port: int) -> None:
    """Inspect and switch the agent's current project from the CLI.

    The current project is what `slp skill` and `slp mcp` operate on, so this
    lets a CLI/desktop user see and change that target without the TUI.
    """
    if not _require_server(host, port):
        return

    base = _base_url(host, port)
    sub = getattr(args, "project_command", None)

    if sub == "current":
        project = _resolve_current_project(host, port)
        if not project:
            console.print("[dim]No active project.[/dim]")
            return
        console.print(
            f"[green]Current project:[/green] {project.get('name')} "
            f"(id={project.get('id')})"
        )
        return

    if sub in (None, "list"):
        try:
            resp = requests.get(f"{base}/v1/projects", timeout=10)
            resp.raise_for_status()
            projects = resp.json().get("projects", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if not projects:
            console.print("[dim]No projects found.[/dim]")
            return
        table = PrettyTable()
        table.field_names = ["#", "ID", "Name", "Description", "Model", "Skills", "MCP", "Current"]
        table.align = "l"
        for col in ("Skills", "MCP"):
            table.align[col] = "r"
        for i, p in enumerate(projects, 1):
            table.add_row([
                i,
                p.get("id", ""),
                p.get("name", ""),
                p.get("description") or "-",
                p.get("model_name") or "default",
                len(p.get("skills", [])),
                len(p.get("mcp", [])),
                "yes" if p.get("current") else "",
            ])
        console.print(table)
        return

    if sub == "switch":
        try:
            index = int(args.index)
        except (TypeError, ValueError):
            console.print("[yellow]Usage: slp project switch <#>  (see 'slp project list')[/yellow]")
            return
        try:
            projects = requests.get(f"{base}/v1/projects", timeout=10).json().get("projects", [])
        except RequestException as e:
            console.print(f"[red]Request failed: {e}[/red]")
            return
        if index < 1 or index > len(projects):
            console.print(f"[yellow]Invalid index {index}. There are {len(projects)} project(s).[/yellow]")
            return
        project = projects[index - 1]
        with console.status(f"Switching to '{project.get('name')}'...", spinner="dots"):
            try:
                resp = requests.post(
                    f"{base}/v1/models/load",
                    json={"mode": "inference", "project_id": project["id"]},
                    timeout=300,
                )
                resp.raise_for_status()
            except RequestException as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")  # type: ignore[union-attr]
                except Exception:
                    pass
                console.print(f"[red]Failed to switch project{': ' + detail if detail else ''}[/red]")
                return
        console.print(f"[green]Switched to project:[/green] {project.get('name')} (id={project['id']})")
        return

    console.print("[yellow]Usage: slp project <list|current|switch>[/yellow]")


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
    run_parser.add_argument("--model", "-m", help="Model name (default: $SLP_BASE_MODEL_NAME or slp-orchestra-mini)")
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

    # Skill management commands (companion to the desktop app)
    skill_parser = subparsers.add_parser("skill", help="Manage skills for the active project")
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command", help="Skill commands")
    skill_add_parser = skill_subparsers.add_parser(
        "add", help="Add a skill (large content via --file, piped stdin, or $EDITOR)"
    )
    skill_add_parser.add_argument("text", nargs="?", default=None, help="Skill text (omit to use --file, stdin, or an editor)")
    skill_add_parser.add_argument("--file", "-f", help="Read skill content from a file")
    skill_add_parser.add_argument("--name", help="Skill name (auto-generated if omitted)")
    skill_subparsers.add_parser("list", help="List skills for the active project")
    skill_remove_parser = skill_subparsers.add_parser("remove", help="Remove a skill by index")
    skill_remove_parser.add_argument("index", help="Skill index from 'slp skill list'")

    # MCP management commands (companion to the desktop app)
    mcp_parser = subparsers.add_parser("mcp", help="Manage MCP servers for the active project")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")
    mcp_add_parser = mcp_subparsers.add_parser(
        "add", help="Register an MCP server: a URL (remote) or a command (local stdio)"
    )
    mcp_add_parser.add_argument("target", nargs="+", help="<url> for remote, or <command> [args...] for local")
    mcp_add_parser.add_argument("--local", action="store_true", help="Force local (stdio) registration")
    mcp_add_parser.add_argument("--remote", action="store_true", help="Force remote (URL) registration")
    mcp_subparsers.add_parser("list", help="List MCP servers for the active project")
    mcp_remove_parser = mcp_subparsers.add_parser("remove", help="Remove an MCP server by index")
    mcp_remove_parser.add_argument("index", help="Server index from 'slp mcp list'")

    # Project commands (companion to the desktop app)
    project_parser = subparsers.add_parser("project", help="List, current, or switch the current project")
    project_subparsers = project_parser.add_subparsers(dest="project_command", help="Project commands")
    project_subparsers.add_parser("list", help="List all projects with IDs and details")
    project_subparsers.add_parser("current", help="Show the current project and its ID")
    project_switch_parser = project_subparsers.add_parser("switch", help="Switch the current project by index")
    project_switch_parser.add_argument("index", help="Project index from 'slp project list'")

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
    elif args.command == "skill":
        cmd_skill(args, host, port)
    elif args.command == "mcp":
        cmd_mcp(args, host, port)
    elif args.command == "project":
        cmd_project(args, host, port)
    else:
        cmd_run(args, host, port, model_name, project_dir)


if __name__ == "__main__":
    main()
