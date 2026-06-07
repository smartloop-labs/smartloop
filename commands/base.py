"""Command — base class inherited by all CLI command classes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from requests.exceptions import RequestException

from smartloop.constants import SLP_PRIMARY
from smartloop.server import is_server_running
from smartloop.utils.log_utils import print_logo

from commands.console import console, logger, settings


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
    """Print an in-place block progress bar matching the TUI style."""
    if total <= 0:
        return
    ratio = min(downloaded / total, 1.0)
    pct = ratio * 100
    filled = int(ratio * _BAR_WIDTH)
    empty = _BAR_WIDTH - filled

    bar = f"{_PINK}{'█' * filled}{_DIM}{'░' * empty}{_NC}"
    label = f"{_DIM}Downloading {filename}  {_format_bytes(downloaded)} / {_format_bytes(total)}{_NC}"

    # \033[K clears rest of line to prevent old text bleeding through
    sys.stdout.write(f"{_HIDE_CURSOR}\r\033[K{label}\n\r\033[K{bar} {_PINK}{pct:3.0f}%{_NC}\033[1A\r")
    sys.stdout.flush()


def _clear_progress() -> None:
    """Clear the two-line progress display."""
    sys.stdout.write(f"\r\033[K\n\r\033[K\033[1A\r{_SHOW_CURSOR}")
    sys.stdout.flush()


class Command:
    """Base class for all CLI commands."""

    args: object
    host: str
    port: int
    developer_token: str

    # Seconds to wait for the auto-started agent to answer /health. The server
    # blocks on a model preload during startup, so this must comfortably exceed
    # a cold model load. Override with SLP_SERVER_START_TIMEOUT.
    _SERVER_START_TIMEOUT = int(os.environ.get("SLP_SERVER_START_TIMEOUT", "180"))

    def execute(self) -> None:
        """Execute the primary command action."""
        raise NotImplementedError

    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _require_server(self) -> bool:
        """Ensure the API agent is reachable; auto-start if needed."""
        if self.port and is_server_running(self.host, self.port):
            return True

        from version import __version__
        print_logo(version=__version__, console=console)

        try:
            home_dir = Path(settings.home_dir)
            log_file = home_dir / "server.log"
            os.makedirs(home_dir, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as lf:
                if getattr(sys, "frozen", False):
                    cmd = [sys.executable, "agent", "start"]
                else:
                    # Launch via -m, NOT the script path: running
                    # ".../site-packages/commands/cli.py" puts commands/ on
                    # sys.path[0], where commands/token.py shadows the stdlib
                    # `token` module and breaks inspect/numpy imports.
                    cmd = [sys.executable, "-m", "commands.cli", "agent", "start"]
                # Propagate the resolved host/port so the auto-started agent binds
                # the same address the CLI is polling (e.g. when 'run -p' overrides
                # the default port).
                cmd += ["--host", str(self.host), "--port", str(self.port)]
                # Force UTF-8 for the child's stdout/stderr. Otherwise on Windows
                # the agent's streams default to the locale codec (cp1252) and
                # printing non-Latin-1 characters (e.g. '→') during bootstrap
                # raises UnicodeEncodeError, which surfaces as "Bootstrap failed".
                env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
                # start_new_session detaches the agent into its own session so it
                # outlives the CLI and is shared by multiple CLI instances (the
                # model stays warm). Stop it explicitly with 'slp agent stop'.
                proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=lf, start_new_session=True, env=env
                )

            # The server only answers /health once its lifespan finishes the
            # (blocking) model preload, which can take well over 30s on a cold
            # GPU load. Poll up to SERVER_START_TIMEOUT, but bail out immediately
            # if the spawned process dies so a crash surfaces fast instead of
            # waiting out the whole window.
            with console.status("Please wait while agent is heating up...", spinner="dots") as status:
                for i in range(self._SERVER_START_TIMEOUT):
                    time.sleep(1)
                    if is_server_running(self.host, self.port):
                        return True
                    if proc.poll() is not None:
                        logger.error(
                            f"Agent process exited (code {proc.returncode}) before "
                            f"becoming ready."
                        )
                        self._print_server_log_tail(log_file)
                        return False
                    if i % 5 == 4:
                        status.update("Still waiting for agent...")

            logger.error(
                f"Agent failed to start within {self._SERVER_START_TIMEOUT}s. "
                f"Check {log_file} for details."
            )
            self._print_server_log_tail(log_file)
            return False
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            return False

    @staticmethod
    def _print_server_log_tail(log_file: Path, lines: int = 15) -> None:
        """Echo the tail of the server log so the real failure cause is visible."""
        try:
            tail = log_file.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return
        if tail:
            console.print("[dim]--- server.log (tail) ---[/dim]")
            for line in tail:
                console.print(f"[dim]{line}[/dim]")

    def _is_ready(self) -> bool:
        """Return True when the server has a loaded model and at least one project."""
        try:
            health = requests.get(f"{self._base_url()}/health", timeout=5).json()
            if not health.get("model_loaded"):
                return False
        except RequestException:
            return False
        try:
            data = requests.get(f"{self._base_url()}/v1/projects", timeout=10).json()
            if not data.get("projects"):
                return False
        except RequestException:
            return False
        return True

    def _resolve_project_id(self) -> str | None:
        """Return the current project ID, resolving from the server if needed."""
        pid = getattr(self, "project_id", None)
        if pid:
            return pid
        try:
            data = requests.get(f"{self._base_url()}/v1/projects", timeout=10).json()
            for p in data.get("projects", []):
                if p.get("current"):
                    return p["id"]
        except RequestException:
            pass
        return None

    def _ensure_ready(self) -> bool:
        """Guarantee the server has a model + project (fast no-op if already ready)."""
        if self._is_ready():
            return True
        if getattr(self.args, "model", None):
            self._init()
        else:
            self._bootstrap()
        return self._is_ready()

    def _init(self) -> None:
        """Authenticated init — download a specific model via /init."""
        payload = {"cwd": self.project_dir}
        if m := getattr(self.args, "model", None):
            payload["model_name"] = m
        if self.developer_token:
            payload["developer_token"] = self.developer_token
        try:
            with requests.post(
                f"{self._base_url()}/v1/init",
                json=payload,
                stream=True,
                timeout=600,
            ) as resp:
                self._consume_sse_stream(resp)
        except RequestException as e:
            console.print(f"[red]API Error: {e}[/red]")

    def _bootstrap(self) -> None:
        """Unauthenticated bootstrap — download model + create default project."""
        try:
            with requests.post(
                f"{self._base_url()}/v1/bootstrap",
                json={"cwd": self.project_dir},
                stream=True,
                timeout=1800,
            ) as resp:
                self._consume_sse_stream(resp)
        except RequestException as e:
            console.print(f"[red]API Error: {e}[/red]")

    def _consume_sse_stream(self, resp: requests.Response) -> None:
        """Read an SSE response and render download progress / status messages."""
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
