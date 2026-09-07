"""Server lifecycle management for Smartloop.

Provides start_server, stop_server, and get_status as importable functions:

    from smartloop import start_server, stop_server, get_status
"""

import logging
import os
import signal
import time

from pathlib import Path

import httpx

from smartloop.config import AppSettings
from smartloop.api import start_server_foreground

logger = logging.getLogger(__name__)

_settings = AppSettings()

PID_FILE = Path(os.path.join(_settings.home_dir, "server.pid"))


def get_server_url(host: str, port: int) -> str:
    """Get the base URL for the API server."""
    return f"http://{host}:{port}"


def is_server_running(host: str, port: int) -> bool:
    """Check if the API server is already running."""
    try:
        response = httpx.get(f"{get_server_url(host, port)}/health", timeout=2)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


def read_pid_file() -> int | None:
    """Read PID from the pid file if it exists and the process is alive."""
    try:
        if PID_FILE.exists():
            pid = int(PID_FILE.read_text().strip())
            try:
                os.kill(pid, 0)
                return pid
            except OSError:
                PID_FILE.unlink(missing_ok=True)
    except (ValueError, IOError):
        pass
    return None


def write_pid_file(pid: int):
    """Write PID to the pid file."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid))
    

def stop_server() -> bool:
    """Stop the background API server if running.

    Returns True if the server was stopped, False otherwise.
    """
    pid = read_pid_file()
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)

            still_running = False
            for _ in range(20):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.5)
                    still_running = True
                except OSError:
                    still_running = False
                    break

            if still_running:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

            PID_FILE.unlink(missing_ok=True)
            logger.info("Server stopped (PID %d)", pid)
            return True
        except OSError as e:
            logger.error("Could not stop server: %s", e)
            PID_FILE.unlink(missing_ok=True)
    else:
        logger.info("No server running")
    return False


def get_status(host: str, port: int) -> dict:
    """Return server status as a dictionary.

    Returns a dict with at least {"running": bool}. When the server is running,
    additional keys are included: host, port, pid, model_loaded, model_name,
    quantization, n_ctx, n_gpu_layers, flash_attn, model_size_bytes,
    memory_percent, and active_project (when available).
    """
  
    if not port:
        return {"running": False}

    if not is_server_running(host, port):
        return {"running": False}

    pid = read_pid_file()
    result: dict = {"running": True, "host": host, "port": port, "pid": pid}

    try:
        health = httpx.get(f"{get_server_url(host, port)}/health", timeout=5).json()
        result["model_loaded"] = health.get("model_loaded", False)
        result["model_name"] = health.get("model_name")
        for key in ("quantization", "n_ctx", "n_gpu_layers", "flash_attn",
                     "model_size_bytes", "memory_percent"):
            if health.get(key) is not None:
                result[key] = health[key]
    except httpx.HTTPError:
        pass

    try:
        data = httpx.get(f"{get_server_url(host, port)}/v1/projects", timeout=10).json()
        current = next((p for p in data.get("projects", []) if p.get("current")), None)
        if current:
            result["active_project"] = {
                "id": current.get("id"),
                "name": current.get("name"),
                "model_name": current.get("model_name"),
            }
    except httpx.HTTPError:
        pass

    # Under a supervisor there is a process per project rather than one server,
    # so status is a tree: the supervisor, the model host, then a row per
    # project.  Absent under the --no-supervised fallback, where /health
    # already says everything there is to say.
    try:
        agents = httpx.get(f"{get_server_url(host, port)}/agents", timeout=5).json()
        if agents.get("supervised"):
            names = _project_names(host, port)
            for agent in agents.get("agents", []):
                agent["name"] = names.get(agent.get("project_id"))
            result["supervised"] = True
            result["supervisor"] = agents.get("supervisor", {})
            result["model_host"] = agents.get("model_host", {})
            result["agents"] = agents.get("agents", [])
    except (httpx.HTTPError, ValueError):
        pass

    return result


def _project_names(host: str, port: int) -> dict:
    """``{project_id: name}``, so the status table reads in project names."""
    try:
        data = httpx.get(f"{get_server_url(host, port)}/v1/projects", timeout=10).json()
    except (httpx.HTTPError, ValueError):
        return {}
    return {p.get("id"): p.get("name") for p in data.get("projects", [])}


def start_server(host: str, port: int, debug: bool = False, service: bool = True,
                 supervised: bool = True) -> None:
    """Start the API server in the foreground (blocking).

    Args:
        host: Host to bind to (default: API_HOST env or 127.0.0.1).
        port: Port to bind to (default: SLP_API_PORT env or 38540).
        debug: Enable debug mode (loads base model with LoRA adapters).
        service: Auto-restart on crash with exponential backoff.
        supervised: Run a process per project behind this port, with the model
            factored out into a host they share.  The default.  The client
            contract is unchanged either way; what changes is the blast radius
            of one project's runaway job.  Pass False for the single shared
            server — one process, and one project's crash takes the rest down
            with it.
    """
    os.environ["SLP_SUPERVISOR"] = "1" if supervised else "0"

    start_server_foreground(host, port, service=service)
