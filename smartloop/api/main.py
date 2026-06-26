"""FastAPI application with lifespan management"""

import asyncio
import logging
import os
import sys
import json
import uvicorn
import psutil

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import re as _re
import time as _time

from smartloop.api.config import api_settings
from smartloop.api.routers import auth_router, chat_router, documents_router, projects_router, assets_router, mcp_servers_router, models_router, init_router, skills_router, users_router, feedback_router, project_io_router
from smartloop.api.services import get_model_service, get_api_client
        
from smartloop.config import AppSettings
from smartloop.workspace import Workspace
from smartloop.utils.log_utils import print_logo
from smartloop.platform.token_agent import run_token_refresh_loop
from smartloop import __version__

_app_settings = AppSettings()

_app_settings.version = __version__

# Port file for persisting actual bound port (shared in home directory)
PORT_FILE = os.path.join(_app_settings.home_dir, "server.port")

# Root logger is configured by smartloop.utils.logging_setup.setup_root_logging
# inside run_server / start_server_foreground.  Until that runs, fall back to a
# permissive root level so import-time logs aren't dropped.
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# Redact tokens from uvicorn access logs
_TOKEN_RE = _re.compile(r"(access_token|refresh_token)=[^&\s]+")

class _TokenRedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.args = tuple(
                _TOKEN_RE.sub(r"\1=<redacted>", str(a)) if isinstance(a, str) else a
                for a in record.args
            )
        return True

logging.getLogger("uvicorn.access").addFilter(_TokenRedactFilter())


def _is_llama_cpp_instance(llm) -> bool:
    """Check whether llm is a LlamaCpp instance without eager importing native deps."""
    try:
        from smartloop.llms.llama_cpp import LlamaCpp
        return isinstance(llm, LlamaCpp)
    except Exception:
        return False

def _get_model_file_path() -> str | None:
    """Return the expected GGUF model file path from the config, or None."""
    if not os.path.exists(Workspace.config_file):
        return None
    try:
        workspace = Workspace.load()  # ensure config is loaded and cached

        active_model = [m for m in workspace.models if m.is_active]
        
        if not active_model:
            return None
        
        active_model = next(iter(active_model))

        return os.path.join(workspace.models_dir, active_model.model_id, f"{active_model.model_id}.gguf")
    except Exception:
        return None


async def _wait_for_config_and_load(model_service):
    """Background task: poll for config and model file, then load the model."""
    logger.info("Waiting for config and model file to be ready...")
    while True:
        if not os.path.exists(Workspace.config_file):
            await asyncio.sleep(2)
            continue
        model_file = _get_model_file_path()
        if model_file and os.path.exists(model_file):
            break
        await asyncio.sleep(2)
    logger.info("Config and model file detected! Loading model...")
    model_service.load_model()
    logger.info("Model loaded successfully.")


def _bootstrap_orchestrator_model(model_service) -> None:
    """Download the orchestrator base model via the unauthenticated bootstrap
    source and register it in the workspace.

    Used as a fallback when the orchestrator GGUF is missing on disk (e.g. a
    workspace created before the model was bootstrapped). Mirrors the core of
    the ``POST /v1/bootstrap`` endpoint without the SSE streaming.
    """
    from smartloop.model.downloader.artifact_downloader import SLPArtifactDownloader

    workspace = model_service._ensure_workspace()
    downloader = SLPArtifactDownloader(workspace, logger)
    bootstrap_models = downloader._bootstrap_source.list_models()
    if not bootstrap_models:
        raise RuntimeError("No bootstrap models available from the bootstrap source")

    target = bootstrap_models[0]
    name = target.get("name")
    model_id = SLPArtifactDownloader._resolve_model_id(name)

    logger.info(f"Bootstrapping orchestrator model '{name}'...")
    downloader.bootstrap_download(name)

    # Register (or reuse) the workspace entry and mark it downloaded so the
    # subsequent load_model() and config watchers find it. add_model is
    # alias-aware, so an existing legacy-named entry is reused, not duplicated.
    workspace.add_model(name, model_id)
    ws_model = workspace.get_model(name)
    if ws_model is not None:
        ws_model.models_dir = workspace.models_dir
        ws_model.downloaded = True
    workspace.save()
    logger.info(f"Orchestrator model '{name}' bootstrapped successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Smartloop API server...")
    logger.info(f"Debug mode setting: {api_settings.debug_mode}")

    workspace = Workspace.load()

    if api_settings.debug_mode:
        logger.info("🔧 Debug mode enabled - will load base model with LoRA adapters")

    token_refresh_task = asyncio.create_task(run_token_refresh_loop())

    config_watcher_task = None
    model_service = get_model_service()

    if workspace:
        logger.info("Pre-loading model for current project...")
        try:
            model_service.load_model()
            logger.info("Model pre-loaded successfully")
        except FileNotFoundError as e:
            # The orchestrator base model GGUF isn't on disk yet. Pull it via
            # the unauthenticated bootstrap source and retry, rather than
            # leaving the server running with no model loaded.
            logger.warning(f"Orchestrator model missing ({e}); attempting bootstrap download...")
            try:
                _bootstrap_orchestrator_model(model_service)
                model_service.load_model()
                logger.info("Model pre-loaded successfully after bootstrap")
            except Exception as e2:
                logger.error(f"Failed to bootstrap and pre-load model: {e2}")
        except Exception as e:
            logger.error(f"Failed to pre-load model: {e}")
    else:
        config_watcher_task = asyncio.create_task(
            _wait_for_config_and_load(model_service)
        )

    # Set up tasks (indexer, build, train) — the downloader task
    # is registered lazily when /v1/init is called.
    try:
        if workspace:
            model_service.setup_tasks(workspace)
            logger.info("Task queue initialized")
    except Exception as e:
        logger.warning(f"Could not initialize agents: {e}")

    yield

    # Shutdown
    token_refresh_task.cancel()
    if config_watcher_task and not config_watcher_task.done():
        config_watcher_task.cancel()

    logger.info("Shutting down Smartloop API server...")

    model_service.shutdown_tasks()
    model_service.unload_model()

    logger.info("Model unloaded, shutdown complete")

    # Stop the centralized log listener last so the lines above still flush
    # through the rotating file handler.
    listener = getattr(app.state, "log_listener", None)
    if listener is not None:
        try:
            listener.stop()
        except Exception:
            pass


app = FastAPI(
    title="Smartloop",
    description="Smartloop local LLM inference framework - OpenAI-compatible API server",
    version=_app_settings.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
origins = api_settings.cors_origins.split(",") if api_settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(projects_router)
app.include_router(assets_router)
app.include_router(mcp_servers_router)
app.include_router(models_router)
app.include_router(init_router)
app.include_router(skills_router)
app.include_router(users_router)
app.include_router(feedback_router)
app.include_router(project_io_router)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    get_api_client().flush_token_usage()

    model_service = get_model_service()
    result = {
        "status": "healthy",
        "model_loaded": model_service.is_loaded(),
        "model_name": model_service._model_name,
    }

    llm = model_service.get_llm()
    if llm is not None:
        try:
            if _is_llama_cpp_instance(llm):
                result["n_ctx"] = llm.get_max_tokens()
                # Model file size
                model_path = llm.model.model_path
                if model_path:
                    model_size = os.path.getsize(model_path)
                    result["model_size_bytes"] = model_size
                # n_gpu_layers and flash_attn from the Llama instance
                n_gpu = getattr(llm.model, "n_gpu_layers", None)
                if n_gpu is not None:
                    result["n_gpu_layers"] = n_gpu
                result["flash_attn"] = getattr(llm.model, "flash_attn", False)
                # Quantization type from GGUF metadata
                # GGUF file_type is a numeric code — map to human-readable labels
                _GGUF_FTYPE_MAP = {
                    "0": "F32", "1": "F16", "2": "Q4_0", "3": "Q4_1",
                    "7": "Q8_0", "8": "Q5_0", "9": "Q5_1", "10": "Q2_K",
                    "11": "Q3_K_S", "12": "Q3_K_M", "13": "Q3_K_L",
                    "14": "Q4_K_S", "15": "Q4_K_M", "16": "Q5_K_S",
                    "17": "Q5_K_M", "18": "Q6_K", "19": "IQ2_XXS",
                    "20": "IQ2_XS", "21": "IQ3_XXS", "22": "IQ1_S",
                    "23": "IQ4_NL", "24": "IQ3_S", "25": "IQ2_S",
                    "26": "IQ4_XS", "27": "IQ1_M", "28": "BF16",
                }
                metadata = getattr(llm.model, "metadata", {}) or {}
                for key, value in metadata.items():
                    if "file_type" in key or "ftype" in key:
                        if isinstance(value, bytes):
                            raw = value.decode()
                        elif isinstance(value, int):
                            raw = str(value)
                        else:
                            raw = str(value)
                        result["quantization"] = _GGUF_FTYPE_MAP.get(raw.strip(), raw)
                        break
                # Fallback: check GGUF filename for quant hint (e.g. Q4_K_M)
                if "quantization" not in result and model_path:
                    import re
                    match = re.search(r'[_\-\.]((?:I?Q\d[_A-Z\d]*|F16|F32|BF16))', os.path.basename(model_path), re.IGNORECASE)
                    if match:
                        result["quantization"] = match.group(1).upper()
        except Exception:
            pass

    # Indexer status + indexed documents
    try:
        indexer_status = model_service._task_queue.get_status("indexer")
        if indexer_status and "error" not in indexer_status:
            result["indexing"] = {
                "status": indexer_status.get("status"),
                "queued": indexer_status.get("queued", 0),
            }
        workspace = Workspace.get_instance()
        if workspace and workspace.current_project:
            from smartloop.workspace import DocumentStatus
            result["indexing"]["documents"] = [
                {"id": doc.id, "path": doc.path, "status": doc.status.value}
                for doc in workspace.current_project.documents
                if doc.status == DocumentStatus.INDEXED
            ]
    except Exception:
        pass

    # Memory pressure: orchestrator process footprint vs device memory.
    # On CUDA the models live in VRAM (not process RSS), so use the GPU's
    # own used/total.  On MPS and CPU, mmap'd GGUF weights are reflected
    # in process RSS, so RSS / total is accurate.
    try:
        cuda_mem = False
        try:
            import torch
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                result["memory_percent"] = int((total_b - free_b) / total_b * 100)
                cuda_mem = True
        except Exception:
            pass
        if not cuda_mem:
            process = psutil.Process()
            mem_rss = process.memory_info().rss
            mem_total = psutil.virtual_memory().total
            result["memory_percent"] = int(mem_rss / mem_total * 100)
    except Exception:
        pass

    return result


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SLP API - OpenAI-compatible chat completion service",
        "docs": "/docs",
        "health": "/health",
    }


class _PortCapturingServer(uvicorn.Server):
    """Uvicorn server that writes the actual bound port to PORT_FILE after startup."""

    async def startup(self, sockets=None):
        await super().startup(sockets)
        if self.servers:
            for srv in self.servers:
                if srv.sockets:
                    actual_port = srv.sockets[0].getsockname()[1]
                    os.makedirs(os.path.dirname(PORT_FILE), exist_ok=True)
                    with open(PORT_FILE, "w") as f:
                        f.write(str(actual_port))
                    break


def _install_centralized_logging() -> None:
    """Install the shared log queue, root QueueHandler, and parent listener.

    The same queue is later picked up by ``LLMExecutor`` via
    ``logging_setup.get_log_queue`` so spawn worker processes log through
    the same listener.  Stashes the listener on ``app.state`` so the
    lifespan shutdown can stop it.
    """
    from smartloop.utils.logging_setup import (
        make_log_queue, setup_root_logging, start_listener,
    )
    queue = make_log_queue()
    setup_root_logging(queue, level="DEBUG")
    # main.py (the CLI entry point) clamps "smartloop" to WARNING to keep the
    # interactive `slp` shell quiet — but start_server_foreground is launched
    # from that same main.py, so the server inherits that gag.  Lift it only
    # when the user has explicitly asked for DEBUG; at INFO (the default, used
    # by the shipped Electron build) the gag stays so server.log only
    # accumulates uvicorn access lines + warnings/errors, not the full
    # orchestrator / DAG / planner / explorer / MCP trace.
    log_level = (api_settings.log_level or "INFO").upper()
    if log_level == "DEBUG":
        logging.getLogger("smartloop").setLevel(logging.DEBUG)
    listener = start_listener(
        queue,
        home_dir=_app_settings.home_dir,
        file_level=log_level,
    )
    # Module-level code already attached _TokenRedactFilter to
    # ``uvicorn.access`` at import time; setup_root_logging only resets root
    # handlers, so that filter survives this call.
    app.state.log_listener = listener
    app.state.log_queue = queue


def run_server():
    """Run the uvicorn server (for development/direct execution)."""
    _install_centralized_logging()

    # Write PID file for tracking
    pid_file = os.path.join(_app_settings.home_dir, "server.pid")
    os.makedirs(os.path.dirname(pid_file), exist_ok=True)
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    print_logo(
        version=_app_settings.version,
        host=api_settings.api_host,
        port=api_settings.api_port,
        model=api_settings.default_model
    )

    try:
        config = uvicorn.Config(
            app,
            host=api_settings.api_host,
            port=api_settings.api_port,
            log_level="info",
            log_config=None,
        )
        server = _PortCapturingServer(config)
        server.run()
    finally:
        # Clean up PID and port files on exit
        try:
            os.unlink(pid_file)
        except:
            pass
        try:
            os.unlink(PORT_FILE)
        except OSError:
            pass


def start_server_foreground(host: str, port: int, service: bool = True) -> None:
    """
    Start the API server using uvicorn in the foreground (blocking).

    Note: We run uvicorn directly instead of via gunicorn because MPS (Metal)
    contexts don't survive fork operations on macOS.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        service: If True, auto-restart the server on crash with backoff
    """

    # Print logo before starting
    print_logo(version=_app_settings.version,)

    # PID file for tracking
    pid_file = Path(_app_settings.home_dir) / "server.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))

    _install_centralized_logging()

    max_backoff = 60  # seconds
    backoff = 1
    backoff_reset_threshold = 120  # reset backoff after 2 min of stable uptime

    try:
        while True:
            start_ts = _time.monotonic()
            try:
                config = uvicorn.Config(app, host=host, port=port, log_level="info", log_config=None)
                server = _PortCapturingServer(config)
                server.run()
            except Exception:
                if not service:
                    raise
                logger.exception("Server crashed")

            if not service:
                break

            uptime = _time.monotonic() - start_ts
            if uptime >= backoff_reset_threshold:
                backoff = 1

            logger.warning(
                "Server exited after %.1fs — restarting in %ds ...", uptime, backoff
            )
            _time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

            # Re-write PID file for the same process
            pid_file.write_text(str(os.getpid()))
    finally:
        # Clean up PID and port files on exit
        pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    run_server()
