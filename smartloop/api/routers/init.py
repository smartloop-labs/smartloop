"""Init router – initialise workspace and stream model download progress via SSE."""

import json
import logging
import os
import asyncio
import time
from queue import Queue
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from smartloop.api.config import api_settings
from smartloop.api.services import ModelService, get_model_service
from smartloop.config import SUPPORTED_MODELS
from smartloop.constants import canonical_model_name, display_model_name
from smartloop.config.model_config import ModelConfig
from smartloop.model.downloader import SLPArtifactDownloader, DownloadProgress
from smartloop.platform import ModelAccessError, get_platform_service
from smartloop.workspace.context import projects_delegated

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["init"])

# SSE keepalive comment — keeps the connection alive without producing a client-visible event
_SSE_KEEPALIVE = ": keepalive\n\n"


class InitRequest(BaseModel):
    """Request body for workspace initialization."""
    model_name: Optional[str] = Field(default=None, description="Model to initialize")
    developer_token: Optional[str] = Field(default=None, description="Developer token for model download")
    cwd: Optional[str] = Field(default=None, description="Working directory for the default project; defaults to ~/smartloop/<name> when omitted")
    project_id: Optional[str] = Field(default=None, description="Project the download was requested for; scopes the resulting enablement to it")


def _sse_event(data: dict, event: str = "progress") -> str:
    """Format a dict as an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _status_for(exc: Exception) -> int:
    """Classify a stream failure as the HTTP status it would have been.

    Entitlement is settled before the stream opens, so anything reaching here
    failed mid-transfer — an expired signed URL, a full disk, a dropped socket.
    The client still has to tell "sign in" apart from "try again", and it should
    not have to do that by matching on message text, which is written for a
    person and gets reworded.  Platform errors carry the upstream status (see
    ``_raise_for_status``); the rest fall back to their type.

    Only :class:`ModelAccessError` earns a 403.  ``PermissionError`` used to,
    which was too wide by half: it is an ``OSError`` subclass, so an unwritable
    models directory arrived as "not on your plan" and the desktop client turned
    that into a sign-in prompt for what is a local disk problem.  A platform
    refusal carries ``status_code`` and is answered above regardless.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    if isinstance(exc, ModelAccessError):
        return 403
    # Transfers that died on a dropped socket, a stale signed URL or an upstream
    # 5xx are aggregated into a ConnectionError by ``_aggregate_error``.
    if isinstance(exc, ConnectionError):
        return 502
    return 500


def _error_event(exc: Exception) -> str:
    """Terminal SSE frame for a failed stream, tagged with a status code."""
    return _sse_event(
        {"status": "error", "code": _status_for(exc), "message": str(exc)},
        event="error",
    )


def _seed_model_catalog() -> None:
    """Fire-and-forget best-effort cache of the cloud model catalog.

    Scheduled on the executor rather than awaited: list_models() is a sync
    cloud call with a 30s timeout, and blocking on it would stall the SSE
    stream. Seeding is off the critical path, so we never wait on it.
    """
    def _log_result(future) -> None:
        exc = future.exception()
        if exc is not None:
            logger.warning("Could not seed model catalog cache", exc_info=exc)

    try:
        future = asyncio.get_event_loop().run_in_executor(
            None, get_platform_service().list_models
        )
        future.add_done_callback(_log_result)
    except Exception:
        logger.warning("Could not schedule model catalog seed", exc_info=True)


def _gguf_exists(model_dir: str) -> bool:
    """Check whether a .gguf file exists in model_dir or its bin/ subdirectory."""
    if not os.path.isdir(model_dir):
        return False
    if any(f.endswith(".gguf") for f in os.listdir(model_dir)):
        return True
    bin_dir = os.path.join(model_dir, "bin")
    return os.path.isdir(bin_dir) and any(f.endswith(".gguf") for f in os.listdir(bin_dir))


async def _download_and_stream(downloader, model_id: str, model_name: str,
                                is_bootstrap: bool = False,
                                artifacts: dict[str, str] | None = None):
    """Download model files and stream progress.

    *artifacts* carries a signed-URL map already resolved by the caller (see
    :meth:`SLPArtifactDownloader.resolve_artifacts`), so entitlement is settled
    before this stream opens and the URLs are minted once.
    """
    progress_queue: Queue[dict] = Queue()
    _last_progress_time: dict[str, float] = {}

    def on_progress(progress: DownloadProgress):
        now = time.monotonic()
        if progress.status != "completed" and now - _last_progress_time.get(progress.filename, 0) < 0.5:
            return
        _last_progress_time[progress.filename] = now
        d = progress.to_dict()
        d["status"] = "downloading"
        if progress.total > 0:
            d["percent"] = round(progress.downloaded / progress.total * 100, 1)
            d["message"] = f"Downloading {progress.filename}... {d['percent']}%"
        progress_queue.put(d)

    yield _sse_event({"status": "downloading", "message": f"Downloading model {display_model_name(model_name)}..."})

    loop = asyncio.get_event_loop()

    def _do_download():
        if is_bootstrap:
            downloader.bootstrap_download(progress_callback=on_progress)
        else:
            downloader.download(model_id=model_id, progress_callback=on_progress,
                                artifacts=artifacts)

    download_task = loop.run_in_executor(None, _do_download)

    # Emit a keepalive when no progress has flowed for a few seconds so the SSE
    # connection survives download stalls (e.g. a dropped connection being
    # retried) without the client or a proxy timing out the idle stream.
    idle_ticks = 0
    while not download_task.done():
        drained = False
        while not progress_queue.empty():
            yield _sse_event(progress_queue.get_nowait())
            drained = True
        if drained:
            idle_ticks = 0
        else:
            idle_ticks += 1
            if idle_ticks >= 10:  # ~5s of silence
                yield _SSE_KEEPALIVE
                idle_ticks = 0
        await asyncio.sleep(0.5)

    while not progress_queue.empty():
        yield _sse_event(progress_queue.get_nowait())

    download_task.result()

    yield _sse_event({"status": "download_complete", "message": f"Model {display_model_name(model_name)} downloaded."})


async def _load_orchestrator_and_stream(model_service):
    """Load the orchestrator model into memory with SSE keepalives.

    Delegates to ``model_service.load_model()``, which always loads the small
    base model (``api_settings.default_model``) as the orchestrator/main model
    — regardless of which model was just downloaded or is active.  A larger
    downloaded model is loaded lazily on demand for ModelTier.LARGE steps.
    """
    name = api_settings.default_model
    yield _sse_event({"status": "loading", "message": f"Loading orchestrator model {display_model_name(name)}..."})

    loop = asyncio.get_event_loop()
    load_task = loop.run_in_executor(None, model_service.load_model)

    while not load_task.done():
        yield _SSE_KEEPALIVE
        await asyncio.sleep(2)

    load_task.result()  # propagate any load error to the stream
    yield _sse_event({
        "status": "model_loaded",
        "message": f"Orchestrator model {display_model_name(model_service._model_name)} loaded.",
    })


async def _ensure_project(model_service, workspace, cwd=None):
    """Create bare project if none exist (no skills/MCP — LLM not loaded yet)."""
    if not model_service._model.workspace.projects:
        yield _sse_event({"status": "creating_project", "message": "Creating default project..."})
        model_service.create_project(
            name="general_chat",
            description="Default general chat project",
            system=True,
            cwd=cwd,
        )
        yield _sse_event({"status": "project_created", "message": "Project created."}, event="project")


async def _setup_project_services(model_service):
    """Install MCP servers and generate skills — requires LLM to be loaded."""
    current = model_service._model.workspace.current_project
    if not current or current.system:
        return
    yield _sse_event({"status": "setup", "message": "Setting up project services..."})
    model_service._setup_project_tools(current)
    # Re-fetch: _setup_project_tools may have replaced model_service._model
    # via _ensure_model(), so `current` may point to a stale workspace.
    refreshed = model_service._model.workspace.current_project
    if refreshed:
        refreshed.system = True
    model_service._model.save()
    yield _sse_event({"status": "setup_complete", "message": "Project setup complete."})


def _get_project_info(workspace, model_name: str) -> Optional[dict]:
    """Return project info dict for the current project."""
    current = workspace.current_project
    if not current:
        return None
    current_model = workspace.get_model()
    return {
        "id": current.id,
        "name": current.name,
        "description": current.description,
        "model_name": current_model.model_name if current_model else model_name,
        "current": current.current,
        "system": current.system,
    }


@router.post("/init")
async def init(
    request: InitRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Initialize the workspace and download the model.

    Downloads the GGUF via the Smartloop artifact endpoints.
    Returns a streaming SSE response with real-time download progress.
    """
    workspace = model_service._ensure_workspace()
    target = canonical_model_name(request.model_name) or api_settings.default_model

    # Register the model in the workspace if not already present
    ws_model = workspace.get_model(target)
    if not ws_model:
        model_cls = SUPPORTED_MODELS.get(target)
        if not model_cls:
            raise HTTPException(status_code=400, detail=f"Unknown model: {target}")
        # Register a project-scoped download disabled: the caller's follow-up
        # PATCH enables it and promotes the app-default from there (see
        # Workspace.promote_model_to_workspace_default), so a cancelled
        # download strands nothing. A project-less init (CLI/TUI) sends no
        # PATCH, so it registers enabled workspace-wide.
        ws_model = workspace.add_model(target, model_cls.model_id(),
                                       enabled=request.project_id is None)

    model_dir = ws_model.model_dir

    # Settle entitlement before the response starts. There is no credential
    # pre-flight — FREE models download anonymously and the platform is the
    # authority on the rest — but the answer has to arrive as a status code:
    # once StreamingResponse begins, the request has already committed to 200
    # and a refusal can only be described inside the body. Resolving here also
    # means the signed URLs are minted once, which matters because the download
    # route is rate-limited per IP.
    downloader = None
    artifacts = None
    if not _gguf_exists(model_dir):
        downloader = SLPArtifactDownloader(workspace, logger)
        try:
            artifacts = await asyncio.get_event_loop().run_in_executor(
                None, downloader.resolve_artifacts, ws_model.model_id
            )
        except PermissionError as exc:
            raise HTTPException(status_code=getattr(exc, "status_code", 403), detail=str(exc))
        except ConnectionError as exc:
            raise HTTPException(status_code=getattr(exc, "status_code", 502), detail=str(exc))

    async def event_stream():
        try:
            # 1. Download if needed
            if downloader is None:
                yield _sse_event({"status": "model_ready", "message": f"Model {display_model_name(target)} already available."})
                model_service.mark_model_downloaded(target, workspace=workspace)
            else:
                async for event in _download_and_stream(downloader, ws_model.model_id, target,
                                                        artifacts=artifacts):
                    yield event

                model_service.mark_model_downloaded(target, workspace=workspace)

                # A model earns its place among the app-defaults by arriving,
                # not by being switched on.  This is the only branch where a
                # download actually ran, and it is the event the promotion
                # belongs to: the model is new to the workspace, so "on by
                # default from now on" is the answer the user just gave by
                # fetching it.  Switching on a model that was already here is
                # project-scoped and deliberately does not reach this (see
                # ``update_project_model``).
                #
                # It has to happen here rather than in that route for a second
                # reason: ``/v1/init`` names no project in its path, so under
                # supervision it is answered by the process that owns
                # ``config.json``.  The project route is proxied to the child
                # pinned to that project, which cannot write the workspace
                # record at all — a promotion there was discarded at save.
                workspace.promote_model_to_workspace_default(target)
                workspace.save()

            # ``mark_model_downloaded`` persisted the workspace, so config.json
            # is on disk and the platform service can bind the workspace DB and
            # persist the catalog.
            _seed_model_catalog()

            # 2. Ensure model pipeline
            model_service._ensure_model()

            # 3. Create bare project (no skills yet — LLM not loaded)
            async for event in _ensure_project(model_service, workspace, cwd=request.cwd):
                yield event

            # 4. Start background tasks (indexer, etc.) if not already running.
            #    Under supervision no indexer is registered at all (see
            #    ``ModelService.setup_tasks``), so asking whether one is
            #    running would re-run the whole setup on every call —
            #    replacing the MCP task and orphaning the loop it had
            #    already started.
            if (not projects_delegated()
                    and model_service._task_queue.get_task("indexer") is None):
                model_service.setup_tasks(workspace)

            # 5. Load model (only if a GGUF is available)
            if not _gguf_exists(model_dir):
                yield _sse_event({"status": "download_complete", "message": f"Model {display_model_name(target)} downloaded. Build a GGUF to load it."})
            elif model_service._llm is not None and model_service._model_name == api_settings.default_model:
                yield _sse_event({"status": "model_loaded", "message": f"Orchestrator model {display_model_name(model_service._model_name)} already loaded."})
            else:
                async for event in _load_orchestrator_and_stream(model_service):
                    yield event

                # 6. Setup MCP servers and skills (requires LLM)
                async for event in _setup_project_services(model_service):
                    yield event

            # 7. Done
            yield _sse_event({
                "status": "completed",
                "message": "Model initialized.",
                "model_name": target,
                "project": _get_project_info(model_service._model.workspace, target),
            }, event="complete")

        except Exception as exc:
            logger.exception("Init failed")
            yield _error_event(exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/bootstrap")
async def bootstrap(
    request: Optional[InitRequest] = None,
    model_service: ModelService = Depends(get_model_service),
):
    """Bootstrap the workspace: download a model and create a default project.

    No authentication required.  Intended for first-run scenarios
    (Electron installer, brew install) where the user has no login token.
    Returns a streaming SSE response with real-time progress.
    """

    async def event_stream():
        try:
            # 1. Ensure workspace
            yield _sse_event({"status": "checking", "message": "Checking workspace..."})
            workspace = model_service._ensure_workspace()

            # Fast path: already loaded
            if model_service._llm is not None and workspace.current_project:
                _seed_model_catalog()
                active = workspace.get_model()
                yield _sse_event({
                    "status": "completed",
                    "message": "Model initialized.",
                    "model_name": active.model_name if active else model_service._model_name,
                    "project": _get_project_info(workspace, active.model_name if active else model_service._model_name),
                }, event="complete")
                return

            # 2. Check for existing GGUF on disk — prefer the active model
            existing_model_name: Optional[str] = None
            active = workspace.active_model
            if active:
                active.models_dir = workspace.models_dir
                if active.model_dir and _gguf_exists(active.model_dir):
                    existing_model_name = active.model_name
                    if not active.downloaded:
                        active.downloaded = True

            if not existing_model_name:
                for wm in workspace.models:
                    wm.models_dir = workspace.models_dir
                    md = wm.model_dir
                    if md and _gguf_exists(md):
                        existing_model_name = wm.model_name
                        if not wm.downloaded:
                            wm.downloaded = True
                        break
            if existing_model_name:
                yield _sse_event({"status": "model_ready", "message": f"Model {display_model_name(existing_model_name)} already available."})
            else:
                # 3. Bootstrap download (GGUF only, no build needed)
                downloader = SLPArtifactDownloader(workspace, logger)
                bootstrap_models = downloader._bootstrap_source.list_models()
                target = bootstrap_models[0]
                bootstrap_name = target.get("name")

                bootstrap_model_id = SLPArtifactDownloader._resolve_model_id(bootstrap_name)
                workspace.add_model(bootstrap_name, bootstrap_model_id)
                ws_model = workspace.get_model(bootstrap_name)
                ws_model.models_dir = workspace.models_dir

                async for event in _download_and_stream(downloader, bootstrap_model_id, bootstrap_name, is_bootstrap=True):
                    yield event

                model_service.mark_model_downloaded(bootstrap_name, workspace=workspace)
                existing_model_name = bootstrap_name

            # Both branches above mutated this workspace -- a downloaded flag
            # on a model whose weights were already on disk, or a freshly
            # registered bootstrap model -- and only the download branch used
            # to persist it, so the flag the other branch set never reached
            # disk.  Saving here covers both; config.json is then on disk for
            # the platform service to bind the workspace DB against.
            workspace.save()
            _seed_model_catalog()

            # 4. Ensure model pipeline
            model_service._ensure_model()

            # 5. Start background tasks (indexer, etc.) if not already running.
            #    Under supervision no indexer is registered at all (see
            #    ``ModelService.setup_tasks``), so asking whether one is
            #    running would re-run the whole setup on every call —
            #    replacing the MCP task and orphaning the loop it had
            #    already started.
            if (not projects_delegated()
                    and model_service._task_queue.get_task("indexer") is None):
                model_service.setup_tasks(workspace)

            # 6. Create bare project (no skills yet — LLM not loaded)
            async for event in _ensure_project(model_service, workspace, cwd=(request.cwd if request else None)):
                yield event

            # 7. Load the orchestrator model (always the small base model)
            async for event in _load_orchestrator_and_stream(model_service):
                yield event

            # 8. Setup MCP servers and skills (requires LLM)
            async for event in _setup_project_services(model_service):
                yield event

            # 9. Done
            yield _sse_event({
                "status": "completed",
                "message": "Model initialized.",
                "model_name": existing_model_name,
                "project": _get_project_info(model_service._model.workspace, existing_model_name),
            }, event="complete")

        except Exception as exc:
            logger.exception("Bootstrap failed")
            yield _error_event(exc)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
