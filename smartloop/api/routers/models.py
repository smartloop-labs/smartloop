"""Models management router"""

import asyncio
import enum
import json
import logging
from queue import Queue, Empty

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from smartloop.api.services import ModelService, get_model_service
from smartloop.config import SUPPORTED_MODELS
from smartloop.config.app_settings import AppSettings
from smartloop.platform import PlatformService, get_platform_service

try:
    from smartloop.trainer import BuildProgress, TrainingProgress
except ImportError:
    BuildProgress = None
    TrainingProgress = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["models"])


def _sse_event(data: dict, event: str = "progress") -> str:
    """Format a dict as an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

class Mode(str, enum.Enum):
    inference = "inference"
    train = "train"
    build = "build"

class _ModelLoadRequest(BaseModel):
    mode: Mode = Mode.inference  # "inference" | "train" | "build"
    project_id: str | None = None
    quantize: str | None = None
    model_name: str | None = None
    developer_token: str | None = None


@router.post("/models/load")
async def load_model(
    request: _ModelLoadRequest = _ModelLoadRequest(),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Load the model.

    - **mode=inference** (default): loads the GGUF/quantised model for serving
      and keeps it in memory until `/v1/models/unload` is called.
    - **mode=train**: unloads any running inference model, runs the full training
      pipeline (downloads artefacts + LoRA fine-tuning), then clears the model
      from memory so the next inference load picks up fresh weights.
    """
    mode = request.mode.value if isinstance(request.mode, Mode) else request.mode.strip().lower()
    if mode not in {m.value for m in Mode}:
        raise HTTPException(status_code=400, detail="mode must be 'inference', 'train', or 'build'")

    try:
        loop = asyncio.get_event_loop()

        # Switch to the requested project, or load current project's model
        if request.project_id:
            try:
                model_service.set_current_project(request.project_id)
            except (ValueError, RuntimeError) as e:
                logger.warning(f"Project not found: {request.project_id}, using current project")

        if mode == "train":
            train_queue: Queue[dict] = Queue()

            def on_train_progress(progress: TrainingProgress):
                train_queue.put(progress.to_dict())

            async def train_event_stream():
                loop = asyncio.get_event_loop()
                train_task = loop.run_in_executor(
                    None,
                    lambda: model_service.train_model(
                        request.model_name,
                        progress_callback=on_train_progress,
                    ),
                )

                yield _sse_event({"stage": "preparing", "current": 0, "total": 0, "message": "Starting training..."})

                while not train_task.done():
                    try:
                        item = train_queue.get_nowait()
                        yield _sse_event(item)
                    except Empty:
                        await asyncio.sleep(0.1)

                # Drain remaining events
                while not train_queue.empty():
                    yield _sse_event(train_queue.get_nowait())

                # Check for exceptions
                try:
                    train_task.result()
                except Exception as exc:
                    logger.exception("Model training failed")
                    yield _sse_event({"status": "error", "message": str(exc)}, event="error")
                    return

                yield _sse_event({"status": "completed", "message": "Training complete. Model unloaded – call /v1/models/load to load for inference."}, event="complete")

            return StreamingResponse(train_event_stream(), media_type="text/event-stream")
        elif mode == "build":
            progress_queue: Queue[dict] = Queue()

            def on_progress(progress: BuildProgress):
                progress_queue.put(progress.to_dict())

            async def build_event_stream():
                loop = asyncio.get_event_loop()
                build_task = loop.run_in_executor(
                    None,
                    lambda: model_service.build_model(
                        request.model_name,
                        request.quantize,
                        request.developer_token,
                        progress_callback=on_progress,
                    ),
                )

                yield _sse_event({"stage": "preparing", "current": 0, "total": 0, "message": "Starting build..."})

                while not build_task.done():
                    try:
                        item = progress_queue.get_nowait()
                        yield _sse_event(item)
                    except Empty:
                        await asyncio.sleep(0.1)

                # Drain remaining events
                while not progress_queue.empty():
                    yield _sse_event(progress_queue.get_nowait())

                # Check for exceptions
                try:
                    build_task.result()
                except Exception as exc:
                    logger.exception("Model build failed")
                    yield _sse_event({"status": "error", "message": str(exc)}, event="error")
                    return

                yield _sse_event({"status": "completed", "message": "Build complete."}, event="complete")

            return StreamingResponse(build_event_stream(), media_type="text/event-stream")
        else:
            # load_model resolves model from the current project
            await loop.run_in_executor(None, model_service.load_model)
            project_name = model_service.get_current_project_name() or "default"
            return {"status": "ok", "message": f"Model loaded for inference (project: {project_name})."}
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except (EnvironmentError, ConnectionError) as e:
        raise HTTPException(status_code=502, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Error in /models/load (mode=%s)", mode)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model(model_service: ModelService = Depends(get_model_service)):
    """Unload the model from memory to free resources."""
    project_name = model_service.get_current_project_name() or "default"
    model_service.unload_model()
    return {"status": "ok", "message": f"Model unloaded (project: {project_name})."}


class ModelEntry(BaseModel):
    name: str
    model_id: str
    size: str | None = None
    description: str | None = None
    access_level: str | None = None
    is_accessible: bool = False
    sha256: str | None = None
    enabled: bool = True
    capabilities: list[str] = []


@router.get("/models", response_model=list[ModelEntry])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
    svc: PlatformService = Depends(get_platform_service),
):
    """List available models from the cloud API (cached locally), enriched with workspace state."""
    try:
        models = svc.list_models()
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable")

    try:
        workspace = model_service._ensure_workspace()
        ws_lookup = {m.model_name: m for m in workspace.models}
    except Exception:
        ws_lookup = {}

    settings = AppSettings()
    bootstrap_model = settings.default_model

    result = []
    for m in models:
        entry = ModelEntry(**m) if isinstance(m, dict) else m
        if entry.name == bootstrap_model:
            continue
        ws_model = ws_lookup.get(entry.name)
        if ws_model:
            entry.enabled = ws_model.enabled
            entry.capabilities = ws_model.effective_capabilities
        else:
            model_cls = SUPPORTED_MODELS.get(entry.name)
            if model_cls and hasattr(model_cls, 'default_capabilities'):
                entry.capabilities = model_cls.default_capabilities()
        result.append(entry)

    # Best model first: rank by model version (a newer version wins even with
    # fewer params), then parameter size. Falls back to (0, 0) for models with
    # no class entry so they sort last, in their original catalogue order.
    def _rank(entry: ModelEntry) -> tuple[int, float]:
        cls = SUPPORTED_MODELS.get(entry.name)
        if cls is None:
            return (0, 0.0)
        return (cls.model_version(), float(cls.param_size_in_b()))

    result.sort(key=_rank, reverse=True)
    return result


class WorkspaceModelEntry(BaseModel):
    model_name: str
    model_id: str
    downloaded: bool
    is_active: bool
    enabled: bool
    capabilities: list[str] = []


@router.get("/models/workspace", response_model=list[WorkspaceModelEntry])
async def list_workspace_models(
    model_service: ModelService = Depends(get_model_service),
):
    """List models registered in the local workspace."""
    try:
        settings = AppSettings()
        bootstrap_model = settings.default_model
        workspace = model_service._ensure_workspace()
        # Best model first: largest size, then newer version (see _rank).
        models = sorted(
            (m for m in workspace.models if m.model_name != bootstrap_model),
            key=lambda m: m._rank, reverse=True,
        )
        return [
            WorkspaceModelEntry(
                model_name=m.model_name,
                model_id=m.model_id,
                downloaded=m.downloaded,
                is_active=m.is_active,
                enabled=m.enabled,
                capabilities=m.effective_capabilities,
            )
            for m in models
        ]
    except Exception as e:
        logger.exception("Error listing workspace models")
        raise HTTPException(status_code=500, detail=str(e))


class _UpdateWorkspaceModelRequest(BaseModel):
    enabled: bool | None = None
    capabilities: list[str] | None = None


@router.patch("/models/workspace/{model_name}")
async def update_workspace_model(
    model_name: str,
    request: _UpdateWorkspaceModelRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Update a workspace model's enabled state.

    ``capabilities`` in the request body is accepted for backward
    compatibility but ignored — capabilities are read-only from the
    model class (see ``WorkspaceModel.effective_capabilities``).
    """
    try:
        workspace = model_service._ensure_workspace()
        model = workspace.get_model(model_name)
        if model is None:
            raise ValueError(f"Model not found: {model_name}")
        if request.enabled is not None:
            model.enabled = request.enabled
            # Deselecting a model must also free any VRAM it still holds — the
            # enabled flag only gates future loads, not an already-resident
            # instance (see ModelService.unload_model_by_name).
            if request.enabled is False:
                model_service.unload_model_by_name(model_name)
        workspace.save()
        return {
            "status": "ok",
            "model_name": model_name,
            "enabled": model.enabled,
            "capabilities": model.effective_capabilities,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating workspace model")
        raise HTTPException(status_code=500, detail=str(e))
