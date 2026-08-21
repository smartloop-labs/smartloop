"""Models management router"""

import asyncio
import logging

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from smartloop.api.services import ModelService, get_model_service
from smartloop.config import SUPPORTED_MODELS
from smartloop.config.app_settings import AppSettings
from smartloop.constants import canonical_model_name
from smartloop.platform import PlatformService, get_platform_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["models"])

class _ModelLoadRequest(BaseModel):
    project_id: str | None = None

@router.post("/models/load")
async def load_model(
    request: _ModelLoadRequest = _ModelLoadRequest(),
    model_service: ModelService = Depends(get_model_service),
):
    """
    Load the GGUF/quantised model for serving and keep it in memory until
    `/v1/models/unload` is called.
    """
    try:
        loop = asyncio.get_event_loop()

        # Switch to the requested project, or load current project's model
        if request.project_id:
            try:
                model_service.set_current_project(request.project_id)
            except (ValueError, RuntimeError):
                logger.warning(f"Project not found: {request.project_id}, using current project")

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
        logger.exception("Error in /models/load")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model(model_service: ModelService = Depends(get_model_service)):
    """Unload the model from memory to free resources."""
    project_name = model_service.get_current_project_name() or "default"
    model_service.unload_model()
    return {"status": "ok", "message": f"Model unloaded (project: {project_name})."}

class AccessLevel(str, Enum):
    free = "free"
    paid = "pro"
    premium = "premium"

class ModelEntry(BaseModel):
    name: str
    model_id: str
    size: str | None = None
    description: str | None = None
    access_level: AccessLevel | None = None
    is_accessible: bool = False
    sha256: str | None = None
    enabled: bool = True
    capabilities: list[str] = []


@router.get("/models", response_model=list[ModelEntry])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
    svc: PlatformService = Depends(get_platform_service),
):
    """List available models from the cloud API (cached locally), enriched with workspace state.

    Paid models (Pro/premium) are listed only when the account's plan grants
    them: the cloud computes ``is_accessible`` per-user (see
    ``_refresh_platform_cache`` in the auth router), so a free account never
    sees a Pro-only model.  Free models are always listed — anonymous callers
    get ``is_accessible=False`` on everything, so gating them on that flag
    would leave a logged-out user with an empty catalog.

    A model already downloaded into this workspace is always listed too,
    whatever its access level: the weights are on disk and stay usable after a
    plan downgrade or a logout.
    """
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
        # The cloud catalog may still publish a model under its old name.
        entry.name = canonical_model_name(entry.name)
        if entry.name == bootstrap_model:
            continue
        ws_model = ws_lookup.get(entry.name)
        # An unset access_level means the cloud didn't classify the model —
        # treat it as free rather than hiding it.
        is_paid = entry.access_level not in (None, AccessLevel.free)
        if is_paid and not entry.is_accessible and not (ws_model and ws_model.downloaded):
            continue
        if ws_model:
            entry.enabled = ws_model.enabled
            entry.capabilities = ws_model.effective_capabilities
        else:
            model_cls = SUPPORTED_MODELS.get(entry.name)
            if model_cls and hasattr(model_cls, 'default_capabilities'):
                entry.capabilities = model_cls.default_capabilities()
        result.append(entry)

    result.sort(key=lambda entry: (entry.access_level == AccessLevel.free, SUPPORTED_MODELS.get(entry.name).param_size_in_b() if SUPPORTED_MODELS.get(entry.name) else 0), reverse=True)

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
            (m for m in workspace.models
             if canonical_model_name(m.model_name) != bootstrap_model),
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
        # An older client may still address the model by its previous name.
        model_name = canonical_model_name(model_name)
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
