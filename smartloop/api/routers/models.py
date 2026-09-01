"""Models management router"""

import asyncio
import logging

from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, field_validator

from smartloop.api.services import ModelService, get_model_service
from smartloop.config import SUPPORTED_MODELS
from smartloop.config.app_settings import AppSettings
from smartloop.constants import canonical_model_name
from smartloop.platform import PlatformService, get_platform_service
from smartloop.types import ModelCapability

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["models"])


def _display_capabilities(caps) -> list:
    """The capabilities a model advertises to the client.

    ``ModelCapability.THINKING`` is internal-only: it drives the setting and
    the picker toggle, not routing — no step asks for a "thinking model" — so
    it is stripped from the list returned in the model listing.
    """
    return [c for c in caps if c != ModelCapability.THINKING]

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
    # Not on any plan: staff/preview builds the catalog publishes alongside the
    # public tiers.  Listed like the rest — access is enforced where the weights
    # are served (/v1/init and the signed-URL fetch both answer 403), not by
    # hiding the entry.
    internal = "internal"

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
    # Reasoning is a per-model setting, not a per-request one: a model the user
    # switches thinking on for reasons on every call.  ``thinking`` is null for
    # a model whose capabilities lack "thinking" — the picker shows the toggle
    # only when the model can honor it.
    thinking: bool | None = None

    @field_validator("access_level", mode="before")
    @classmethod
    def _tolerate_unknown_access_level(cls, value):
        """Drop an access level this build doesn't know instead of failing.

        The catalog is server-side and gains tiers without a client release —
        ``internal`` above arrived that way and 500'd the whole picker, because
        one unparseable entry fails the response model for every model in the
        list.  An unknown tier degrades to ``None`` (the client's "unspecified"
        case) so the rest of the catalog still renders; the download path
        enforces access regardless.
        """
        if value is None or isinstance(value, AccessLevel):
            return value
        try:
            return AccessLevel(value)
        except ValueError:
            logger.warning(
                "[models] unknown access_level %r in catalog — treating as unspecified",
                value,
            )
            return None


@router.get("/models", response_model=list[ModelEntry])
async def list_models(
    model_service: ModelService = Depends(get_model_service),
    svc: PlatformService = Depends(get_platform_service),
    project_id: str | None = Query(default=None, description="Resolve enabled state against this project; defaults to workspace (app-default) state."),
):
    """List available models from the cloud API (cached locally), enriched with local state.

    The whole catalog is returned, Pro/premium included: the picker shows what
    an upgrade would buy, and the client decides how to present a model the
    plan doesn't grant.  ``access_level`` and the per-user ``is_accessible``
    (computed by the cloud — see ``_refresh_platform_cache`` in the auth
    router) carry that state.  Free models download with no token at all, so
    ``is_accessible`` is not a download gate; an anonymous caller gets
    ``is_accessible=False`` on everything and can still download the free tier.

    Access is enforced where the weights are actually served: ``/v1/init``
    answers 403 for a model off the plan, and the platform answers 403 again
    when the signed URLs are fetched.  Nothing is hidden here to enforce it.

    Only ``enabled`` resolves against ``project_id``; download state is
    workspace-wide.
    """
    try:
        models = svc.list_models()
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable")

    workspace = None
    try:
        workspace = model_service._ensure_workspace()
        ws_lookup = {m.model_name: m for m in workspace.models}
    except Exception:
        ws_lookup = {}

    project = None
    if project_id and workspace is not None:
        project = next((p for p in workspace.projects if p.id == project_id), None)

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
        if ws_model:
            entry.enabled = workspace.effective_model_enabled(entry.name, project)
            entry.capabilities = _display_capabilities(ws_model.effective_capabilities)
            entry.thinking = workspace.effective_model_thinking(entry.name, project)
        else:
            model_cls = SUPPORTED_MODELS.get(entry.name)
            if model_cls and hasattr(model_cls, 'default_capabilities'):
                entry.capabilities = _display_capabilities(model_cls.default_capabilities())
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
    thinking: bool | None = None


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
                capabilities=_display_capabilities(m.effective_capabilities),
                thinking=workspace.effective_model_thinking(m.model_name),
            )
            for m in models
        ]
    except Exception as e:
        logger.exception("Error listing workspace models")
        raise HTTPException(status_code=500, detail=str(e))


class _UpdateWorkspaceModelRequest(BaseModel):
    enabled: bool | None = None
    thinking: bool | None = None
    capabilities: list[str] | None = None


@router.patch("/models/workspace/{model_name}")
async def update_workspace_model(
    model_name: str,
    request: _UpdateWorkspaceModelRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Update a workspace model's enabled state, or whether it reasons.

    ``thinking`` is rejected for a model whose capabilities lack "thinking":
    its chat template never reads ``enable_thinking``, so storing the
    preference would leave the picker showing a toggle that does nothing.

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
            # This edits the app-default that new projects are seeded from.  A
            # project with no entry for the model reads through to that default,
            # so pin the current answer everywhere before moving it — the change
            # is meant for projects created later, not existing ones.
            workspace.pin_model_for_existing_projects(model_name)
            model.enabled = request.enabled
            # Deselecting a model must also free any VRAM it still holds — the
            # enabled flag only gates future loads, not an already-resident
            # instance (see ModelService.unload_model_by_name).
            if request.enabled is False:
                model_service.unload_model_by_name(model_name)
        if request.thinking is not None:
            if not workspace.supports_thinking(model_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"{model_name} does not support thinking",
                )
            workspace.pin_model_for_existing_projects(model_name)
            model.thinking = request.thinking
            # Unlike ``enabled`` — which only gates future loads — this has to
            # reach the instance that is already resident, or the setting would
            # not take effect until the model is next loaded (for the
            # always-resident orchestrator, never).
            model_service.apply_thinking_setting(model_name)
        workspace.save()
        return {
            "status": "ok",
            "model_name": model_name,
            "enabled": model.enabled,
            "thinking": workspace.effective_model_thinking(model_name),
            "capabilities": _display_capabilities(model.effective_capabilities),
        }
    except HTTPException:
        # A status we chose deliberately (e.g. the 400 above) — the catch-all
        # below would otherwise re-wrap it as an opaque 500.
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating workspace model")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Per-project model settings.
#
# Model settings are scoped to a project (the project sidebar owns them), seeded
# from the workspace (app-default) registry at project creation.  The legacy
# ``/v1/models/workspace`` endpoints above remain as thin workspace-level
# (app-default) accessors and are deprecated in favor of these.
# ---------------------------------------------------------------------------


class ProjectModelEntry(BaseModel):
    model_name: str
    model_id: str
    downloaded: bool
    is_active: bool
    enabled: bool
    capabilities: list[str] = []
    settings: dict = {}


class _UpdateProjectModelRequest(BaseModel):
    enabled: bool | None = None
    is_active: bool | None = None
    settings: dict | None = None


def _resolve_project(model_service: ModelService, project_id: str) -> tuple:
    """Return (workspace, project), raising 404 when the project is unknown."""
    workspace = model_service._ensure_workspace()
    project = next((p for p in workspace.projects if p.id == project_id), None)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    return workspace, project


@router.get("/projects/{project_id}/models", response_model=list[ProjectModelEntry])
async def list_project_models(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """List models registered for a project, with per-project state.

    Entries are seeded from the workspace (app-default) registry; a project's
    own settings override those defaults.  Capabilities are read-only from the
    model class, and so is ``downloaded`` — weights are shared workspace-wide.
    """
    try:
        workspace, project = _resolve_project(model_service, project_id)
        workspace.sync_project_models(project)
        models = sorted(
            (m for m in workspace.models if not workspace.is_bootstrap_model(m.model_name)),
            key=lambda m: m._rank, reverse=True,
        )
        return [
            ProjectModelEntry(
                model_name=m.model_name,
                model_id=m.model_id,
                downloaded=workspace.effective_model_downloaded(m.model_name),
                is_active=workspace.effective_model_active(m.model_name, project),
                enabled=workspace.effective_model_enabled(m.model_name, project),
                capabilities=_display_capabilities(m.effective_capabilities),
                settings=dict(pm.settings) if (pm := workspace.project_model(project, m.model_name)) else {},
            )
            for m in models
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error listing project models")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}/models/{model_name}")
async def update_project_model(
    project_id: str,
    model_name: str,
    request: _UpdateProjectModelRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Update a project's model settings (enabled/active/settings).

    Per-project overrides take precedence over the workspace (app-default)
    registry.  Disabling a model also unloads any resident instance — VRAM is
    shared across projects, so the unload stays global.  Download state is not
    settable here: weights are added and removed workspace-wide, so the
    ``downloaded`` field in the response is read-only.
    """
    try:
        # An older client may still address the model by its previous name.
        model_name = canonical_model_name(model_name)
        workspace, project = _resolve_project(model_service, project_id)
        if workspace.is_bootstrap_model(model_name):
            # The orchestrator model is always enabled and is filtered out of
            # the list responses above, so it is not a per-project setting.
            # Reject explicitly rather than writing an entry that would shadow
            # its always-on workspace state.
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} is always enabled and cannot be configured per project.",
            )
        model = workspace.get_model(model_name)
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")

        workspace.sync_project_models(project)
        pm = workspace.project_model(project, model_name)

        if request.enabled is not None:
            pm.enabled = request.enabled
            if request.enabled:
                # Enabling here also makes the model an app-default, so projects
                # created later start with it.  Existing projects are pinned
                # first and keep what they resolve to today.
                workspace.promote_model_to_workspace_default(model_name)
            else:
                # Deselecting must also free any VRAM it still holds — the
                # enabled flag only gates future loads, not an already-resident
                # instance.
                model_service.unload_model_by_name(model_name)

        if request.is_active is not None:
            if request.is_active:
                for other in project.model_settings:
                    other.is_active = (other is pm)
            else:
                pm.is_active = False

        if request.settings is not None:
            pm.settings.update(request.settings)

        workspace.save()
        return {
            "status": "ok",
            "project_id": project_id,
            "model_name": model_name,
            "enabled": pm.enabled,
            "downloaded": workspace.effective_model_downloaded(model_name),
            "is_active": pm.is_active,
            "capabilities": _display_capabilities(model.effective_capabilities),
            "settings": dict(pm.settings),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating project model")
        raise HTTPException(status_code=500, detail=str(e))
