"""Web search enablement.

Two switches, and the orchestrator requires both:

* workspace — ``PATCH /v1/workspace/websearch`` turns the feature off
  everywhere. This is the one Settings shows.
* project — ``PATCH /v1/projects/{id}/websearch`` silences it for a single
  project. This is what the Cmd/Ctrl+Alt+S shortcut toggles.

Neither says anything about whether a search can actually run: that also needs
credentials and a provider registered on the account, and is reported as
``available``. The search itself is served by the hosted ``/v1/websearch``
endpoint.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from smartloop.api.services import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["web-search"])


class WebSearchEnabledRequest(BaseModel):
    enabled: bool


class WebSearchConfigResponse(BaseModel):
    project_id: str
    enabled: bool


class WorkspaceWebSearchResponse(BaseModel):
    enabled: bool
    #: Whether the account is signed in. Web search needs a token, and this is
    #: not something a toggle can fix — clients disable the control instead of
    #: leaving it on with a warning.
    signed_in: bool
    #: Whether a search would run right now: signed in, and both switches on.
    available: bool


@router.get("/projects/{project_id}/websearch", response_model=WebSearchConfigResponse)
async def get_web_search_config(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Return whether web search is enabled for the project."""
    try:
        return model_service.get_web_search_config(project_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error reading web search config")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}/websearch", response_model=WebSearchConfigResponse)
async def set_web_search_enabled(
    project_id: str,
    request: WebSearchEnabledRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Enable/disable web search (Pro feature) for the project."""
    try:
        return model_service.set_web_search_enabled(project_id, request.enabled)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating web search config")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workspace/websearch", response_model=WorkspaceWebSearchResponse)
async def get_workspace_web_search_config(
    model_service: ModelService = Depends(get_model_service),
):
    """Return the workspace-wide web search switch."""
    try:
        return model_service.get_workspace_web_search_config()
    except Exception as e:
        logger.exception("Error reading workspace web search config")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/workspace/websearch", response_model=WorkspaceWebSearchResponse)
async def set_workspace_web_search_enabled(
    request: WebSearchEnabledRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Enable/disable web search across the whole workspace."""
    try:
        return model_service.set_workspace_web_search_enabled(request.enabled)
    except Exception as e:
        logger.exception("Error updating workspace web search config")
        raise HTTPException(status_code=500, detail=str(e))
