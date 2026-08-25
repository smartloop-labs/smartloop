"""Web search enablement (project-level).

``PATCH /v1/projects/{id}/websearch`` is the user's own switch, toggled by the
Cmd/Ctrl+Alt+S shortcut. It says nothing about whether a search can actually
run: that depends on the account being signed in with a provider registered,
which the hosted ``GET /v1/websearch/status`` answers and the orchestrator
checks before searching. There is deliberately no second local switch — it
would only duplicate what the endpoint already knows.
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
