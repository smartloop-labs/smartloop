"""Web search enablement router (project-level).

Web search is a Pro feature. This router only toggles whether the
orchestrator runs the WebSearchAgent for a project; the search itself is served
by the hosted ``/v1/websearch`` endpoint (gated by the account's plan).
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
