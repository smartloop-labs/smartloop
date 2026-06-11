"""Skills CRUD router"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from smartloop.api.model.project import SkillRequest, SkillResponse, UpdateSkillRequest
from smartloop.api.services import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["skills"])


@router.get("/projects/{project_id}/skills")
async def list_skills(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """List all skills for a project."""
    try:
        skills = model_service.list_skills(project_id)
        return {"skills": [SkillResponse(**s) for s in skills]}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error listing skills")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/skills", response_model=SkillResponse)
async def add_skill(
    project_id: str,
    request: SkillRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Add a skill to a project."""
    try:
        result = model_service.add_skill(
            project_id, content=request.content,
            name=request.name, description=request.description,
        )
        return SkillResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error adding skill")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/skills/{skill_id}", response_model=SkillResponse)
async def get_skill(
    project_id: str,
    skill_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Get a single skill by ID."""
    try:
        result = model_service.get_skill(project_id, skill_id)
        return SkillResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error getting skill")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}/skills/{skill_id}", response_model=SkillResponse)
async def update_skill(
    project_id: str,
    skill_id: str,
    request: UpdateSkillRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Update a skill's content."""
    try:
        result = model_service.update_skill(project_id, skill_id, content=request.content)
        return SkillResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error updating skill")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}/skills/{skill_id}")
async def delete_skill(
    project_id: str,
    skill_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a skill by ID."""
    try:
        result = model_service.delete_skill(project_id, skill_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting skill")
        raise HTTPException(status_code=500, detail=str(e))
