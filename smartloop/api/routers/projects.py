"""Project CRUD router"""

import logging

from hashlib import md5
from fastapi import APIRouter, Depends, HTTPException

from smartloop.api.model import (
    CreateProjectRequest,
    UpdateProjectRequest,
    ProjectResponse,
    ProjectListResponse,
    DeleteProjectResponse,
)
from smartloop.api.services import ModelService, get_model_service
from smartloop.workspace.workspace import Skill

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["projects"])


@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    request: CreateProjectRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Create a new project."""
    try:
        result = model_service.create_project(
            name=request.name,
            description=request.description,
            system=request.system,
            skills=[Skill(id=md5(s.content.encode()).hexdigest(), name=s.name or f"skill-{md5(s.content.encode()).hexdigest()[:6]}", description=s.description, content=s.content) for s in request.skills],
            cwd=request.cwd,
        )
        return ProjectResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error creating project")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    model_service: ModelService = Depends(get_model_service),
):
    """List all projects."""
    try:
        projects = model_service.list_projects()
        return ProjectListResponse(
            projects=[ProjectResponse(**p) for p in projects]
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error listing projects")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Update a project."""
    try:
        result = model_service.update_project(
            project_id=project_id,
            name=request.name,
            description=request.description,
            skills=[Skill(id=md5(s.content.encode()).hexdigest(), name=s.name or f"skill-{md5(s.content.encode()).hexdigest()[:6]}", description=s.description, content=s.content) for s in request.skills],
        )
        return ProjectResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error updating project")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}", response_model=DeleteProjectResponse)
async def delete_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a project by ID."""
    try:
        result = model_service.delete_project(project_id)
        return DeleteProjectResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting project")
        raise HTTPException(status_code=500, detail=str(e))
