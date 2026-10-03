"""Storage usage and cleanup router.

Reports what the SLP home directory is spending disk on and reclaims the parts
that are safe to lose.  The desktop app used to do this itself in Electron JS;
these endpoints exist so the framework — which owns the workspace record and
the loaded models — makes those decisions instead.
"""
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from smartloop.api.services import ModelService, get_model_service, storage_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["storage"])


class CleanupRequest(BaseModel):
    """Categories to reclaim.  Nothing is deleted unless asked for explicitly."""

    models: bool = Field(default=False, description="Delete downloaded models that aren't in use")
    model_ids: Optional[list[str]] = Field(
        default=None,
        description="Restrict the model sweep to these ids; omit for every removable model",
    )
    versions: bool = Field(default=False, description="Delete framework installs other than the running one")
    logs: bool = Field(default=False, description="Empty server.log and delete its rotated backups")
    cache: bool = Field(default=False, description="Empty the regenerable cache directories")
    orphan_projects: bool = Field(
        default=False, description="Delete project directories the workspace no longer references"
    )
    dry_run: bool = Field(default=False, description="Report what would be removed without removing it")


@router.get("/storage/usage")
async def get_storage_usage(model_service: ModelService = Depends(get_model_service)):
    """Report disk usage under the SLP home directory, by category."""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: storage_service.compute_usage(model_service=model_service)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error computing storage usage")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/storage/cleanup")
async def cleanup_storage(
    request: CleanupRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Reclaim disk space in the requested categories.

    Returns what was removed, what failed, and a fresh usage report, so the
    caller doesn't need a follow-up request to refresh its view.
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: storage_service.run_cleanup(
                model_service=model_service,
                models=request.models,
                model_ids=request.model_ids,
                versions=request.versions,
                logs=request.logs,
                cache=request.cache,
                orphan_projects=request.orphan_projects,
                dry_run=request.dry_run,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error cleaning up storage")
        raise HTTPException(status_code=500, detail=str(e))
