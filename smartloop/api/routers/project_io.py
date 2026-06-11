"""Project export/import router."""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse

from smartloop.api.model import ProjectResponse
from smartloop.api.services.project_io import export_project, import_project

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["project-io"])


@router.post("/projects/{project_id}/export")
async def export_project_endpoint(project_id: str):
    """Export a project as a downloadable zip archive."""
    try:
        zip_path = export_project(project_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error exporting project")
        raise HTTPException(status_code=500, detail=str(e))

    filename = os.path.basename(str(zip_path))

    def _iter_and_cleanup():
        try:
            with open(zip_path, "rb") as f:
                while chunk := f.read(64 * 1024):
                    yield chunk
        finally:
            try:
                os.unlink(zip_path)
            except OSError:
                pass

    return StreamingResponse(
        _iter_and_cleanup(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/projects/import", response_model=ProjectResponse)
async def import_project_endpoint(
    file: UploadFile = File(...),
    name: Optional[str] = Form(default=None, description="Override the project name"),
):
    """Import a project from a zip archive."""
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()

        result = import_project(tmp.name, name=name)
        return ProjectResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error importing project")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
