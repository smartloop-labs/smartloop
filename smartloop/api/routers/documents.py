"""Document management router"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from smartloop.api.model import (
    AddDocumentRequest,
    DocumentResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
)
from smartloop.api.services import ModelService, get_model_service
from smartloop.workspace import DocumentInfo, DocumentStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["documents"])

_executor = ThreadPoolExecutor(max_workers=1)

_TERMINAL_STATUSES = {
    DocumentStatus.INDEXED.value,
    DocumentStatus.FAILED.value,
    DocumentStatus.DELETED.value,
    DocumentStatus.PROCESSED.value,
}
_POLL_INTERVAL = 1.0   # seconds between status checks
_STREAM_TIMEOUT = 300  # max seconds before the stream closes


def _sse_event(data: dict, event: str = "progress") -> str:
    """Format a dict as an SSE event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_document_progress(
    doc_ids: list[str],
    model_service: ModelService,
    project_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events until all documents reach a terminal status."""
    last_statuses: dict[str, str] = {did: DocumentStatus.PENDING.value for did in doc_ids}
    deadline = time.monotonic() + _STREAM_TIMEOUT

    # Emit an initial event so the client sees the queued documents immediately.
    yield _sse_event(
        {
            "documents": [{"id": did, "status": DocumentStatus.PENDING.value} for did in doc_ids],
            "message": f"{len(doc_ids)} document(s) queued for processing",
        },
        event="queued",
    )

    loop = asyncio.get_event_loop()

    while time.monotonic() < deadline:
        await asyncio.sleep(_POLL_INTERVAL)

        doc_infos: list[DocumentInfo] = await loop.run_in_executor(
            _executor,
            model_service.get_document_statuses,
            doc_ids,
            project_id,
        )

        changed: list[DocumentInfo] = []
        for info in doc_infos:
            if info.status != last_statuses.get(info.id):
                last_statuses[info.id] = info.status
                changed.append(info)

        if changed:
            yield _sse_event({"documents": [i.model_dump() for i in changed]}, event="progress")

        all_done = all(last_statuses.get(did) in _TERMINAL_STATUSES for did in doc_ids)
        if all_done:
            yield _sse_event(
                {
                    "documents": [doc.model_dump() for doc in doc_infos],
                    "message": "Processing complete",
                },
                event="done",
            )
            return

    # Timeout reached — tell the client to poll on their own.
    yield _sse_event({"message": "Stream timeout — poll /documents for status"}, event="timeout")


async def _sse_done_immediately(message: str) -> AsyncGenerator[str, None]:
    """Yield a single SSE 'done' event for requests that need no processing."""
    yield _sse_event({"documents": [], "message": message}, event="done")


@router.get("/projects/{project_id}/documents", response_model=DocumentListResponse)
async def list_documents_for_project(
    project_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """List documents for a specific project."""
    try:
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            _executor,
            model_service.get_documents_for_project,
            project_id,
        )
        return DocumentListResponse(documents=[DocumentResponse(**d) for d in docs])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error listing documents for project")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/{project_id}/documents",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE stream of document processing progress",
            "content": {
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "example": (
                            "event: queued\ndata: {\"documents\": [...]}\n\n"
                            "event: progress\ndata: {\"documents\": [...]}\n\n"
                            "event: done\ndata: {\"documents\": [...], \"message\": \"Processing complete\"}\n\n"
                        ),
                    }
                }
            },
        }
    },
)
async def add_document_to_project(
    project_id: str,
    request: AddDocumentRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """Add a document source to a specific project and stream progress via SSE.

    Returns a text/event-stream response with the same events as POST /documents.
    """
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: model_service.add_document_async(request.source, project_id=project_id, cwd=request.cwd, generate_filename=request.generate_filename),
        )
        doc_ids = [d["id"] for d in result.get("documents", [])]
        if not doc_ids:
            return StreamingResponse(
                _sse_done_immediately(result.get("message", "Document(s) already exist")),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        return StreamingResponse(
            _stream_document_progress(doc_ids, model_service, project_id=project_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error adding document to project")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/documents/upload")
async def upload_document_file(
    project_id: str,
    file: UploadFile = File(...),
    cwd: str = Form(...),
    model_service: ModelService = Depends(get_model_service),
):
    """Upload a file directly into the project's documents folder.

    The file is dropped into ``<cwd>/documents/`` and the indexer picks it
    up automatically within a few seconds.  Returns immediately with
    ``status=accepted``; poll ``GET /v1/projects/{project_id}/documents``
    to track indexing progress.
    """
    try:
        file_bytes = await file.read()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: model_service.upload_document_to_inbox(
                project_id, file.filename or "upload", file_bytes, cwd=cwd
            ),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error uploading document file")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/projects/{project_id}/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document_from_project(
    project_id: str,
    document_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a document by ID from a specific project."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            model_service.delete_document,
            document_id,
            project_id,
        )
        return DeleteDocumentResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting document from project")
        raise HTTPException(status_code=500, detail=str(e))
