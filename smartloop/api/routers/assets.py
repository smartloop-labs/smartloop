"""Asset management router — upload files and reference them in chat messages."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from smartloop.api.model import AssetUploadResponse, AssetResponse, DeleteAssetResponse
from smartloop.api.services import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["assets"])

_executor = ThreadPoolExecutor(max_workers=2)


@router.post("/assets", response_model=AssetUploadResponse)
async def upload_asset(
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
):
    """Upload a file as a reusable asset.

    Supported document types (pdf, csv, docx, pptx, xlsx, txt) are automatically
    converted to markdown and stored for efficient injection into chat context.
    Other file types are stored as-is.

    Returns an ``asset_id`` that can be referenced in
    ``message.attachments`` of a chat completion request.
    """
    try:
        file_bytes = await file.read()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            lambda: model_service.upload_asset(
                file_bytes,
                file.filename or "upload",
                file.content_type,
            ),
        )
        return AssetUploadResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        logger.exception("Error uploading asset")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{asset_id}", response_model=AssetResponse)
async def get_asset(
    asset_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Retrieve asset metadata and content (markdown) by ID."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            model_service.get_asset,
            asset_id,
        )
        return AssetResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error retrieving asset")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/assets/{asset_id}", response_model=DeleteAssetResponse)
async def delete_asset(
    asset_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete an asset by ID."""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            model_service.delete_asset,
            asset_id,
        )
        return DeleteAssetResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Error deleting asset")
        raise HTTPException(status_code=500, detail=str(e))
