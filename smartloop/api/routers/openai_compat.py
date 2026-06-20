"""OpenAI-compatible passthrough endpoint for the local vision model.

MarkItDown (and the document parser's scanned-PDF OCR fallback) talk to this
endpoint to have the vision-capable model describe / transcribe images.  The
backing model is selected by the ``"vision"`` capability exactly like the
orchestrator (``ModelService.get_vision_llm``), so whichever model the user has
tagged vision-capable is used automatically.

This is mounted under ``/vision/v1`` to avoid clashing with the orchestrator's
``/v1/chat/completions`` (which runs the full agent pipeline, not a raw image
transcription).
"""

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from smartloop.api.services import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision/v1", tags=["vision"])

# Single worker: the underlying llama.cpp model is not reentrant; the
# ModelService inference semaphore further bounds concurrency.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-ocr")


class VisionChatRequest(BaseModel):
    """Subset of the OpenAI chat-completions request we honor.

    ``messages`` content may be a plain string or a list of OpenAI content
    parts (``{"type": "text", ...}`` / ``{"type": "image_url", ...}``); we pass
    it through to llama.cpp's vision chat handler unchanged.
    """
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    stream: bool = False


def _resolve_vision_llm(model_service: ModelService):
    llm = model_service.get_vision_llm()
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="No vision-capable model available (none downloaded/enabled or vision disabled).",
        )
    return llm


def _run_completion(model_service: ModelService, llm, req: VisionChatRequest) -> Dict[str, Any]:
    """Run a blocking vision chat completion under the inference semaphore."""
    max_tokens = req.max_tokens or 2048
    with model_service._inference_lock:
        return llm.model.create_chat_completion(
            messages=req.messages,
            max_tokens=max_tokens,
            temperature=req.temperature,
            stream=False,
        )


@router.get("/health")
async def vision_health(model_service: ModelService = Depends(get_model_service)):
    """Report whether a usable multimodal vision model is available.

    Triggers the lazy load (which is needed for OCR anyway) and returns
    ``available: false`` fast when there's no vision model or no mmproj — so the
    parser can skip the per-page OCR loop entirely instead of failing per page.
    """
    loop = asyncio.get_event_loop()
    try:
        llm = await loop.run_in_executor(_executor, model_service.get_vision_llm)
    except Exception as e:
        logger.warning(f"Vision health check failed: {e}")
        llm = None
    return {"available": llm is not None}


async def _stream_completion(model_service: ModelService, llm, req: VisionChatRequest):
    """Yield SSE chunks by running the blocking stream in the executor."""
    max_tokens = req.max_tokens or 2048
    loop = asyncio.get_event_loop()

    def _produce() -> List[Dict[str, Any]]:
        with model_service._inference_lock:
            return list(llm.model.create_chat_completion(
                messages=req.messages,
                max_tokens=max_tokens,
                temperature=req.temperature,
                stream=True,
            ))

    chunks = await loop.run_in_executor(_executor, _produce)
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
async def create_vision_completion(
    request: VisionChatRequest,
    model_service: ModelService = Depends(get_model_service),
):
    """OpenAI-compatible chat completion backed by the local vision model."""
    llm = _resolve_vision_llm(model_service)

    if request.stream:
        return StreamingResponse(
            _stream_completion(model_service, llm, request),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_completion, model_service, llm, request
        )
    except Exception as e:
        logger.error(f"Vision completion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vision completion failed: {e}")

    # llama.cpp already returns an OpenAI-shaped response; backfill an id/model
    # for clients that expect them.
    result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
    result.setdefault("object", "chat.completion")
    result.setdefault("created", int(time.time()))
    result.setdefault("model", request.model or "vision")
    return result
