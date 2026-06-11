"""Feedback router for thumbs up/down on conversation responses."""

import logging
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query

from smartloop.api.model import FeedbackRequest, FeedbackResponse
from smartloop.api.services import ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["feedback"])

@router.get("/feedback")
async def get_feedback(
    session_id: str = Query(..., description="Session ID to load feedback for"),
    project_id: str | None = Query(default=None, description="Project ID"),
    model_service: ModelService = Depends(get_model_service),
) -> Dict[str, str]:
    """Return a map of message_id -> rating for a session."""
    try:
        return model_service.get_feedback_ratings(session_id, project_id)
    except ValueError:
        # Model or project not loaded yet — return empty feedback
        return {}
    except Exception as e:
        logger.exception("Error loading feedback")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    model_service: ModelService = Depends(get_model_service),
):
    try:
        result = model_service.submit_feedback(
            session_id=request.session_id,
            message_id=request.message_id,
            project_id=request.project_id,
            rating=request.rating,
        )
        return FeedbackResponse(**result)
    except ValueError as e:
        logger.warning(f"[feedback] Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Error submitting feedback")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/feedback")
async def delete_feedback(
    session_id: str = Query(..., description="Session ID to delete feedback for"),
    message_id: str = Query(..., description="Message ID to delete feedback for"),
    project_id: str | None = Query(default=None, description="Project ID"),
    model_service: ModelService = Depends(get_model_service),
):
    """Delete feedback ratings for a message ID."""
    try:
        model_service.delete_feedback_ratings(session_id, message_id, project_id)
        return {"detail": "Feedback deleted successfully"}
    except Exception as e:
        logger.exception("Error deleting feedback")
        raise HTTPException(status_code=500, detail=str(e))