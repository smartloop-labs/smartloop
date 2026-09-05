"""Users router – user profile from the Smartloop platform API.

Endpoints:
    GET  /v1/users/me – fetch authenticated user's full profile
    PATCH /v1/users/me – update mutable profile fields (default_temperature, default_memory)
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from smartloop.platform import PlatformService, get_platform_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/users", tags=["users"])


class UserRead(BaseModel):
    """User profile returned by the platform API."""

    email: str
    name: Optional[str] = None

    is_premium: Optional[bool] = False
    is_paid: Optional[bool] = False
    number_of_tokens_processed: Optional[int] = 0
    default_temperature: float = 0.3
    default_memory: bool = True


class UserUpdate(BaseModel):
    """Mutable user profile fields."""

    default_temperature: Optional[float] = None
    default_memory: Optional[bool] = None


def _user_read(data: dict) -> UserRead:
    return UserRead(
        email=data["email"],
        name=data.get("name"),
        is_premium=data.get("is_premium", False),
        is_paid=data.get("is_paid", False),
        number_of_tokens_processed=data.get("number_of_tokens_processed", 0),
        default_temperature=data.get("default_temperature", 0.3),
        default_memory=data.get("default_memory", True),
    )


@router.get("/me", response_model=UserRead)
    # Sync on purpose: the platform client is synchronous, and a 401 now costs
    # a refresh and a retry on top of the original call.  On the event loop
    # that stalls every in-flight chat stream; declared sync, FastAPI runs it
    # in the threadpool.  See ``auth_status`` in the auth router.
def get_me(svc: PlatformService = Depends(get_platform_service)):
    """Return the authenticated user's profile from the platform API."""
    try:
        data = svc.get_current_user()
    except PermissionError:
        raise HTTPException(status_code=401, detail="Not authenticated") from None
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable") from None

    return _user_read(data)


@router.patch("/me", response_model=UserRead)
    # Sync on purpose: the platform client is synchronous, and a 401 now costs
    # a refresh and a retry on top of the original call.  On the event loop
    # that stalls every in-flight chat stream; declared sync, FastAPI runs it
    # in the threadpool.  See ``auth_status`` in the auth router.
def update_me(body: UserUpdate, svc: PlatformService = Depends(get_platform_service)):
    """Update the authenticated user's profile via the platform API."""
    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail="No fields to update")

    try:
        data = svc.update_current_user(payload)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Not authenticated") from None
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable") from None

    return _user_read(data)
