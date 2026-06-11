"""Users router – user profile from the Smartloop platform API.

Endpoints:
    GET  /v1/users/me – fetch authenticated user's full profile
    PATCH /v1/users/me – update user profile fields (e.g. selected_model)
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from smartloop.config.app_settings import AppSettings
from smartloop.platform.client import SLPPlatformClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/users", tags=["users"])


class UserRead(BaseModel):
    """User profile returned by the platform API."""

    email: str
    name: Optional[str] = None

    is_premium: Optional[bool] = False
    is_paid: Optional[bool] = False
    number_of_tokens_processed: Optional[int] = 0
    selected_model: Optional[str] = None
    default_temperature: float = 0.3
    default_memory: bool = True


class UserUpdate(BaseModel):
    """Mutable user profile fields."""

    selected_model: Optional[str] = None
    default_temperature: Optional[float] = None
    default_memory: Optional[bool] = None


@router.get("/me", response_model=UserRead)
async def get_me():
    """Return the authenticated user's profile from the platform API."""
    settings = AppSettings()
    client = SLPPlatformClient(settings.slp_api_base)

    try:
        data = client.get_current_user()
    except PermissionError:
        raise HTTPException(status_code=401, detail="Not authenticated") from None
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable") from None

    return UserRead(
        email=data["email"],
        name=data.get("name"),
        is_premium=data.get("is_premium", False),
        is_paid=data.get("is_paid", False),
        number_of_tokens_processed=data.get("number_of_tokens_processed", 0),
        selected_model=data.get("selected_model"),
        default_temperature=data.get("default_temperature", 0.3),
        default_memory=data.get("default_memory", True),
    )


@router.patch("/me", response_model=UserRead)
async def update_me(body: UserUpdate):
    """Update the authenticated user's profile via the platform API."""
    settings = AppSettings()
    client = SLPPlatformClient(settings.slp_api_base)

    payload = body.model_dump(exclude_none=True)
    if not payload:
        raise HTTPException(status_code=422, detail="No fields to update")

    try:
        data = client.update_current_user(payload)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Not authenticated") from None
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable") from None

    return UserRead(
        email=data["email"],
        name=data.get("name"),
        is_premium=data.get("is_premium", False),
        is_paid=data.get("is_paid", False),
        number_of_tokens_processed=data.get("number_of_tokens_processed", 0),
        selected_model=data.get("selected_model"),
        default_temperature=data.get("default_temperature", 0.3),
        default_memory=data.get("default_memory", True),
    )

