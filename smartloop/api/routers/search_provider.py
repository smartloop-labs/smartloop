"""Search provider registration router.

Users register a web search provider (e.g. Brave) and its API token so the
WebSearchAgent can search directly through that provider instead of the hosted
endpoint.  Tokens are stored encrypted in the credential store.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from smartloop.auth import credential_store
from smartloop.search.providers import get_search_provider, supported_providers

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["search"])


class SearchProviderRequest(BaseModel):
    provider: str
    token: str


class SearchProviderResponse(BaseModel):
    provider: Optional[str] = None
    configured: bool = False
    supported: list[str]


def _active_provider() -> Optional[str]:
    provider = get_search_provider()
    return provider.name if provider is not None else None


@router.get("/search/provider", response_model=SearchProviderResponse)
async def get_provider_config():
    """Return the active search provider (or None) and supported options."""
    return SearchProviderResponse(
        provider=_active_provider(),
        configured=_active_provider() is not None,
        supported=supported_providers(),
    )


@router.put("/search/provider", response_model=SearchProviderResponse)
async def register_provider(request: SearchProviderRequest):
    """Register a search provider and its API token (stored encrypted).

    ``provider`` must be one of the supported names (e.g. ``brave``).  The
    token is never returned by the API afterwards — only the configured state.
    """
    name = (request.provider or "").strip().lower()
    if name not in supported_providers():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported search provider {request.provider!r}. "
                f"Supported: {', '.join(supported_providers())}."
            ),
        )
    token = (request.token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    credential_store.set_search_provider_config(name, token)
    logger.info("[search] registered provider %r", name)
    return SearchProviderResponse(
        provider=_active_provider(),
        configured=True,
        supported=supported_providers(),
    )


@router.delete("/search/provider", response_model=SearchProviderResponse)
async def clear_provider():
    """Remove the registered search provider and token."""
    credential_store.clear_search_provider_config()
    logger.info("[search] cleared search provider registration")
    return SearchProviderResponse(
        provider=_active_provider(),
        configured=_active_provider() is not None,
        supported=supported_providers(),
    )
