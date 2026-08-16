"""Pluggable web search providers for the WebSearchAgent.

A user-registered provider (e.g. Brave) is used when configured; otherwise the
agent falls back to the hosted Smartloop endpoint.  Registration lives in the
encrypted credential store (or env vars) — see
``smartloop.auth.credential_store``.

Providers:
  - ``brave``   — Brave Search API (https://api.search.brave.com)
  - ``hosted``  — Smartloop's hosted ``/v1/websearch`` endpoint (default)
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


class SearchProviderError(Exception):
    """Base class for web-search provider failures."""


class SearchAuthError(SearchProviderError):
    """Provider rejected the credentials (401/403) or the API key is invalid."""


class NoSearchCredentials(SearchProviderError):
    """No credentials available to call the provider."""


class SearchProvider(ABC):
    """Base class for a web search backend.  ``search`` returns result URLs."""

    name: str = ""

    @abstractmethod
    def search(self, query: str, count: int) -> List[str]:
        """Return up to *count* result URLs for *query*.

        Raises ``SearchAuthError`` when the provider rejects the configured
        token, and ``NoSearchCredentials`` when none is available.
        """


class BraveSearchProvider(SearchProvider):
    """Search via the Brave Search API using a user-registered API key."""

    name = "brave"
    _ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(self, query: str, count: int) -> List[str]:
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params = {"q": query, "count": max(1, min(count, 20))}
        with httpx.Client(timeout=15) as client:
            resp = client.get(self._ENDPOINT, headers=headers, params=params)
        if resp.status_code in (401, 403):
            raise SearchAuthError(
                f"Brave Search rejected the API key (HTTP {resp.status_code})"
            )
        resp.raise_for_status()
        data = resp.json()
        results = (data.get("web") or {}).get("results") or []
        links = [r.get("url") for r in results if r.get("url")]
        logger.info("[search] brave returned %d link(s) for %r", len(links), query)
        return links


class HostedSearchProvider(SearchProvider):
    """Fallback: the hosted Smartloop ``/v1/websearch`` endpoint.

    Gated by the account's plan — a 401/403 means the account isn't on a plan
    that includes web search.
    """

    name = "hosted"
    _PATH = "/v1/websearch"

    def search(self, query: str, count: int) -> List[str]:
        from smartloop.auth import credential_store
        from smartloop.config.app_settings import AppSettings

        token = credential_store.get_active_token()
        if not token:
            raise NoSearchCredentials(
                "no Smartloop credentials available to call the hosted endpoint"
            )
        base = AppSettings().slp_api_base.rstrip("/")
        url = f"{base}{self._PATH}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        params = {"q": query, "count": max(1, min(count, 20))}
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers, params=params)
        if resp.status_code in (401, 403):
            raise SearchAuthError(
                f"hosted endpoint rejected the token (HTTP {resp.status_code})"
            )
        resp.raise_for_status()
        data = resp.json()
        links = (
            [u for u in data if isinstance(u, str) and u.strip()]
            if isinstance(data, list)
            else []
        )
        logger.info("[search] hosted endpoint returned %d link(s) for %r", len(links), query)
        return links


#: Registered provider name → factory.  ``None`` from resolution = hosted.
_PROVIDER_FACTORIES = {
    "brave": lambda token: BraveSearchProvider(token),
}


def supported_providers() -> List[str]:
    """Names of the user-registrable providers (excludes the hosted fallback)."""
    return sorted(_PROVIDER_FACTORIES)


def _env_provider() -> Optional[Tuple[str, str]]:
    """Resolve the provider from environment variables, if any.

    ``SLP_SEARCH_PROVIDER=brave`` with the token in ``SLP_BRAVE_API_KEY`` (or
    the generic ``SLP_SEARCH_PROVIDER_TOKEN``).
    """
    name = os.environ.get("SLP_SEARCH_PROVIDER", "").strip().lower()
    if not name:
        return None
    token = os.environ.get(f"SLP_{name.upper()}_API_KEY", "") or os.environ.get(
        "SLP_SEARCH_PROVIDER_TOKEN", ""
    )
    return (name, token.strip())


def _stored_provider() -> Optional[Tuple[str, str]]:
    """Resolve the provider from the encrypted credential store."""
    from smartloop.auth import credential_store

    return credential_store.get_search_provider_config()


def get_search_provider() -> Optional[SearchProvider]:
    """Resolve the configured search provider.

    Resolution order: env vars first, then the encrypted credential store.
    Returns ``None`` when no provider/token is configured (the caller falls
    back to the hosted endpoint) or the provider name is unknown.
    """
    config = _env_provider() or _stored_provider()
    if not config:
        return None
    name, token = config
    if not name or not token:
        return None
    factory = _PROVIDER_FACTORIES.get(name.lower())
    if factory is None:
        logger.warning(
            "[search] unknown provider %r (supported: %s) — falling back to hosted",
            name, ", ".join(supported_providers()) or "none",
        )
        return None
    try:
        return factory(token)
    except Exception as e:
        logger.warning("[search] provider %r init failed (%s) — falling back to hosted", name, e)
        return None


def resolve_provider() -> SearchProvider:
    """Return the configured provider, or the hosted fallback."""
    return get_search_provider() or HostedSearchProvider()
