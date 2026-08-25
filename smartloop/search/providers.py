"""Web search client for the WebSearchAgent.

Search runs against Smartloop's hosted ``/v1/websearch`` endpoint, which
resolves the account's registered provider server-side. Providers used to be
registered here — a Brave API key in the local credential store — but that tied
web search to one machine and left the hosted API unable to tell whether search
was usable. Registration now lives on the account (managed from the web app),
so this side stays deliberately blind: it asks for links and treats a 401/403
as "search is off".
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

#: How long a status answer is trusted. The orchestrator asks once per query,
#: so an uncached check would put a network round-trip in front of every turn;
#: a provider being connected or removed is not something that needs to be
#: noticed within the minute.
_STATUS_TTL_SECONDS = 60.0

_status_cache: Optional[Tuple[float, bool]] = None
_status_lock = threading.Lock()


def _user_agent() -> str:
    """Identify this client to the API.

    The hosted endpoint records each call in the account's access log, and
    httpx's default agent ("python-httpx/x.y") would make every entry look
    like an anonymous script. Naming ourselves lets the owner tell their own
    agent apart from anything else using their token.
    """
    from smartloop import __version__

    return f"smartloop-cli/{__version__} ({platform.system().lower()})"


class SearchProviderError(Exception):
    """Base class for web-search failures."""


class SearchAuthError(SearchProviderError):
    """The endpoint refused the request (401/403).

    Means web search is unavailable for this account — the plan doesn't
    include it, or no search provider has been registered. Either way the
    agent silently skips searching.
    """


class NoSearchCredentials(SearchProviderError):
    """No Smartloop credentials available to call the endpoint."""


class HostedSearchProvider:
    """Smartloop's hosted ``/v1/websearch`` endpoint."""

    name = "hosted"
    _PATH = "/v1/websearch"

    def search(self, query: str, count: int) -> List[str]:
        """Return up to *count* result URLs for *query*."""
        from smartloop.auth import credential_store
        from smartloop.config.app_settings import AppSettings

        token = credential_store.get_active_token()
        if not token:
            raise NoSearchCredentials(
                "no Smartloop credentials available to call the hosted endpoint"
            )
        base = AppSettings().slp_api_base.rstrip("/")
        url = f"{base}{self._PATH}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        }
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


def resolve_provider() -> HostedSearchProvider:
    """Return the search client. Always the hosted endpoint."""
    return HostedSearchProvider()


def _status_path() -> str:
    from smartloop.config.app_settings import AppSettings

    return f"{AppSettings().slp_api_base.rstrip('/')}/v1/websearch/status"


def hosted_search_available(ttl: float = _STATUS_TTL_SECONDS) -> bool:
    """Whether the hosted endpoint would actually serve a search.

    Asks ``GET /v1/websearch/status``, which answers for the whole account:
    signed in, and a provider registered. Checking it up front means the
    orchestrator skips the search step outright rather than running the query
    normalizer, calling the endpoint and discarding a 403.

    Fails open. A status check that cannot complete says nothing about whether
    search works, and the search itself already degrades safely — it returns
    "" on a 403. Refusing to search because a health check timed out would be
    the worse failure.
    """
    global _status_cache

    now = time.monotonic()
    with _status_lock:
        if _status_cache is not None and now - _status_cache[0] < ttl:
            return _status_cache[1]

    from smartloop.auth import credential_store

    token = credential_store.get_active_token()
    if not token:
        with _status_lock:
            _status_cache = (now, False)
        return False

    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(
                _status_path(),
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "User-Agent": _user_agent(),
                },
            )
        if resp.status_code in (401, 403):
            enabled = False
        else:
            resp.raise_for_status()
            enabled = bool(resp.json().get("enabled"))
    except Exception as e:
        logger.info("[search] status check failed (%s) — assuming available", e)
        return True

    with _status_lock:
        _status_cache = (now, enabled)
    if not enabled:
        logger.info("[search] hosted endpoint reports web search unavailable")
    return enabled


def reset_status_cache() -> None:
    """Forget the cached status — used by tests and after re-authenticating."""
    global _status_cache
    with _status_lock:
        _status_cache = None
