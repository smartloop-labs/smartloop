"""Web search client for the WebSearchAgent.

Search runs against Smartloop's hosted ``/v1/websearch`` endpoint, which runs
the query server-side on Smartloop's own credentials. Providers used to be
configured here — a Brave API key in the local credential store — but that tied
web search to one machine and left the hosted API unable to tell whether search
was usable. Nothing is configured on this side any more: it asks for links and
treats a 401/403 (unavailable) or 429 (the month's allowance is spent) as
"search is off".

The HTTP itself belongs to :class:`~smartloop.platform.client.SLPPlatformClient`
like every other platform call; what lives here is the search-specific policy —
when a failure means "off" versus "unknown", and how long a status answer is
trusted.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

#: How long a status answer is trusted. The orchestrator asks once per query,
#: so an uncached check would put a network round-trip in front of every turn;
#: a provider being connected or removed is not something that needs to be
#: noticed within the minute.
_STATUS_TTL_SECONDS = 60.0

_status_cache: Optional[Tuple[float, bool]] = None
_status_lock = threading.Lock()


def _platform_client():
    """Return a platform client bound to the configured API base."""
    from smartloop.config.app_settings import AppSettings
    from smartloop.platform.client import SLPPlatformClient

    return SLPPlatformClient(AppSettings().slp_api_base)


class SearchProviderError(Exception):
    """Base class for web-search failures."""


class SearchAuthError(SearchProviderError):
    """The endpoint refused the request (401/403).

    Means web search is unavailable for this account — search is not
    configured on the server, or the month's allowance is spent. Either way
    the agent silently skips searching.
    """


class NoSearchCredentials(SearchProviderError):
    """No Smartloop credentials available to call the endpoint."""


class HostedSearchProvider:
    """Smartloop's hosted ``/v1/websearch`` endpoint."""

    name = "hosted"

    def search(self, query: str, count: int) -> List[str]:
        """Return up to *count* result URLs for *query*."""
        from smartloop.platform import auth as platform_auth

        if not platform_auth.has_credentials():
            raise NoSearchCredentials(
                "no Smartloop credentials available to call the hosted endpoint"
            )

        try:
            links = _platform_client().web_search(query, max(1, min(count, 20)))
        except PermissionError as e:
            raise SearchAuthError(str(e)) from e

        logger.info("[search] hosted endpoint returned %d link(s) for %r", len(links), query)
        return links


def resolve_provider() -> HostedSearchProvider:
    """Return the search client. Always the hosted endpoint."""
    return HostedSearchProvider()


def hosted_search_available(ttl: float = _STATUS_TTL_SECONDS) -> bool:
    """Whether the hosted endpoint would actually serve a search.

    Asks ``GET /v1/websearch/status``, which answers for the whole account:
    signed in, with searches left this month. Checking it up front means the
    orchestrator skips the search step outright rather than running the query
    normalizer, calling the endpoint and discarding a 403.

    Fails open. A status check that cannot complete says nothing about whether
    search works, and the search itself already degrades safely — it returns
    "" on a 403. Refusing to search because a health check timed out would be
    the worse failure. Only an explicit 401/403 (``PermissionError``) is taken
    as a real "no".
    """
    global _status_cache

    now = time.monotonic()
    with _status_lock:
        if _status_cache is not None and now - _status_cache[0] < ttl:
            return _status_cache[1]

    from smartloop.platform import auth as platform_auth

    if not platform_auth.has_credentials():
        with _status_lock:
            _status_cache = (now, False)
        return False

    try:
        enabled = _platform_client().web_search_status()
    except PermissionError:
        enabled = False
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
