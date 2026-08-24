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
from typing import List

import httpx

logger = logging.getLogger(__name__)


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
