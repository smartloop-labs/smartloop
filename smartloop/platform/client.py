"""HTTP client for the Smartloop platform API (api.smartloop.ai).

Handles only JSON control-plane calls — no file downloading.
"""

import logging
import platform as _platform
from typing import Callable, Optional

import httpx

from smartloop.platform import auth as platform_auth

logger = logging.getLogger(__name__)

# URL path constants
_BOOTSTRAP_MODELS = "/v1/bootstrap/models"
_MODEL_DOWNLOAD = "/v1/models/{}/download"
_USAGE_TOKENS = "/v1/usage/tokens"
_USERS_ME = "/v1/users/me"
_JWT_REFRESH = "/v1/auth/jwt/refresh"
_MODEL_UPLOAD = "/v1/models/{}/upload"
_MODELS = "/v1/models"
_RELEASES_LATEST = "/v1/releases/latest"
_MCP_CATALOG = "/v1/mcp-catalog"
_WEBSEARCH = "/v1/websearch"
_WEBSEARCH_STATUS = "/v1/websearch/status"

_TIMEOUT = 30

#: Web search runs in front of a chat turn, so it gets its own budgets rather
#: than the 30s control-plane default: a search the user is waiting on, and a
#: status check that gates every query and fails open if it cannot finish.
_SEARCH_TIMEOUT = 15
_SEARCH_STATUS_TIMEOUT = 5


def _user_agent() -> str:
    """Identify this client to the API, e.g. ``smartloop-cli/1.2.3 (darwin)``."""
    from smartloop import __version__

    return f"smartloop-cli/{__version__} ({_platform.system().lower()})"


def _raise_for_refused_search(resp: httpx.Response) -> None:
    """Raise ``PermissionError`` if the search endpoint refused the request.

    Only 401/403 means refused. ``_raise_for_status`` would map every failure
    to whatever class it is given, which would turn a 500 into "search is off"
    and get that cached — so the auth statuses are singled out here and the
    rest fall through to a ``ConnectionError``.
    """
    if resp.status_code in (401, 403):
        raise PermissionError(
            f"hosted endpoint refused the request (HTTP {resp.status_code})"
        )


def _raise_for_status(resp: httpx.Response, error_map: dict[int, str] | None = None,
                      error_cls: type[Exception] = ConnectionError) -> None:
    if resp.is_success:
        return
    if error_map and resp.status_code in error_map:
        raise error_cls(error_map[resp.status_code])
    raise error_cls(f"API error ({resp.status_code}): {resp.reason_phrase}")


class SLPPlatformClient:
    """Thin HTTP client for Smartloop platform API calls.

    Parameters
    ----------
    base_url : str
        API base URL, e.g. ``https://api.smartloop.ai``.
    auth_headers_fn : callable, optional
        Returns auth headers dict. Defaults to ``platform_auth.auth_headers``.
    """

    def __init__(
        self,
        base_url: str,
        auth_headers_fn: Callable[[], dict[str, str]] = platform_auth.auth_headers,
    ):
        self._base_url = base_url.rstrip("/")
        self._auth_headers_fn = auth_headers_fn

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def list_bootstrap_models(self) -> list[dict]:
        """GET /v1/bootstrap/models (no auth). Returns the array of model entries."""
        try:
            resp = httpx.get(self._url(_BOOTSTRAP_MODELS), timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {404: "Bootstrap endpoint not found on server."})

        data = resp.json()
        if not data:
            raise ValueError("Bootstrap API returned no models")
        return data

    def list_models(self) -> list[dict]:
        """GET /v1/models (auth optional). Returns list of model entries."""
        try:
            resp = httpx.get(self._url(_MODELS), headers=self._auth_headers_fn(),
                             timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code == 401:
            try:
                resp = httpx.get(self._url(_MODELS), timeout=_TIMEOUT)
            except httpx.TransportError as e:
                raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp)
        return resp.json()

    def get_latest_release(self, channel: str = "stable") -> Optional[dict]:
        """GET /v1/releases/latest (no auth). Returns the LatestReleaseResponse dict.

        We poll WITHOUT ``current_version`` so the cached payload is
        client-agnostic; update_available/update_required are computed locally
        per request.  Returns the full wrapper ``{update_available, update_required,
        release}``; callers use the ``release`` field (which may be ``None`` when
        no published release exists for the channel).
        """
        try:
            resp = httpx.get(self._url(_RELEASES_LATEST),
                             params={"channel": channel}, timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp)
        return resp.json()

    def list_mcp_catalog(self) -> list[dict]:
        """GET /v1/mcp-catalog (auth required). Returns list of curated MCP server catalog entries."""
        try:
            resp = httpx.get(self._url(_MCP_CATALOG), headers=self._auth_headers_fn(),
                             timeout=_TIMEOUT)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Check your SLP_DEVELOPER_TOKEN.",
        }, PermissionError)
        return resp.json()

    def get_model_artifacts(self, model_name: str) -> dict[str, str]:
        """GET /v1/models/{model_name}/download (auth required).

        Returns ``{filename: signed_url}`` map.
        """
        try:
            resp = httpx.get(self._url(_MODEL_DOWNLOAD.format(model_name)),
                             headers=self._auth_headers_fn(), timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Check your SLP_DEVELOPER_TOKEN.",
            403: f"Model '{model_name}' is not available on your current plan. Please upgrade to access this model.",
            404: f"Model '{model_name}' not found on server.",
            500: f"Server error while fetching model '{model_name}' artifacts."
        }, PermissionError)

        artifacts = resp.json().get("artifacts", {})
        if not artifacts:
            raise ValueError(f"API returned no artifacts for model '{model_name}'")
        return artifacts

    def get_current_user(self) -> dict:
        """GET /v1/users/me (auth required). Returns user profile dict."""
        try:
            resp = httpx.get(self._url(_USERS_ME), headers=self._auth_headers_fn(),
                             timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Token may be expired or invalid.",
        }, PermissionError)
        return resp.json()

    def update_current_user(self, payload: dict) -> dict:
        """PATCH /v1/users/me (auth required). Returns updated user profile dict."""
        try:
            resp = httpx.patch(self._url(_USERS_ME), headers=self._auth_headers_fn(),
                               json=payload, timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Token may be expired or invalid.",
        }, PermissionError)
        return resp.json()

    def refresh_access_token(self, refresh_token: str) -> dict:
        """POST /v1/auth/jwt/refresh with a refresh token.

        Returns dict with ``access_token`` and optionally ``refresh_token``.
        """
        try:
            resp = httpx.post(self._url(_JWT_REFRESH),
                              json={"refresh_token": refresh_token}, timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Refresh token may be expired or revoked.",
        }, PermissionError)

        data = resp.json()
        if not data.get("access_token"):
            raise ValueError("Refresh response missing access_token")
        return data

    def get_model_upload_url(self, model_name: str, filename: str) -> dict:
        """POST /v1/models/{name}/upload (auth required). Returns {url, blob_path, expires_at}."""
        try:
            resp = httpx.post(self._url(_MODEL_UPLOAD.format(model_name)),
                              headers=self._auth_headers_fn(),
                              json={"filename": filename}, timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Check your SLP_DEVELOPER_TOKEN.",
            404: "Model upload endpoint not found on server.",
        }, PermissionError)
        return resp.json()

    def push_token_usage(self, record: dict, timeout: int = 10) -> None:
        """POST /v1/usage/tokens (auth required). Fire-and-forget telemetry."""
        try:
            resp = httpx.post(self._url(_USAGE_TOKENS), headers=self._auth_headers_fn(),
                              json=record, timeout=timeout)
            _raise_for_status(resp)
        except Exception as e:
            logger.error("Failed to push token usage: %s", e)

    # -------------------------------------------------------------- web search

    def _search_headers(self) -> dict[str, str]:
        """Auth headers plus this client's identity, for the search endpoints.

        The search endpoints record each call in the account's access log, and
        httpx's default agent ("python-httpx/x.y") would make every entry look
        like an anonymous script; naming ourselves lets the owner tell their
        own agent apart from anything else using their token. The other calls
        on this client don't send it yet.
        """
        return {
            **self._auth_headers_fn(),
            "Accept": "application/json",
            "User-Agent": _user_agent(),
        }

    def search_web(self, query: str, count: int) -> list:
        """GET /v1/websearch (auth required). Returns the raw JSON body.

        The endpoint resolves the account's registered search provider
        server-side; this client never sees a provider name or an API key.
        Raises ``PermissionError`` on 401/403 — meaning search is unavailable
        for this account — and ``ConnectionError`` for anything else.
        """
        try:
            resp = httpx.get(self._url(_WEBSEARCH), headers=self._search_headers(),
                             params={"q": query, "count": count},
                             timeout=_SEARCH_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_refused_search(resp)
        _raise_for_status(resp)
        return resp.json()

    def get_web_search_status(self) -> dict:
        """GET /v1/websearch/status (auth required). Returns ``{"enabled": bool}``.

        Answers for the whole account: signed in, with a provider registered.
        Raises ``PermissionError`` on 401/403 and ``ConnectionError``
        otherwise, so a caller can tell "search is off" from "could not ask".
        """
        try:
            resp = httpx.get(self._url(_WEBSEARCH_STATUS),
                             headers=self._search_headers(),
                             timeout=_SEARCH_STATUS_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_refused_search(resp)
        _raise_for_status(resp)
        return resp.json()
