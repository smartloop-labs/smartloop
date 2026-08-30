"""HTTP client for the Smartloop platform API (api.smartloop.ai).

Handles only JSON control-plane calls — no file downloading.
"""

import logging
import platform
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

#: Web search sits in front of a user-visible turn, so it gets a tighter
#: budget than the control-plane calls: a slow search is worth abandoning,
#: and the status check runs ahead of every query.
_SEARCH_TIMEOUT = 15
_SEARCH_STATUS_TIMEOUT = 5


def _user_agent() -> str:
    """Identify this client to the API.

    The platform records each call in the account's access log, and httpx's
    default agent ("python-httpx/x.y") would make every entry look like an
    anonymous script. Naming ourselves lets the owner tell their own agent
    apart from anything else using their token.

    Deliberately not "smartloop-cli": these calls come from the local API
    server, background agents and the downloader as much as from the CLI. What
    the log needs from this header is the product and version — *which* caller
    it was is already the path on the same log line.
    """
    from smartloop import __version__

    return f"smartloop/{__version__} ({platform.system().lower()})"


class ModelAccessError(PermissionError):
    """The account's plan does not grant this model (HTTP 403).

    A subclass of ``PermissionError`` so existing handlers keep catching it,
    but distinguishable from a 401: a 401 means "sign in", a 403 means the
    caller (signed in or anonymous) reached the endpoint and the model is not
    on their plan.  Callers map it to an HTTP 403 so the UI can offer an
    upgrade instead of a login.
    """


def _raise_for_status(resp: httpx.Response, error_map: dict[int, str] | None = None,
                      error_cls: type[Exception] = ConnectionError) -> None:
    """Raise *error_cls* for a non-2xx response, tagged with the upstream status.

    The status is attached as ``status_code`` on the exception so callers can
    classify the failure without re-deriving it from the message.  The API
    router turns it back into an HTTP status for its own client, and a message
    is a bad channel for that: it is written for a person to read, and it gets
    reworded.
    """
    if resp.is_success:
        return
    if error_map and resp.status_code in error_map:
        message = error_map[resp.status_code]
    else:
        message = f"API error ({resp.status_code}): {resp.reason_phrase}"
    exc = error_cls(message)
    exc.status_code = resp.status_code
    raise exc


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

    def _headers(self, auth: Optional[dict[str, str]] = None) -> dict[str, str]:
        """Request headers. Every request identifies itself; *auth* is merged in.

        The endpoints that take no auth pass nothing rather than flipping a
        flag — they are a minority, and which of them is unauthenticated is
        worth seeing at the call site.
        """
        return {"User-Agent": _user_agent(), **(auth or {})}

    def list_bootstrap_models(self) -> list[dict]:
        """GET /v1/bootstrap/models (no auth). Returns the array of model entries."""
        try:
            resp = httpx.get(self._url(_BOOTSTRAP_MODELS),
                             headers=self._headers(), timeout=_TIMEOUT)
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
            resp = httpx.get(self._url(_MODELS), headers=self._headers(self._auth_headers_fn()),
                             timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code == 401:
            try:
                resp = httpx.get(self._url(_MODELS),
                                 headers=self._headers(), timeout=_TIMEOUT)
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
            resp = httpx.get(self._url(_RELEASES_LATEST), headers=self._headers(),
                             params={"channel": channel}, timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp)
        return resp.json()

    def list_mcp_catalog(self) -> list[dict]:
        """GET /v1/mcp-catalog (auth required). Returns list of curated MCP server catalog entries."""
        try:
            resp = httpx.get(self._url(_MCP_CATALOG), headers=self._headers(self._auth_headers_fn()),
                             timeout=_TIMEOUT)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Check your SLP_DEVELOPER_TOKEN.",
        }, PermissionError)
        return resp.json()

    def get_model_artifacts(self, model_name: str) -> dict[str, str]:
        """GET /v1/models/{model_name}/download (auth optional).

        Returns ``{filename: signed_url}`` map.

        Auth headers are sent when we have them, but their absence is not an
        error: the platform serves free models to anonymous callers and answers
        403 for anything the caller's plan doesn't grant.  That 403 is raised as
        :class:`ModelAccessError` so callers can tell "not on your plan" apart
        from the 401 that means the token itself is bad.

        Failures are split by whether retrying could change the answer, because
        the two lead somewhere different.  ``ConnectionError`` covers the
        transient ones (429, 5xx): the download route is rate-limited per IP and
        anonymous downloads now reach that limit, so a 429 means "wait a moment",
        not "upgrade your plan".  ``PermissionError`` covers the verdicts about
        this caller and this model (401/403/404), which no amount of retrying
        fixes.  Every one of them carries ``status_code``.
        """
        try:
            resp = httpx.get(self._url(_MODEL_DOWNLOAD.format(model_name)),
                             headers=self._headers(self._auth_headers_fn()), timeout=_TIMEOUT)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code == 429 or resp.status_code >= 500:
            _raise_for_status(resp, {
                429: "Too many download requests. Please wait a moment and try again.",
                500: f"Server error while fetching model '{model_name}' artifacts.",
            }, ConnectionError)

        if resp.status_code == 403:
            exc = ModelAccessError(
                f"Model '{model_name}' is not available on your current plan. "
                "Please upgrade to access this model."
            )
            exc.status_code = 403
            raise exc

        _raise_for_status(resp, {
            # Reaches an end user in the app, so it says what they can do about
            # it rather than naming an env var they have never set.
            401: "Sign in to download this model.",
            404: f"Model '{model_name}' not found on server.",
        }, PermissionError)

        artifacts = resp.json().get("artifacts", {})
        if not artifacts:
            raise ValueError(f"API returned no artifacts for model '{model_name}'")
        return artifacts

    def get_current_user(self) -> dict:
        """GET /v1/users/me (auth required). Returns user profile dict."""
        try:
            resp = httpx.get(self._url(_USERS_ME), headers=self._headers(self._auth_headers_fn()),
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
            resp = httpx.patch(self._url(_USERS_ME), headers=self._headers(self._auth_headers_fn()),
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
            resp = httpx.post(self._url(_JWT_REFRESH), headers=self._headers(),
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
                              headers=self._headers(self._auth_headers_fn()),
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
            resp = httpx.post(self._url(_USAGE_TOKENS), headers=self._headers(self._auth_headers_fn()),
                              json=record, timeout=timeout)
            _raise_for_status(resp)
        except Exception as e:
            logger.error("Failed to push token usage: %s", e)

    def web_search(self, query: str, count: int,
                   timeout: int = _SEARCH_TIMEOUT) -> list[str]:
        """GET /v1/websearch (auth required). Returns the result URLs.

        The endpoint resolves the account's registered search provider
        server-side, so this side never sees a provider name or an API key.
        A 401/403 means search is unavailable for the account — no plan, or no
        provider registered — and raises ``PermissionError``; every other
        failure is a ``ConnectionError``, which callers must be free to treat
        as transient rather than as "search is off".
        """
        try:
            resp = httpx.get(self._url(_WEBSEARCH), headers=self._headers(self._auth_headers_fn()),
                             params={"q": query, "count": count}, timeout=timeout)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code in (401, 403):
            raise PermissionError(
                f"Web search endpoint rejected the token (HTTP {resp.status_code})")
        _raise_for_status(resp)

        data = resp.json()
        if not isinstance(data, list):
            return []
        return [u for u in data if isinstance(u, str) and u.strip()]

    def web_search_status(self, timeout: int = _SEARCH_STATUS_TIMEOUT) -> bool:
        """GET /v1/websearch/status (auth required). Whether search would serve.

        Answers for the whole account: signed in, with a provider registered.
        Raises ``PermissionError`` on 401/403 and ``ConnectionError`` on
        anything else, keeping "search is off" distinguishable from "the check
        did not complete" — see :func:`smartloop.search.providers.hosted_search_available`,
        which fails open on the latter.
        """
        try:
            resp = httpx.get(self._url(_WEBSEARCH_STATUS), headers=self._headers(self._auth_headers_fn()),
                             timeout=timeout)
        except httpx.TransportError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code in (401, 403):
            raise PermissionError(
                f"Web search status rejected the token (HTTP {resp.status_code})")
        _raise_for_status(resp)
        return bool(resp.json().get("enabled"))
