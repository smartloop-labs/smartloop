"""HTTP client for the Smartloop platform API (api.smartloop.ai).

Handles only JSON control-plane calls — no file downloading.
"""

import logging
from typing import Callable

import httpx

from smartloop.platform import auth as platform_auth

logger = logging.getLogger(__name__)

# URL path constants
_BOOTSTRAP_MODELS = "/v1/bootstrap/models"
_MODEL_DOWNLOAD = "/v1/models/{}/download"
_USAGE_TOKENS = "/usage/tokens"
_USERS_ME = "/v1/users/me"
_JWT_REFRESH = "/v1/auth/jwt/refresh"
_MODEL_UPLOAD = "/v1/models/{}/upload"
_MODELS = "/v1/models"

_TIMEOUT = 30


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
        except httpx.ConnectError as e:
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
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        if resp.status_code == 401:
            try:
                resp = httpx.get(self._url(_MODELS), timeout=_TIMEOUT)
            except httpx.ConnectError as e:
                raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp)
        return resp.json()

    def get_model_artifacts(self, model_name: str) -> dict[str, str]:
        """GET /v1/models/{model_name}/download (auth required).

        Returns ``{filename: signed_url}`` map.
        """
        try:
            resp = httpx.get(self._url(_MODEL_DOWNLOAD.format(model_name)),
                             headers=self._auth_headers_fn(), timeout=_TIMEOUT)
        except httpx.ConnectError as e:
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
        except httpx.ConnectError as e:
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
        except httpx.ConnectError as e:
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
        except httpx.ConnectError as e:
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
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection error: {e}") from e

        _raise_for_status(resp, {
            401: "Authentication failed. Check your SLP_DEVELOPER_TOKEN.",
            404: "Model upload endpoint not found on server.",
        }, PermissionError)
        return resp.json()

    def push_token_usage(self, record: dict, timeout: int = 10) -> None:
        """POST /usage/tokens (auth required). Fire-and-forget telemetry."""
        httpx.post(self._url(_USAGE_TOKENS), headers=self._auth_headers_fn(),
                   json=record, timeout=timeout)
