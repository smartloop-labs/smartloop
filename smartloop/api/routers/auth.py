"""Auth router – credential management and OAuth callback for the local API server.

Endpoints:
    GET    /v1/auth/status    – check credential state
    POST   /v1/auth/callback  – receive tokens from webapp after OAuth flow
    GET    /v1/auth/callback  – receive tokens via query params (Safari fallback)
    POST   /v1/auth/token     – manually set a token (developer_token or access_token)
    DELETE /v1/auth/token     – clear stored credentials
"""

import logging
from enum import Enum
from typing import Literal, Optional

from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from smartloop.auth import credential_store
from smartloop.platform import get_platform_service
from smartloop.config.app_settings import AppSettings
from smartloop.workspace.context import owns_workspace, pinned_project_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/auth", tags=["auth"])


# -- request / response models ------------------------------------------------


class TokenType(str, Enum):
    developer_token = "developer_token"
    access_token = "access_token"


class ClearScope(str, Enum):
    all = "all"
    developer_token = "developer_token"
    access_token = "access_token"


class TokenSetRequest(BaseModel):
    token: str = Field(min_length=1, description="The token value to store")
    type: TokenType = Field(
        default=TokenType.developer_token,
        description="Token type: developer_token (CLI/scripts) or access_token (OAuth/Electron)",
    )


class UserInfo(BaseModel):
    email: str
    name: Optional[str] = None


class AuthStatusResponse(BaseModel):
    authenticated: bool
    method: Optional[Literal["developer_token", "access_token"]] = None
    has_developer_token: bool
    has_access_token: bool
    user: Optional[UserInfo] = None


# -- endpoints ----------------------------------------------------------------


_last_refreshed_token: Optional[str] = None


def _refresh_platform_cache(refresh_user: bool = True) -> None:
    """Force-refresh the cached model catalog (and optionally the user) on an
    auth change.

    The catalog always refreshes so each model's ``is_accessible`` — which the
    cloud computes on the fly per-user — reflects the current auth state instead
    of a stale entry. On logout pass ``refresh_user=False``: the cached user
    profile is intentionally left in place (stale is fine; already-downloaded
    models stay usable), and we avoid a pointless 401 round-trip.

    Best-effort and independent: a failed user fetch doesn't block the catalog
    refresh. A single login delivers the token redundantly (POST + Safari GET
    fallback, plus retries), so we dedupe on the credential value — only a
    genuine change triggers a refetch.
    """
    global _last_refreshed_token
    token = credential_store.get_access_token() or credential_store.get_token()
    if token and token == _last_refreshed_token:
        return

    # Whether hosted search will serve is an answer about the account, cached for
    # a minute.  Signing in with that cache holding a "no" from the logged-out
    # state left search skipping itself for the first minute of the session —
    # ``reset_status_cache`` existed for exactly this and nothing called it.
    from smartloop.search import providers

    providers.reset_status_cache()

    svc = get_platform_service()
    if refresh_user:
        try:
            svc.get_current_user(force=True)
        except Exception:
            logger.warning("Failed to refresh cached user", exc_info=True)
    try:
        svc.list_models(force=True)
    except Exception:
        logger.warning("Failed to refresh cached model catalog", exc_info=True)
    _last_refreshed_token = token


def _resolve_user() -> Optional[UserInfo]:
    """Return the logged-in user's profile. Returns None on failure.

    Resolves the user through the platform service layer so the cached
    user info is used instead of an uncached client call.
    """
    try:
        data = get_platform_service().get_current_user()
        return UserInfo(email=data["email"], name=data.get("name"))
    except PermissionError as exc:
        # ``get_current_user`` raises PermissionError for every refusal that is
        # not transient — 403 and 404 as much as 401 — so the status has to be
        # read rather than assumed.  Only a 401 is a statement about the token.
        # Discarding one on a 404 meant an API deploy that moved a route logged
        # everyone out of the desktop client.
        if getattr(exc, "status_code", None) != 401:
            logger.warning(
                "Could not resolve user (HTTP %s) — leaving credentials alone",
                getattr(exc, "status_code", "?"),
            )
            return None

        # A 401 that survived :meth:`SLPPlatformClient._authed`, which redeems the
        # refresh token and retries before any exception is raised.  So by the
        # time one arrives here the refresh has already been tried and failed —
        # there was nothing to refresh with, the refresh itself was refused, or
        # the fresh token was refused too.
        #
        # This used to do the redeeming itself, which is why it was the only path
        # in the app that recovered from an expired token; that logic now lives
        # in the client, where every endpoint gets it.  What is left is the part
        # that was always local to auth status: drop the dead access token so
        # later requests fall back to the developer token.  A refresh token that
        # was itself rejected has already been cleared by the refresh agent.
        #
        # Dropping it is workspace-owner work.  Routing normally keeps this
        # endpoint on the supervisor, but a project is resolved from a
        # ``project_id`` query parameter on *any* path, so a stray
        # ``/v1/auth/status?project_id=X`` is proxied to an agent child — which
        # would delete the whole workspace's access token on the strength of one
        # project's 401.  A child also cannot tell a dead credential from one the
        # owner has already replaced, because it never saw the replacement.
        if not owns_workspace():
            logger.info(
                "401 from the platform API in project agent %s — leaving the "
                "credential to the workspace owner", pinned_project_id(),
            )
            return None

        if credential_store.get_access_token():
            credential_store.clear_access_token()
            logger.warning("Cleared expired access token after 401 from platform API")
        return None
    except Exception:
        logger.warning("Could not resolve user from platform API", exc_info=True)
        return None


@router.get("/status", response_model=AuthStatusResponse)
def auth_status():
    """Return the current credential state without exposing token values.

    Deliberately ``def`` rather than ``async def``, along with every other
    handler in this router: they resolve the user through the platform API with
    a synchronous client, and on the worst path — a 401, a refresh, a retry —
    that is three 30-second calls.  On the event loop those stall every chat
    stream in flight; declared sync, FastAPI runs them in the threadpool.
    """
    has_dev = credential_store.get_token() is not None
    has_access = credential_store.get_access_token() is not None
    authenticated = has_dev or has_access

    # _resolve_user may clear an expired access token on 401,
    # so re-read credential state afterward.
    user = _resolve_user() if authenticated else None

    has_dev = credential_store.get_token() is not None
    has_access = credential_store.get_access_token() is not None
    authenticated = has_dev or has_access

    if has_access:
        method = "access_token"
    elif has_dev:
        method = "developer_token"
    else:
        method = None

    return AuthStatusResponse(
        authenticated=authenticated,
        method=method,
        has_developer_token=has_dev,
        has_access_token=has_access,
        user=user,
    )


@router.post("/token", response_model=AuthStatusResponse)
def set_token(request: TokenSetRequest):
    """Store a token in the credential store.

    Defaults to developer_token (used by ``slp token set``).
    Pass ``"type": "access_token"`` for OAuth/JWT tokens.
    """
    if request.type == TokenType.developer_token:
        credential_store.set_token(request.token)
        logger.info("Developer token updated via /v1/auth/token")
    else:
        credential_store.set_access_token(request.token)
        logger.info("Access token updated via /v1/auth/token")

    _refresh_platform_cache()
    return auth_status()


@router.delete("/token", response_model=AuthStatusResponse)
def clear_token(
    type: ClearScope = Query(
        default=ClearScope.all,
        description="Which credentials to clear: all, developer_token, or access_token",
    ),
):
    """Clear stored credentials.

    By default clears everything (logout).  Pass ``?type=developer_token``
    or ``?type=access_token`` to clear only one credential type.
    """
    if type in (ClearScope.all, ClearScope.developer_token):
        credential_store.clear_token()
        logger.info("Developer token cleared via /v1/auth/token")

    if type in (ClearScope.all, ClearScope.access_token):
        credential_store.clear_access_token()
        credential_store.clear_refresh_token()
        logger.info("Access/refresh tokens cleared via /v1/auth/token")

    # Refresh the catalog (now anonymous) so model is_accessible reflects the
    # logout. Keep the cached user profile as-is — stale is acceptable and
    # already-downloaded models remain usable.
    _refresh_platform_cache(refresh_user=False)
    return auth_status()


# -- OAuth callback ------------------------------------------------------------


class OAuthCallbackRequest(BaseModel):
    access_token: str = Field(description="JWT access token from the OAuth flow")
    refresh_token: Optional[str] = Field(
        default=None,
        description="Refresh token for renewing the access token",
    )


@router.post("/callback", response_model=AuthStatusResponse)
def oauth_callback(request: OAuthCallbackRequest):
    """Receive tokens from the webapp after a completed OAuth flow.

    The cloud API handles the actual OAuth code exchange and redirects the
    browser to the webapp with the minted JWT.  The webapp then POSTs the
    token(s) here so the local server can store them in the credential store.
    """
    credential_store.set_access_token(request.access_token)
    if request.refresh_token:
        credential_store.set_refresh_token(request.refresh_token)

    _log_token_received("POST", request.access_token, request.refresh_token)
    _refresh_platform_cache()
    return auth_status()


APP_URL = AppSettings().slp_app_url if hasattr(AppSettings(), "slp_app_url") else "https://app.smartloop.ai"


def _log_token_received(method: str, access_token: str, refresh_token: Optional[str] = None):
    """Log token receipt with partially redacted values for debugging."""
    at = f"{access_token[:4]}...{access_token[-4:]}" if len(access_token) > 8 else "***"
    rt = "none"
    if refresh_token:
        rt = f"{refresh_token[:4]}...{refresh_token[-4:]}" if len(refresh_token) > 8 else "***"
    logger.info("OAuth tokens stored via /v1/auth/callback (%s) access=%s refresh=%s", method, at, rt)


@router.get("/callback")
def oauth_callback_get(
    access_token: str = Query(description="JWT access token from the OAuth flow"),
    refresh_token: Optional[str] = Query(default=None, description="Refresh token"),
):
    """Safari fallback: receive tokens via GET query params and redirect to success page.

    Safari blocks mixed-content fetch (HTTPS→HTTP) so the webapp falls back
    to a full-page navigation with tokens as query params.
    """
    credential_store.set_access_token(access_token)
    if refresh_token:
        credential_store.set_refresh_token(refresh_token)

    _log_token_received("GET", access_token, refresh_token)
    _refresh_platform_cache()
    return RedirectResponse(url=f"{APP_URL}/redirect/?mode=desktop")
