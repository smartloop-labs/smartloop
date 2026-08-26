"""Keeps the access token fresh.

Two entry points over the same operation: a background loop started from the
API server lifespan, and :func:`refresh_access_token_now` for callers that
have just been told their access token is dead and want it back immediately.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional

from smartloop.auth import credential_store
from smartloop.config import AppSettings
from smartloop.platform.client import SLPPlatformClient

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 45 * 60  # 45 minutes


def _jwt_exp(token: str) -> Optional[float]:
    """The ``exp`` claim of a JWT, or ``None`` if it has none to read.

    Claims only — the signature is the server's to verify.  We are deciding
    whether to bother asking for a new token, not whether to trust this one.
    """
    if not token or token.count(".") != 2:
        return None
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    try:
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _needs_refresh() -> bool:
    """True when the stored access token will not survive until the next tick.

    The loop wakes on a fixed interval, so anything expiring inside that
    interval has to be renewed now — there is no later chance before it dies.
    """
    if not credential_store.get_refresh_token():
        return False
    access = credential_store.get_access_token()
    if not access:
        # Nothing to use.  This is the state a 401 used to leave behind, and
        # it is exactly when a refresh is most needed.
        return True
    exp = _jwt_exp(access)
    if exp is None:
        # Opaque token — no expiry to read, so renew on the interval as before.
        return True
    return exp - time.time() <= REFRESH_INTERVAL_SECONDS


def refresh_access_token_now(client: Optional[SLPPlatformClient] = None) -> bool:
    """Redeem the stored refresh token for a new access token.

    Returns True when a new access token was stored.  A 401 from the refresh
    endpoint means the refresh token itself is dead — retrying cannot help, so
    both tokens are cleared and the user has to log in again.  Any other
    failure leaves the credentials alone to be retried later.
    """
    refresh_token = credential_store.get_refresh_token()
    if not refresh_token:
        return False
    if client is None:
        client = SLPPlatformClient(AppSettings().slp_api_base)
    try:
        data = client.refresh_access_token(refresh_token)
    except PermissionError:
        credential_store.clear_access_token()
        credential_store.clear_refresh_token()
        logger.info(
            "Cleared tokens after 401 from refresh endpoint — re-login required"
        )
        return False
    except Exception:
        logger.warning("Token refresh failed — will retry next cycle", exc_info=True)
        return False

    access = (data or {}).get("access_token")
    if not access:
        logger.warning("Refresh endpoint returned no access token")
        return False
    credential_store.set_access_token(access)
    if data.get("refresh_token"):
        credential_store.set_refresh_token(data["refresh_token"])
    logger.info("Tokens refreshed via /v1/auth/jwt/refresh")
    return True


async def run_token_refresh_loop() -> None:
    """Renew the access token before it expires, starting immediately.

    The sleep is at the *end*: sleeping first meant nothing refreshed for the
    first interval after start, so an access token that expired in that window
    was never renewed — and the 401 it caused deleted it.  Checking on entry
    also repairs that state on the next restart.

    Skips silently when no refresh token is present (user hasn't logged in).
    """
    settings = AppSettings()
    client = SLPPlatformClient(settings.slp_api_base)

    while True:
        try:
            if _needs_refresh():
                refresh_access_token_now(client)
        except Exception:  # a bad cycle must not kill the loop
            logger.warning("Token refresh cycle failed", exc_info=True)
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
