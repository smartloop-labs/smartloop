"""Background agent that keeps the access token fresh.

Periodically loads the refresh token from the credential store and
calls the SLP cloud API refresh endpoint to obtain new tokens.
Runs as an asyncio task started from the API server lifespan.
"""

import asyncio
import logging

from smartloop.auth import credential_store
from smartloop.config import AppSettings
from smartloop.platform.client import SLPPlatformClient

logger = logging.getLogger(__name__)

REFRESH_INTERVAL_SECONDS = 45 * 60  # 45 minutes


async def run_token_refresh_loop() -> None:
    """Refresh stored tokens on a fixed interval using the refresh token.

    Skips silently when no refresh token is present (user hasn't logged in).
    """
    settings = AppSettings()
    client = SLPPlatformClient(settings.slp_api_base)

    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)

        refresh_token = credential_store.get_refresh_token()
        if not refresh_token:
            continue

        try:
            data = client.refresh_access_token(refresh_token)
            credential_store.set_access_token(data["access_token"])
            if data.get("refresh_token"):
                credential_store.set_refresh_token(data["refresh_token"])
            logger.info("Tokens refreshed via /v1/auth/jwt/refresh")
        except PermissionError:
            # 401 — refresh token is expired or revoked; clear tokens so we
            # stop retrying.  User will need to re-authenticate.
            credential_store.clear_access_token()
            credential_store.clear_refresh_token()
            logger.info("Cleared tokens after 401 from refresh endpoint — re-login required")
        except Exception:
            logger.warning("Token refresh failed — will retry next cycle", exc_info=True)
