"""Background agent that keeps the local release cache warm.

Periodically polls the SLP cloud API for the latest published release on each
configured channel and stores it in the workspace DB, so the Electron client
sees update notifications promptly even when otherwise idle.  Runs as an
asyncio task started from the API server lifespan.
"""

import asyncio
import logging

from smartloop.config import AppSettings
from smartloop.platform import get_platform_service

logger = logging.getLogger(__name__)

RELEASE_POLL_INTERVAL_SECONDS = 60 * 60  # 1 hour


async def run_release_poll_loop() -> None:
    """Refresh the cached latest release for every configured channel on an interval."""
    settings = AppSettings()

    while True:
        for channel in settings.release_channels:
            try:
                svc = get_platform_service()
                svc.get_latest_release(channel, force=True)
                logger.info("Refreshed release cache for channel %r", channel)
            except (ConnectionError, PermissionError):
                logger.warning(
                    "Release poll failed for channel %r — will retry next cycle",
                    channel,
                )
            except Exception:
                logger.warning(
                    "Unexpected error polling releases for channel %r",
                    channel,
                    exc_info=True,
                )

        await asyncio.sleep(RELEASE_POLL_INTERVAL_SECONDS)
