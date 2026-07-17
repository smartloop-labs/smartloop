"""Cache-through service for Smartloop platform API calls.

Sits between the API routers and the platform client + local DB.
On success, writes responses to the local DB. On connection failure,
falls back to cached data.
"""

import logging
import time
from typing import Callable, Optional, TypeVar

from smartloop.platform.client import SLPPlatformClient
from smartloop.platform.workspace_db import WorkspaceDB

logger = logging.getLogger(__name__)

T = TypeVar("T")

# TTL in seconds
USER_TTL = 5 * 60       # 5 minutes
MODELS_TTL = 15 * 60    # 15 minutes
RELEASES_TTL = 60 * 60  # 1 hour


def _fetch_or_cache(
    *,
    ttl: float,
    get_cached: Callable[[], Optional[T]],
    get_timestamp: Callable[[], Optional[float]],
    fetch_remote: Callable[[], T],
    save_cache: Callable[[T], None],
    force: bool = False,
) -> T:
    """Try cloud first, fall back to cache on connection error.

    If a fresh cache entry exists (within *ttl*), return it without
    hitting the cloud.  Otherwise fetch from cloud, update cache, and
    return.  On ``ConnectionError``, return stale cache if available.

    When *force* is True, the TTL short-circuit is skipped and the cloud
    is always queried (used to refresh per-user state on login).
    """
    # Check if cache is fresh enough
    if not force:
        ts = get_timestamp()
        if ts is not None and (time.time() - ts) < ttl:
            cached = get_cached()
            if cached is not None:
                return cached

    # Try cloud
    try:
        data = fetch_remote()
    except ConnectionError:
        # Offline — serve stale cache if we've ever fetched successfully
        if get_timestamp() is not None:
            cached = get_cached()
            if cached is not None:
                logger.info("Platform API unreachable, serving cached data")
                return cached
        raise
    except PermissionError:
        raise

    # Cloud succeeded — update cache
    try:
        save_cache(data)
    except Exception:
        logger.warning("Failed to write cache", exc_info=True)

    return data


class PlatformService:
    """Coordinates between the cloud platform client and local DB cache.

    When ``db`` is ``None`` the service degrades to a pure proxy — used
    in the brief window before a workspace exists on disk and there is
    nowhere to persist the cache yet.
    """

    def __init__(self, client: SLPPlatformClient, db: Optional[WorkspaceDB] = None) -> None:
        self._client = client
        self._db = db

    def get_current_user(self, force: bool = False) -> dict:
        """Return the current user's profile (cached or fresh)."""
        if self._db is None:
            return self._client.get_current_user()
        return _fetch_or_cache(
            ttl=USER_TTL,
            get_cached=self._db.get_user,
            get_timestamp=self._db.user_fetched_at,
            fetch_remote=self._client.get_current_user,
            save_cache=self._db.save_user,
            force=force,
        )

    def update_current_user(self, payload: dict) -> dict:
        """Update user profile on the cloud and refresh cache."""
        data = self._client.update_current_user(payload)
        if self._db is not None:
            try:
                self._db.save_user(data)
            except Exception:
                logger.warning("Failed to cache user update", exc_info=True)
        return data

    def list_models(self, force: bool = False) -> list[dict]:
        """Return the model catalog (cached or fresh)."""
        if self._db is None:
            return self._client.list_models()
        return _fetch_or_cache(
            ttl=MODELS_TTL,
            get_cached=self._db.list_models,
            get_timestamp=self._db.models_fetched_at,
            fetch_remote=self._client.list_models,
            save_cache=self._db.save_models,
            force=force,
        )

    def get_latest_release(self, channel: str = "stable", force: bool = False) -> dict:
        """Return the latest-release envelope for *channel* (cached or fresh).

        Mirrors the cloud ``LatestReleaseResponse`` shape
        ``{update_available, update_required, release}``.  We poll the cloud
        WITHOUT ``current_version`` and cache only the inner ``release`` (so one
        cache entry serves every client); ``update_available`` /
        ``update_required`` are left null/False here and computed per-request by
        the caller.  ``release`` is ``None`` when the channel has no published
        release.

        Set *force* to bypass the TTL and re-poll the cloud — used by the
        client's explicit "check for updates" action, mirroring
        :meth:`get_current_user`.
        """
        def _fetch_release() -> Optional[dict]:
            return (self._client.get_latest_release(channel) or {}).get("release")

        if self._db is None:
            release = _fetch_release()
        else:
            release = _fetch_or_cache(
                ttl=RELEASES_TTL,
                get_cached=lambda: self._db.get_release(channel),
                get_timestamp=lambda: self._db.release_fetched_at(channel),
                fetch_remote=_fetch_release,
                save_cache=lambda rel: self._db.save_release(channel, rel),
                force=force,
            )
        return {"update_available": None, "update_required": False, "release": release}
