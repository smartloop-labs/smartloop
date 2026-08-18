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
MCP_CATALOG_TTL = 60 * 60  # 1 hour — the curated catalog changes rarely


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

    def list_mcp_catalog(self, force: bool = False) -> list[dict]:
        """Return the curated MCP server catalog (cached or fresh).

        Cached so the Connections list still renders — and a curated install can
        still resolve its slug — when the platform API is unreachable or the
        token has gone stale. Neither state implies the user is offline: they
        can still reach a third-party MCP server and complete OAuth.
        """
        if self._db is None:
            return self._client.list_mcp_catalog()
        return _fetch_or_cache(
            ttl=MCP_CATALOG_TTL,
            get_cached=self._db.list_mcp_catalog,
            get_timestamp=self._db.mcp_catalog_fetched_at,
            fetch_remote=self._client.list_mcp_catalog,
            save_cache=self._db.save_mcp_catalog,
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

        NOTE: this does NOT reuse :func:`_fetch_or_cache`.  That helper treats a
        cached value of ``None`` as "nothing cached", but for releases ``None``
        is a *legitimate* cached value meaning "this channel has no published
        release yet" (e.g. an unreleased ``beta``).  Conflating the two would
        make the TTL fast-path and the offline fallback misfire for empty
        channels — re-polling the cloud on every request, and returning an error
        offline for a state we already know.  So we discriminate on the fetch
        *timestamp* (present iff we've ever successfully polled), which the DB
        records even when the release itself is ``None``.
        """
        def _fetch_release() -> Optional[dict]:
            return (self._client.get_latest_release(channel) or {}).get("release")

        def _envelope(release: Optional[dict]) -> dict:
            return {"update_available": None, "update_required": False, "release": release}

        if self._db is None:
            return _envelope(_fetch_release())

        fetched_at = self._db.release_fetched_at(channel)
        have_cache = fetched_at is not None

        # Fresh cache within TTL — serve without hitting the cloud. Works even
        # when the cached release is None (channel has nothing published).
        if not force and have_cache and (time.time() - fetched_at) < RELEASES_TTL:
            return _envelope(self._db.get_release(channel))

        try:
            release = _fetch_release()
        except ConnectionError:
            # Offline — serve whatever we last learned, including a known-empty
            # channel. Only re-raise if we've genuinely never fetched.
            if have_cache:
                logger.info("Platform API unreachable, serving cached release for %r", channel)
                return _envelope(self._db.get_release(channel))
            raise

        try:
            self._db.save_release(channel, release)
        except Exception:
            logger.warning("Failed to write release cache", exc_info=True)
        return _envelope(release)
