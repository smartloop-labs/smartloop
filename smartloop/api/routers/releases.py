"""Release update-check router.

Exposes the current published Electron app release for a channel, cached
locally from the cloud platform API.  The client (Electron) polls this to show
a "new version available" notification; stage 2 will point electron-updater at
the returned ``feed_url``.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from smartloop.config.app_settings import AppSettings
from smartloop.platform import PlatformService, get_platform_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["releases"])


class ReleaseArtifact(BaseModel):
    platform: str
    arch: str
    kind: str
    url: str
    sha512: str
    size: int


class ReleaseInfo(BaseModel):
    version: str
    name: str
    notes: str = ""
    channel: str
    published_at: str | None = None
    artifacts: list[ReleaseArtifact] = []
    min_version: str | None = None


class LatestReleaseResponse(BaseModel):
    update_available: bool | None = None
    update_required: bool = False
    release: ReleaseInfo | None = None
    feed_url: str
    channel: str


@router.get("/releases/latest", response_model=LatestReleaseResponse)
async def latest_release(
    current_version: str | None = None,
    channel: str | None = None,
    force: bool = False,
    svc: PlatformService = Depends(get_platform_service),
):
    """Return the latest published release for *channel* (cached locally).

    When *current_version* is supplied, ``update_available`` /
    ``update_required`` are computed locally by semver comparison, mirroring the
    cloud ``/releases/latest`` contract.  ``feed_url`` is derived from
    ``SLP_API_BASE`` so it tracks dev/prod automatically.

    Pass ``force=true`` (used by an explicit "check for updates" action) to
    bypass the local cache TTL and re-poll the cloud immediately.
    """
    # Local import keeps semver optional at module load and avoids any import cycle.
    from smartloop.platform import semver

    settings = AppSettings()
    channel = channel or settings.release_channels[0]
    if channel not in settings.release_channels:
        raise HTTPException(status_code=400, detail=f"Unknown channel {channel!r}")

    try:
        wrapper = svc.get_latest_release(channel, force=force)
    except ConnectionError:
        raise HTTPException(status_code=502, detail="Platform API unavailable")

    feed_url = f"{settings.slp_api_base.rstrip('/')}/v1/releases/feed/{channel}"

    # PlatformService returns the cloud wrapper {update_available, update_required, release};
    # the inner `release` may be None when no published release exists for the channel.
    release = (wrapper or {}).get("release")
    if not release:
        return LatestReleaseResponse(feed_url=feed_url, channel=channel)

    info = ReleaseInfo(**release)

    update_available = None
    update_required = False
    if current_version:
        update_available = semver.is_newer(info.version, current_version)
        if info.min_version:
            update_required = semver.is_newer(info.min_version, current_version)

    return LatestReleaseResponse(
        update_available=update_available,
        update_required=update_required,
        release=info,
        feed_url=feed_url,
        channel=channel,
    )
