"""Keeps the access token fresh.

Two entry points over the same operation: a background loop started from the
API server lifespan, and :func:`refresh_access_token_now` for callers that
have just been told their access token is dead and want it back immediately.

Both run in the workspace owner and nowhere else.  Under supervision this
process is one of N+1 sharing a single credential file — a supervisor plus an
agent child per project — and the refresh token is single-use: redeeming it
twice gets the loser a 401, which reads as "the refresh token is dead" and
clears both credentials.  Nothing here can see another process, so the guard is
ownership rather than locking; see :func:`refresh_access_token_now`.
"""

import asyncio
import base64
import json
import logging
import threading
import time
from typing import Optional

from smartloop.auth import credential_store
from smartloop.config import AppSettings
from smartloop.platform.client import SLPPlatformClient
from smartloop.workspace.context import owns_workspace, pinned_project_id

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


#: Serializes redemption.  Two callers reach this — the background loop on the
#: event loop, and a request thread whose call just came back 401 — and with
#: single-use rotating refresh tokens the second redemption of the same token is
#: refused, which clears both credentials and logs the user out.
#:
#: Two, and only two, because a pinned process declines before reaching here
#: (:func:`refresh_access_token_now`).  Without that gate there would be one
#: more per agent child, on a lock that does not span processes and so would
#: not order them.
_refresh_lock = threading.Lock()

#: How long a failed redemption suppresses the next attempt.  The lock alone
#: only collapses callers when the refresh *succeeds* — on failure the stored
#: token is unchanged, so every queued waiter falls through and pays the full
#: timeout in turn.  With the auth endpoints now served from the threadpool,
#: ten concurrent 401s against a degraded platform meant ten serialized
#: redemptions holding ten threadpool slots.  One attempt per window is enough
#: to discover the platform is still down.
_FAILED_REFRESH_COOLDOWN_SECONDS = 30.0

_last_failure_at: float = 0.0


def refresh_access_token_now(client: Optional[SLPPlatformClient] = None) -> bool:
    """Redeem the stored refresh token for a new access token.

    Returns True when a new access token is in place.  Concurrent callers
    collapse onto one redemption: whoever holds the lock does the work, and the
    rest find the token already replaced and report success without spending the
    refresh token a second time.

    Workspace-owner work.  Everything above that keeps two *threads* from
    double-redeeming is process-local, so it says nothing about two
    *processes* — and under supervision there are N+1 of them against one
    credential file.  A child racing the supervisor's background loop spends
    the rotated token twice, and the refusal that comes back is read by
    :func:`_redeem` as a dead refresh token, clearing both credentials: a full
    logout of every project, which is the failure this function exists to
    prevent.

    So a pinned process declines, and it costs it nothing.  The retry in
    :meth:`SLPPlatformClient._authed` is keyed on the credentials having
    *changed*, not on what this returned, so declining leaves the caller with
    the 401 it already had — exactly what it had before the refresh was
    widened to every call site.  And because :mod:`credential_store` caches
    nothing, a child picks up a token the owner rotated on its very next read,
    with no redemption and no message passing.

    This is a statement about process topology, not a safety property: it holds
    because only the owner writes, not because writing is safe.
    """
    global _last_failure_at

    if not owns_workspace():
        logger.debug(
            "[token] refresh declined in project agent %s — redemption "
            "belongs to the workspace owner", pinned_project_id(),
        )
        return False

    before = credential_store.get_access_token()
    with _refresh_lock:
        current = credential_store.get_access_token()
        if current and current != before:
            # Someone refreshed while we waited for the lock.  Their token is the
            # live one; redeeming again would burn a rotated refresh token for a
            # result we already have.
            return True

        if time.monotonic() - _last_failure_at < _FAILED_REFRESH_COOLDOWN_SECONDS:
            # A redemption failed moments ago.  Whatever stopped it — the
            # platform is down, the refresh token is gone — will not have been
            # fixed by the time it took us to reach the lock.
            return False

        ok = _redeem(client)
        _last_failure_at = 0.0 if ok else time.monotonic()
        return ok


def _redeem(client: Optional[SLPPlatformClient] = None) -> bool:
    """Do the actual redemption.  Callers hold :data:`_refresh_lock`.

    A 401 from the refresh endpoint means the refresh token itself is dead —
    retrying cannot help, so both tokens are cleared and the user has to log in
    again.  Any other failure leaves the credentials alone to be retried later;
    ``_raise_for_status`` guarantees a 429 or 5xx arrives as ``ConnectionError``
    rather than as the ``PermissionError`` that would clear them.
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
                # Off the event loop.  Redemption is a synchronous httpx POST
                # with a 10s budget, taken behind a lock a request thread may
                # already hold.  Awaiting it inline stalls this loop for as
                # long as that takes — and under supervision this loop belongs
                # to the supervisor, which is proxying every project's chat
                # stream through it byte by byte.  One expired token would
                # pause generation for every project at once.
                await asyncio.to_thread(refresh_access_token_now, client)
        except Exception:  # a bad cycle must not kill the loop
            logger.warning("Token refresh cycle failed", exc_info=True)
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
