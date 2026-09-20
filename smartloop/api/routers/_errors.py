"""Turning a framework exception back into the HTTP status it stands for.

Shared because the alternative is each router deciding for itself, and the
decisions drifted: ``PermissionError`` means four different things depending on
who raised it, and a router that reads it as one of them answers "sign in" to a
full disk.
"""

import errno

from smartloop.platform import ModelAccessError


def _is_disk_full(exc: Exception) -> bool:
    """True when *exc* — or the failure it wraps — is an out-of-space OSError.

    The download path wraps a raw ENOSPC in a classified ``ConnectionError``
    (``_classified`` keeps the cause as ``__cause__``), so the errno is not
    always on the exception itself — follow the chain before declaring it a
    network failure.
    """
    while exc is not None:
        if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
            return True
        exc = exc.__cause__
    return False


def status_for(exc: Exception) -> int:
    """Classify *exc* as the HTTP status it would have been.

    Platform errors carry the upstream status (see
    :func:`smartloop.platform.client._raise_for_status`) and are answered with
    it, so the client never has to match on message text — that is written for a
    person to read, and it gets reworded.

    Only :class:`ModelAccessError` earns a 403.  ``PermissionError`` used to,
    which was too wide by half: it is an ``OSError`` subclass, so an unwritable
    models directory arrived as "not on your plan" and the desktop client turned
    that into a sign-in prompt for what is a local disk problem.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    if isinstance(exc, ModelAccessError):
        return 403
    if _is_disk_full(exc):
        # The most retryable failure there is (free some space and try again),
        # and not a network problem — so neither 500 nor 502 says it right.
        # 507 is how the client tells "refill the disk" from "it broke".
        return 507
    # Transfers that died on a dropped socket, a stale signed URL or an upstream
    # 5xx are aggregated into a ConnectionError by ``_aggregate_error``.
    if isinstance(exc, ConnectionError):
        return 502
    return 500
