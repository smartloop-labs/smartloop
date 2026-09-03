"""Turning a framework exception back into the HTTP status it stands for.

Shared because the alternative is each router deciding for itself, and the
decisions drifted: ``PermissionError`` means four different things depending on
who raised it, and a router that reads it as one of them answers "sign in" to a
full disk.
"""

from smartloop.platform import ModelAccessError


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
    # Transfers that died on a dropped socket, a stale signed URL or an upstream
    # 5xx are aggregated into a ConnectionError by ``_aggregate_error``.
    if isinstance(exc, ConnectionError):
        return 502
    return 500
