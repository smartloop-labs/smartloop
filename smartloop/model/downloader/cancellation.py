"""Asking a download in progress to stop, and finding the one to ask.

A transfer runs on a thread pool handed to ``loop.run_in_executor``, and that
future cannot be cancelled once a worker has picked it up.  So stopping is
cooperative: the byte loops consult a flag, and this module is where the flag
lives and how a second request finds it.

Two registries now cover a download, answering different questions.
:mod:`~smartloop.model.downloader.inflight` is keyed by directory and asks "is
it safe to delete this path", which needs containment in both directions and
must tolerate two transfers into one tree.  This one is keyed by ``model_id``
and asks "which download did the user just click stop on", which needs
identity.  Neither collapses into the other.

Process-local, for the same reason ``inflight`` is (see its docstring): a token
is meaningful exactly as long as the thread it can interrupt, and a row that
outlived the process would offer a cancel button for a download that is not
running.  That holds today because ``/v1/init`` and ``DELETE /v1/downloads/…``
are both workspace-level, so ``supervisor.routing.project_to_proxy`` leaves them
with the process that owns the workspace.  A route that grew a ``?project_id=``
would be proxied to a child and would signal an empty registry.
"""

import threading
import time


class DownloadCancelled(Exception):
    """The caller asked this download to stop.  Not a failure.

    Subclasses ``Exception`` directly, and that is load-bearing in three
    directions at once:

    * not ``ConnectionError`` — ``artifact_downloader._is_transient_error``
      ends in ``isinstance(exc, ConnectionError)``, so a cancel would be
      classified "try again" and answered 502;
    * not ``OSError``/``PermissionError`` — ``_download_artifacts`` re-raises
      those immediately to fail the whole batch fast, which is right for an
      unwritable disk and wrong as a description of a cancel;
    * not ``RuntimeError`` — ``storage.py`` and the download router map that to
      503 "busy, try later".

    ``status_code`` is read by ``api.routers._errors.status_for`` through its
    existing ``getattr`` branch, so nothing there needs to learn about this
    class.  499 is nginx's "client closed request": no standard code says "you
    asked me to stop and I did", and 499 is the one clients already read as a
    deliberate abort rather than a fault.
    """

    status_code = 499


class DownloadInProgress(RuntimeError):
    """This model is already downloading, so a second transfer was refused.

    Nothing serialized downloads on ``model_id`` before this: two calls wrote
    the same directory and each deleted the other's ``.part`` on its way in
    (``_download_file`` unlinks a pre-existing sidecar rather than resuming
    onto it), so both made progress and neither finished.
    """

    status_code = 409


class CancelToken:
    """One download's stop flag, plus the two ways it gets consulted.

    A wrapper rather than a bare :class:`threading.Event` for two reasons.
    :meth:`check` keeps every call site in the byte loops one line, which is
    what makes it affordable to put one in each of them.  And :meth:`wait` is a
    drop-in for ``time.sleep``, so the downloader's retry backoff becomes
    interruptible by substitution rather than by restructuring — without it a
    cancel arriving early in a 30s backoff waits the whole backoff out.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        #: Set once the transfer's threads have stopped *and* its bytes have
        #: been discarded — not merely when the flag was raised.  The route
        #: offers a bounded wait on this so a caller can ask for "stopped" in
        #: the strong sense; tests use it to avoid sleeping on a guess.
        self.done = threading.Event()

    def cancel(self) -> None:
        """Ask this download to stop.  Idempotent."""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self, what: str = "download") -> None:
        """Raise :class:`DownloadCancelled` if a stop has been requested."""
        if self._event.is_set():
            raise DownloadCancelled(f"{what} cancelled")

    def wait(self, timeout: float) -> bool:
        """Sleep for *timeout*, returning early and True if cancelled."""
        return self._event.wait(timeout)


_lock = threading.Lock()

#: ``model_id`` -> the token for the transfer writing that model's directory.
_active: "dict[str, CancelToken]" = {}

#: How often :func:`begin` re-checks for a departing entry.  Short enough that
#: a hand-off costs nothing perceptible, long enough not to spin.
_POLL_INTERVAL = 0.05


def begin(model_id: str, *, wait: float = 10.0) -> CancelToken:
    """Claim *model_id* for a new download, or raise :class:`DownloadInProgress`.

    Waits up to *wait* seconds for an existing entry to leave before refusing.
    That grace is what makes cancel-then-immediately-re-enable work: the entry
    is held until the cancelled transfer's bytes are gone (see :func:`end`), and
    a user who clicks stop and then start again is well inside it.  Without the
    wait, the second click would be answered "already downloading" for a
    download that is in the act of deleting itself.
    """
    deadline = time.monotonic() + wait
    while True:
        with _lock:
            if model_id not in _active:
                token = CancelToken()
                _active[model_id] = token
                return token
        if time.monotonic() >= deadline:
            raise DownloadInProgress(f"Already downloading: {model_id}")
        time.sleep(_POLL_INTERVAL)


def find(model_id: str) -> "CancelToken | None":
    """The token for *model_id*'s running download, if there is one."""
    with _lock:
        return _active.get(model_id)


def request_cancel(model_id: str) -> "CancelToken | None":
    """Signal *model_id*'s download to stop, returning its token.

    ``None`` means nothing was running — which the route reports as a plain
    outcome rather than an error, because the client races by construction: a
    progress frame arrives, the user clicks, and the transfer finishes before
    the request lands.
    """
    token = find(model_id)
    if token is not None:
        token.cancel()
    return token


def end(model_id: str, token: CancelToken) -> None:
    """Release *model_id*, but only if *token* is still the one registered.

    The identity check is what stops a slow wind-down from unregistering a
    newer download: a cancelled transfer can be inside a 60s socket read while
    its replacement has already claimed the key, and an unconditional ``pop``
    there would leave the live download with no token for the next cancel to
    find.

    Callers release *after* the bytes are discarded, not when the threads stop.
    Holding it across the cleanup is what keeps a second download from creating
    the directory into a tree that is about to be removed.
    """
    with _lock:
        if _active.get(model_id) is token:
            del _active[model_id]
    token.done.set()


def active_keys() -> "list[str]":
    """The ``model_id``s with a download registered, for status and tests."""
    with _lock:
        return sorted(_active)


def reset_for_tests() -> None:
    """Drop every registration.  Tests only."""
    with _lock:
        _active.clear()
