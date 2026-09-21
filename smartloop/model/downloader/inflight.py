"""Which model directories a download is currently writing to.

On disk a live download and one that died look identical: a directory holding
a ``.part`` and no finished GGUF.  ``storage_service._weights_state`` cannot
tell them apart, so the disk sweep read a transfer in progress as abandoned
and deleted it out from under the downloader.  This module carries the one
fact the filesystem does not: someone is still writing.

Process-local and advisory, deliberately.  No lock is held across a transfer,
so consulting the registry costs microseconds and a cleanup never blocks
behind a multi-GB download.  It is also why the state lives here rather than
in a database: "a transfer is running" is true exactly as long as a thread is,
and a row that outlived the process would claim a dead download was live
forever — wedging cleanup on a full disk, which is when it matters most.
"""

import os
import threading
from contextlib import contextmanager
from typing import Iterator

_lock = threading.Lock()

#: Canonical directory path -> how many transfers are writing to it.  A count
#: rather than a set: nothing serializes downloads on model_id, so the same
#: model can be fetched twice concurrently and the first to finish must not
#: unmark the directory while the second is still writing.
_active: dict[str, int] = {}


def _key(path: str) -> str:
    """Canonical form of *path*, as both sides of the registry must agree.

    ``realpath`` is not cosmetic here: the marker is written with the
    downloader's ``local_dir`` and read with the sweep's scan path, and on
    macOS a temp directory under ``/var`` resolves to ``/private/var``.
    ``normcase`` covers callers on a case-insensitive filesystem.
    """
    return os.path.normcase(os.path.realpath(path))


@contextmanager
def downloading(local_dir: str) -> Iterator[None]:
    """Mark *local_dir* as being written to for the duration of the block."""
    key = _key(local_dir)
    with _lock:
        _active[key] = _active.get(key, 0) + 1
    try:
        yield
    finally:
        with _lock:
            remaining = _active.get(key, 0) - 1
            if remaining > 0:
                _active[key] = remaining
            else:
                _active.pop(key, None)


def is_downloading(path: str) -> bool:
    """True while a transfer is writing at, below, or above *path*.

    Containment in both directions, because the caller is asking "is it safe to
    delete this" and both answers are no.  Below: removing ``models/<org>``
    destroys a download into ``models/<org>/<repo>``.  Above: removing
    ``<model_dir>/bin`` takes files out of a live transfer — weights can land
    there, as ``_weights_state`` and the init router's ``_gguf_exists`` both
    know.  The mapping holds one entry per running transfer, so the scan is
    free either way.
    """
    key = _key(path)
    with _lock:
        return any(
            active == key
            or active.startswith(key + os.sep)      # a transfer below *path*
            or key.startswith(active + os.sep)      # *path* inside a transfer
            for active in _active
        )
