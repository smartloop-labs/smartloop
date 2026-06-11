"""Path utilities for cross-platform (WSL / Windows / Linux) compatibility."""

import sys
from pathlib import PurePosixPath, PureWindowsPath


def normalize_path(path: str) -> str:
    """Convert a Windows-style path to a WSL path when running on Linux.

    On native Windows the path is returned unchanged.  On WSL/Linux a
    client-supplied path like ``C:\\Users\\foo\\bar`` is translated to
    ``/mnt/c/Users/foo/bar`` so that it resolves correctly on the host.
    """
    win = PureWindowsPath(path)
    if win.drive and sys.platform != "win32":
        drive = win.drive[0].lower()
        return str(PurePosixPath("/mnt", drive, *win.parts[1:]))
    return path
