"""PyInstaller runtime hook: force UTF-8 I/O on Windows.

Two problems this fixes, both rooted in Windows defaulting to cp1252:

1. Reads: some libraries read bundled text files (e.g. jinja chat
   templates) via Path.read_text() without specifying encoding, which
   fails on UTF-8 content.

2. Writes: the frozen exe is built with console=True, so when the server
   runs attached to a real console, sys.stdout/sys.stderr use the console
   code page (cp1252). Status/log strings containing characters like
   "→" are written there directly via rich.Console.print(), raising
   UnicodeEncodeError ("'charmap' codec can't encode character ...") that
   propagates and aborts bootstrap.

Runtime hooks run before any application imports, so reconfiguring the
streams here means modules that capture sys.stdout (e.g. rich.Console)
inherit UTF-8.
"""
import sys

if sys.platform == "win32":
    import pathlib

    _original_read_text = pathlib.Path.read_text

    def _read_text_utf8(self, encoding=None, errors=None):
        if encoding is None:
            encoding = "utf-8"
        return _original_read_text(self, encoding=encoding, errors=errors)

    pathlib.Path.read_text = _read_text_utf8

    # Force stdout/stderr to UTF-8. errors="replace" guarantees a stray
    # un-encodable character degrades to "?" instead of crashing. Streams
    # may be None under a windowed build or lack reconfigure() if already
    # wrapped, so guard accordingly.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
