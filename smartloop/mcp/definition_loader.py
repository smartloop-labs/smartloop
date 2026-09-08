"""Loading MCP connection definitions from TOML.

Two sources, later merged over earlier:

1. ``<home>/mcp.toml`` — the user's own file, if they wrote one.
2. ``<home>/mcp.d/*.toml`` — drop-ins, read in name order.  This is where a
   connection goes: one file per server, added and removed without touching
   anything else.

A definition whose ``slug`` matches an earlier one replaces it.  A file that
fails to parse or validate is skipped with a logged reason and the rest still
load: one malformed drop-in must not take the other connections down with it.

``mcp.toml.example`` beside this module documents the format with worked
examples; nothing reads it.  A bundled ``mcp.toml`` used to ship curated crawl
plans here — crawling is gone (see :mod:`smartloop.mcp.definitions`), and a
server's tools are enumerated at registration instead.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import tomllib
from typing import Dict, List, Optional, Tuple

from smartloop.mcp.definitions import MCPDefinition, retired_fields

logger = logging.getLogger(__name__)

#: Where a user's own definitions go, relative to the SLP home directory.
USER_DEFINITIONS_FILE = "mcp.toml"
USER_DEFINITIONS_DIR = "mcp.d"


def user_definition_paths(home_dir: Optional[str] = None) -> List[str]:
    """Every user-supplied definition file, in the order they are merged."""
    if home_dir is None:
        try:
            from smartloop.config.app_settings import AppSettings
            home_dir = AppSettings().home_dir
        except Exception:
            return []
    paths: List[str] = []
    single = os.path.join(home_dir, USER_DEFINITIONS_FILE)
    if os.path.isfile(single):
        paths.append(single)
    paths.extend(sorted(glob.glob(os.path.join(home_dir, USER_DEFINITIONS_DIR, "*.toml"))))
    return paths


def parse_definitions(text: str, source: str) -> Dict[str, MCPDefinition]:
    """Parse one TOML document into ``{key: MCPDefinition}``.

    Raises on a malformed document — callers decide whether one bad file is
    fatal (it is, for an upload the user is watching) or not (it isn't, for the
    drop-in directory).  The retired indexing fields are not malformed: they
    are dropped, and logged once per entry that carries them.
    """
    document = tomllib.loads(text)
    found: Dict[str, MCPDefinition] = {}
    entries = document.get("mcp") or []
    if not isinstance(entries, list):
        raise ValueError("'mcp' must be a list of [[mcp]] tables")
    for raw in entries:
        if not isinstance(raw, dict):
            raise ValueError("each [[mcp]] must be a table")
        retired = retired_fields(raw)
        definition = MCPDefinition(**raw)
        found[definition.key] = definition
        if retired:
            # The connection installed; the indexing it asked for no longer
            # exists.  Said out loud so the file can be tidied, rather than
            # leaving someone to wonder why nothing gets crawled.
            logger.warning(
                "[mcp-definitions] %s: '%s' declares %s, which no longer does "
                "anything — a connection's tools are called when a question "
                "needs them, nothing is indexed ahead of time",
                source, definition.key, ", ".join(retired),
            )
        logger.debug("[mcp-definitions] %s: loaded '%s'", source, definition.key)
    return found


def load_definitions(home_dir: Optional[str] = None) -> Dict[str, MCPDefinition]:
    """Every definition the user's files declare, by key."""
    registry: Dict[str, MCPDefinition] = {}
    for path in user_definition_paths(home_dir):
        try:
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8")
            registry.update(parse_definitions(text, path))
        except FileNotFoundError:
            continue
        except Exception as e:
            # One bad drop-in must not take the working connections down.
            logger.warning("[mcp-definitions] Skipping %s: %s", path, e)
    return registry


# --------------------------------------------------------------------- cache

_lock = threading.Lock()
_cache: Optional[Dict[str, MCPDefinition]] = None
_cache_stamp: Optional[Tuple[Tuple[str, float], ...]] = None


def _stamp(home_dir: Optional[str]) -> Tuple[Tuple[str, float], ...]:
    """(path, mtime) per definition file, so a dropped-in file is noticed.

    A handful of stat calls, which is what lets someone add a file and have the
    next request see it rather than restarting the app.
    """
    stamp: List[Tuple[str, float]] = []
    for path in user_definition_paths(home_dir):
        try:
            stamp.append((path, os.path.getmtime(path)))
        except OSError:
            continue
    return tuple(stamp)


def definitions(home_dir: Optional[str] = None) -> Dict[str, MCPDefinition]:
    """The definition registry, reloaded when any file has changed."""
    global _cache, _cache_stamp
    stamp = _stamp(home_dir)
    with _lock:
        if _cache is None or stamp != _cache_stamp:
            _cache = load_definitions(home_dir)
            _cache_stamp = stamp
            logger.info(
                "[mcp-definitions] %d definition(s) loaded: %s",
                len(_cache), ", ".join(sorted(_cache)) or "none",
            )
        return _cache


def reload_definitions() -> Dict[str, MCPDefinition]:
    """Drop the cache and read every definition file again."""
    global _cache, _cache_stamp
    with _lock:
        _cache = None
        _cache_stamp = None
    return definitions()
