"""Loading crawl plans from TOML.

The plans used to be Python literals in :mod:`smartloop.mcp.crawl_plans`, which
made adding one a code change and shipping one a release.  They are data — a
list of tool names and how to walk between them — so they live in ``mcp.toml``
and are validated against the Pydantic models on load.

Three sources, later ones merged over earlier:

1. ``smartloop/mcp/mcp.toml`` — bundled with the package.
2. ``<home>/mcp.toml`` — the user's own file, if they wrote one.
3. ``<home>/mcp.d/*.toml`` — drop-ins, read in name order.  This is where a
   custom server's plan goes: one file per connection, added and removed
   without touching anything else.

A plan whose ``slug`` matches an earlier one replaces it, or appends to it when
it sets ``extend = true`` — so a user can add a step to the built-in Notion
walk without restating the other eleven.

A file that fails to parse or validate is skipped with a logged reason and the
rest still load: one malformed drop-in must not take the Notion connection down
with it.
"""

from __future__ import annotations

import glob
import logging
import os
import threading
import tomllib
from typing import Any, Dict, List, Optional, Tuple

from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep, MCPDefinition

logger = logging.getLogger(__name__)

#: The bundled plans, alongside this module.
BUNDLED_PLANS_PATH = os.path.join(os.path.dirname(__file__), "mcp.toml")

#: Where a user's own plans go, relative to the SLP home directory.
USER_PLANS_FILE = "mcp.toml"
USER_PLANS_DIR = "mcp.d"

# Fields whose value is a list of names and may therefore contain "@alias".
_ALIASABLE = ("tools", "detail_tools", "detail_args", "parent_args", "needs_arguments")


def user_plan_paths(home_dir: Optional[str] = None) -> List[str]:
    """Every user-supplied plan file, in the order they are merged."""
    if home_dir is None:
        try:
            from smartloop.config.app_settings import AppSettings
            home_dir = AppSettings().home_dir
        except Exception:
            return []
    paths: List[str] = []
    single = os.path.join(home_dir, USER_PLANS_FILE)
    if os.path.isfile(single):
        paths.append(single)
    paths.extend(sorted(glob.glob(os.path.join(home_dir, USER_PLANS_DIR, "*.toml"))))
    return paths


def _expand_aliases(value: Any, aliases: Dict[str, List[str]]) -> Any:
    """Replace ``"@name"`` entries in a list with the alias's contents.

    Expansion is in place, so ``["@notion_fetch", "my_tool"]`` reads as the
    alias followed by the extra name — the order a plan author wrote is the
    order the crawler tries.
    """
    if not isinstance(value, list):
        return value
    expanded: List[Any] = []
    for item in value:
        if isinstance(item, str) and item.startswith("@"):
            name = item[1:]
            if name not in aliases:
                raise ValueError(f"unknown alias '{item}'")
            expanded.extend(aliases[name])
        else:
            expanded.append(item)
    return expanded


def _step_from_toml(raw: Dict[str, Any], aliases: Dict[str, List[str]]) -> CrawlStep:
    data = dict(raw)
    for key in _ALIASABLE:
        if key in data:
            data[key] = _expand_aliases(data[key], aliases)
    # `for_each` holds its list one level down, and a shared watchlist is
    # exactly the kind of thing worth naming once.
    if isinstance(data.get("for_each"), dict):
        data["for_each"] = {
            name: _expand_aliases(values, aliases)
            for name, values in data["for_each"].items()
        }
    # `parent` takes one label or several; TOML has no scalar-or-list type, so
    # both spellings are accepted and normalized by the model.
    return CrawlStep(**data)


def parse_definitions(text: str, source: str) -> Dict[str, MCPDefinition]:
    """Parse one TOML document into ``{key: MCPDefinition}``.

    Only ``[[mcp]]`` entries carry connection details, so a file written in the
    older indexing-only ``[[plan]]`` shape yields nothing here.
    """
    found: Dict[str, MCPDefinition] = {}
    parse_plans(text, source, definitions=found)
    return found


def parse_plans(
    text: str,
    source: str,
    definitions: Optional[Dict[str, MCPDefinition]] = None,
) -> List[Tuple[CrawlPlan, bool]]:
    """Parse one TOML document into ``(plan, extend)`` pairs.

    *definitions*, when given, collects the full server definitions alongside —
    the crawl only needs the plans, but installing a server from a file needs
    the connection too.

    Raises on a malformed document — callers decide whether one bad file is
    fatal (it isn't, for the drop-in directory) or worth surfacing.
    """
    if definitions is None:
        definitions = {}
    document = tomllib.loads(text)
    aliases_raw = document.get("alias") or {}
    if not isinstance(aliases_raw, dict):
        raise ValueError("'alias' must be a table of name = [names]")
    aliases: Dict[str, List[str]] = {
        name: list(value) for name, value in aliases_raw.items()
        if isinstance(value, list)
    }

    plans: List[Tuple[CrawlPlan, bool]] = []

    # `[[mcp]]` is the shape a file uses to describe a server: what it is called,
    # how to reach it, and — optionally — how to index it.  `[[plan]]` is the
    # earlier indexing-only spelling, still accepted so a file written against
    # it keeps working.
    for raw in document.get("mcp") or []:
        if not isinstance(raw, dict):
            raise ValueError("each [[mcp]] must be a table")
        entry = dict(raw)
        extend = bool(entry.pop("extend", False))
        raw_steps = entry.pop("index", None) or entry.pop("step", None) or []
        entry["index"] = [_step_from_toml(s, aliases) for s in raw_steps]
        definition = MCPDefinition(**entry)
        definitions[definition.key] = definition
        plans.append((
            CrawlPlan.model_construct(
                slug=definition.key, steps=definition.index,
                match=definition.plan.match, derived=False,
            ) if extend else definition.plan,
            extend,
        ))
        logger.debug("[mcp-plans] %s: loaded '%s'", source, definition.key)

    for raw_plan in document.get("plan") or []:
        if not isinstance(raw_plan, dict):
            raise ValueError("each [[plan]] must be a table")
        entry = dict(raw_plan)
        extend = bool(entry.pop("extend", False))
        raw_steps = entry.pop("step", None) or []
        entry["steps"] = [_step_from_toml(s, aliases) for s in raw_steps]
        if extend:
            # An extension's steps are validated against the plan they are
            # merged into, not on their own — a step whose parent lives in the
            # built-in plan is legal there and only there.
            plans.append((
                CrawlPlan.model_construct(
                    slug=entry.get("slug", ""),
                    steps=tuple(entry["steps"]),
                    match=tuple(entry.get("match", ())),
                    derived=False,
                ),
                True,
            ))
        else:
            plans.append((CrawlPlan(**entry), False))
        logger.debug("[mcp-plans] %s: loaded plan '%s'", source, entry.get("slug"))
    return plans


def _merge(registry: Dict[str, CrawlPlan], plan: CrawlPlan, extend: bool, source: str) -> None:
    existing = registry.get(plan.slug)
    if not extend or existing is None:
        if extend:
            logger.warning(
                "[mcp-plans] %s: plan '%s' sets extend but there is nothing to "
                "extend — treating it as a new plan", source, plan.slug,
            )
            registry[plan.slug] = CrawlPlan(
                slug=plan.slug, steps=plan.steps, match=plan.match,
            )
            return
        registry[plan.slug] = plan
        return

    # Appending, so the built-in steps stay ahead of the added ones and a new
    # step can fan out over any of them.  Re-validated as a whole: that is what
    # catches an added step colliding with a built-in label, or naming a parent
    # that doesn't exist.
    registry[plan.slug] = CrawlPlan(
        slug=existing.slug,
        steps=existing.steps + plan.steps,
        match=tuple(dict.fromkeys(existing.match + plan.match)),
    )


def load_plans(
    home_dir: Optional[str] = None,
    definitions: Optional[Dict[str, MCPDefinition]] = None,
) -> Dict[str, CrawlPlan]:
    """Build the plan registry from the bundled file plus any user files.

    *definitions*, when given, collects the server definitions from the same
    files — the same pass, since re-reading them separately would be a second
    place for the two views to disagree.
    """
    registry: Dict[str, CrawlPlan] = {}
    paths = [BUNDLED_PLANS_PATH, *user_plan_paths(home_dir)]

    for path in paths:
        try:
            with open(path, "rb") as fh:
                text = fh.read().decode("utf-8")
            for plan, extend in parse_plans(text, path, definitions=definitions):
                _merge(registry, plan, extend, path)
        except FileNotFoundError:
            if path == BUNDLED_PLANS_PATH:
                logger.error(
                    "[mcp-plans] Bundled plans missing at %s — only derived "
                    "plans will be available", path,
                )
        except Exception as e:
            # One bad drop-in must not take the working plans down with it.
            level = logger.error if path == BUNDLED_PLANS_PATH else logger.warning
            level("[mcp-plans] Skipping %s: %s", path, e)

    return registry


# --------------------------------------------------------------------- cache

_lock = threading.Lock()
_cache: Optional[Dict[str, CrawlPlan]] = None
_definitions: Dict[str, MCPDefinition] = {}
_cache_stamp: Optional[Tuple[Tuple[str, float], ...]] = None


def _stamp(home_dir: Optional[str]) -> Tuple[Tuple[str, float], ...]:
    """(path, mtime) for every plan file, so a dropped-in file is noticed.

    A handful of stat calls per crawl, which is what lets someone add a plan and
    hit re-index rather than restart the app to see it take effect.
    """
    stamp: List[Tuple[str, float]] = []
    for path in [BUNDLED_PLANS_PATH, *user_plan_paths(home_dir)]:
        try:
            stamp.append((path, os.path.getmtime(path)))
        except OSError:
            continue
    return tuple(stamp)


def plans(home_dir: Optional[str] = None) -> Dict[str, CrawlPlan]:
    """The plan registry, reloaded when any plan file has changed."""
    global _cache, _cache_stamp, _definitions
    stamp = _stamp(home_dir)
    with _lock:
        if _cache is None or stamp != _cache_stamp:
            found: Dict[str, MCPDefinition] = {}
            _cache = load_plans(home_dir, definitions=found)
            _definitions = found
            _cache_stamp = stamp
            logger.info(
                "[mcp-plans] %d crawl plan(s) loaded: %s",
                len(_cache), ", ".join(sorted(_cache)) or "none",
            )
        return _cache


def definitions(home_dir: Optional[str] = None) -> Dict[str, MCPDefinition]:
    """Server definitions from the same files, by key.

    Only entries written as ``[[mcp]]`` appear — those are the ones that carry
    a connection.
    """
    plans(home_dir)
    return _definitions


def reload_plans() -> Dict[str, CrawlPlan]:
    """Drop the cache and read every plan file again."""
    global _cache, _cache_stamp
    with _lock:
        _cache = None
        _cache_stamp = None
    return plans()
