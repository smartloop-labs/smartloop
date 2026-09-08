"""What to call *next*, once a server has answered.

A listing tool answers "which records", not "what do they say".  ``notion-search``
comes back with page titles and ids; the pitch document's text is behind a
second call, ``notion-fetch(page_id=…)``.  Nothing in a tool's schema says the
two go together — that pairing is knowledge about the server, and this is where
it lives:

    [[plan]]
    slug = "notion"
    match = ["notion"]
    detail_tools = ["notion-fetch", "API-get-block-children", "get_page"]
    detail_args  = ["page_id", "pageId", "block_id", "id", "url"]
    detail_arguments = { include_discussions = true }

Given a payload from any listing tool, :func:`follow_up_calls` returns the
per-record calls that fetch the bodies — so an answer about a Notion page is
written from the page rather than from its title.

This is the surviving half of the crawl plans.  Those plans walked a server on a
schedule and vectorized what came back; the walk is gone, and what is kept is
the part that was never really about crawling: which tool fetches a record, and
which argument it takes the record's id under.  Applied to the call a question
just made, not to a background pass.

Plans are data (``call_plans.toml`` beside this module, plus ``[[plan]]``
entries in any user file under ``~/.smartloop/mcp.d/``), so a server can be
taught without a code change.  A connection with no plan simply gets no
follow-up, which is right for the many servers whose listing *is* the answer.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import tomllib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from smartloop.mcp.definitions import normalize_name

logger = logging.getLogger(__name__)

#: The bundled plans, alongside this module.
BUNDLED_PLANS_PATH = os.path.join(os.path.dirname(__file__), "call_plans.toml")

# Argument names a per-record detail call takes the record's identifier under.
ID_ARGS: Tuple[str, ...] = (
    "id", "page_id", "pageId", "url", "uri", "identifier", "key", "gid",
)

# Keys a record's identifier is read from, in order of preference.
#
# Real identifiers first, bare spellings ahead of entity-qualified ones: a
# server returning both means `id` by the record itself and `<entity>_id` by
# something it points at, while the ones returning only `design_id` / `page_id`
# have no other identity to offer.  `url` and `uri` come last, as the fallback
# for listings that carry no id at all (hosted Notion is one).
RECORD_ID_KEYS: Tuple[str, ...] = (
    "id", "uuid", "gid", "key", "identifier",
    "page_id", "design_id", "shortLink",
    "url", "uri",
)

# Keys whose value is the record list inside an envelope object.
RECORD_LIST_KEYS: Tuple[str, ...] = (
    "results", "items", "records", "data", "nodes", "edges", "entries",
    "pages", "documents", "issues", "cards", "boards", "lists", "projects",
    "teams", "databases", "objects", "value", "content",
)

# How many records one answer follows up on.  A search returns twenty hits and
# the question is about one or two of them; twenty round-trips to find that out
# is a worse answer, later.
DEFAULT_MAX_RECORDS = 3


class CallPlan(BaseModel):
    """How to fetch the body of a record one of *this* server's listings named."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slug: str = Field(min_length=1)
    # Substrings matched (case-insensitively) against an installed server's
    # name, package and URL.  `slug` is always tried.
    match: Tuple[str, ...] = ()
    # Candidate names for the tool that fetches one record, best first.
    detail_tools: Tuple[str, ...] = ()
    # Argument names that tool takes the record's id under.
    detail_args: Tuple[str, ...] = ID_ARGS
    # Fixed arguments the detail call also needs — the content type to return,
    # or the operation on a server where one tool covers several.  Filtered
    # against the tool's schema, so an argument it doesn't declare is dropped
    # rather than rejecting the call.
    detail_arguments: Dict[str, Any] = Field(default_factory=dict)
    max_records: int = Field(default=DEFAULT_MAX_RECORDS, ge=1)

    @property
    def match_terms(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys((self.slug.lower(), *(m.lower() for m in self.match))))


# ------------------------------------------------------------------ loading

_lock = threading.Lock()
_cache: Optional[Dict[str, CallPlan]] = None


def parse_plans(text: str, source: str) -> Dict[str, CallPlan]:
    """Parse ``[[plan]]`` entries out of one TOML document."""
    document = tomllib.loads(text)
    found: Dict[str, CallPlan] = {}
    for raw in document.get("plan") or []:
        if not isinstance(raw, dict):
            raise ValueError("each [[plan]] must be a table")
        plan = CallPlan(**raw)
        found[plan.slug.lower()] = plan
        logger.debug("[call-plans] %s: loaded '%s'", source, plan.slug)
    return found


def load_plans(home_dir: Optional[str] = None) -> Dict[str, CallPlan]:
    """The bundled plans, with any user file merged over them."""
    from smartloop.mcp.definition_loader import user_definition_paths

    registry: Dict[str, CallPlan] = {}
    for path in [BUNDLED_PLANS_PATH, *user_definition_paths(home_dir)]:
        try:
            with open(path, "rb") as fh:
                registry.update(parse_plans(fh.read().decode("utf-8"), path))
        except FileNotFoundError:
            if path == BUNDLED_PLANS_PATH:
                logger.error("[call-plans] Bundled plans missing at %s", path)
        except Exception as e:
            # One bad file must not cost every other server its follow-up.
            level = logger.error if path == BUNDLED_PLANS_PATH else logger.warning
            level("[call-plans] Skipping %s: %s", path, e)
    return registry


def plans(home_dir: Optional[str] = None) -> Dict[str, CallPlan]:
    """The plan registry, loaded once."""
    global _cache
    with _lock:
        if _cache is None:
            _cache = load_plans(home_dir)
            logger.info(
                "[call-plans] %d follow-up plan(s) loaded: %s",
                len(_cache), ", ".join(sorted(_cache)) or "none",
            )
        return _cache


def reload_plans() -> Dict[str, CallPlan]:
    """Drop the cache and read the plan files again."""
    global _cache
    with _lock:
        _cache = None
    return plans()


# ------------------------------------------------------------ plan lookup

def _server_haystack(server: Any) -> str:
    parts = [
        str(getattr(server, "name", "") or ""),
        str(getattr(server, "pkg", "") or ""),
        str(getattr(server, "catalog_slug", "") or ""),
    ]
    remote = getattr(server, "remote_config", None)
    if remote is not None:
        parts.append(str(getattr(remote, "server_url", "") or ""))
    config = getattr(server, "config", None)
    if config is not None:
        parts.append(str(getattr(config, "command", "") or ""))
        parts.extend(str(a) for a in (getattr(config, "args", None) or []))
    return " ".join(parts).lower()


def plan_for(server: Any) -> Optional[CallPlan]:
    """The follow-up plan for *server*, or None when it has no plan.

    The catalog slug wins — it is provenance the backend established — and the
    declared match terms are the fallback for a hand-added connection.
    """
    registry = plans()
    if not registry:
        return None
    slug = (getattr(server, "catalog_slug", None) or "").strip().lower()
    if slug and slug in registry:
        return registry[slug]
    haystack = _server_haystack(server)
    for plan in registry.values():
        if any(term and term in haystack for term in plan.match_terms):
            return plan
    return None


# --------------------------------------------------------- schema helpers

def tool_schema(server: Any, tool_name: str) -> Dict[str, Any]:
    """Return a tool's input schema from *server*, or ``{}``."""
    for tool in getattr(server, "tools", None) or []:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        if name != tool_name:
            continue
        schema = (
            tool.get("input_schema") if isinstance(tool, dict)
            else getattr(tool, "input_schema", None)
        )
        return schema or {}
    return {}


def _properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props if isinstance(props, dict) else {}


def resolve_tool(tool_names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    """The first of *candidates* the server actually exposes, or None.

    Matching widens in three passes so the closest name always wins: exact
    (normalized) equality, then a prefix/suffix match — which is what pairs a
    plan's ``fetch`` with a server's ``notion-fetch`` — and finally a substring
    match.  Candidate order decides ties, so a plan lists its preferred name
    first.
    """
    if not tool_names or not candidates:
        return None
    normalized = [(name, normalize_name(name)) for name in tool_names if name]
    for candidate in candidates:
        target = normalize_name(candidate)
        if not target:
            continue
        for name, norm in normalized:
            if norm == target:
                return name
        for name, norm in normalized:
            if norm.endswith(target) or norm.startswith(target):
                return name
        for name, norm in normalized:
            if target in norm:
                return name
    return None


def resolve_argument(
    schema: Dict[str, Any], candidates: Sequence[str],
) -> Optional[str]:
    """The first of *candidates* the tool declares, or its lone required arg.

    A detail tool that takes exactly one required argument takes the record's
    id under whatever it happens to call it.
    """
    props = _properties(schema)
    if props:
        by_norm = {normalize_name(k): k for k in props}
        for candidate in candidates:
            match = by_norm.get(normalize_name(candidate))
            if match:
                return match
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
        return required[0]
    return None


def filter_arguments(
    schema: Dict[str, Any], arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Drop arguments the tool doesn't declare.

    A plan states what it *would* like to pass; a server that doesn't take it
    would reject the whole call, so unknown keys go.  A tool with no declared
    properties is passed them as given — an empty schema is telling us nothing,
    not telling us "no arguments".
    """
    if not arguments:
        return {}
    props = _properties(schema)
    if not props:
        return dict(arguments)
    by_norm = {normalize_name(k): k for k in props}
    resolved: Dict[str, Any] = {}
    for key, value in arguments.items():
        match = by_norm.get(normalize_name(key))
        if match:
            resolved[match] = value
    return resolved


def coerce_argument(schema: Dict[str, Any], name: str, value: Any) -> Any:
    """Cast a record id to what the tool says the argument is.

    monday.com's board ids come back as strings and its item tools declare
    numbers; sending the string is rejected by the server, and the id is the
    one argument a follow-up cannot do without.
    """
    declared = _properties(schema).get(name)
    if not isinstance(declared, dict):
        return value
    types = []
    raw = declared.get("type")
    if isinstance(raw, str):
        types.append(raw)
    elif isinstance(raw, list):
        types.extend(t for t in raw if isinstance(t, str))
    if "string" in types or not types:
        return value
    if "number" in types or "integer" in types:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            try:
                return float(str(value))
            except (TypeError, ValueError):
                return value
    return value


# -------------------------------------------------------- record extraction

def _first_str(data: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    """The first of *keys* whose value is a usable scalar."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
        # Notion-style rich values: {"title": {"content": "..."}}
        if isinstance(value, dict):
            nested = _first_str(value, ("content", "plain_text", "text", "name", "title"))
            if nested:
                return nested
    return None


def _unwrap_records(data: Any) -> List[Any]:
    """The record list inside whatever envelope a server wrapped it in."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in RECORD_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _unwrap_records(value)
            if nested:
                return nested
    # A single record answered on its own is a list of one.
    if any(k in data for k in RECORD_ID_KEYS):
        return [data]
    return []


def record_ids(payload: str, limit: int = DEFAULT_MAX_RECORDS) -> List[str]:
    """Ids of the records *payload* lists, in order, at most *limit* of them.

    A tool result is text; most servers put JSON in it, and the ones that don't
    have no ids to follow up on.  Both are ordinary — an unparseable payload
    means no follow-up, not an error.
    """
    text = (payload or "").strip()
    if not text:
        return []
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1), default=-1,
    )
    if start == -1:
        return []
    try:
        data = json.loads(text[start:])
    except Exception:
        return []

    ids: List[str] = []
    seen: set = set()
    for item in _unwrap_records(data):
        if isinstance(item, dict) and isinstance(item.get("node"), dict):
            item = item["node"]          # GraphQL edges
        if not isinstance(item, dict):
            continue
        record_id = _first_str(item, RECORD_ID_KEYS)
        if not record_id or record_id in seen:
            continue
        seen.add(record_id)
        ids.append(record_id)
        if len(ids) >= limit:
            break
    return ids


# ------------------------------------------------------------- the follow-up

def follow_up_calls(
    server: Any, tool_name: str, payload: str,
) -> List[Tuple[str, Dict[str, Any]]]:
    """``[(detail_tool, arguments), …]`` to fetch the bodies *payload* names.

    Empty — the common case, and never an error — when the connection has no
    plan, the plan's detail tool is not on this server, the call that produced
    *payload* was already that detail tool, or the payload carries no ids.
    """
    plan = plan_for(server)
    if plan is None or not plan.detail_tools:
        return []

    names = [
        (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
        for t in (getattr(server, "tools", None) or [])
    ]
    detail_tool = resolve_tool([n for n in names if n], plan.detail_tools)
    if not detail_tool or detail_tool == tool_name:
        return []

    schema = tool_schema(server, detail_tool)
    id_arg = resolve_argument(schema, plan.detail_args)
    if not id_arg:
        return []

    fixed = filter_arguments(schema, plan.detail_arguments)
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for record_id in record_ids(payload, plan.max_records):
        arguments = dict(fixed)
        arguments[id_arg] = coerce_argument(schema, id_arg, record_id)
        calls.append((detail_tool, arguments))
    return calls
