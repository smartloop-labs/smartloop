"""Pre-defined crawl plans for document-type MCP servers.

Registering an MCP server indexes its *tools* (see
:class:`~smartloop.db.mcp_tool_indexer.MCPToolIndexer`), which is enough to
resolve a step to a tool on demand.  It leaves the server's *content* behind a
live call: nothing about the user's Notion pages, Linear issues or Trello cards
is in context until the router happens to pick the right tool for a query and
that call happens to succeed.

A crawl plan closes that gap.  It is a declarative, read-only walk over the
listing tools a person would call to see "everything" on a server.  The results
are vectorized as documents (see
:class:`~smartloop.db.mcp_document_indexer.MCPDocumentIndexer`) and refreshed on
a cycle, so retrieval has the server's content from the first message instead of
depending on a tool call landing.

A plan is resolved for a server by:

1. the curated catalog slug it was installed from (``catalog_slug`` — the
   provenance the backend established itself, so it is checked first);
2. a loose match of the plan's ``match`` substrings against the server's name,
   package and URL, for a hand-added Notion/Linear/Trello connection;
3. :func:`generic_plan` — steps derived from the tools the server actually
   reported, used when nothing curated matches.

Tool names are matched *loosely* (see :func:`resolve_tool`) because the same
server is published under several naming conventions (``search``,
``notion-search``, ``notion_search``), and every step is skipped when the
server has no tool for it — so a plan can list candidates freely without
breaking servers that expose only some of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Argument names a per-record detail call takes the record's identifier under.
ID_ARGS: Tuple[str, ...] = (
    "id", "page_id", "pageId", "url", "uri", "identifier", "key", "gid",
)

# Keys a record's identifier is read from, in order of preference.
RECORD_ID_KEYS: Tuple[str, ...] = (
    "id", "uuid", "gid", "key", "identifier", "url", "uri", "page_id", "shortLink",
)

# Keys a record's human-readable title is read from, in order of preference.
RECORD_TITLE_KEYS: Tuple[str, ...] = (
    "title", "name", "subject", "displayName", "display_name", "summary", "label",
)

# Keys whose value is the record list inside an envelope object.
RECORD_LIST_KEYS: Tuple[str, ...] = (
    "results", "items", "records", "data", "nodes", "edges", "entries",
    "pages", "documents", "issues", "cards", "boards", "lists", "projects",
    "teams", "databases", "objects", "value", "content",
)


@dataclass(frozen=True)
class CrawlStep:
    """One read-only listing call in a crawl plan.

    ``tools`` are candidate tool names, first match on the server wins.
    ``arguments`` are filtered against the resolved tool's input schema, so an
    argument the server doesn't declare is dropped rather than rejected.

    A step with ``parent`` set fans out over the records an earlier step
    produced, passing each parent record's id as the first of ``parent_args``
    the tool declares — that is how boards → lists → cards is expressed.

    ``detail_tools`` fetch the full body of each record (a Notion page's
    content behind a search hit), again by whichever of ``detail_args`` the
    tool declares.
    """

    label: str
    tools: Tuple[str, ...]
    arguments: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None
    parent_args: Tuple[str, ...] = ID_ARGS
    detail_tools: Tuple[str, ...] = ()
    detail_args: Tuple[str, ...] = ID_ARGS
    # Records kept from this step's listing calls.  A crawl is a background
    # refresh, not an export: the cap bounds both the embedding work and the
    # number of follow-up detail calls.
    max_records: int = 40
    # Parent records fanned out over, when ``parent`` is set.
    max_parents: int = 8


@dataclass(frozen=True)
class CrawlPlan:
    """An ordered set of steps that crawls one kind of server."""

    slug: str
    steps: Tuple[CrawlStep, ...]
    # Substrings matched (case-insensitively) against a server's name, package
    # and URL when it carries no catalog slug.
    match: Tuple[str, ...] = ()
    # True for the derived fallback plan, so callers can tell a curated plan
    # (crawl by default) from a guessed one (crawl only when asked).
    derived: bool = False


# --------------------------------------------------------------- curated plans

# Notion's page fetch, shared by every listing step below: a listing returns
# titles and ids, the body has to be fetched per page.
_NOTION_FETCH = (
    "notion-fetch", "notion_fetch", "fetch", "get_page", "retrieve_page",
    "notion_retrieve_page",
)

NOTION_PLAN = CrawlPlan(
    slug="notion",
    match=("notion",),
    steps=(
        # The dedicated listing tools first: they take no required argument, so
        # they are the reliable way to enumerate what the connection can see.
        # Their results overlap heavily (a recent page is usually also a private
        # or shared one) — the crawler dedupes by record id across steps, so the
        # overlap costs nothing.
        CrawlStep(
            label="recent-pages",
            tools=("notion-list-recent-pages", "list_recent_pages"),
            arguments={"limit": 50},
            detail_tools=_NOTION_FETCH,
        ),
        CrawlStep(
            label="private-pages",
            tools=("notion-list-private-pages", "list_private_pages"),
            arguments={"limit": 50},
            detail_tools=_NOTION_FETCH,
        ),
        CrawlStep(
            label="shared-pages",
            tools=("notion-list-shared-pages", "list_shared_pages"),
            arguments={"limit": 50},
            detail_tools=_NOTION_FETCH,
        ),
        CrawlStep(
            label="favorite-pages",
            tools=("notion-list-favorite-pages", "list_favorite_pages"),
            arguments={"limit": 50},
            detail_tools=_NOTION_FETCH,
        ),
        # Search last, and only as a fallback for the Notion servers that ship
        # no listing tools: the hosted server makes `query` required, so an
        # empty-query call there simply fails and the step is skipped.
        CrawlStep(
            label="pages",
            tools=("search", "notion-search", "notion_search", "search_pages"),
            arguments={"query": ""},
            detail_tools=_NOTION_FETCH,
        ),
        CrawlStep(
            label="databases",
            tools=("list_databases", "notion_list_databases", "get_databases",
                   "notion-list-databases"),
        ),
        CrawlStep(
            label="meeting-notes",
            tools=("notion-query-meeting-notes", "query_meeting_notes"),
            detail_tools=_NOTION_FETCH,
        ),
    ),
)

LINEAR_PLAN = CrawlPlan(
    slug="linear",
    match=("linear",),
    steps=(
        CrawlStep(
            label="teams",
            tools=("list_teams", "linear_list_teams", "get_teams"),
            max_records=20,
        ),
        CrawlStep(
            label="projects",
            tools=("list_projects", "linear_list_projects", "get_projects"),
            detail_tools=("get_project", "linear_get_project"),
        ),
        CrawlStep(
            label="issues",
            tools=("list_issues", "linear_list_issues", "get_issues",
                   "list_my_issues"),
            arguments={"limit": 50},
            detail_tools=("get_issue", "linear_get_issue"),
        ),
        CrawlStep(
            label="documents",
            tools=("list_documents", "linear_list_documents", "get_documents"),
            detail_tools=("get_document", "linear_get_document"),
        ),
    ),
)

TRELLO_PLAN = CrawlPlan(
    slug="trello",
    match=("trello",),
    steps=(
        CrawlStep(
            label="boards",
            tools=("list_boards", "get_boards", "trello_list_boards",
                   "get_my_boards", "list_boards_by_user"),
            max_records=20,
        ),
        CrawlStep(
            label="lists",
            tools=("get_lists", "list_lists", "trello_get_lists",
                   "get_lists_by_board_id", "get_board_lists"),
            parent="boards",
            parent_args=("board_id", "boardId", "idBoard", "id"),
        ),
        CrawlStep(
            label="cards",
            tools=("get_cards_by_list_id", "list_cards", "get_cards",
                   "trello_get_cards", "get_list_cards"),
            parent="lists",
            parent_args=("list_id", "listId", "idList", "id"),
            max_records=60,
        ),
    ),
)

PLANS: Dict[str, CrawlPlan] = {
    plan.slug: plan for plan in (NOTION_PLAN, LINEAR_PLAN, TRELLO_PLAN)
}


# ------------------------------------------------------------- tool resolution

def _normalize(name: str) -> str:
    """Lowercase *name* with separators stripped: ``notion-search`` → ``notionsearch``."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def resolve_tool(
    tool_names: Sequence[str], candidates: Sequence[str]
) -> Optional[str]:
    """Return the first of *candidates* the server actually exposes, or None.

    Matching widens in three passes so the closest name always wins: exact
    (normalized) equality, then a prefix/suffix match — which is what pairs a
    plan's ``search`` with a server's ``notion-search`` — and finally a
    substring match.  Candidate order decides ties, so a plan lists its
    preferred name first.
    """
    if not tool_names or not candidates:
        return None
    normalized = [(name, _normalize(name)) for name in tool_names if name]

    for candidate in candidates:
        target = _normalize(candidate)
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


def schema_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    props = schema.get("properties") if isinstance(schema, dict) else None
    return props if isinstance(props, dict) else {}


def resolve_argument(
    schema: Dict[str, Any], candidates: Sequence[str]
) -> Optional[str]:
    """Return the first of *candidates* the tool's schema declares, or None.

    Falls back to the tool's single required property when no candidate
    matches by name — a detail tool that takes exactly one required argument
    takes the record's id under whatever it happens to call it.
    """
    props = schema_properties(schema)
    if props:
        by_norm = {_normalize(k): k for k in props}
        for candidate in candidates:
            match = by_norm.get(_normalize(candidate))
            if match:
                return match
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list) and len(required) == 1 and isinstance(required[0], str):
        return required[0]
    return None


def filter_arguments(
    schema: Dict[str, Any], arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Drop arguments the tool doesn't declare.

    A plan states what it *would* like to pass (``{"limit": 50}``); servers that
    don't take it would reject the whole call, so unknown keys are dropped.  A
    tool with no declared properties is passed the arguments as given — the
    schema is then telling us nothing, not telling us "no arguments".
    """
    if not arguments:
        return {}
    props = schema_properties(schema)
    if not props:
        return dict(arguments)
    by_norm = {_normalize(k): k for k in props}
    resolved: Dict[str, Any] = {}
    for key, value in arguments.items():
        match = by_norm.get(_normalize(key))
        if match:
            resolved[match] = value
    return resolved


# --------------------------------------------------------------- generic plan

# Leading verbs of a tool that lists or searches content (as opposed to
# fetching one known record, which is what a detail call does).
_LIST_VERBS = frozenset({
    "list", "search", "browse", "query", "find", "all", "recent",
})
# Tools whose leading verb reads a *single* record are only useful as detail
# calls, so they are never crawl entry points.
_DETAIL_VERBS = frozenset({"get", "fetch", "read", "retrieve", "describe", "show"})

_MAX_GENERIC_STEPS = 6


def _leading_verb(tool_name: str) -> str:
    parts = re.split(r"[_\-\s.]+", (tool_name or "").strip().lower())
    return parts[0] if parts else ""


def _is_plural_getter(tool_name: str) -> bool:
    """True for ``get_boards`` / ``fetch-all-pages``: a detail verb that
    nonetheless returns a collection."""
    norm = _normalize(tool_name)
    return norm.endswith("s") or "all" in norm


def generic_plan(server: Any) -> CrawlPlan:
    """Derive a crawl plan from the tools *server* reported.

    Used for a server with no curated plan.  Only listing/search tools that can
    be called with no required arguments become steps (a required argument we
    can't fill would just error), capped at :data:`_MAX_GENERIC_STEPS` so an
    unusually large server can't turn one refresh cycle into hundreds of calls.
    """
    steps: List[CrawlStep] = []
    for tool in getattr(server, "tools", None) or []:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        if not name:
            continue
        verb = _leading_verb(name)
        if verb not in _LIST_VERBS and not (
            verb in _DETAIL_VERBS and _is_plural_getter(name)
        ):
            continue
        schema = tool_schema(server, name)
        required = schema.get("required") if isinstance(schema, dict) else None
        arguments: Dict[str, Any] = {}
        if isinstance(required, list) and required:
            # A search tool whose only requirement is the query string can be
            # called with an empty query; anything else needs an argument we
            # would have to invent, so it is not a crawl entry point.
            query_args = [r for r in required if _normalize(r) in ("query", "q", "searchquery", "text")]
            if len(required) > len(query_args) or not query_args:
                continue
            arguments = {query_args[0]: ""}
        steps.append(CrawlStep(label=name, tools=(name,), arguments=arguments))
        if len(steps) >= _MAX_GENERIC_STEPS:
            break

    return CrawlPlan(
        slug=f"generic:{getattr(server, 'name', 'server')}",
        steps=tuple(steps),
        derived=True,
    )


# --------------------------------------------------------------- plan lookup

def _server_haystack(server: Any) -> str:
    remote = getattr(server, "remote_config", None)
    parts = [
        getattr(server, "catalog_slug", None),
        getattr(server, "name", None),
        getattr(server, "pkg", None),
        getattr(remote, "server_url", None) if remote else None,
    ]
    return " ".join(p for p in parts if p).lower()


def resolve_plan(server: Any) -> CrawlPlan:
    """Return the crawl plan for *server* — curated when one matches, else derived."""
    slug = (getattr(server, "catalog_slug", None) or "").strip().lower()
    if slug in PLANS:
        return PLANS[slug]

    haystack = _server_haystack(server)
    for plan in PLANS.values():
        if any(token in haystack for token in plan.match):
            return plan

    return generic_plan(server)


def should_crawl(server: Any) -> bool:
    """True when *server*'s content should be crawled on the refresh cycle.

    ``server.crawl`` is the explicit answer when set.  Left unset (the default),
    a server is crawled when a *curated* plan matches it: those are the
    document-type connections whose content is worth having in context ahead of
    a query.  Everything else — a weather or shell server, where "all the
    content" is not a thing — stays on-demand until someone opts it in.
    """
    if not getattr(server, "enabled", True) or getattr(server, "system", False):
        return False
    explicit = getattr(server, "crawl", None)
    if explicit is not None:
        return bool(explicit)
    return not resolve_plan(server).derived
