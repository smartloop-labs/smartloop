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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class CrawlStep(BaseModel):
    """One read-only listing call in a crawl plan.

    ``tools`` are candidate tool names, first match on the server wins.
    ``arguments`` are filtered against the resolved tool's input schema, so an
    argument the server doesn't declare is dropped rather than rejected.

    A step with ``parent`` set fans out over the records earlier steps
    produced, passing each parent record's id as the first of ``parent_args``
    the tool declares — that is how boards → lists → cards, or teams →
    projects → issues → comments, is expressed.  ``parent`` may name several
    steps, in which case their records are pooled and deduped by id: a Linear
    issue's comments hang off whichever issue listing found the issue, and the
    workspace-wide, per-team and per-project listings all feed the same
    ``list_comments`` fan-out.

    ``detail_tools`` fetch the full body of each record (a Notion page's
    content behind a search hit), again by whichever of ``detail_args`` the
    tool declares.

    Plans are authored in TOML (see :mod:`smartloop.mcp.plan_loader`), so this
    model is what a hand-written file is validated against — ``extra="forbid"``
    turns a misspelled key into an error naming it, rather than a field that
    silently does nothing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)
    tools: Tuple[str, ...] = Field(min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    parent: Optional[Union[str, Tuple[str, ...]]] = None
    parent_args: Tuple[str, ...] = ID_ARGS
    detail_tools: Tuple[str, ...] = ()
    detail_args: Tuple[str, ...] = ID_ARGS
    # Arguments without which this step is pointless.  Unknown arguments are
    # normally dropped so a step still runs in reduced form, but a step that
    # exists *because* of one argument — the same search narrowed to databases
    # — degrades into a duplicate of another step when it is dropped, so it is
    # skipped instead.
    needs_arguments: Tuple[str, ...] = ()
    # Records kept from this step's listing calls.  A crawl is a background
    # refresh, not an export: the cap bounds both the embedding work and the
    # number of follow-up detail calls.
    max_records: int = Field(default=40, ge=1)
    # Parent records fanned out over, when ``parent`` is set.
    max_parents: int = Field(default=8, ge=1)

    @property
    def parent_labels(self) -> Tuple[str, ...]:
        """The step labels this step fans out over, ``()`` for a root step."""
        if not self.parent:
            return ()
        if isinstance(self.parent, str):
            return (self.parent,)
        return tuple(self.parent)


class CrawlPlan(BaseModel):
    """An ordered set of steps that crawls one kind of server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slug: str = Field(min_length=1)
    steps: Tuple[CrawlStep, ...] = ()
    # Substrings matched (case-insensitively) against a server's name, package
    # and URL when it carries no catalog slug.
    match: Tuple[str, ...] = ()
    # True for the derived fallback plan, so callers can tell a curated plan
    # (crawl by default) from a guessed one (crawl only when asked).
    derived: bool = False

    @model_validator(mode="after")
    def _check_step_graph(self) -> "CrawlPlan":
        """Reject a plan whose fan-out graph can't run.

        Both failures here are silent at crawl time — the step simply never
        produces anything — and both are easy to write in a hand-authored
        file, so they are errors at load time instead:

        * a duplicate label, which would make one step's records overwrite the
          other's in the fan-out bookkeeping;
        * a ``parent`` that names a step which doesn't exist, or one declared
          *later*, since a step can only fan out over records an earlier step
          already collected.
        """
        seen: set = set()
        for step in self.steps:
            if step.label in seen:
                raise ValueError(
                    f"plan '{self.slug}': duplicate step label '{step.label}'"
                )
            for parent in step.parent_labels:
                if parent not in seen:
                    raise ValueError(
                        f"plan '{self.slug}': step '{step.label}' fans out over "
                        f"'{parent}', which is not an earlier step in this plan"
                    )
            seen.add(step.label)
        return self


# --------------------------------------------------------------- curated plans
#
# The curated plans are data, not code: they live in ``mcp.toml`` next to this
# module and are loaded and validated by :mod:`smartloop.mcp.plan_loader`.  A
# user's own plans are merged over them from ``~/.smartloop/mcp.d/*.toml``,
# which is what lets a custom server be indexed without a code change.


def plans() -> Dict[str, "CrawlPlan"]:
    """Every curated plan, by slug.  Reloaded when a plan file changes."""
    from smartloop.mcp.plan_loader import plans as _plans
    return _plans()


# ------------------------------------------------------------- tool resolution

def normalize_name(name: str) -> str:
    """Lowercase *name* with separators stripped: ``notion-search`` → ``notionsearch``.

    The one place tool-name spelling is normalised.  Anything comparing tool or
    argument names goes through it, so ``notion-search``/``notion_search``/
    ``NotionSearch`` are one name everywhere rather than in each caller's own
    approximation of the rule.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# Internal alias, kept because this module reads better with the short name.
_normalize = normalize_name


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


def declares_argument(schema: Dict[str, Any], candidates: Sequence[str]) -> bool:
    """True when the tool declares any of *candidates* as a property.

    Unlike :func:`resolve_argument` this never falls back to "the single
    required property": the question here is whether the tool really
    understands this specific argument, not what a record id could go in.
    """
    props = schema_properties(schema)
    if not props:
        # An empty schema tells us nothing; assume the argument is understood
        # rather than skipping a step on no evidence.
        return True
    by_norm = {_normalize(k) for k in props}
    return any(_normalize(c) in by_norm for c in candidates)


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

# Root steps (a listing that needs no argument) and child steps (a listing
# whose one required argument is a parent record's id) are capped separately:
# the children are the tree walk, and it is worth some budget of its own.
_MAX_GENERIC_STEPS = 6
_MAX_GENERIC_CHILD_STEPS = 6

_QUERY_ARGS = ("query", "q", "searchquery", "text")


def _leading_verb(tool_name: str) -> str:
    parts = re.split(r"[_\-\s.]+", (tool_name or "").strip().lower())
    return parts[0] if parts else ""


def _is_plural_getter(tool_name: str) -> bool:
    """True for ``get_boards`` / ``fetch-all-pages``: a detail verb that
    nonetheless returns a collection."""
    norm = _normalize(tool_name)
    return norm.endswith("s") or "all" in norm


def _entity_of_id_arg(arg_name: str) -> str:
    """``teamId`` → ``team``; ``''`` for an argument that isn't an entity id.

    A bare ``id`` names no entity, so it says nothing about which listing could
    supply it and is not a link the walk can follow.
    """
    norm = _normalize(arg_name)
    if not norm.endswith("id") or norm in ("id", "uuid", "guid", "gid"):
        return ""
    return norm[:-2]


def _tool_names(server: Any) -> List[str]:
    names = []
    for tool in getattr(server, "tools", None) or []:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        if name:
            names.append(name)
    return names


def generic_plan(server: Any) -> CrawlPlan:
    """Derive a crawl plan from the tools *server* reported.

    Used for a server with no curated plan, and built in two passes so the
    derived plan walks the server's tree the way a curated one does:

    1. *roots* — listing/search tools callable with no required argument.  A
       required argument we can't fill would just error, so those are not entry
       points.
    2. *children* — listings whose one required argument is another entity's id
       (``list_issues(teamId)``), linked to the root step whose name carries
       that entity (``list_teams``).  Without this pass the whole per-team,
       per-board, per-project half of a server is unreachable: those tools are
       exactly the ones pass 1 has to reject.

    Each pass is capped so an unusually large server can't turn one refresh
    cycle into hundreds of calls.
    """
    steps: List[CrawlStep] = []
    deferred: List[Tuple[str, str, Dict[str, Any]]] = []

    for name in _tool_names(server):
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
            # called with an empty query.
            query_args = [r for r in required if _normalize(r) in _QUERY_ARGS]
            others = [r for r in required if r not in query_args]
            if others:
                # Everything else has to come from somewhere.  A single id-shaped
                # requirement can come from a parent listing, so the tool is held
                # back for pass 2 rather than dropped.
                entity = _entity_of_id_arg(others[0]) if len(others) == 1 else ""
                if entity:
                    deferred.append((
                        name, entity,
                        {query_args[0]: ""} if query_args else {},
                    ))
                continue
            if not query_args:
                continue
            arguments = {query_args[0]: ""}
        if len(steps) < _MAX_GENERIC_STEPS:
            steps.append(CrawlStep(label=name, tools=(name,), arguments=arguments))

    children = 0
    for name, entity, arguments in deferred:
        if children >= _MAX_GENERIC_CHILD_STEPS:
            break
        # The parent is the root listing whose own name is about that entity:
        # `teamId` → `list_teams`, `board_id` → `get_boards`.
        parents = tuple(
            step.label for step in steps
            if entity in _normalize(step.label) and step.label != name
        )
        if not parents:
            continue
        steps.append(CrawlStep(
            label=name,
            tools=(name,),
            arguments=arguments,
            parent=parents,
            # The entity name is the only reliable hint; the schema resolver
            # picks whichever spelling the tool declares.
            parent_args=(f"{entity}_id", f"{entity}Id", f"{entity}id", entity),
        ))
        children += 1

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
    registry = plans()
    slug = (getattr(server, "catalog_slug", None) or "").strip().lower()
    if slug in registry:
        return registry[slug]

    haystack = _server_haystack(server)
    for plan in registry.values():
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
