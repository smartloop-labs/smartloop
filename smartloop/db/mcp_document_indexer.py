"""
MCPDocumentIndexer — crawls a document-type MCP server and vectorizes its
content into a per-project ``mcp_documents`` collection.

The tool index (:mod:`smartloop.db.mcp_tool_indexer`) makes a server's *tools*
retrievable; this makes its *content* retrievable.  For a Notion, Linear or
Trello connection that is the difference between "the model can call a tool if
the router picks the right one" and "the pages, issues and cards are already in
context when the question is asked".

The walk itself is declared by a crawl plan (see
:mod:`smartloop.mcp.crawl_plans`): read-only listing calls, optional per-record
detail calls, and a fan-out for parent/child shapes like boards → lists → cards.
Nothing here calls an LLM — a crawl is pure I/O, so it runs on a background
cycle without competing for the model.

Each crawled record becomes one document:

* ``document_id``   — ``<server_id>:<hash of step + record key>``, stable across
  crawls so an unchanged record is neither re-embedded nor duplicated.
* ``document_name`` — ``<server>/<step>/<title>``, which is what retrieval shows
  as the source.
* metadata          — ``server_id``, ``server_name``, ``plan``, ``step``,
  ``record_id``, ``source_url``, ``tool_name``, ``crawled_at``.

A manifest next to the store (``.mcp_documents_manifest.json``) records the
content hash of every indexed document per server, so a refresh re-embeds only
what changed and deletes what the server no longer returns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from smartloop.db.store import ChunkResult
from smartloop.db.store_factory import StoreFactory
from smartloop.mcp.crawl_plans import (
    RECORD_ID_KEYS,
    RECORD_LIST_KEYS,
    RECORD_TITLE_KEYS,
    CrawlPlan,
    CrawlStep,
    coerce_argument,
    declares_argument,
    filter_arguments,
    resolve_argument,
    render_url,
    resolve_plan,
    resolve_tool,
    tool_schema,
)
from smartloop.mcp.read_only_calls import (
    ToolCallResult,
    call_read_only,
    ensure_tool_caller,
)
from smartloop.utils.mcp_result_formatter import ResultFormatter

logger = logging.getLogger(__name__)

MCP_DOCUMENTS_COLLECTION = "mcp_documents"

_MANIFEST_FILENAME = ".mcp_documents_manifest.json"

# Characters per chunk / overlap — same sizing as the workspace indexer, so
# every collection in a project feeds the embedding model comparable spans.
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Hard ceiling on the text kept for one record.  A crawl is a summary of the
# server's content, not a mirror of it: a single 2MB Notion page would otherwise
# dominate a project's index.
MAX_RECORD_CHARS = 20_000

# Tool calls one server's crawl may make in a cycle.  A plan walks a tree — a
# listing per team, a detail fetch per issue, a comment listing per issue — and
# the per-step caps bound each branch but not their product.  This bounds the
# whole walk, so an unusually large workspace costs a bounded number of
# requests and the rest is picked up on the next cycle.
MAX_CRAWL_CALLS = 400

# Consecutive failed calls within one step before the step gives up for this
# cycle.  A fan-out asks the same tool the same question once per parent, so a
# tool that rejects the shape of the call rejects all of them — walking the
# remaining parents just spends the budget on a known answer.
MAX_STEP_FAILURES = 3


# ------------------------------------------------------------------ helpers

@dataclass
class CrawlRecord:
    """One record returned by a crawl step."""

    record_id: Optional[str]
    title: Optional[str]
    text: str
    url: Optional[str] = None


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split *text* into overlapping chunks of at most *size* characters."""
    if len(text) <= size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def _first_str(data: Dict[str, Any], keys: Sequence[str]) -> Optional[str]:
    """Return the first key in *keys* whose value is a usable scalar."""
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


def _nested_record_lists(data: Dict[str, Any], depth: int = 2) -> List[Any]:
    """Pool the record lists a known key names *below* this level.

    monday.com's ``workspace_info`` answers with ``{"workspace": {...},
    "folders": [...], "root_items": {"boards": [...]}}``: the boards are the
    records, two keys down, under an envelope name no other server uses.  The
    search stops at the shallowest level that has any match and pools every
    match there, so a board in a folder and a board at the root arrive as one
    list rather than one of them winning.

    Only the vocabulary in ``RECORD_LIST_KEYS`` counts, at any depth — pooling
    whatever lists happen to be nested would turn a record's own sub-objects
    (an issue's labels, a user's teams) into records of their own.
    """
    pooled: List[Any] = []
    for key in RECORD_LIST_KEYS:
        value = data.get(key)
        if isinstance(value, list) and value and all(isinstance(i, dict) for i in value):
            pooled.extend(value)
    if pooled or depth <= 1:
        return pooled
    for value in data.values():
        if isinstance(value, dict):
            pooled.extend(_nested_record_lists(value, depth - 1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    pooled.extend(_nested_record_lists(item, depth - 1))
    return pooled


def _unwrap_records(data: Any) -> List[Any]:
    """Find the record list inside a parsed tool result.

    Handles the shapes servers actually return: a bare list, an envelope naming
    its list (``{"results": [...]}``, ``{"issues": {"nodes": [...]}}``), an
    envelope that names it something new or keeps it a level or two down, and a
    single object.
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in RECORD_LIST_KEYS:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = _unwrap_records(value)
            if nested:
                return nested
    # A single-key envelope around something else (GraphQL "data" wrappers).
    if len(data) == 1:
        only = next(iter(data.values()))
        if isinstance(only, (dict, list)):
            nested = _unwrap_records(only)
            if nested:
                return nested
    # An envelope that names its list with a word this module has not seen:
    # ``{"workspaces": [...], "nextCursor": ..., "hasMore": false}``.  Every
    # server pluralizes its own entities, so enumerating the names is a losing
    # game — one list-of-objects key and no identity of its own is taken as the
    # envelope it is.  The identity check is what keeps ``{"id": ...,
    # "labels": [...]}`` a single record: a record has an id, an envelope
    # doesn't.
    if not _first_str(data, RECORD_ID_KEYS):
        listed = [
            value for value in data.values()
            if isinstance(value, list) and value
            and all(isinstance(item, dict) for item in value)
        ]
        if len(listed) == 1:
            return listed[0]
        # Or names them a level or two further in: monday.com hands back the
        # boards of a workspace under `data.root_items.boards`, and they are the
        # only place a board id can be read from on that server.
        nested = _nested_record_lists(data)
        if nested:
            return nested
    return [data]


def _record_text(record: Any) -> str:
    """Render one record as the plain key/value text that gets embedded.

    Ids, URLs and timestamps are stripped (they are noise in an embedding and
    are kept in metadata instead) via the same compaction the live tool path
    uses, so crawled content reads like tool output rather than raw JSON.
    """
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, (dict, list)):
        return str(record)
    try:
        compacted = ResultFormatter.compact_json(record)
        return ResultFormatter.flatten_to_text(compacted).strip()
    except Exception:
        return json.dumps(record, default=str)[:MAX_RECORD_CHARS]


def _unwrap_record(item: Any) -> Any:
    """Unwrap a record the server wrapped in a type discriminator.

    Canva's folder listing answers with ``{"type": "design", "design": {...}}``
    — the identity, the title and the body all one level down.  Read as-is such
    a record has no id to fan out from and no title to name a document by, so it
    is keyed by a hash of its own text and the same design reached from a search
    is documented a second time.

    Unwrapped only when the outer object has no identity of its own *and* the
    inner one does, so a record that merely happens to carry a nested object
    keeps its own shape.
    """
    if not isinstance(item, dict) or _first_str(item, RECORD_ID_KEYS):
        return item
    nested = [value for value in item.values() if isinstance(value, dict)]
    if len(nested) == 1 and _first_str(nested[0], RECORD_ID_KEYS):
        return nested[0]
    return item


def _parse_records(payload: str) -> List[CrawlRecord]:
    """Parse a tool result payload into crawl records.

    JSON (the common case) is unwrapped into one record per object.  Anything
    else — markdown from a Notion page fetch, a plain-text listing — is one
    record carrying the whole payload.
    """
    payload = (payload or "").strip()
    if not payload:
        return []

    data: Any = None
    if payload[0] in "[{":
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            data = None

    if data is None:
        return [CrawlRecord(record_id=None, title=None, text=payload[:MAX_RECORD_CHARS])]

    records: List[CrawlRecord] = []
    for item in _unwrap_records(data):
        item = _unwrap_record(item)
        text = _record_text(item)
        if not text:
            continue
        record_id = title = url = None
        if isinstance(item, dict):
            record_id = _first_str(item, RECORD_ID_KEYS)
            title = _first_str(item, RECORD_TITLE_KEYS)
            url = _first_str(
                item, ("url", "uri", "link", "shortUrl", "html_url", "edit_url"),
            )
        records.append(CrawlRecord(
            record_id=record_id,
            title=title,
            text=text[:MAX_RECORD_CHARS],
            url=url,
        ))
    return records


def _merge_listing_and_detail(listing: str, detail: str) -> str:
    """Combine a listing row with the record's fetched body.

    The listing row carries fields the body may not have (an issue's status and
    assignee), so it is kept — but for a page fetch the body repeats the row
    verbatim ("title: Roadmap" twice), and duplicated lines are pure dilution in
    an embedding.  Only the listing lines the body doesn't already contain
    survive.
    """
    detail = (detail or "").strip()
    if not detail:
        return listing
    listing = (listing or "").strip()
    if not listing:
        return detail
    extra = [
        line for line in listing.splitlines()
        if line.strip() and line.strip() not in detail
    ]
    return "\n".join([*extra, detail]).strip() if extra else detail


# Failures that will repeat on every cycle: the tool is gated behind a paid
# plan, the grant lacks the scope, or the arguments the plan can build don't
# satisfy the schema.  Steps that fail this way are remembered and skipped
# until an explicit refresh, so a periodic crawl stops paying for a call whose
# answer cannot change on its own.
_PERMANENT_FAILURE_RE = re.compile(
    r"requires a .* plan|upgrade your plan|not entitled|"
    r"input validation error|invalid arguments|"
    r"validation_error|failed validation|"
    r"unauthorized|forbidden|permission denied|not found",
    re.IGNORECASE,
)


def _is_permanent_failure(reason: Optional[str]) -> bool:
    return bool(reason and _PERMANENT_FAILURE_RE.search(reason))


def _unsatisfied_required(
    schema: Dict[str, Any], arguments: Dict[str, Any]
) -> Optional[str]:
    """Return a required argument the call cannot satisfy, or None.

    A plan supplies ``{"query": ""}`` to mean "list everything", which works on
    servers that treat an empty query as unfiltered — and is a guaranteed
    validation error on the ones that require a non-empty string.  Checking the
    schema first turns that into a skipped step instead of a failed round-trip
    on every refresh.
    """
    required = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(required, list):
        return None
    for name in required:
        if not isinstance(name, str):
            continue
        value = arguments.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return name
    return None


def _has_required(schema: Dict[str, Any]) -> bool:
    """True when the tool declares required arguments.

    A bare retry of such a tool is a guaranteed error, so it is skipped rather
    than spending a round-trip to learn that again.
    """
    required = schema.get("required") if isinstance(schema, dict) else None
    return bool(required)


# Characters kept from one step's failure reason in the per-step log line.  A
# server's validation error can run to hundreds of characters of JSON; the part
# that names the offending field comes first, and the rest turns a summary of
# twelve steps into an unreadable wall.
MAX_STEP_REASON_CHARS = 120


def _format_step_details(details: Sequence[dict]) -> str:
    """Render a crawl's per-step outcomes as one log line.

    ``step/tool: <records>`` for a step that produced something, with the fresh
    count when it differs (a listing whose records another step already
    documented is not the same event as a listing that returned nothing), and
    ``step/tool: 0 (why)`` for one that didn't.
    """
    parts: List[str] = []
    for detail in details:
        label = detail.get("step") or "?"
        tool = detail.get("tool")
        prefix = f"{label}/{tool}" if tool else label
        records = detail.get("records") or 0
        if records:
            fresh = detail.get("new")
            suffix = str(records)
            if fresh is not None and fresh != records:
                suffix = f"{records} ({fresh} new)"
        else:
            reason = " ".join(str(detail.get("reason") or "").split())
            if len(reason) > MAX_STEP_REASON_CHARS:
                reason = reason[:MAX_STEP_REASON_CHARS] + "..."
            suffix = f"0 ({reason})" if reason else "0"
        parts.append(f"{prefix}: {suffix}")
    return "; ".join(parts) or "plan had no steps"


def _doc_id(server_id: str, step_label: str, key: str) -> str:
    """Stable per-record document id, scoped to the server."""
    digest = hashlib.md5(f"{step_label}|{key}".encode("utf-8")).hexdigest()[:16]
    return f"{server_id}:{digest}"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ------------------------------------------------------------------ indexer

class MCPDocumentIndexer:
    """Crawls MCP servers and indexes their content into ``mcp_documents``."""

    def __init__(
        self,
        project_dir: str,
        logger: Optional[logging.Logger] = None,
        base_dir: Optional[str] = None,
        tool_caller: Optional[Any] = None,
    ):
        self._project_dir = project_dir
        self._logger = logger or logging.getLogger(__name__)
        self._store = StoreFactory.create_store(
            project_dir=project_dir,
            collection_name=MCP_DOCUMENTS_COLLECTION,
            logger=self._logger,
            base_dir=base_dir,
        )
        self._manifest_path = os.path.join(project_dir, _MANIFEST_FILENAME)
        self._manifest: Dict[str, dict] = self._load_manifest()
        self._lock = threading.Lock()
        self._tool_caller = tool_caller
        # Server ids with a crawl in flight.  The crawl runs on a background
        # task, so "is this server still being indexed?" is a question the API
        # can only answer if the walk says so while it is happening.
        self._crawling: set = set()

    # ---------------------------------------------------------------- public

    def crawl_server(
        self,
        server: Any,
        ttl: float = 0.0,
        force: bool = False,
        on_server_updated: Optional[Callable[[Any], None]] = None,
    ) -> Dict[str, Any]:
        """Crawl one server's content and index what changed.

        *ttl* skips a server crawled less than that many seconds ago, so a
        forced refresh (project switch, a newly registered server) doesn't
        re-walk every connection.  *force* ignores it.

        *on_server_updated* is called when the crawl mutated the server — an
        OAuth token refresh, or the crawl timestamp — so the caller can persist
        the workspace.

        Returns ``{"indexed", "skipped", "removed", "records", "steps",
        "plan", "reason"}``.
        """
        server_id = getattr(server, "id", None)
        with self._lock:
            if server_id:
                self._crawling.add(server_id)
            from smartloop.db.embeddings import indexing_burst

            try:
                with indexing_burst():
                    return self._crawl_server(server, ttl, force, on_server_updated)
            finally:
                self._crawling.discard(server_id)

    def search(
        self,
        query: str,
        n_results: int = 8,
        max_distance: float = 0.8,
    ) -> List[ChunkResult]:
        """Semantic search over crawled MCP content."""
        try:
            return self._store.query_with_metadata(
                query, n_results=n_results, max_distance=max_distance
            )
        except Exception as e:
            self._logger.warning(f"[mcp-crawl] Search failed: {e}")
            return []

    def remove_server(self, server_id: str) -> int:
        """Delete every crawled document for a server.  Returns the count."""
        with self._lock:
            entry = self._manifest.pop(server_id, None)
            doc_ids = list((entry or {}).get("docs", {}).keys())
            for doc_id in doc_ids:
                try:
                    self._store.delete_document(doc_id)
                except Exception:
                    pass
            if entry is not None:
                self._save_manifest()
            if doc_ids:
                self._logger.info(
                    f"[mcp-crawl] Removed {len(doc_ids)} crawled document(s) "
                    f"for server {server_id}"
                )
            return len(doc_ids)

    def count(self) -> int:
        """Total number of crawled content chunks in this project."""
        try:
            return self._store.count()
        except Exception as e:
            self._logger.warning(f"[mcp-crawl] count failed: {e}")
            return 0

    def is_crawling(self, server_id: str) -> bool:
        """True while a crawl of *server_id* is in flight."""
        return server_id in self._crawling

    def server_status(self, server_id: str) -> Dict[str, Any]:
        """Crawl state for one server.

        Kept separate from :meth:`status` because the per-server view is read
        on every poll of a connection's detail page, and counting the whole
        collection to answer it would be wasted work.
        """
        entry = self._manifest.get(server_id) or {}
        return {
            "crawled_at": entry.get("crawled_at"),
            "documents": len(entry.get("docs", {})),
            "records": entry.get("records", 0),
            "plan": entry.get("plan"),
            "crawling": self.is_crawling(server_id),
        }

    def status(self) -> Dict[str, Any]:
        """Per-server crawl state, for the /health and diagnostics endpoints."""
        return {
            "chunks": self.count(),
            "servers": {
                server_id: self.server_status(server_id)
                for server_id in self._manifest
            },
        }

    def last_crawl_at(self, server_id: str) -> Optional[float]:
        entry = self._manifest.get(server_id) or {}
        return entry.get("crawled_at")

    # ---------------------------------------------------------------- crawl

    def _crawl_server(
        self,
        server: Any,
        ttl: float,
        force: bool,
        on_server_updated: Optional[Callable[[Any], None]],
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "indexed": 0, "skipped": 0, "removed": 0,
            "records": 0, "steps": 0, "plan": None, "reason": None,
            "calls": 0, "truncated": False,
            # One entry per plan step: which tool it resolved to, how many
            # records it produced, and why it produced none.  A crawl that
            # silently indexes nothing is the failure mode that matters here,
            # so the outcome is reported rather than logged at debug and lost.
            "details": [],
        }
        server_id = getattr(server, "id", None)
        if not server_id:
            stats["reason"] = "no server id"
            return stats

        entry = self._manifest.get(server_id) or {}
        last = entry.get("crawled_at") or 0.0
        now = time.time()
        if not force and ttl and (now - last) < ttl:
            stats["reason"] = "fresh"
            stats["skipped"] = len(entry.get("docs", {}))
            return stats

        plan = resolve_plan(server)
        stats["plan"] = plan.slug
        if not plan.steps:
            stats["reason"] = "no crawlable listing tools"
            self._logger.info(
                "[mcp-crawl] %s: nothing to crawl (no listing tool matched)",
                getattr(server, "name", server_id),
            )
            return stats

        tool_names = [
            (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
            for t in (getattr(server, "tools", None) or [])
        ]
        tool_names = [n for n in tool_names if n]
        if not tool_names:
            stats["reason"] = "server has no discovered tools"
            return stats

        # Steps a previous crawl found permanently blocked (paid-plan tool,
        # missing scope, arguments the schema rejects).  A forced crawl ignores
        # the memory and retries everything, which is the escape hatch when the
        # plan or permissions change.
        blocked: Dict[str, str] = dict(entry.get("blocked") or {})
        new_blocked: Dict[str, str] = {}

        credentials_changed = False
        records_by_step: Dict[str, List[CrawlRecord]] = {}
        documents: Dict[str, dict] = {}
        # Record ids already documented in this crawl.  Plans deliberately
        # overlap (a Notion page is listed as recent *and* private *and*
        # shared), so without this the same page is embedded several times and
        # its body fetched once per listing.
        seen_records: set = set()

        def call(tool_name: str, arguments: Dict[str, Any]):
            """Spend one unit of the crawl's call budget.

            Past the budget the walk stops making requests rather than
            failing: what it already indexed stands, and the branches it did
            not reach are picked up on the next cycle.  The exhaustion reason
            is deliberately not one ``_is_permanent_failure`` recognises — a
            step cut short by the budget must not be remembered as blocked.
            """
            if stats["calls"] >= MAX_CRAWL_CALLS:
                stats["truncated"] = True
                return None, False, "call budget for this cycle used up"
            stats["calls"] += 1
            return self._call_tool(server, tool_name, arguments)

        for step in plan.steps:
            tool = resolve_tool(tool_names, step.tools)
            if not tool:
                stats["details"].append({
                    "step": step.label, "tool": None, "records": 0,
                    "reason": "no matching tool on this server",
                })
                continue

            blocked_reason = None if force else blocked.get(step.label)
            if blocked_reason:
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": f"skipped (last failure: {blocked_reason})",
                })
                continue

            schema = tool_schema(server, tool)
            if step.needs_arguments and not declares_argument(schema, step.needs_arguments):
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": f"tool declares none of {list(step.needs_arguments)}, "
                              f"which is the only thing this step adds",
                })
                continue

            arg_sets, parents = self._step_arguments(step, schema, records_by_step)
            if arg_sets is None:
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": f"nothing to fan out from "
                              f"'{'/'.join(step.parent_labels)}'"
                              if step.parent_labels
                              else "no arguments could be built",
                })
                continue

            # Drop the calls the tool's own schema would reject.  A plan says
            # `{"query": ""}` to mean "list everything", which works where an
            # empty query is unfiltered and is a guaranteed validation error
            # where the string must be non-empty — cheaper to notice here than
            # to re-learn over the network every cycle.
            unsatisfied = {
                name for name in (
                    _unsatisfied_required(schema, args) for args in arg_sets
                ) if name
            }
            arg_sets = [
                args for args in arg_sets
                if not _unsatisfied_required(schema, args)
            ]
            if not arg_sets:
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": f"tool requires {sorted(unsatisfied)} and the plan "
                              f"has no value for it",
                })
                continue

            # What a failed call is retried with: the plan's own arguments may
            # not fit this server's variant of the tool.  A fan-out step keeps
            # the parent id — without it the retry is a different, unscoped
            # question, asked once per parent.
            retry_keys = set(step.arguments) | set(filter_arguments(schema, step.arguments))

            records: List[CrawlRecord] = []
            failure: Optional[str] = None
            consecutive_failures = 0
            for arguments in arg_sets:
                payload, changed, error = call(tool, arguments)
                credentials_changed = credentials_changed or changed
                retry_args = {
                    k: v for k, v in arguments.items() if k not in retry_keys
                }
                if payload is None and retry_args != arguments and not _has_required(schema):
                    # Only worth retrying when the tool declares nothing as
                    # required — otherwise the retry fails for sure.
                    payload, changed, error = call(tool, retry_args)
                    credentials_changed = credentials_changed or changed
                if not payload:
                    reason = error or "no content returned"
                    failure = failure or reason
                    consecutive_failures += 1
                    # The tool rejected the call itself, not this parent: the
                    # next parent is the identical call with a different id in
                    # it.  Stop rather than proving it once per parent.
                    if _is_permanent_failure(reason):
                        self._logger.info(
                            "[mcp-crawl] step '%s' abandoned after %s",
                            step.label, reason,
                        )
                        break
                    if consecutive_failures >= MAX_STEP_FAILURES:
                        self._logger.info(
                            "[mcp-crawl] step '%s' abandoned after %d consecutive "
                            "failures (last: %s)",
                            step.label, consecutive_failures, reason,
                        )
                        break
                    continue
                consecutive_failures = 0
                records.extend(_parse_records(payload))
                if len(records) >= step.max_records:
                    break

            records = records[:step.max_records]
            records_by_step[step.label] = records
            if not records:
                reason = failure or "returned no records"
                if _is_permanent_failure(reason):
                    new_blocked[step.label] = reason
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": reason,
                })
                continue

            # Keep the step's own record list intact for child steps to fan out
            # over, but only document what hasn't been seen yet.
            fresh = []
            for record in records:
                # Ids first; a record with neither id nor title (a plain-text
                # listing) still falls back to its own content, so two steps
                # that resolved to the same tool don't index it twice.
                key = record.record_id or record.title or _content_hash(record.text)
                if key in seen_records:
                    continue
                seen_records.add(key)
                fresh.append(record)

            stats["steps"] += 1
            stats["details"].append({
                "step": step.label, "tool": tool, "records": len(records),
                "new": len(fresh), "reason": None,
            })
            if not fresh:
                continue

            detail_tool = resolve_tool(tool_names, step.detail_tools) if step.detail_tools else None
            detail_schema = tool_schema(server, detail_tool) if detail_tool else {}
            detail_arg = (
                resolve_argument(detail_schema, step.detail_args)
                if detail_tool else None
            )
            # The record id is not always the whole call: a tool that dispatches
            # on an `action` needs to be told which operation this is.
            detail_base = filter_arguments(detail_schema, step.detail_arguments)

            for index, record in enumerate(fresh):
                text = record.text
                if detail_tool and detail_arg and record.record_id:
                    detail_payload, changed, _ = call(
                        detail_tool,
                        {
                            **detail_base,
                            detail_arg: coerce_argument(
                                detail_schema, detail_arg, record.record_id,
                            ),
                        },
                    )
                    credentials_changed = credentials_changed or changed
                    detail_text = self._detail_text(detail_payload)
                    if detail_text:
                        # Keep whatever the listing row adds (status, assignee,
                        # dates) and append the body: the row alone is rarely the
                        # answer, and the body alone can lose the metadata.
                        text = _merge_listing_and_detail(
                            record.text, detail_text,
                        )[:MAX_RECORD_CHARS]

                key = record.record_id or record.title or f"{index}"
                doc_id = _doc_id(server_id, step.label, key)
                title = record.title or record.record_id or f"{step.label} {index + 1}"
                documents[doc_id] = {
                    "name": f"{getattr(server, 'name', 'mcp')}/{step.label}/{title}",
                    "text": text,
                    "step": step.label,
                    "tool_name": tool,
                    "record_id": record.record_id or "",
                    # A step may know a durable address for its records where
                    # the listing only handed back an ephemeral one.
                    "source_url": (
                        render_url(step.url_template, record.record_id)
                        or record.url or ""
                    ),
                    "title": title,
                }

            stats["records"] += len(fresh)
            if parents:
                self._logger.debug(
                    "[mcp-crawl] %s: step '%s' fanned out over %d parent(s)",
                    getattr(server, "name", server_id), step.label, parents,
                )

        indexed, skipped = self._index_documents(server, plan, documents, entry, now)
        stats["indexed"], stats["skipped"] = indexed, skipped
        stats["removed"] = self._remove_stale(entry, documents)

        self._manifest[server_id] = {
            "crawled_at": now,
            "plan": plan.slug,
            "records": stats["records"],
            # Carried forward so a blocked step stays skipped across restarts;
            # a forced crawl starts the memory over.
            "blocked": new_blocked if force else {**blocked, **new_blocked},
            "docs": {
                doc_id: _content_hash(doc["text"]) for doc_id, doc in documents.items()
            },
        }
        self._save_manifest()

        try:
            server.last_crawl_at = now
        except Exception:
            pass
        if on_server_updated is not None and (credentials_changed or documents):
            try:
                on_server_updated(server)
            except Exception as e:
                self._logger.debug(f"[mcp-crawl] server update callback failed: {e}")

        self._logger.info(
            "[mcp-crawl] %s (plan=%s): %d record(s) over %d step(s) in %d call(s)%s "
            "— indexed=%d, unchanged=%d, removed=%d",
            getattr(server, "name", server_id), plan.slug, stats["records"],
            stats["steps"], stats["calls"],
            " (call budget reached)" if stats["truncated"] else "",
            stats["indexed"], stats["skipped"], stats["removed"],
        )
        # Unconditionally, because a crawl where one step of twelve quietly
        # stopped producing logs exactly like a healthy one otherwise — the
        # summary above counts the steps that worked and says nothing about the
        # rest.  Escalated to a warning only when the whole crawl came back
        # empty, which is still the failure that matters most.
        steps = _format_step_details(stats["details"])
        if stats["records"]:
            self._logger.info(
                "[mcp-crawl] %s steps: %s",
                getattr(server, "name", server_id), steps,
            )
        else:
            self._logger.warning(
                "[mcp-crawl] %s produced no content — steps: %s",
                getattr(server, "name", server_id), steps,
            )
        return stats

    def _step_arguments(
        self,
        step: CrawlStep,
        schema: Dict[str, Any],
        records_by_step: Dict[str, List[CrawlRecord]],
    ) -> Tuple[Optional[List[Dict[str, Any]]], int]:
        """Return the argument sets to call a step's tool with.

        A root step is one call.  A child step is one call per parent record
        (capped by ``max_parents``).  ``None`` means the step can't run — the
        parent produced nothing, or the tool doesn't declare an argument the
        parent id could go in.
        """
        base = filter_arguments(schema, step.arguments)

        if step.for_each:
            # Fan out over values the plan named.  Filtered like any other
            # argument, so a plan that names an argument this server's variant
            # of the tool doesn't declare is skipped rather than rejected once
            # per value.
            (name, values), = step.for_each.items()
            if not declares_argument(schema, (name,)):
                self._logger.debug(
                    "[mcp-crawl] step '%s': tool does not declare '%s'",
                    step.label, name,
                )
                return None, 0
            arg_name = resolve_argument(schema, (name,)) or name
            return (
                [
                    {**base, arg_name: coerce_argument(schema, arg_name, value)}
                    for value in values[:step.max_parents]
                ],
                min(len(values), step.max_parents),
            )

        if not step.parent_labels:
            return [base], 0

        # A step may name several parents (an issue's comments hang off
        # whichever listing found the issue).  Pool them in the order the plan
        # lists them and dedupe by id, so the fan-out budget is spent on
        # distinct records rather than on the overlap between listings.
        parents: List[CrawlRecord] = []
        seen: set = set()
        for label in step.parent_labels:
            for record in records_by_step.get(label) or []:
                if not record.record_id or record.record_id in seen:
                    continue
                seen.add(record.record_id)
                parents.append(record)
        if not parents:
            return None, 0
        parent_arg = resolve_argument(schema, step.parent_args)
        if not parent_arg:
            self._logger.debug(
                "[mcp-crawl] step '%s': no argument for parent id on this tool",
                step.label,
            )
            return None, 0

        arg_sets = [
            {
                **base,
                parent_arg: coerce_argument(schema, parent_arg, parent.record_id),
            }
            for parent in parents[:step.max_parents]
        ]
        return (arg_sets or None), len(arg_sets)

    @staticmethod
    def _detail_text(payload: Optional[str]) -> str:
        """Render a per-record detail call's payload as text to append."""
        if not payload:
            return ""
        records = _parse_records(payload)
        return "\n\n".join(r.text for r in records if r.text).strip()

    def _call_tool(
        self, server: Any, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolCallResult:
        """Call one read-only tool, through the shared background-call path.

        The 401 retry, the failure-reason extraction and the transport itself
        all live in :mod:`smartloop.mcp.read_only_calls`, so a crawl and the
        account probe reach a server the same way.
        """
        return call_read_only(
            self._ensure_tool_caller(), server, tool_name, arguments,
            self._logger, label="mcp-crawl",
        )

    def _ensure_tool_caller(self):
        """Return the MCP tool caller, building the default one lazily."""
        self._tool_caller = ensure_tool_caller(self._tool_caller, self._logger)
        return self._tool_caller

    # ---------------------------------------------------------------- index

    def _index_documents(
        self,
        server: Any,
        plan: CrawlPlan,
        documents: Dict[str, dict],
        previous: dict,
        crawled_at: float,
    ) -> Tuple[int, int]:
        """Embed the documents whose content changed.  Returns (indexed, skipped)."""
        known = (previous or {}).get("docs", {})
        indexed = skipped = 0
        server_id = getattr(server, "id", "")
        server_name = getattr(server, "name", "") or ""

        for doc_id, doc in documents.items():
            content_hash = _content_hash(doc["text"])
            if known.get(doc_id) == content_hash:
                skipped += 1
                continue

            if doc_id in known:
                try:
                    self._store.delete_document(doc_id)
                except Exception:
                    pass

            chunks = _chunk_text(doc["text"])
            chunk_dicts = [
                {
                    "text": chunk,
                    "document_id": doc_id,
                    "document_name": doc["name"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "server_id": server_id,
                    "server_name": server_name,
                    "plan": plan.slug,
                    "step": doc["step"],
                    "tool_name": doc["tool_name"],
                    "record_id": doc["record_id"],
                    "source_url": doc["source_url"],
                    "title": doc["title"],
                    "crawled_at": crawled_at,
                }
                for i, chunk in enumerate(chunks)
            ]
            try:
                self._store.add_chunks(chunk_dicts)
                indexed += 1
            except Exception as e:
                self._logger.warning(
                    f"[mcp-crawl] Failed to index '{doc['name']}': {e}"
                )
        return indexed, skipped

    def _remove_stale(self, previous: dict, documents: Dict[str, dict]) -> int:
        """Delete documents the server no longer returns."""
        removed = 0
        for doc_id in (previous or {}).get("docs", {}):
            if doc_id in documents:
                continue
            try:
                self._store.delete_document(doc_id)
                removed += 1
            except Exception:
                pass
        return removed

    # ------------------------------------------------------------- manifest

    def _load_manifest(self) -> Dict[str, dict]:
        if os.path.isfile(self._manifest_path):
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if isinstance(v, dict)}
            except Exception as e:
                self._logger.warning(f"[mcp-crawl] Could not load manifest: {e}")
        return {}

    def _save_manifest(self) -> None:
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as e:
            self._logger.warning(f"[mcp-crawl] Could not save manifest: {e}")
