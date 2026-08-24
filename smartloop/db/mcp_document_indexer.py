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
    filter_arguments,
    resolve_argument,
    resolve_plan,
    resolve_tool,
    tool_schema,
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


def _content_to_text(content: Optional[Sequence[dict]]) -> str:
    """Join an MCP tool result's content blocks into one string."""
    if not content:
        return ""
    parts: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text") or "")
            else:
                parts.append(json.dumps(block, default=str))
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p).strip()


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


def _unwrap_records(data: Any) -> List[Any]:
    """Find the record list inside a parsed tool result.

    Handles the three shapes servers actually return: a bare list, an envelope
    naming its list (``{"results": [...]}``, ``{"issues": {"nodes": [...]}}``),
    and a single object.
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
        text = _record_text(item)
        if not text:
            continue
        record_id = title = url = None
        if isinstance(item, dict):
            record_id = _first_str(item, RECORD_ID_KEYS)
            title = _first_str(item, RECORD_TITLE_KEYS)
            url = _first_str(item, ("url", "uri", "link", "shortUrl", "html_url"))
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
        with self._lock:
            return self._crawl_server(server, ttl, force, on_server_updated)

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

    def status(self) -> Dict[str, Any]:
        """Per-server crawl state, for the /health and diagnostics endpoints."""
        return {
            "chunks": self.count(),
            "servers": {
                server_id: {
                    "crawled_at": entry.get("crawled_at"),
                    "documents": len(entry.get("docs", {})),
                    "records": entry.get("records", 0),
                    "plan": entry.get("plan"),
                }
                for server_id, entry in self._manifest.items()
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
            arg_sets, parents = self._step_arguments(step, schema, records_by_step)
            if arg_sets is None:
                stats["details"].append({
                    "step": step.label, "tool": tool, "records": 0,
                    "reason": f"nothing to fan out from '{step.parent}'"
                              if step.parent else "no arguments could be built",
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

            records: List[CrawlRecord] = []
            failure: Optional[str] = None
            for arguments in arg_sets:
                payload, changed, error = self._call_tool(server, tool, arguments)
                credentials_changed = credentials_changed or changed
                if payload is None and arguments and not _has_required(schema):
                    # The plan's arguments may not fit this server's variant of
                    # the tool.  Only worth retrying bare when the tool declares
                    # nothing as required — otherwise the retry fails for sure.
                    payload, changed, error = self._call_tool(server, tool, {})
                    credentials_changed = credentials_changed or changed
                if not payload:
                    failure = failure or error or "no content returned"
                    continue
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
                key = record.record_id or record.title
                if key and key in seen_records:
                    continue
                if key:
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
            detail_arg = (
                resolve_argument(tool_schema(server, detail_tool), step.detail_args)
                if detail_tool else None
            )

            for index, record in enumerate(fresh):
                text = record.text
                if detail_tool and detail_arg and record.record_id:
                    detail_payload, changed, _ = self._call_tool(
                        server, detail_tool, {detail_arg: record.record_id}
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
                    "source_url": record.url or "",
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
            "[mcp-crawl] %s (plan=%s): %d record(s) over %d step(s) — "
            "indexed=%d, unchanged=%d, removed=%d",
            getattr(server, "name", server_id), plan.slug, stats["records"],
            stats["steps"], stats["indexed"], stats["skipped"], stats["removed"],
        )
        if not stats["records"]:
            # Nothing indexed is the failure that matters, so the reason each
            # step gave is logged rather than left in a return value nobody
            # reads.
            self._logger.warning(
                "[mcp-crawl] %s produced no content — steps: %s",
                getattr(server, "name", server_id),
                "; ".join(
                    f"{d['step']}"
                    f"{'/' + d['tool'] if d.get('tool') else ''}: {d.get('reason')}"
                    for d in stats["details"]
                ) or "plan had no steps",
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
        if not step.parent:
            return [base], 0

        parents = records_by_step.get(step.parent) or []
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
            {**base, parent_arg: parent.record_id}
            for parent in parents[:step.max_parents]
            if parent.record_id
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
    ) -> Tuple[Optional[str], bool, Optional[str]]:
        """Call one read-only tool.

        Returns ``(payload, credentials_changed, error)``.  A failure is a skip,
        not an exception: a crawl covers what it can reach and the next cycle
        tries again — but the reason travels back to the caller so a crawl that
        indexes nothing can say why.  An expired OAuth token is refreshed once
        and the call retried, mirroring the live tool path.
        """
        caller = self._ensure_tool_caller()
        if caller is None:
            return None, False, "no MCP tool caller available"
        try:
            access_token = caller.get_access_token(server)
            result = caller.call_tool(
                server=server, tool_name=tool_name,
                arguments=arguments, access_token=access_token,
            )
            credentials_changed = False
            if not result.success and result.error and "401" in str(result.error):
                new_token = caller.refresh_access_token(server)
                if new_token:
                    credentials_changed = True
                    result = caller.call_tool(
                        server=server, tool_name=tool_name,
                        arguments=arguments, access_token=new_token,
                    )
            if not result.success or result.is_error or not result.content:
                reason = self._failure_reason(result)
                self._logger.info(
                    "[mcp-crawl] %s.%s(%s) returned nothing: %s",
                    getattr(server, "name", "?"), tool_name,
                    ", ".join(arguments) or "-", reason,
                )
                return None, credentials_changed, reason
            return _content_to_text(result.content) or None, credentials_changed, None
        except Exception as e:
            self._logger.info(
                "[mcp-crawl] %s.%s raised: %s",
                getattr(server, "name", "?"), tool_name, e,
            )
            return None, False, str(e)[:200]

    @staticmethod
    def _failure_reason(result: Any) -> str:
        """Short reason a tool call produced nothing, for the step report.

        MCP servers report an argument rejection as a *successful* call with
        ``is_error`` set and the message in the content, so the content is read
        too — otherwise the most common crawl failure ("query is required")
        shows up as a bare "no content".
        """
        if getattr(result, "error", None):
            return str(result.error)[:200]
        content = getattr(result, "content", None)
        if content:
            text = _content_to_text(content)
            if text:
                return text[:200]
        return "empty result"

    def _ensure_tool_caller(self):
        """Return the MCP tool caller, building the default one lazily.

        Only its transport-level helpers are used (``call_tool``,
        ``get_access_token``, ``refresh_access_token``); the argument extractor
        and result mapper it takes for the live path are LLM-backed and have no
        place in a crawl, so they are left unset.
        """
        if self._tool_caller is None:
            try:
                from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller
                self._tool_caller = MCPToolCaller(
                    arg_extractor=None, result_mapper=None,
                )
            except Exception as e:
                self._logger.warning(f"[mcp-crawl] No MCP tool caller available: {e}")
                return None
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
