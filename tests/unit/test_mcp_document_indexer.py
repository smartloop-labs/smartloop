"""Crawling a document-type MCP server into the vector index.

The crawl is what puts Notion pages / Linear issues / Trello cards in context
before a query needs them, so what matters here is: the right calls are made,
the records land as documents with usable text, a refresh re-embeds only what
changed, and content the server stopped returning is dropped.
"""

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smartloop.db.mcp_document_indexer import MCPDocumentIndexer


def _tool(name, schema=None):
    return SimpleNamespace(name=name, description=name, input_schema=schema or {})


def _server(name="notion", slug="notion", tools=(), **fields):
    fields.setdefault("pkg", None)
    fields.setdefault("enabled", True)
    fields.setdefault("system", False)
    fields.setdefault("crawl", None)
    fields.setdefault("remote_config", None)
    fields.setdefault("oauth_credentials", None)
    fields.setdefault("last_crawl_at", None)
    return SimpleNamespace(
        id="srv1", name=name, catalog_slug=slug, tools=list(tools), **fields,
    )


def _result(payload, success=True, is_error=False, error=None):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(
        success=success,
        is_error=is_error,
        error=error,
        content=[{"type": "text", "text": text}],
    )


class FakeCaller:
    """Stands in for MCPToolCaller, recording calls and replaying results."""

    def __init__(self, responses):
        # responses: {tool_name: result | callable(arguments) -> result}
        self._responses = responses
        self.calls = []

    def get_access_token(self, server):
        return "tok"

    def refresh_access_token(self, server):
        return None

    def call_tool(self, server, tool_name, arguments=None, access_token=None):
        self.calls.append((tool_name, dict(arguments or {})))
        handler = self._responses.get(tool_name)
        if handler is None:
            return _result("", success=False, error="no such tool")
        if callable(handler):
            return handler(arguments or {})
        return handler


def _indexer(tmp_dir, caller):
    """Build an indexer with a mocked store (no embedding model needed)."""
    idx = MCPDocumentIndexer.__new__(MCPDocumentIndexer)
    idx._project_dir = str(tmp_dir)
    idx._logger = MagicMock()
    idx._store = MagicMock()
    idx._manifest_path = str(Path(tmp_dir) / ".mcp_documents_manifest.json")
    idx._manifest = {}
    import threading
    idx._lock = threading.Lock()
    idx._crawling = set()
    idx._tool_caller = caller
    return idx


def _indexed_docs(store):
    """{document_id: joined text} for everything add_chunks received."""
    docs = {}
    for call in store.add_chunks.call_args_list:
        for chunk in call[0][0]:
            docs.setdefault(chunk["document_id"], []).append(chunk["text"])
    return {k: "\n".join(v) for k, v in docs.items()}


class TestNotionStyleCrawl:
    """search → fetch, which is the shape every document server shares."""

    def _crawl(self, tmp, extra=None):
        server = _server(tools=[
            _tool("notion-search", {"properties": {"query": {"type": "string"}}}),
            _tool("notion-fetch", {"properties": {"id": {"type": "string"}}}),
        ])
        responses = {
            "notion-search": _result({"results": [
                {"id": "p1", "title": "Roadmap", "url": "https://notion.so/p1"},
                {"id": "p2", "title": "Onboarding"},
            ]}),
            "notion-fetch": lambda args: _result(
                f"# Body of {args.get('id')}\n\nsome content"
            ),
        }
        responses.update(extra or {})
        caller = FakeCaller(responses)
        idx = _indexer(tmp, caller)
        stats = idx.crawl_server(server, force=True)
        return idx, caller, stats

    def test_records_are_indexed_as_documents(self):
        with tempfile.TemporaryDirectory() as td:
            idx, caller, stats = self._crawl(td)
            assert stats["records"] == 2
            assert stats["indexed"] == 2
            docs = _indexed_docs(idx._store)
            assert len(docs) == 2
            assert any("Roadmap" in text for text in docs.values())

    def test_detail_call_body_is_appended_to_the_listing_row(self):
        with tempfile.TemporaryDirectory() as td:
            idx, caller, _ = self._crawl(td)
            assert ("notion-fetch", {"id": "p1"}) in caller.calls
            text = "\n".join(_indexed_docs(idx._store).values())
            assert "Body of p1" in text
            assert "Roadmap" in text  # listing fields survive

    def test_listing_row_is_not_duplicated_inside_the_body(self):
        """A page fetch repeats the listing row; embedding it twice dilutes it."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-fetch", {"properties": {"id": {}}, "required": ["id"]}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result([
                    {"id": "p1", "type": "page", "title": "Roadmap"},
                ]),
                "notion-fetch": _result(
                    {"title": "Roadmap", "type": "page", "text": "Q3: ship it"}
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            body = "\n".join(_indexed_docs(idx._store).values())
            assert body.count("title: Roadmap") == 1
            assert "Q3: ship it" in body

    def test_listing_fields_absent_from_the_body_are_kept(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_issues", {"properties": {"limit": {}}}),
                _tool("get_issue", {"properties": {"id": {}}, "required": ["id"]}),
            ])
            caller = FakeCaller({
                "list_issues": _result([
                    {"id": "i1", "title": "Fix login", "state": "In Progress"},
                ]),
                "get_issue": _result({"description": "steps to reproduce"}),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            body = "\n".join(_indexed_docs(idx._store).values())
            assert "state: In Progress" in body
            assert "steps to reproduce" in body

    def test_document_name_carries_server_step_and_title(self):
        with tempfile.TemporaryDirectory() as td:
            idx, _, _ = self._crawl(td)
            names = [
                chunk["document_name"]
                for call in idx._store.add_chunks.call_args_list
                for chunk in call[0][0]
            ]
            assert "notion/pages/Roadmap" in names

    def test_ids_and_urls_stay_in_metadata_not_in_the_embedded_text(self):
        with tempfile.TemporaryDirectory() as td:
            idx, _, _ = self._crawl(td)
            chunk = idx._store.add_chunks.call_args_list[0][0][0][0]
            assert chunk["record_id"] == "p1"
            assert chunk["source_url"] == "https://notion.so/p1"
            assert chunk["server_name"] == "notion"
            assert chunk["plan"] == "notion"
            assert "https://notion.so/p1" not in chunk["text"]

    def test_unchanged_records_are_not_re_embedded_on_refresh(self):
        with tempfile.TemporaryDirectory() as td:
            idx, _, first = self._crawl(td)
            idx._store.reset_mock()
            server = _server(tools=[
                _tool("notion-search", {"properties": {"query": {}}}),
                _tool("notion-fetch", {"properties": {"id": {}}}),
            ])
            caller = FakeCaller({
                "notion-search": _result({"results": [
                    {"id": "p1", "title": "Roadmap", "url": "https://notion.so/p1"},
                    {"id": "p2", "title": "Onboarding"},
                ]}),
                "notion-fetch": lambda args: _result(
                    f"# Body of {args.get('id')}\n\nsome content"
                ),
            })
            idx._tool_caller = caller
            second = idx.crawl_server(server, force=True)
            assert second["skipped"] == 2
            assert second["indexed"] == 0
            idx._store.add_chunks.assert_not_called()

    def test_changed_record_is_re_indexed(self):
        with tempfile.TemporaryDirectory() as td:
            idx, _, _ = self._crawl(td)
            idx._store.reset_mock()
            server = _server(tools=[
                _tool("notion-search", {"properties": {"query": {}}}),
                _tool("notion-fetch", {"properties": {"id": {}}}),
            ])
            idx._tool_caller = FakeCaller({
                "notion-search": _result({"results": [
                    {"id": "p1", "title": "Roadmap v2", "url": "https://notion.so/p1"},
                    {"id": "p2", "title": "Onboarding"},
                ]}),
                "notion-fetch": lambda args: _result(
                    f"# Body of {args.get('id')}\n\nsome content"
                ),
            })
            stats = idx.crawl_server(server, force=True)
            assert stats["indexed"] == 1
            assert stats["skipped"] == 1

    def test_records_the_server_stopped_returning_are_removed(self):
        with tempfile.TemporaryDirectory() as td:
            idx, _, _ = self._crawl(td)
            server = _server(tools=[
                _tool("notion-search", {"properties": {"query": {}}}),
                _tool("notion-fetch", {"properties": {"id": {}}}),
            ])
            idx._tool_caller = FakeCaller({
                "notion-search": _result({"results": [
                    {"id": "p1", "title": "Roadmap", "url": "https://notion.so/p1"},
                ]}),
                "notion-fetch": lambda args: _result(
                    f"# Body of {args.get('id')}\n\nsome content"
                ),
            })
            stats = idx.crawl_server(server, force=True)
            assert stats["removed"] == 1
            assert idx._store.delete_document.called


class TestFanOut:
    def test_child_steps_walk_each_parent_record(self):
        """Trello's boards → lists → cards, the shape that needs a fan-out."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="trello", slug="trello", tools=[
                _tool("list_boards"),
                _tool("get_lists", {"properties": {"board_id": {"type": "string"}}}),
                _tool("get_cards_by_list_id", {"properties": {"list_id": {"type": "string"}}}),
            ])
            caller = FakeCaller({
                "list_boards": _result([{"id": "b1", "name": "Sprint"}]),
                "get_lists": lambda args: _result(
                    [{"id": f"l-{args['board_id']}", "name": "Doing"}]
                ),
                "get_cards_by_list_id": lambda args: _result(
                    [{"id": "c1", "name": f"Card in {args['list_id']}"}]
                ),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            assert ("get_lists", {"board_id": "b1"}) in caller.calls
            assert ("get_cards_by_list_id", {"list_id": "l-b1"}) in caller.calls
            assert stats["steps"] == 3
            text = "\n".join(_indexed_docs(idx._store).values())
            assert "Card in l-b1" in text

    def test_child_step_runs_when_its_required_argument_is_the_parent_id(self):
        """The parent id arrives at call time, not from the plan's arguments."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="trello", slug="trello", tools=[
                _tool("list_boards"),
                _tool("get_lists", {
                    "properties": {"board_id": {"type": "string"}},
                    "required": ["board_id"],
                }),
            ])
            caller = FakeCaller({
                "list_boards": _result([{"id": "b1", "name": "Sprint"}]),
                "get_lists": _result([{"id": "l1", "name": "Doing"}]),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            assert ("get_lists", {"board_id": "b1"}) in caller.calls

    def test_linear_walks_teams_then_issues_per_team(self):
        """list_teams → list_issues(teamId), the tree walk the plan is built on.

        The unscoped `list_issues` returns one page of the workspace; the
        per-team calls are what reach the issues that page cut off.
        """
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_issues", {"properties": {
                    "teamId": {"type": "string"}, "limit": {"type": "number"},
                }}),
            ])
            issues = {
                "core": [{"id": "i-core", "title": "Core issue"}],
                "web": [{"id": "i-web", "title": "Web issue"}],
            }
            caller = FakeCaller({
                "list_teams": _result([
                    {"id": "core", "name": "Core"}, {"id": "web", "name": "Web"},
                ]),
                "list_issues": lambda args: _result(
                    issues.get(args.get("teamId"), [{"id": "i0", "title": "Workspace issue"}])
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            assert ("list_issues", {"teamId": "core", "limit": 50}) in caller.calls
            assert ("list_issues", {"teamId": "web", "limit": 50}) in caller.calls
            text = "\n".join(_indexed_docs(idx._store).values())
            assert "Core issue" in text and "Web issue" in text
            assert "Workspace issue" in text

    def test_a_step_pools_the_records_of_every_parent_it_names(self):
        """Comments hang off whichever issue listing found the issue."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_issues", {"properties": {"teamId": {}, "limit": {}}}),
                _tool("list_comments", {"properties": {"issueId": {"type": "string"}}}),
            ])
            caller = FakeCaller({
                "list_teams": _result([{"id": "core", "name": "Core"}]),
                "list_issues": lambda args: _result(
                    [{"id": "i-core", "title": "Scoped"}] if args.get("teamId")
                    else [{"id": "i-all", "title": "Unscoped"}]
                ),
                "list_comments": lambda args: _result(
                    [{"id": f"c-{args['issueId']}", "body": f"note on {args['issueId']}"}]
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            commented = {
                args["issueId"] for tool, args in caller.calls if tool == "list_comments"
            }
            # `issues` found i-all, `team-issues` found i-core: both feed the walk.
            assert commented == {"i-all", "i-core"}

    def test_the_same_parent_is_only_walked_once_across_pooled_parents(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_issues", {"properties": {"teamId": {}, "limit": {}}}),
                _tool("list_comments", {"properties": {"issueId": {}}}),
            ])
            caller = FakeCaller({
                "list_teams": _result([{"id": "core", "name": "Core"}]),
                # Every listing returns the same issue, as the real overlap does.
                "list_issues": _result([{"id": "i1", "title": "Shared"}]),
                "list_comments": _result([{"id": "c1", "body": "note"}]),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            assert [c[0] for c in caller.calls].count("list_comments") == 1

    def test_a_fan_out_retry_keeps_the_parent_id(self):
        """A failed per-team call must not be retried as an unscoped one.

        Dropping the parent id turns the retry into a different question, asked
        once per parent — the plan's own arguments are what get dropped.
        """
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_issues", {"properties": {"teamId": {}, "limit": {}}}),
            ])
            caller = FakeCaller({
                "list_teams": _result([{"id": "core", "name": "Core"}]),
                "list_issues": lambda args: (
                    _result("", success=False, error="limit is not supported")
                    if "limit" in args else _result([{"id": "i1", "title": "Issue"}])
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            issue_calls = [args for tool, args in caller.calls if tool == "list_issues"]
            # The team step drops `limit` and keeps `teamId`...
            assert {"teamId": "core"} in issue_calls
            # ...while the one genuinely unscoped step is the only bare retry.
            assert issue_calls.count({}) == 1

    def test_a_rejected_fan_out_is_not_proved_once_per_parent(self):
        """The failure that made 30 of 36 crawl calls useless.

        `notion-get-comments` rejected the shape of the call, not the page id
        in it — so every remaining page was the identical rejected call.
        """
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-get-comments", {"properties": {"page_id": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    [{"id": f"p{i}", "title": f"Page {i}"} for i in range(20)]
                ),
                "notion-get-comments": _result(
                    '{"code":"validation_error","message":"body failed validation"}',
                    success=True, is_error=True,
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            assert [c[0] for c in caller.calls].count("notion-get-comments") == 1

    def test_a_step_gives_up_after_repeated_transient_failures(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-get-comments", {"properties": {"page_id": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    [{"id": f"p{i}", "title": f"Page {i}"} for i in range(20)]
                ),
                # Not a recognisably permanent reason, so it takes a few tries
                # before the step concludes the tool isn't going to answer.
                "notion-get-comments": _result("", success=False, error="socket timeout"),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            assert [c[0] for c in caller.calls].count("notion-get-comments") == 3

    def test_one_failed_parent_does_not_end_a_working_fan_out(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-get-comments", {"properties": {"page_id": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    [{"id": f"p{i}", "title": f"Page {i}"} for i in range(5)]
                ),
                # Only p1 is missing its comments; the rest still answer.
                "notion-get-comments": lambda args: (
                    _result("", success=False, error="socket timeout")
                    if args.get("page_id") == "p1"
                    else _result([{"id": f"c-{args['page_id']}", "body": "note"}])
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            assert [c[0] for c in caller.calls].count("notion-get-comments") == 5

    def test_a_rejected_step_stays_skipped_next_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            tools = [
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-get-comments", {"properties": {"page_id": {}}}),
            ]
            responses = {
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-get-comments": _result(
                    '{"code":"validation_error"}', success=True, is_error=True,
                ),
            }
            idx = _indexer(td, FakeCaller(responses))
            idx.crawl_server(_server(tools=tools), force=True)

            caller = FakeCaller(responses)
            idx._tool_caller = caller
            idx.crawl_server(_server(tools=tools))
            assert "notion-get-comments" not in [c[0] for c in caller.calls]

    def test_the_crawl_stops_at_its_call_budget(self):
        with tempfile.TemporaryDirectory() as td:
            from smartloop.db import mcp_document_indexer as mod
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_issues", {"properties": {"teamId": {}, "limit": {}}}),
                _tool("get_issue", {"properties": {"id": {}}}),
            ])
            caller = FakeCaller({
                "list_teams": _result([{"id": f"t{i}"} for i in range(25)]),
                "list_issues": _result([{"id": f"i{i}", "title": f"Issue {i}"} for i in range(40)]),
                "get_issue": _result("body"),
            })
            idx = _indexer(td, caller)
            original = mod.MAX_CRAWL_CALLS
            mod.MAX_CRAWL_CALLS = 5
            try:
                stats = idx.crawl_server(server, force=True)
            finally:
                mod.MAX_CRAWL_CALLS = original

            assert len(caller.calls) == 5
            assert stats["truncated"] is True
            # What it did reach still lands, and nothing is remembered as blocked.
            assert stats["records"] > 0
            assert not idx._manifest[server.id]["blocked"]

    def test_a_step_can_fan_out_over_values_the_plan_names(self):
        """Servers that can't be enumerated.

        A market-data server has no `list_*` tool — every call needs a ticker —
        so there is no walk to derive. The subjects come from the plan instead.
        """
        with tempfile.TemporaryDirectory() as td:
            from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep

            plan = CrawlPlan(slug="yfinance", steps=(CrawlStep(
                label="quotes",
                tools=("get_stock_info",),
                for_each={"ticker": ("AAPL", "MSFT")},
            ),))
            server = _server(name="yahoo-finance", slug=None, tools=[
                _tool("get_stock_info", {
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                }),
            ])
            caller = FakeCaller({
                "get_stock_info": lambda args: _result(
                    {"symbol": args["ticker"], "name": f"{args['ticker']} Inc"}
                ),
            })
            idx = _indexer(td, caller)
            with patch("smartloop.db.mcp_document_indexer.resolve_plan", return_value=plan):
                idx.crawl_server(server, force=True)

            assert ("get_stock_info", {"ticker": "AAPL"}) in caller.calls
            assert ("get_stock_info", {"ticker": "MSFT"}) in caller.calls
            text = "\n".join(_indexed_docs(idx._store).values())
            assert "AAPL Inc" in text and "MSFT Inc" in text

    def test_a_named_fan_out_is_skipped_when_the_tool_has_no_such_argument(self):
        # Otherwise the plan's argument is dropped and the same bare call is
        # made once per value.
        with tempfile.TemporaryDirectory() as td:
            from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep

            plan = CrawlPlan(slug="yfinance", steps=(CrawlStep(
                label="quotes",
                tools=("get_stock_info",),
                for_each={"ticker": ("AAPL", "MSFT")},
            ),))
            server = _server(name="yahoo-finance", slug=None, tools=[
                _tool("get_stock_info", {"properties": {"symbol": {}}}),
            ])
            caller = FakeCaller({"get_stock_info": _result([{"id": "x"}])})
            idx = _indexer(td, caller)
            with patch("smartloop.db.mcp_document_indexer.resolve_plan", return_value=plan):
                idx.crawl_server(server, force=True)
            assert caller.calls == []

    def test_child_step_is_skipped_when_the_parent_returned_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="trello", slug="trello", tools=[
                _tool("list_boards"),
                _tool("get_lists", {"properties": {"board_id": {}}}),
            ])
            caller = FakeCaller({"list_boards": _result([])})
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)
            assert [c[0] for c in caller.calls].count("get_lists") == 0
            assert stats["records"] == 0


class TestResilience:
    def test_plan_arguments_that_dont_fit_fall_back_to_a_bare_call(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_issues", {"properties": {"limit": {"type": "number"}}}),
            ])
            attempts = []

            def handler(args):
                attempts.append(dict(args))
                if args:
                    return _result("", success=False, error="bad request")
                return _result([{"id": "i1", "title": "Fix login"}])

            caller = FakeCaller({"list_issues": handler})
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)
            assert attempts == [{"limit": 50}, {}]
            assert stats["records"] == 1

    def test_failing_step_does_not_abort_the_crawl(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="linear", slug="linear", tools=[
                _tool("list_teams"),
                _tool("list_projects"),
            ])
            caller = FakeCaller({
                "list_teams": _result("", success=False, error="500 boom"),
                "list_projects": _result([{"id": "pr1", "name": "Core"}]),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)
            assert stats["records"] == 1
            assert "Core" in "\n".join(_indexed_docs(idx._store).values())

    def test_non_json_payload_is_indexed_as_one_record(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result("Page one\nPage two")})
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)
            assert stats["records"] == 1
            assert "Page one" in "\n".join(_indexed_docs(idx._store).values())

    def test_server_with_no_matching_tool_reports_why(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="weather", slug=None, tools=[_tool("get_forecast")])
            idx = _indexer(td, FakeCaller({}))
            stats = idx.crawl_server(server, force=True)
            assert stats["reason"] == "no crawlable listing tools"
            idx._store.add_chunks.assert_not_called()

    def test_expired_token_is_refreshed_once_and_the_call_retried(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            calls = []

            def handler(args):
                calls.append(args)
                if len(calls) == 1:
                    return _result("", success=False, error="HTTP 401 Unauthorized")
                return _result([{"id": "p1", "title": "Roadmap"}])

            caller = FakeCaller({"notion-search": handler})
            caller.refresh_access_token = lambda server: "new-token"
            idx = _indexer(td, caller)
            saved = []
            stats = idx.crawl_server(
                server, force=True, on_server_updated=saved.append,
            )
            assert stats["records"] == 1
            assert saved == [server]   # refreshed credentials must be persisted


class TestOverlapAndReporting:
    def test_a_page_listed_by_two_steps_is_indexed_once(self):
        """Notion lists the same page as recent *and* private *and* shared."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-list-private-pages", {"properties": {"limit": {}}}),
                _tool("notion-fetch", {"properties": {"id": {}}, "required": ["id"]}),
            ])
            page = {"id": "p1", "title": "Roadmap"}
            caller = FakeCaller({
                "notion-list-recent-pages": _result([page]),
                "notion-list-private-pages": _result([page]),
                "notion-fetch": _result("# Roadmap body"),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            assert stats["indexed"] == 1
            assert len(_indexed_docs(idx._store)) == 1
            # And the body was fetched once, not once per listing.
            assert [c[0] for c in caller.calls].count("notion-fetch") == 1

    def test_failed_call_is_not_retried_bare_when_arguments_are_required(self):
        """A tool that needs its argument would only fail again without it."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="trello", slug="trello", tools=[
                _tool("list_boards"),
                _tool("get_lists", {
                    "properties": {"board_id": {"type": "string"}},
                    "required": ["board_id"],
                }),
            ])
            caller = FakeCaller({
                "list_boards": _result([{"id": "b1", "name": "Sprint"}]),
                "get_lists": _result("", success=False, error="500 boom"),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            calls = [c for c in caller.calls if c[0] == "get_lists"]
            assert calls == [("get_lists", {"board_id": "b1"})]

    def test_each_step_reports_the_tool_it_used_and_why_it_found_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    "", success=True, is_error=True,
                ),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            by_step = {d["step"]: d for d in stats["details"]}
            assert by_step["recent-pages"]["tool"] == "notion-list-recent-pages"
            assert by_step["recent-pages"]["reason"]
            # A step whose tool the server doesn't have says so.
            assert by_step["private-pages"]["reason"] == "no matching tool on this server"

    def test_argument_rejection_reported_from_the_content_block(self):
        """MCP servers return argument errors as is_error content, not error."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    "query is required", success=True, is_error=True,
                ),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)
            reasons = [d["reason"] for d in stats["details"] if d["step"] == "recent-pages"]
            assert "query is required" in reasons[0]

    def test_step_outcomes_are_logged_even_when_the_crawl_succeeded(self):
        """A partly-working crawl used to log exactly like a healthy one.

        The summary line counts the steps that produced records and says nothing
        about the rest, and the per-step reasons were logged only when the whole
        crawl came back empty — so one step of twelve quietly stopping was
        invisible.
        """
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-list-shared-pages", {"properties": {"limit": {}}}),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-list-shared-pages": _result([]),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            logged = " ".join(
                str(call[0]) for call in idx._logger.info.call_args_list
            )
            assert "steps:" in logged
            # The step that worked and the one that returned nothing both named.
            assert "recent-pages" in logged
            assert "shared-pages" in logged
            # The crawl did index something, so this is not a warning.
            idx._logger.warning.assert_not_called()

    def test_the_step_line_names_the_tool_and_the_fresh_count(self):
        from smartloop.db.mcp_document_indexer import _format_step_details

        line = _format_step_details([
            {"step": "boards", "tool": "trelloReadBoard", "records": 5, "new": 5},
            # A listing whose records another step already documented is not the
            # same event as a listing that returned nothing.
            {"step": "workspace-boards", "tool": "trelloReadBoard",
             "records": 5, "new": 0},
            {"step": "inbox-cards", "tool": "trelloReadInbox", "records": 0,
             "reason": "returned no records"},
            {"step": "databases", "tool": None, "records": 0,
             "reason": "no matching tool on this server"},
        ])
        assert line == (
            "boards/trelloReadBoard: 5; "
            "workspace-boards/trelloReadBoard: 5 (0 new); "
            "inbox-cards/trelloReadInbox: 0 (returned no records); "
            "databases: 0 (no matching tool on this server)"
        )

    def test_a_long_failure_reason_is_truncated_in_the_step_line(self):
        # A server's validation error runs to hundreds of characters of JSON.
        # The field name comes first; the rest would bury the other eleven steps.
        from smartloop.db.mcp_document_indexer import (
            MAX_STEP_REASON_CHARS,
            _format_step_details,
        )

        line = _format_step_details([
            {"step": "page-comments", "tool": "notion-get-comments",
             "records": 0, "reason": "body failed validation: " + "x" * 900},
        ])
        assert line.startswith("page-comments/notion-get-comments: 0 (body failed")
        assert line.endswith("...)")
        assert len(line) < MAX_STEP_REASON_CHARS + 80

    def test_a_multiline_reason_becomes_one_line(self):
        # Reasons arrive as raw server output, newlines and all; one step must
        # not break the line the other steps share.
        from smartloop.db.mcp_document_indexer import _format_step_details

        line = _format_step_details([
            {"step": "pages", "tool": "notion-search", "records": 0,
             "reason": "query is\n  required\n"},
        ])
        assert line == "pages/notion-search: 0 (query is required)"

    def test_a_plan_with_no_steps_still_says_something(self):
        from smartloop.db.mcp_document_indexer import _format_step_details

        assert _format_step_details([]) == "plan had no steps"


class TestWastedCallsAreAvoided:
    def test_step_whose_required_argument_cannot_be_filled_is_not_called(self):
        """notion-search requires a non-empty query; the plan only has ""."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-search", {
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            assert "notion-search" not in [c[0] for c in caller.calls]
            reason = next(d["reason"] for d in stats["details"] if d["step"] == "pages")
            assert "requires ['query']" in reason

    def test_plan_gated_tool_is_not_retried_next_cycle(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-query-meeting-notes", {"properties": {}}),
            ])
            responses = {
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-query-meeting-notes": _result(
                    "This tool requires a Business plan or higher.",
                    success=True, is_error=True,
                ),
            }
            idx = _indexer(td, FakeCaller(responses))
            idx.crawl_server(server, force=True)

            caller = FakeCaller(responses)
            idx._tool_caller = caller
            stats = idx.crawl_server(server)   # the periodic cycle
            assert "notion-query-meeting-notes" not in [c[0] for c in caller.calls]
            reason = next(
                d["reason"] for d in stats["details"] if d["step"] == "meeting-notes"
            )
            assert reason.startswith("skipped (last failure:")

    def test_a_forced_refresh_retries_a_blocked_step(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-query-meeting-notes", {"properties": {}}),
            ])
            blocked = _result(
                "This tool requires a Business plan or higher.",
                success=True, is_error=True,
            )
            idx = _indexer(td, FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-query-meeting-notes": blocked,
            }))
            idx.crawl_server(server, force=True)

            # Plan upgraded: a forced refresh must pick the content up.
            caller = FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-query-meeting-notes": _result([{"id": "m1", "title": "Standup"}]),
            })
            idx._tool_caller = caller
            idx.crawl_server(server, force=True)
            assert "notion-query-meeting-notes" in [c[0] for c in caller.calls]
            assert "Standup" in "\n".join(_indexed_docs(idx._store).values())

    def test_a_transient_empty_result_is_not_remembered_as_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-list-favorite-pages", {"properties": {"limit": {}}}),
            ])
            idx = _indexer(td, FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-list-favorite-pages": _result([]),
            }))
            idx.crawl_server(server, force=True)

            caller = FakeCaller({
                "notion-list-recent-pages": _result([{"id": "p1", "title": "A"}]),
                "notion-list-favorite-pages": _result([{"id": "f1", "title": "Fav"}]),
            })
            idx._tool_caller = caller
            idx.crawl_server(server)
            assert "notion-list-favorite-pages" in [c[0] for c in caller.calls]


class TestEnvelopeUnwrapping:
    """Finding the record list inside whatever the server wrapped it in.

    Every server pluralizes its own entities, so a fixed list of envelope key
    names is always one server behind.  Trello answers `trelloReadWorkspace`
    with `{"workspaces": [...], "nextCursor": ..., "hasMore": false}`, and
    before this the whole envelope became one id-less record — which read as a
    successful step and left the boards fan-out with nothing to walk.
    """

    def test_an_unknown_plural_key_is_still_the_record_list(self):
        from smartloop.db.mcp_document_indexer import _unwrap_records

        records = _unwrap_records({
            "workspaces": [{"id": "w1", "name": "One"}, {"id": "w2", "name": "Two"}],
            "nextCursor": "abc",
            "hasMore": False,
        })
        assert [r["id"] for r in records] == ["w1", "w2"]

    def test_a_record_with_a_list_field_is_not_mistaken_for_an_envelope(self):
        # The discriminator is identity: a record carries an id, an envelope
        # doesn't.  Without it a single card would unwrap into its own labels.
        from smartloop.db.mcp_document_indexer import _unwrap_records

        card = {"id": "c1", "name": "Card", "labels": [{"name": "urgent"}]}
        assert _unwrap_records(card) == [card]

    def test_an_empty_listing_produces_no_records_rather_than_the_envelope(self):
        # An empty Trello inbox is `{"cards": [], ...}`.  Returning the envelope
        # would index a document whose whole content is "hasMore: false".
        from smartloop.db.mcp_document_indexer import _unwrap_records

        assert _unwrap_records({"cards": [], "nextCursor": None, "hasMore": False}) == []

    def test_a_step_fans_out_over_an_unknown_envelope(self):
        """The end of the same story: workspaces → boards actually walks."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="Trello", slug="trello", tools=[
                _tool("trelloReadWorkspace", {
                    "properties": {"action": {}, "limit": {}}, "required": ["action"],
                }),
                _tool("trelloReadBoard", {
                    "properties": {
                        "action": {}, "workspaceId": {}, "filter": {}, "limit": {},
                    },
                    "required": ["action"],
                }),
            ])
            caller = FakeCaller({
                "trelloReadWorkspace": _result({
                    "workspaces": [{"id": "ari:...workspace/w1", "name": "Team"}],
                    "hasMore": False,
                }),
                "trelloReadBoard": lambda args: _result({
                    "boards": [{"id": "b1", "name": "Board One"}],
                    "hasMore": False,
                }),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            by_workspace = [
                args for tool, args in caller.calls
                if args.get("action") == "list_by_workspace"
            ]
            assert by_workspace, "workspace-boards never fanned out"
            assert by_workspace[0]["workspaceId"] == "ari:...workspace/w1"
            reasons = {d["step"]: d.get("reason") for d in stats["details"]}
            assert reasons["workspace-boards"] is None


class TestRecordIdentity:
    """Finding the id and the body when the server buries them.

    A record with no id is a record nothing can fan out from, and one keyed by
    a hash of its own text is a record a second listing will document all over
    again.  Both showed up on Canva, whose two listings of the same design
    disagree about where its identity lives.
    """

    def test_an_entity_qualified_id_is_still_the_record_id(self):
        # search-designs answers with `design_id` and no `id`, so the fan-out
        # over designs had nothing to walk and the comments were never reached.
        from smartloop.db.mcp_document_indexer import _parse_records

        records = _parse_records(json.dumps({"items": [
            {"title": "Canva MCP TESTING", "design_id": "DAHS4TDHiec",
             "edit_url": "https://www.canva.com/d/c7n9T"},
        ]}))
        assert [r.record_id for r in records] == ["DAHS4TDHiec"]
        assert records[0].url == "https://www.canva.com/d/c7n9T"

    def test_a_record_wrapped_in_a_type_discriminator_is_unwrapped(self):
        # list-folder-items answers with {"type": "design", "design": {...}}:
        # id, title and body all one level down.  Read as-is the record has no
        # identity, so the same design a search already indexed is documented
        # a second time under a content hash.
        from smartloop.db.mcp_document_indexer import _parse_records

        records = _parse_records(json.dumps({"items": [
            {"type": "design", "design": {
                "id": "DAHS4TDHiec", "title": "Canva MCP TESTING",
                "page_count": 2,
            }},
        ]}))
        assert [r.record_id for r in records] == ["DAHS4TDHiec"]
        assert records[0].title == "Canva MCP TESTING"

    def test_a_record_that_merely_carries_a_nested_object_keeps_its_shape(self):
        # The unwrap only fires when the outer object has no identity and the
        # inner one does — otherwise a Trello card would unwrap into its list.
        from smartloop.db.mcp_document_indexer import _unwrap_record

        card = {"id": "c1", "name": "Card",
                "list": {"id": "l1", "name": "To Do"}}
        assert _unwrap_record(card) == card

    def test_an_id_less_wrapper_around_an_id_less_object_is_left_alone(self):
        from smartloop.db.mcp_document_indexer import _unwrap_record

        item = {"type": "design", "design": {"title": "no id here"}}
        assert _unwrap_record(item) == item

    def test_two_listings_of_one_design_are_indexed_once(self):
        """The end of the story: Canva's search and folder walk agree."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(name="Canva", slug="canva", tools=[
                _tool("search-designs", {"properties": {
                    "ownership": {}, "sort_by": {}, "limit": {},
                    "design_types": {}, "user_intent": {},
                }}),
                _tool("get-design-content", {
                    "properties": {"design_id": {}, "content_types": {},
                                   "user_intent": {}},
                    "required": ["design_id", "content_types"],
                }),
                _tool("list-folder-items", {
                    "properties": {"folder_id": {}, "item_types": {},
                                   "sort_by": {}, "user_intent": {}},
                    "required": ["folder_id"],
                }),
            ])
            design = {"title": "Canva MCP TESTING", "design_id": "DAHS4TDHiec"}
            caller = FakeCaller({
                "search-designs": _result({"items": [design]}),
                "get-design-content": _result("the design body text"),
                "list-folder-items": _result({"items": [
                    {"type": "design", "design": {
                        "id": "DAHS4TDHiec", "title": "Canva MCP TESTING"}},
                ]}),
            })
            idx = _indexer(td, caller)
            stats = idx.crawl_server(server, force=True)

            docs = _indexed_docs(idx._store)
            assert len(docs) == 1, docs
            # And the one document is the one that fetched the body.
            assert "the design body text" in next(iter(docs.values()))
            by_step = {d["step"]: d for d in stats["details"]}
            assert by_step["designs"]["new"] == 1
            assert by_step["root-items"]["new"] == 0


class TestDurableUrls:
    """A stored URL outlives the crawl that found it, so it has to be stable.

    Canva mints a fresh ``/d/<token>`` share link on every call — five crawls of
    one design produced five different tokens — so snapshotting one stores a
    link that may rot, and a share link is a capability rather than an address.
    """

    def test_a_template_is_filled_with_the_record_id(self):
        from smartloop.mcp.crawl_plans import render_url

        assert render_url(
            "https://www.canva.com/design/{id}", "DAHS4TDHiec"
        ) == "https://www.canva.com/design/DAHS4TDHiec"

    def test_nothing_is_rendered_when_it_would_be_a_broken_link(self):
        # Each of these would otherwise emit a URL with a literal {id} in it,
        # or a URL for a record that has no id to identify.
        from smartloop.mcp.crawl_plans import render_url

        assert render_url("https://x/{id}", None) is None
        assert render_url(None, "DAHS4TDHiec") is None
        assert render_url("https://x/{wrong}", "DAHS4TDHiec") is None
        assert render_url("https://x/no-placeholder", "DAHS4TDHiec") is None

    def _crawl(self, td, url_template, payload_url="https://canva.com/d/EPHEMERAL"):
        from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep

        plan = CrawlPlan(slug="c", steps=(CrawlStep(
            label="designs", tools=("search-designs",),
            url_template=url_template,
        ),))
        server = _server(name="Canva", slug="canva", tools=[
            _tool("search-designs", {"properties": {}}),
        ])
        caller = FakeCaller({"search-designs": _result({"items": [
            {"design_id": "DAHS4TDHiec", "title": "Deck", "url": payload_url},
        ]})})
        idx = _indexer(td, caller)
        with patch(
            "smartloop.db.mcp_document_indexer.resolve_plan", return_value=plan
        ):
            idx.crawl_server(server, force=True)
        chunk = idx._store.add_chunks.call_args_list[0][0][0][0]
        return chunk["source_url"]

    def test_the_template_beats_the_url_the_listing_returned(self):
        with tempfile.TemporaryDirectory() as td:
            assert self._crawl(
                td, "https://www.canva.com/design/{id}"
            ) == "https://www.canva.com/design/DAHS4TDHiec"

    def test_without_a_template_the_listing_url_is_kept(self):
        # The common case, and what Notion/Linear/Trello rely on.
        with tempfile.TemporaryDirectory() as td:
            assert self._crawl(td, None) == "https://canva.com/d/EPHEMERAL"

    def test_a_template_that_cannot_be_filled_falls_back(self):
        from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep

        with tempfile.TemporaryDirectory() as td:
            plan = CrawlPlan(slug="c", steps=(CrawlStep(
                label="designs", tools=("search-designs",),
                url_template="https://www.canva.com/design/{id}",
            ),))
            server = _server(name="Canva", slug="canva", tools=[
                _tool("search-designs", {"properties": {}}),
            ])
            # No id anywhere, so there is nothing to build a URL from.
            caller = FakeCaller({"search-designs": _result({"items": [
                {"title": "Untitled", "url": "https://canva.com/d/FALLBACK"},
            ]})})
            idx = _indexer(td, caller)
            with patch(
                "smartloop.db.mcp_document_indexer.resolve_plan", return_value=plan
            ):
                idx.crawl_server(server, force=True)
            chunk = idx._store.add_chunks.call_args_list[0][0][0][0]
            assert chunk["source_url"] == "https://canva.com/d/FALLBACK"


class TestDetailArguments:
    """A detail call is not always just the record's id.

    A server that dispatches on a discriminator — one `trelloReadCard` tool for
    both "list a list's cards" and "get a card" — rejects a detail call that
    carries only the id, once per record.  Expressing the fetch as a child step
    instead doesn't work either: `seen_records` dedupes by record id, and a
    detail listing returns the ids the parent step already produced.
    """

    def _run(self, td, detail_arguments):
        from smartloop.mcp.crawl_plans import CrawlPlan, CrawlStep

        plan = CrawlPlan(slug="dispatch", steps=(CrawlStep(
            label="cards",
            tools=("readCard",),
            arguments={"action": "list"},
            detail_tools=("readCard",),
            detail_arguments=detail_arguments,
            detail_args=("cardId",),
        ),))
        server = _server(name="dispatch", slug="dispatch", tools=[
            _tool("readCard", {
                "properties": {"action": {}, "cardId": {}},
                "required": ["action"],
            }),
        ])
        caller = FakeCaller({
            "readCard": lambda args: (
                _result({"body": "the full card body"})
                if args.get("cardId")
                else _result([{"id": "c1", "name": "Card"}])
            ),
        })
        idx = _indexer(td, caller)
        with patch(
            "smartloop.db.mcp_document_indexer.resolve_plan", return_value=plan
        ):
            idx.crawl_server(server, force=True)
        return caller, idx

    def test_fixed_arguments_travel_with_the_detail_call(self):
        with tempfile.TemporaryDirectory() as td:
            caller, idx = self._run(td, {"action": "get"})
            detail = [args for _, args in caller.calls if "cardId" in args]
            assert detail == [{"action": "get", "cardId": "c1"}]
            body = "\n".join(_indexed_docs(idx._store).values())
            assert "the full card body" in body

    def test_an_argument_the_detail_tool_does_not_declare_is_dropped(self):
        # Filtered like `arguments` is, so one plan can name the discriminator
        # for the servers that need it without breaking the ones that don't.
        with tempfile.TemporaryDirectory() as td:
            caller, _ = self._run(td, {"action": "get", "verbosity": "full"})
            detail = [args for _, args in caller.calls if "cardId" in args]
            assert detail == [{"action": "get", "cardId": "c1"}]


class TestRefreshCycle:
    def test_ttl_skips_a_recently_crawled_server(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result([{"id": "p1", "title": "R"}])})
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            before = len(caller.calls)

            stats = idx.crawl_server(server, ttl=3600)
            assert stats["reason"] == "fresh"
            assert len(caller.calls) == before

    def test_ttl_expiry_crawls_again(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result([{"id": "p1", "title": "R"}])})
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            idx._manifest["srv1"]["crawled_at"] = time.time() - 7200
            before = len(caller.calls)

            stats = idx.crawl_server(server, ttl=3600)
            assert stats["reason"] is None
            assert len(caller.calls) > before

    def test_crawl_records_the_timestamp_on_the_server(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result([{"id": "p1", "title": "R"}])})
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)
            assert server.last_crawl_at is not None

    def test_manifest_survives_a_new_indexer_instance(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result([{"id": "p1", "title": "R"}])})
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            reopened = _indexer(td, caller)
            reopened._manifest = reopened._load_manifest()
            assert "srv1" in reopened._manifest
            stats = reopened.crawl_server(server, force=True)
            assert stats["skipped"] == 1   # hashes matched across instances


class TestRemoval:
    def test_removing_a_server_deletes_its_crawled_documents(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[_tool("notion-search", {"properties": {"query": {}}})])
            caller = FakeCaller({"notion-search": _result([
                {"id": "p1", "title": "A"}, {"id": "p2", "title": "B"},
            ])})
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            removed = idx.remove_server("srv1")
            assert removed == 2
            assert idx._store.delete_document.call_count >= 2
            assert "srv1" not in idx._manifest


class TestMcpSourceAttribution:
    """Crawled MCP content must say which tool produced it — otherwise the
    model has content it cannot attribute when asked where it came from."""

    @staticmethod
    def _chunk(metadata):
        class _C:
            def __init__(self, m):
                self.metadata = m
        return _C(metadata)

    def test_tool_and_server_lead_the_source_line(self):
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "notion/pages/Roadmap",
            self._chunk({"tool_name": "search_pages", "server_name": "Notion"}),
        )
        assert "search_pages" in out
        assert out.startswith("Notion")

    def test_tool_alone_is_enough(self):
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source("linear/issues/Bug", self._chunk({"tool_name": "list_issues"}))
        assert out.startswith("list_issues")

    def test_falls_back_to_the_document_name(self):
        """Chunks indexed before the tool was recorded still render."""
        from smartloop.agents.mcp.explorer import _mcp_source

        assert _mcp_source("plain/doc", self._chunk({})) == "plain/doc"
        assert _mcp_source("plain/doc", self._chunk(None)) == "plain/doc"

    def test_the_record_url_travels_in_the_source_line(self):
        """The one thing in the header the model cannot reconstruct.

        Answering from a live tool call handed the user a link they could click
        straight through to the design; answering from the index handed them
        `<server>/<step>/<title>`, which reads like a file path and opens
        nothing.  The URL is deliberately stripped from the embedded text — a
        link is noise in an embedding — so the header is the only place it can
        travel.
        """
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "Canva/designs/Canva MCP TESTING",
            self._chunk({
                "tool_name": "search-designs", "server_name": "Canva",
                "source_url": "https://www.canva.com/d/0_Dv9PwBsQmYOOc",
            }),
        )
        assert out.endswith("https://www.canva.com/d/0_Dv9PwBsQmYOOc")
        assert "search-designs" in out

    def test_a_record_without_a_url_gains_no_empty_separator(self):
        # source_url is stored as "" when the record had none, and a trailing
        # " — " would read as a truncated link.
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "notion/pages/Roadmap",
            self._chunk({"tool_name": "search_pages", "server_name": "Notion",
                         "source_url": ""}),
        )
        assert out == "Notion · search_pages — notion/pages/Roadmap"

    def test_the_record_id_travels_in_the_source_line(self):
        """The id is the handle a *next* tool call needs.

        "List my designs, then comment on that one" only works if the id came
        back with the listing; otherwise the model re-runs a live search purely
        to recover what the crawl already stored.  `is_noisy_value` drops every
        key ending in `_id`, so `design_id` never survives into a chunk — the
        header is the only place it can travel.
        """
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "Canva/designs/Benefits of Local AI",
            self._chunk({
                "tool_name": "search-designs", "server_name": "Canva",
                "record_id": "DAHTRxZG2Qo",
            }),
        )
        assert "id: DAHTRxZG2Qo" in out

    def test_an_id_the_document_name_already_carries_is_not_repeated(self):
        # A record with no title of its own is already named by its id, and
        # saying it twice reads like two different records.
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "Canva/design-comments/KAHS4n2FFaA",
            self._chunk({"tool_name": "list-comments", "server_name": "Canva",
                         "record_id": "KAHS4n2FFaA"}),
        )
        assert out.count("KAHS4n2FFaA") == 1
        assert "id:" not in out

    def test_the_id_is_the_tool_argument_not_something_read_off_the_url(self):
        """Deriving an id from a URL is wrong on the servers we crawl.

        A Trello card's id is its ARI; its URL carries the shortLink, which
        `trelloReadCard` rejects.  Canva is the same story with a pattern:
        `comment-on-design` requires ^D[a-zA-Z0-9_-]+$ and the /d/ token in the
        URL does not match it.
        """
        from smartloop.agents.mcp.explorer import _mcp_source

        ari = "ari:cloud:trello::card/workspace/5fef637f/624b42a03ce1e666"
        out = _mcp_source(
            "Trello/cards/Donate Clothes Trip",
            self._chunk({
                "tool_name": "trelloReadCard", "server_name": "Trello",
                "record_id": ari,
                "source_url": "https://trello.com/c/dQv6OAbC/28-donate",
            }),
        )
        assert f"id: {ari}" in out
        # Both present and distinct — the shortLink is not offered as the id.
        assert "dQv6OAbC" in out
        assert "id: https" not in out

    def test_a_record_without_an_id_gains_no_empty_separator(self):
        from smartloop.agents.mcp.explorer import _mcp_source

        out = _mcp_source(
            "notion/pages/Roadmap",
            self._chunk({"tool_name": "search_pages", "server_name": "Notion",
                         "record_id": "", "source_url": ""}),
        )
        assert out == "Notion · search_pages — notion/pages/Roadmap"

    def test_indexed_chunks_carry_the_record_id(self):
        """The read path can only surface what indexing stored."""
        import inspect

        from smartloop.db import mcp_document_indexer as mod

        assert '"record_id": doc["record_id"]' in inspect.getsource(mod)

    def test_indexed_chunks_carry_the_record_url(self):
        """The read path can only surface what indexing stored."""
        import inspect

        from smartloop.db import mcp_document_indexer as mod

        src = inspect.getsource(mod)
        assert '"source_url": doc["source_url"]' in src

    def test_indexed_chunks_carry_the_tool_name(self):
        """The read path can only surface what indexing stored."""
        import inspect

        from smartloop.db import mcp_document_indexer as mod

        src = inspect.getsource(mod)
        assert '"tool_name": doc["tool_name"]' in src


class TestNotionComments:
    """Notion anchors comments to the paragraph they are about.

    `include_all_blocks` and `include_resolved` both default to false, and both
    defaults are wrong for a crawl.  Probing a real workspace: the defaults
    returned nothing on all 16 pages, `include_all_blocks` alone reached 6 of a
    page's 10 threads, and the two together reached all 10.  Resolved threads
    are kept because "resolved" is usually where the decision is.
    """

    def test_the_flags_that_reach_block_level_discussions_travel(self):
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-get-comments", {
                    "properties": {"page_id": {}, "include_all_blocks": {},
                                   "include_resolved": {}},
                    "required": ["page_id"],
                }),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    {"results": [{"id": "p1", "title": "The Pitch"}]}
                ),
                "notion-get-comments": _result({"text": "<discussions>…"}),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            args = next(
                a for tool, a in caller.calls if tool == "notion-get-comments"
            )
            assert args["include_all_blocks"] is True
            assert args["include_resolved"] is True

    def test_a_page_fetch_asks_for_the_discussions_with_it(self):
        """The only route to a comment that works with a URL and no page id.

        `notion-fetch` takes "URL or ID", so it is reachable where
        `get-comments` is not, and `include_discussions` inlines the comment
        text into the page body.  A sample rather than the full set — this is
        partial capture, not a replacement for `page-comments`.
        """
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("notion-fetch", {
                    "properties": {"id": {}, "include_discussions": {}},
                    "required": ["id"],
                }),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result({"results": [
                    {"url": "https://app.notion.com/p/abc", "title": "The Pitch"},
                ]}),
                "notion-fetch": _result(
                    {"text": "<comment>what happened to the model?????</comment>"}
                ),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            args = next(a for tool, a in caller.calls if tool == "notion-fetch")
            assert args["include_discussions"] is True
            body = "\n".join(_indexed_docs(idx._store).values())
            assert "what happened to the model?????" in body

    def test_the_flags_are_dropped_where_a_server_does_not_declare_them(self):
        """The OpenAPI server's comment tool takes neither; the step still runs."""
        with tempfile.TemporaryDirectory() as td:
            server = _server(tools=[
                _tool("notion-list-recent-pages", {"properties": {"limit": {}}}),
                _tool("API-retrieve-a-comment", {
                    "properties": {"page_id": {}}, "required": ["page_id"],
                }),
            ])
            caller = FakeCaller({
                "notion-list-recent-pages": _result(
                    {"results": [{"id": "p1", "title": "The Pitch"}]}
                ),
                "API-retrieve-a-comment": _result({"text": "a comment"}),
            })
            idx = _indexer(td, caller)
            idx.crawl_server(server, force=True)

            args = next(
                a for tool, a in caller.calls if tool == "API-retrieve-a-comment"
            )
            assert args == {"page_id": "p1"}
