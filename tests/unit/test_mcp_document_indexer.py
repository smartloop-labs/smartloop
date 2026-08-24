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
from unittest.mock import MagicMock

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
