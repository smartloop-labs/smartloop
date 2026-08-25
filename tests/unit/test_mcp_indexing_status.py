"""What a client polls to know a connection is still being indexed.

Registering an MCP server returns before it is usable: tool discovery and the
content crawl both run on the MCP task.  Until they finish, the connection
answers nothing — so the API has to say so, otherwise the UI shows a Notion
connection that looks connected and knows nothing.
"""

import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from smartloop.api.services.model_service import ModelService
from smartloop.db.mcp_document_indexer import MCPDocumentIndexer
from smartloop.workspace import MCPServer, MCPServerStatus


def _server(**fields):
    fields.setdefault("id", "srv1")
    fields.setdefault("name", "notion")
    fields.setdefault("catalog_slug", "notion")
    return MCPServer(**fields)


def _state(server, **entry):
    entry.setdefault("crawling", False)
    return ModelService._mcp_indexing_state(server, entry)


class TestIndexingState:
    def test_a_server_still_provisioning_is_in_progress(self):
        state = _state(_server(status=MCPServerStatus.PENDING))
        assert state["state"] == "provisioning"
        assert state["in_progress"] is True

    def test_a_crawl_in_flight_reads_as_indexing(self):
        state = _state(_server(), crawling=True)
        assert state["state"] == "indexing"
        assert state["in_progress"] is True

    def test_a_crawlable_server_not_yet_walked_is_queued(self):
        # Provisioned but the crawl cycle hasn't reached it: the connection is
        # not done, and saying "indexed" here is what makes an empty answer
        # look like the server having nothing.
        state = _state(_server())
        assert state["state"] == "queued"
        assert state["in_progress"] is True

    def test_a_crawled_server_is_done(self):
        state = _state(_server(), crawled_at=1000.0, documents=12, records=12)
        assert state["state"] == "indexed"
        assert state["in_progress"] is False
        assert state["documents"] == 12

    def test_a_server_with_no_content_to_crawl_is_not_pending_forever(self):
        # A weather or shell connection is complete once its tools are indexed;
        # it has no content, so nothing will ever set a crawl timestamp.
        state = _state(_server(name="weather", catalog_slug=None))
        assert state["state"] == "disabled"
        assert state["in_progress"] is False

    def test_an_explicit_opt_out_also_reads_as_disabled(self):
        state = _state(_server(crawl=False))
        assert state["state"] == "disabled"
        assert state["in_progress"] is False

    def test_a_failed_provisioning_carries_its_reason(self):
        server = _server(status=MCPServerStatus.FAILED, last_error="401 from server")
        state = _state(server)
        assert state["state"] == "failed"
        assert state["in_progress"] is False
        assert state["error"] == "401 from server"

    def test_the_plan_is_reported_before_the_first_crawl(self):
        # Which plan matched is what tells the UI *what* is about to be indexed.
        assert _state(_server())["plan"] == "notion"


class TestServerStatusEndpointShape:
    def _svc(self, server, entry):
        project = SimpleNamespace(
            id="p1", name="proj", project_dir="/tmp/p1", mcp_servers=[server],
        )
        svc = MagicMock()
        svc._get_project = MagicMock(return_value=project)
        svc._model.workspace.current_project = project
        svc._mcp_server_to_dict = ModelService._mcp_server_to_dict
        svc._mcp_indexing_state = ModelService._mcp_indexing_state
        indexer = MagicMock()
        indexer.server_status = MagicMock(return_value={**entry, "crawling": entry.get("crawling", False)})
        svc._ensure_mcp_document_indexer = MagicMock(return_value=indexer)
        return svc

    def test_the_server_dict_carries_its_indexing_block(self):
        server = _server()
        svc = self._svc(server, {"crawling": True, "documents": 3, "records": 3})
        result = ModelService.mcp_server_status(svc, "srv1", project_id="p1")
        assert result["id"] == "srv1"
        assert result["indexing"]["state"] == "indexing"
        assert result["indexing"]["in_progress"] is True

    def test_an_unknown_server_is_a_not_found(self):
        svc = self._svc(_server(), {})
        try:
            ModelService.mcp_server_status(svc, "nope", project_id="p1")
        except ValueError as e:
            assert "nope" in str(e)
        else:
            raise AssertionError("expected a ValueError the router maps to 404")

    def test_the_list_carries_the_same_block(self):
        # Otherwise watching a freshly registered connection costs one request
        # per server on every poll.
        server = _server()
        svc = self._svc(server, {"crawled_at": 1.0, "documents": 2})
        servers = ModelService.list_mcp_servers_with_status(svc, "p1")
        assert [s["indexing"]["state"] for s in servers] == ["indexed"]

    def test_system_servers_stay_out_of_the_list(self):
        server = _server(system=True)
        svc = self._svc(server, {})
        assert ModelService.list_mcp_servers_with_status(svc, "p1") == []


class TestIsCrawling:
    def _indexer(self, tmp_dir):
        idx = MCPDocumentIndexer.__new__(MCPDocumentIndexer)
        idx._project_dir = tmp_dir
        idx._logger = MagicMock()
        idx._store = MagicMock()
        idx._manifest_path = f"{tmp_dir}/.mcp_documents_manifest.json"
        idx._manifest = {}
        idx._lock = threading.Lock()
        idx._crawling = set()
        idx._tool_caller = None
        return idx

    def test_a_server_reads_as_crawling_only_while_the_walk_runs(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._indexer(td)
            seen = {}

            def _walk(server, ttl, force, cb):
                seen["during"] = idx.is_crawling("srv1")
                return {"records": 0}

            idx._crawl_server = _walk
            assert idx.is_crawling("srv1") is False
            idx.crawl_server(SimpleNamespace(id="srv1"), force=True)
            assert seen["during"] is True
            assert idx.is_crawling("srv1") is False

    def test_a_crawl_that_raises_still_clears_the_flag(self):
        with tempfile.TemporaryDirectory() as td:
            idx = self._indexer(td)

            def _boom(server, ttl, force, cb):
                raise RuntimeError("server went away")

            idx._crawl_server = _boom
            try:
                idx.crawl_server(SimpleNamespace(id="srv1"), force=True)
            except RuntimeError:
                pass
            # A stuck flag would leave the UI spinning until the process restarts.
            assert idx.is_crawling("srv1") is False
