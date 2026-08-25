"""Tests for smartloop.platform.client.SLPPlatformClient."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import pytest

from smartloop.platform.client import SLPPlatformClient


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves canned responses."""

    # Class-level response config — set by fixtures before each test
    routes: dict = {}
    #: (path_with_query, headers) per GET, so tests can assert what was sent
    seen: list = []

    def do_GET(self):
        path = self.path.split("?")[0]
        self.seen.append((self.path, dict(self.headers)))
        if path in self.routes:
            status, body = self.routes[path]
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress stderr


@pytest.fixture
def stub_server():
    """Start a local HTTP server with configurable routes, yield (base_url, handler_cls)."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    _StubHandler.seen = []
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", _StubHandler
    server.shutdown()


class TestListBootstrapModels:
    def test_returns_models(self, stub_server):
        base_url, handler = stub_server
        models = [{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]
        handler.routes = {"/v1/bootstrap/models": (200, models)}

        client = SLPPlatformClient(base_url)
        result = client.list_bootstrap_models()
        assert result == models

    def test_empty_response_raises(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/bootstrap/models": (200, [])}

        client = SLPPlatformClient(base_url)
        with pytest.raises(ValueError, match="no models"):
            client.list_bootstrap_models()

    def test_404_raises_connection_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {}  # no route registered

        client = SLPPlatformClient(base_url)
        with pytest.raises(ConnectionError, match="not found"):
            client.list_bootstrap_models()


class TestListModels:
    def test_returns_models_with_auth(self, stub_server):
        base_url, handler = stub_server
        models = [{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]
        handler.routes = {"/v1/models": (200, models)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        assert client.list_models() == models

    def test_returns_models_without_auth(self, stub_server):
        base_url, handler = stub_server
        models = [{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]
        handler.routes = {"/v1/models": (200, models)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.list_models() == models

    def test_retries_without_auth_on_401(self, stub_server):
        """When the first request (with auth) gets 401, retry without auth succeeds."""
        base_url, handler = stub_server
        models = [{"name": "slp-orchestra-mini", "model_id": "google/gemma-3-1b-it"}]
        call_count = {"n": 0}

        # Override do_GET to return 401 on first call, 200 on second
        original_do_GET = handler.do_GET

        def counting_do_GET(self_handler):
            call_count["n"] += 1
            if call_count["n"] == 1:
                self_handler.send_response(401)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.end_headers()
                self_handler.wfile.write(b'{"detail":"Unauthorized"}')
            else:
                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.end_headers()
                self_handler.wfile.write(json.dumps(models).encode())

        handler.do_GET = counting_do_GET
        try:
            client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer expired"})
            assert client.list_models() == models
            assert call_count["n"] == 2
        finally:
            handler.do_GET = original_do_GET

    def test_connection_error_on_non_401(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/models": (500, {"detail": "Internal Server Error"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError, match="API error"):
            client.list_models()

    def test_connection_error_on_unreachable(self):
        client = SLPPlatformClient("http://127.0.0.1:1", auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.list_models()


class TestGetModelArtifacts:
    def test_returns_artifacts(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/slp-orchestra-mini/download": (200, {
                "artifacts": {"model.gguf": "https://storage/model.gguf"},
            }),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"X-Api-Key": "test"})
        result = client.get_model_artifacts("slp-orchestra-mini")
        assert result == {"model.gguf": "https://storage/model.gguf"}

    def test_empty_artifacts_raises(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/slp-orchestra-mini/download": (200, {"artifacts": {}}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ValueError, match="no artifacts"):
            client.get_model_artifacts("slp-orchestra-mini")

    def test_401_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/slp-orchestra-mini/download": (401, {"error": "unauthorized"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="Authentication failed"):
            client.get_model_artifacts("slp-orchestra-mini")

    def test_404_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="not found"):
            client.get_model_artifacts("nonexistent")


class TestGetLatestRelease:
    def test_returns_wrapper(self, stub_server):
        base_url, handler = stub_server
        wrapper = {
            "update_available": None,
            "update_required": False,
            "release": {"version": "1.0.15", "name": "v", "channel": "stable"},
        }
        handler.routes = {"/v1/releases/latest": (200, wrapper)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.get_latest_release("stable") == wrapper

    def test_null_release_returned(self, stub_server):
        base_url, handler = stub_server
        wrapper = {"update_available": None, "update_required": False, "release": None}
        handler.routes = {"/v1/releases/latest": (200, wrapper)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.get_latest_release("beta")["release"] is None

    def test_error_raises_connection_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/releases/latest": (500, {"detail": "boom"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError, match="API error"):
            client.get_latest_release("stable")

    def test_unreachable_raises_connection_error(self):
        client = SLPPlatformClient("http://127.0.0.1:1", auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.get_latest_release("stable")


class TestGetCurrentUser:
    def test_returns_user_profile(self, stub_server):
        base_url, handler = stub_server
        user_data = {"email": "chris@example.com", "name": "Chris"}
        handler.routes = {"/v1/users/me": (200, user_data)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        result = client.get_current_user()
        assert result == user_data

    def test_401_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (401, {"detail": "Unauthorized"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="expired or invalid"):
            client.get_current_user()

    def test_500_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (500, {"detail": "Internal Server Error"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="API error"):
            client.get_current_user()


class TestWebSearch:
    """Web search goes through this client so it gets the same credential
    injection as every other hosted call."""

    def test_returns_the_link_array(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, ["https://a.example", "https://b.example"])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        assert client.search_web("canyon bikes", 5) == ["https://a.example", "https://b.example"]

    def test_sends_the_query_and_count(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, [])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        client.search_web("canyon bikes", 7)

        path = handler.seen[-1][0]
        assert "q=canyon+bikes" in path or "q=canyon%20bikes" in path
        assert "count=7" in path

    def test_a_developer_token_is_sent_as_an_api_key(self, stub_server):
        """The regression this path had: formatting the header locally sent a
        developer token as a bearer token, and the resulting 401 read as
        "web search is off"."""
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, [])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"X-Api-Key": "sk-dev"})
        client.search_web("q", 5)

        headers = handler.seen[-1][1]
        assert headers["X-Api-Key"] == "sk-dev"
        assert "Authorization" not in headers

    def test_identifies_itself_in_the_user_agent(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, [])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        client.search_web("q", 5)

        assert handler.seen[-1][1]["User-Agent"].startswith("smartloop-cli/")

    def test_401_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (401, {})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError):
            client.search_web("q", 5)

    def test_a_server_error_is_not_a_refusal(self, stub_server):
        """A 500 must not masquerade as "search is off" — the caller caches
        that answer, but fails open on a connection error."""
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (500, {})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.search_web("q", 5)

    def test_transport_error_raises_connection_error(self):
        client = SLPPlatformClient("http://127.0.0.1:1", auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.search_web("q", 5)


class TestWebSearchStatus:
    def test_returns_the_status_body(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (200, {"enabled": True})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        assert client.get_web_search_status() == {"enabled": True}

    def test_a_developer_token_is_sent_as_an_api_key(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (200, {"enabled": True})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"X-Api-Key": "sk-dev"})
        client.get_web_search_status()

        headers = handler.seen[-1][1]
        assert headers["X-Api-Key"] == "sk-dev"
        assert "Authorization" not in headers

    def test_403_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (403, {})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError):
            client.get_web_search_status()

    def test_transport_error_raises_connection_error(self):
        client = SLPPlatformClient("http://127.0.0.1:1", auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.get_web_search_status()
