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

    def do_GET(self):
        path = self.path.split("?")[0]
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
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", _StubHandler
    server.shutdown()


class TestListBootstrapModels:
    def test_returns_models(self, stub_server):
        base_url, handler = stub_server
        models = [{"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"}]
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
        models = [{"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"}]
        handler.routes = {"/v1/models": (200, models)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        assert client.list_models() == models

    def test_returns_models_without_auth(self, stub_server):
        base_url, handler = stub_server
        models = [{"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"}]
        handler.routes = {"/v1/models": (200, models)}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.list_models() == models

    def test_retries_without_auth_on_401(self, stub_server):
        """When the first request (with auth) gets 401, retry without auth succeeds."""
        base_url, handler = stub_server
        models = [{"name": "gemma3-1b", "model_id": "google/gemma-3-1b-it"}]
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
            "/v1/models/gemma3-1b/download": (200, {
                "artifacts": {"model.gguf": "https://storage/model.gguf"},
            }),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"X-Api-Key": "test"})
        result = client.get_model_artifacts("gemma3-1b")
        assert result == {"model.gguf": "https://storage/model.gguf"}

    def test_empty_artifacts_raises(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/gemma3-1b/download": (200, {"artifacts": {}}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ValueError, match="no artifacts"):
            client.get_model_artifacts("gemma3-1b")

    def test_401_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/gemma3-1b/download": (401, {"error": "unauthorized"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="Authentication failed"):
            client.get_model_artifacts("gemma3-1b")

    def test_404_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="not found"):
            client.get_model_artifacts("nonexistent")


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
