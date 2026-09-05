"""Tests for smartloop.platform.client.SLPPlatformClient."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

import pytest

from smartloop.auth import credential_store
from smartloop.platform.client import ModelAccessError, SLPPlatformClient


class _StubHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves canned responses.

    A route's value is ``(status, body)``, or a *list* of them to be served one
    per request — which is how a test says "401 first, then 200", the shape every
    refresh-and-retry assertion needs.  The last entry repeats once the list runs
    out, so a retry that should not have happened still gets an answer rather
    than hanging the test.

    Every request is recorded in :attr:`requests` so a test can assert on what
    went out, not just on what came back — the header the retry carried is the
    whole point of a refresh.
    """

    # Class-level config — reset by the fixture before each test
    routes: dict = {}
    requests: list = []

    def _respond(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.requests.append({
            "method": self.command,
            "path": path,
            "headers": dict(self.headers),
            "body": body,
        })

        entry = self.routes.get(path)
        if entry is None:
            self.send_response(404)
            self.end_headers()
            return
        if isinstance(entry, list):
            status, payload = entry[0] if len(entry) == 1 else entry.pop(0)
        else:
            status, payload = entry

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    do_GET = _respond
    do_POST = _respond
    do_PATCH = _respond

    def log_message(self, *args):
        pass  # suppress stderr


@pytest.fixture
def stub_server():
    """Start a local HTTP server with configurable routes, yield (base_url, handler_cls)."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _StubHandler.routes = {}
    _StubHandler.requests = []
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
        # Reaches an end user in the app, so it says what they can do about it
        # rather than naming an env var they have never set.
        with pytest.raises(PermissionError, match="Sign in"):
            client.get_model_artifacts("slp-orchestra-mini")

    def test_404_raises_permission_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="not found"):
            client.get_model_artifacts("nonexistent")

    def test_403_raises_model_access_error(self, stub_server):
        """403 is "not on your plan", not "your token is bad" — the two lead to
        different UI (upgrade vs. sign in), so they must not collapse into one
        exception."""
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/gemma3-12b/download": (403, {"detail": "forbidden"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ModelAccessError, match="not available on your current plan"):
            client.get_model_artifacts("gemma3-12b")

    def test_free_model_downloads_without_auth_headers(self, stub_server):
        """A free model is served to an anonymous caller; the client must not
        require credentials to ask."""
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/gemma3-4b/download": (200, {
                "artifacts": {"model.gguf": "https://storage/model.gguf"},
            }),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.get_model_artifacts("gemma3-4b") == {
            "model.gguf": "https://storage/model.gguf",
        }

    def test_429_raises_connection_error(self, stub_server):
        """Rate limiting is transient, so it must not be typed as a plan problem.

        The download route is rate-limited per IP, which anonymous downloads now
        reach. Classing a 429 as a PermissionError would tell the user to
        upgrade when what they need is to wait.
        """
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/slp-orchestra-mini/download": (429, {"detail": "rate limited"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError) as exc:
            client.get_model_artifacts("slp-orchestra-mini")
        assert not isinstance(exc.value, PermissionError)
        assert exc.value.status_code == 429

    def test_server_error_raises_connection_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/slp-orchestra-mini/download": (500, {"detail": "boom"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError) as exc:
            client.get_model_artifacts("slp-orchestra-mini")
        assert not isinstance(exc.value, PermissionError)
        assert exc.value.status_code == 500

    def test_errors_carry_the_upstream_status(self, stub_server):
        """Callers classify by status, not by matching message text.

        /v1/init reads this back off the exception to choose its own HTTP
        status, and a message is a bad channel for that: it is written for a
        person and it gets reworded.
        """
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/gemma3-12b/download": (403, {"detail": "upgrade"}),
        }

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ModelAccessError) as exc:
            client.get_model_artifacts("gemma3-12b")
        assert exc.value.status_code == 403


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


class TestWebSearch:
    """401/403 must stay distinguishable from every other failure: the first
    means search is off for the account, the rest are transient."""

    def test_returns_links(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, ["http://a", "http://b"])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
        assert client.web_search("bikes", 5) == ["http://a", "http://b"]

    def test_drops_non_string_and_blank_entries(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, ["http://a", "", None, 3, "  "])}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.web_search("bikes", 5) == ["http://a"]

    def test_non_list_body_returns_empty(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (200, {"results": []})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.web_search("bikes", 5) == []

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_raises_permission_error(self, stub_server, status):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (status, {"detail": "nope"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match=str(status)):
            client.web_search("bikes", 5)

    def test_spent_allowance_raises_permission_error_not_connection_error(
        self, stub_server
    ):
        """429 means the month's searches are used up — stop searching, don't
        retry it as if the endpoint had blipped."""
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (429, {"detail": "limit reached"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="429"):
            client.web_search("bikes", 5)

    def test_server_error_raises_connection_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch": (500, {"detail": "boom"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError, match="API error"):
            client.web_search("bikes", 5)

    def test_unreachable_raises_connection_error(self):
        client = SLPPlatformClient("http://127.0.0.1:1", auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.web_search("bikes", 5)


class TestWebSearchStatus:
    def test_enabled(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (200, {"enabled": True})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.web_search_status() is True

    def test_disabled(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (200, {"enabled": False})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        assert client.web_search_status() is False

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_failure_raises_permission_error(self, stub_server, status):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (status, {"detail": "nope"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match=str(status)):
            client.web_search_status()

    def test_server_error_raises_connection_error(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/websearch/status": (500, {"detail": "boom"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError, match="API error"):
            client.web_search_status()


class TestRequestHeaders:
    def test_user_agent_is_sent_with_and_without_auth(self, stub_server):
        """The account access log would otherwise show python-httpx for every
        call — including the unauthenticated ones."""
        base_url, handler = stub_server
        seen = []

        original_do_GET = handler.do_GET

        def recording_do_GET(self_handler):
            seen.append(dict(self_handler.headers))
            original_do_GET(self_handler)

        handler.do_GET = recording_do_GET
        handler.routes = {
            "/v1/users/me": (200, {"email": "a@b.c"}),
            "/v1/bootstrap/models": (200, [{"name": "sl-mini"}]),
        }
        try:
            client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer jwt"})
            client.get_current_user()
            client.list_bootstrap_models()
        finally:
            handler.do_GET = original_do_GET

        assert len(seen) == 2
        assert all(h.get("User-Agent", "").startswith("smartloop/") for h in seen)
        assert seen[0].get("Authorization") == "Bearer jwt"
        assert "Authorization" not in seen[1], "bootstrap is deliberately unauthenticated"


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

    def test_a_server_error_is_transient_not_a_verdict(self, stub_server):
        """A 500 is the platform having a bad moment, not a bad credential.

        It used to arrive as ``PermissionError`` — the endpoint's *error_cls* was
        applied to every non-2xx — which sent it down the expired-token path: the
        cache-through service refuses to serve stale data for it, and the refresh
        agent clears both tokens on it.  A platform blip logged the user out.
        """
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (500, {"detail": "Internal Server Error"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError, match="API error") as exc:
            client.get_current_user()
        assert not isinstance(exc.value, PermissionError)
        assert exc.value.status_code == 500

    def test_a_rate_limit_is_transient_too(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (429, {"detail": "Too Many Requests"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError):
            client.get_current_user()


class TestRefreshOn401:
    """Every authenticated call redeems the refresh token once on a 401.

    This used to live at a single call site — ``_resolve_user``, behind
    ``/v1/auth/status`` — so a model download answered "Sign in to download this
    model" and web search quietly reported itself off, both while a working
    refresh token sat on disk.  It belongs to the client, where the 401 actually
    arrives and every endpoint gets it.
    """

    def _client(self, base_url, refreshes, token=lambda: "expired"):
        """A client whose refresh swaps the token the next request will send."""
        state = {"token": token()}

        def auth_headers():
            return {"Authorization": f"Bearer {state['token']}"}

        def refresh():
            refreshes.append(state["token"])
            state["token"] = "fresh"
            return True

        return SLPPlatformClient(base_url, auth_headers_fn=auth_headers, refresh_fn=refresh)

    def test_a_401_is_retried_with_the_refreshed_token(self, stub_server):
        base_url, handler = stub_server
        user = {"email": "chris@example.com", "name": "Chris"}
        handler.routes = {"/v1/users/me": [(401, {"detail": "Unauthorized"}), (200, user)]}
        refreshes = []

        client = self._client(base_url, refreshes)
        assert client.get_current_user() == user

        assert len(refreshes) == 1
        assert len(handler.requests) == 2
        # The retry re-read auth_headers_fn rather than reusing the prepared
        # headers — otherwise it would resend the token that just failed.
        assert handler.requests[0]["headers"]["Authorization"] == "Bearer expired"
        assert handler.requests[1]["headers"]["Authorization"] == "Bearer fresh"

    def test_a_403_is_not_a_stale_token(self, stub_server):
        """A plan refusal is about the account, not the credential.

        ``ModelAccessError`` subclasses ``PermissionError``, so keying the retry
        on the exception type would re-mint the same identity and ask again for a
        model the plan still does not grant.  Keyed on the status, it cannot.
        """
        base_url, handler = stub_server
        handler.routes = {"/v1/models/gemma3-12b/download": (403, {"detail": "forbidden"})}
        refreshes = []

        client = self._client(base_url, refreshes)
        with pytest.raises(ModelAccessError):
            client.get_model_artifacts("gemma3-12b")

        assert refreshes == []
        assert len(handler.requests) == 1

    def test_nothing_to_refresh_with_means_no_retry(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {"/v1/models/sl-mini/download": (401, {"detail": "Unauthorized"})}

        client = SLPPlatformClient(
            base_url, auth_headers_fn=lambda: {}, refresh_fn=lambda: False,
        )
        with pytest.raises(PermissionError, match="Sign in"):
            client.get_model_artifacts("sl-mini")

        assert len(handler.requests) == 1

    def test_the_retry_happens_once(self, stub_server):
        """A token the platform keeps refusing must not become a request loop."""
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (401, {"detail": "Unauthorized"})}
        refreshes = []

        client = self._client(base_url, refreshes)
        with pytest.raises(PermissionError):
            client.get_current_user()

        assert len(refreshes) == 1
        assert len(handler.requests) == 2

    def test_the_refresh_goes_to_the_host_that_refused(self, stub_server):
        """The default hook redeems against this client, not ``AppSettings``.

        Resolving the base URL from settings instead means a client built for
        any other host sends its refresh token somewhere it does not belong.
        """
        base_url, handler = stub_server
        handler.routes = {
            "/v1/users/me": [(401, {"detail": "Unauthorized"}), (200, {"email": "c@e.com"})],
            "/v1/auth/jwt/refresh": (200, {"access_token": "fresh"}),
        }

        credential_store.set_access_token("expired")
        credential_store.set_refresh_token("rt")
        client = SLPPlatformClient(base_url)  # no refresh_fn — the default hook
        assert client.get_current_user() == {"email": "c@e.com"}

        assert [r["path"] for r in handler.requests] == [
            "/v1/users/me", "/v1/auth/jwt/refresh", "/v1/users/me",
        ]
        assert credential_store.get_access_token() == "fresh"

    def test_the_catalog_refreshes_before_falling_back_to_anonymous(self, stub_server):
        """``list_models`` degrades to anonymous, but only as a last resort.

        Going straight to the anonymous retry left a signed-in user looking at a
        free-tier catalog — every entry ``is_accessible`` false — with no signal
        that their token had merely expired.
        """
        base_url, handler = stub_server
        models = [{"name": "sl-mini"}]
        handler.routes = {"/v1/models": [(401, {"detail": "Unauthorized"}), (200, models)]}
        refreshes = []

        client = self._client(base_url, refreshes)
        assert client.list_models() == models

        assert len(refreshes) == 1
        assert handler.requests[1]["headers"]["Authorization"] == "Bearer fresh"


class TestRefreshAccessToken:
    """The refresh grant itself — the one call that must never be misread.

    Its failures are acted on destructively: ``refresh_access_token_now`` clears
    both credentials on a ``PermissionError``.  Nothing exercised this endpoint
    directly before; every refresh test mocked it out.
    """

    _PATH = "/v1/auth/jwt/refresh"

    def test_it_posts_the_token_unauthenticated(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {self._PATH: (200, {"access_token": "new", "refresh_token": "rotated"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {"Authorization": "Bearer dead"})
        data = client.refresh_access_token("old-rt")

        assert data == {"access_token": "new", "refresh_token": "rotated"}
        sent = handler.requests[0]
        assert sent["method"] == "POST"
        assert json.loads(sent["body"]) == {"refresh_token": "old-rt"}
        # Sending the dead access token alongside would invite the endpoint to
        # judge it rather than the refresh token.
        assert "Authorization" not in sent["headers"]

    def test_a_401_is_a_dead_refresh_token(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {self._PATH: (401, {"detail": "Unauthorized"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(PermissionError, match="expired or revoked"):
            client.refresh_access_token("old-rt")

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_a_transient_failure_is_not_a_dead_refresh_token(self, stub_server, status):
        """The regression this whole review started from.

        Every non-2xx took the endpoint's ``PermissionError``, and the refresh
        agent clears *both* tokens on that — so one bad minute on the platform
        logged every user out permanently, with no way back but a re-login.
        """
        base_url, handler = stub_server
        handler.routes = {self._PATH: (status, {"detail": "nope"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ConnectionError) as exc:
            client.refresh_access_token("old-rt")
        assert not isinstance(exc.value, PermissionError)

    def test_a_reply_without_an_access_token_raises(self, stub_server):
        base_url, handler = stub_server
        handler.routes = {self._PATH: (200, {"token_type": "Bearer"})}

        client = SLPPlatformClient(base_url, auth_headers_fn=lambda: {})
        with pytest.raises(ValueError, match="missing access_token"):
            client.refresh_access_token("old-rt")


class TestTheRetryFollowsTheCredential:
    """``_authed`` retries when the credentials *changed*, not when the refresh
    reported success. The two come apart in both directions.
    """

    def test_a_dead_oauth_token_falls_back_to_the_developer_token(self, stub_server):
        """Redeeming a dead refresh token clears both OAuth credentials — and
        leaves a working developer token behind.

        Keyed on the refresh's return value that read as failure, so the request
        was abandoned with "Sign in to download this model" for a user who was
        signed in, and the identical next request succeeded.
        """
        base_url, handler = stub_server
        handler.routes = {
            "/v1/models/sl-mini/download": [
                (401, {"detail": "Unauthorized"}),
                (200, {"artifacts": {"m.gguf": "https://blob/m.gguf"}}),
            ],
            "/v1/auth/jwt/refresh": (401, {"detail": "revoked"}),
        }
        credential_store.set_access_token("expired")
        credential_store.set_refresh_token("revoked-rt")
        credential_store.set_token("sk-dev-good")

        client = SLPPlatformClient(base_url)
        assert client.get_model_artifacts("sl-mini") == {"m.gguf": "https://blob/m.gguf"}

        assert credential_store.get_access_token() is None
        assert credential_store.get_refresh_token() is None
        sent = [r["headers"] for r in handler.requests if r["path"].endswith("/download")]
        assert sent[0]["Authorization"] == "Bearer expired"
        assert sent[1]["X-Api-Key"] == "sk-dev-good"

    def test_an_unchanged_credential_is_not_resent(self, stub_server):
        """A refresh that reports success without changing what we send would
        only buy a guaranteed second 401."""
        base_url, handler = stub_server
        handler.routes = {"/v1/users/me": (401, {"detail": "Unauthorized"})}

        client = SLPPlatformClient(
            base_url,
            auth_headers_fn=lambda: {"Authorization": "Bearer same"},
            refresh_fn=lambda: True,
        )
        with pytest.raises(PermissionError):
            client.get_current_user()

        assert len(handler.requests) == 1


class TestTelemetryCannotLogYouOut:
    def test_a_401_on_usage_does_not_redeem_the_refresh_token(self, stub_server):
        """``push_token_usage`` is fire-and-forget and swallows every failure.

        Routed through ``_authed`` it could redeem the refresh token, and a
        refresh endpoint answering 401 clears both credentials — so a telemetry
        POST could log the user out with nothing visible but a log line. Records
        flush in batches, so it would do it once per queued record.
        """
        base_url, handler = stub_server
        handler.routes = {"/v1/usage/tokens": (401, {"detail": "Unauthorized"})}
        credential_store.set_access_token("expired")
        credential_store.set_refresh_token("rt")

        SLPPlatformClient(base_url).push_token_usage({"tokens": 1})

        assert not [r for r in handler.requests if "refresh" in r["path"]]
        assert credential_store.get_refresh_token() == "rt"
