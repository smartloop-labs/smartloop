"""Getting a request from :38540 to the child that owns its project.

The client contract is supposed to be unchanged, which is why this proxies
rather than redirecting: one origin, one auth surface, and the desktop app
never learns that a second port exists. The cost a redirect would have avoided
— streaming SSE through two hops — is paid off by never buffering, which is
what the streaming tests here are for.
"""

import asyncio
import contextlib
import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from smartloop.supervisor.routing import (
    ProjectRoutingMiddleware, forwardable_headers, project_from_body,
    project_to_proxy, read_body, replay_body, response_headers, routes_by_body,
)


class FakeRecord:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def touch(self) -> None:
        pass


class FakePool:
    """Stands in for the agent pool: records what was routed where."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.requested: list[str] = []
        self.fail_with: Exception | None = None

    def ensure(self, project_id: str):
        self.requested.append(project_id)
        if self.fail_with:
            raise self.fail_with
        return FakeRecord(self.base_url)


@contextlib.contextmanager
def serve(app):
    """Run *app* on a real loopback port and yield its base URL.

    A real socket, not ``TestClient``: the ASGI test transport assembles the
    whole response before handing it back, so a streamed body would look
    buffered no matter what the proxy did.
    """
    import uvicorn

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                            log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def child():
    """A stand-in project agent, on a real port, echoing what it received."""
    app = FastAPI()

    @app.get("/v1/projects/{project_id}/documents")
    async def documents(project_id: str):
        return {"served_by": "child", "project_id": project_id}

    @app.get("/v1/echo")
    async def echo(request):  # pragma: no cover - replaced below
        return {}

    @app.api_route("/v1/probe", methods=["GET", "POST"])
    async def probe(request: __import__("fastapi").Request):
        return {
            "method": request.method,
            "query": request.url.query,
            "body": (await request.body()).decode(),
            "saw_header": request.headers.get("x-custom"),
        }

    @app.post("/v1/chat/completions")
    async def chat(request: __import__("fastapi").Request):
        payload = await request.json()
        return {"served_by": "child", "project_id": payload.get("project_id")}

    @app.post("/v1/feedback")
    async def feedback(request: __import__("fastapi").Request):
        payload = await request.json()
        return {"served_by": "child", "project_id": payload.get("project_id")}

    @app.get("/v1/projects/{project_id}/stream")
    async def stream(project_id: str):
        async def body():
            for index in range(5):
                yield f"data: chunk-{index}\n\n".encode()
                await asyncio.sleep(0.05)

        return StreamingResponse(body(), media_type="text/event-stream")

    with serve(app) as base_url:
        yield base_url


@pytest.fixture
def supervisor_app(child):
    """A supervisor app: its own routes, plus routing to the child."""
    app = FastAPI()
    pool = FakePool(child)

    @app.get("/v1/models")
    async def models():
        return {"served_by": "supervisor"}

    @app.get("/v1/projects")
    async def projects():
        return {"served_by": "supervisor"}

    @app.delete("/v1/projects/{project_id}")
    async def delete_project(project_id: str):
        return {"served_by": "supervisor", "project_id": project_id}

    @app.post("/v1/projects/import")
    async def import_project():
        return {"served_by": "supervisor"}

    @app.post("/v1/chat/completions")
    async def chat_here():
        return {"served_by": "supervisor"}

    @app.get("/health")
    async def health():
        return {"served_by": "supervisor"}

    app.add_middleware(ProjectRoutingMiddleware, pool_provider=lambda: pool)
    app.state.pool = pool
    return app


class TestReadingTheProjectFromABody:
    def test_a_chat_request_names_its_project_in_the_body(self):
        assert project_from_body(b'{"project_id": "alpha"}') == "alpha"

    def test_a_body_without_one(self):
        assert project_from_body(b'{"model": "sl-mini"}') is None

    def test_an_empty_body(self):
        assert project_from_body(b"") is None

    def test_a_non_json_body_is_not_an_error(self):
        assert project_from_body(b"not json at all") is None

    def test_a_json_array_is_not_an_error(self):
        assert project_from_body(b'[1, 2, 3]') is None

    def test_an_empty_project_id_does_not_count(self):
        assert project_from_body(b'{"project_id": ""}') is None

    def test_chat_is_routed_by_its_body(self):
        assert routes_by_body({"path": "/v1/chat/completions"}) is True

    def test_feedback_is_routed_by_its_body(self):
        """Its POST names the project in the body; its GET/DELETE in the query."""
        assert routes_by_body({"path": "/v1/feedback"}) is True

    def test_nothing_else_is(self):
        assert routes_by_body({"path": "/v1/models"}) is False
        assert routes_by_body({"path": "/v1/projects"}) is False


class TestWhatOnlyTheWorkspaceCanAnswer:
    """A delete is the one thing a project's own agent cannot do to itself.

    Removing a project means removing it from a list that lives only in
    ``config.json``, which a pinned save never writes — and could not write
    here anyway, since the pinned project is the one leaving the list.
    Forwarded, the child deleted its own directory and persisted nothing.
    Every other write to a project record leaves the project in existence, so
    it still goes to the child that owns it.
    """

    def _scope(self, path: str, method: str = "GET", query: str = "") -> dict:
        return {"type": "http", "method": method, "path": path,
                "query_string": query.encode()}

    def test_a_sub_resource_belongs_to_the_child(self):
        assert project_to_proxy(self._scope("/v1/projects/abc/documents")) == "abc"

    def test_a_nested_sub_resource_too(self):
        assert project_to_proxy(self._scope("/v1/projects/abc/mcp/tools/count")) == "abc"

    def test_a_delete_is_not_forwarded(self):
        assert project_to_proxy(self._scope("/v1/projects/abc", "DELETE")) is None

    def test_a_delete_of_a_sub_resource_still_is(self):
        """Deleting a document is the child's; only the project itself is not."""
        assert project_to_proxy(
            self._scope("/v1/projects/abc/documents/d1", "DELETE")
        ) == "abc"

    def test_a_rename_still_belongs_to_the_child(self):
        """It writes ``web_search_enabled`` and skills — project state, not the index."""
        assert project_to_proxy(self._scope("/v1/projects/abc", "PATCH")) == "abc"

    def test_the_collection_is_not_forwarded(self):
        assert project_to_proxy(self._scope("/v1/projects")) is None

    def test_import_is_a_verb_not_a_project_id(self):
        assert project_to_proxy(self._scope("/v1/projects/import", "POST")) is None

    def test_the_query_string_still_names_a_project(self):
        assert project_to_proxy(self._scope("/v1/feedback", query="project_id=xyz")) == "xyz"

    def test_a_malformed_query_string_is_not_fatal(self):
        scope = self._scope("/v1/models")
        scope["query_string"] = b"\xff\xfe=%%"
        assert project_to_proxy(scope) is None

    def test_a_delete_is_answered_by_the_supervisor(self, supervisor_app):
        response = TestClient(supervisor_app).delete("/v1/projects/alpha")

        assert response.json() == {"served_by": "supervisor", "project_id": "alpha"}
        # Not merely answered here — no agent was started to answer it.
        assert supervisor_app.state.pool.requested == []

    def test_an_import_does_not_start_an_agent_called_import(self, supervisor_app):
        response = TestClient(supervisor_app).post("/v1/projects/import")

        assert response.json() == {"served_by": "supervisor"}
        assert supervisor_app.state.pool.requested == []


class TestHeaderHandling:
    def test_hop_by_hop_headers_are_not_forwarded(self):
        headers = forwardable_headers([
            (b"host", b"localhost:38540"),
            (b"connection", b"keep-alive"),
            (b"content-length", b"12"),
            (b"transfer-encoding", b"chunked"),
            (b"authorization", b"Bearer t"),
            (b"x-custom", b"kept"),
        ])
        assert headers == {"authorization": "Bearer t", "x-custom": "kept"}

    def test_response_hop_by_hop_headers_are_dropped(self):
        import httpx

        headers = httpx.Headers([
            ("content-type", "text/event-stream"),
            ("transfer-encoding", "chunked"),
            ("cache-control", "no-cache"),
        ])
        assert response_headers(headers) == [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache"),
        ]


class TestReplayingABody:
    @pytest.mark.asyncio
    async def test_a_body_read_for_routing_can_still_be_served(self):
        receive = replay_body(b'{"a": 1}')
        assert await receive() == {
            "type": "http.request", "body": b'{"a": 1}', "more_body": False,
        }
        assert (await receive())["type"] == "http.disconnect"

    @pytest.mark.asyncio
    async def test_a_chunked_body_is_reassembled(self):
        messages = [
            {"type": "http.request", "body": b"one", "more_body": True},
            {"type": "http.request", "body": b"two", "more_body": False},
        ]

        async def receive():
            return messages.pop(0)

        assert await read_body(receive) == b"onetwo"


class TestWhatGetsForwarded:
    def test_a_project_scoped_path_goes_to_the_child(self, supervisor_app):
        response = TestClient(supervisor_app).get("/v1/projects/alpha/documents")

        assert response.json() == {"served_by": "child", "project_id": "alpha"}
        assert supervisor_app.state.pool.requested == ["alpha"]

    def test_a_workspace_level_path_is_answered_here(self, supervisor_app):
        response = TestClient(supervisor_app).get("/v1/models")

        assert response.json() == {"served_by": "supervisor"}
        assert supervisor_app.state.pool.requested == []

    def test_the_project_list_itself_is_workspace_level(self, supervisor_app):
        assert TestClient(supervisor_app).get("/v1/projects").json() == \
            {"served_by": "supervisor"}

    def test_health_is_the_supervisors_own(self, supervisor_app):
        assert TestClient(supervisor_app).get("/health").json() == \
            {"served_by": "supervisor"}

    def test_a_chat_request_is_routed_by_its_body(self, supervisor_app):
        response = TestClient(supervisor_app).post(
            "/v1/chat/completions", json={"project_id": "beta", "model": "sl-mini"},
        )

        assert response.json() == {"served_by": "child", "project_id": "beta"}
        assert supervisor_app.state.pool.requested == ["beta"]

    def test_a_feedback_post_reaches_the_project_it_names(self, supervisor_app):
        """The POST used to be answered here, indexing into the wrong project."""
        response = TestClient(supervisor_app).post(
            "/v1/feedback",
            json={"session_id": "s1", "message_id": "m1",
                  "project_id": "gamma", "rating": "up"},
        )

        assert response.json() == {"served_by": "child", "project_id": "gamma"}
        assert supervisor_app.state.pool.requested == ["gamma"]

    def test_a_chat_request_naming_no_project_stays_here(self, supervisor_app):
        response = TestClient(supervisor_app).post(
            "/v1/chat/completions", json={"model": "sl-mini"},
        )

        assert response.json() == {"served_by": "supervisor"}
        assert supervisor_app.state.pool.requested == []

    def test_the_query_string_survives_the_hop(self, supervisor_app):
        response = TestClient(supervisor_app).get(
            "/v1/probe?project_id=alpha&limit=5",
        )
        assert response.json()["query"] == "project_id=alpha&limit=5"

    def test_the_body_and_headers_survive_the_hop(self, supervisor_app):
        response = TestClient(supervisor_app).post(
            "/v1/probe?project_id=alpha",
            content=b'{"hello": "world"}',
            headers={"x-custom": "kept"},
        )
        body = response.json()
        assert body["method"] == "POST"
        assert body["body"] == '{"hello": "world"}'
        assert body["saw_header"] == "kept"


class TestStreamingThroughTwoHops:
    """The cost a redirect would have avoided, paid off by never buffering.

    Driven over real sockets on both hops: the ASGI test transport assembles
    the whole response first, which would make a buffering proxy pass.
    """

    def test_an_sse_response_arrives_chunk_by_chunk(self, supervisor_app):
        import httpx

        arrivals: list[float] = []
        chunks: list[str] = []

        with serve(supervisor_app) as base_url:
            started = time.monotonic()
            with httpx.stream(
                "GET", f"{base_url}/v1/projects/alpha/stream", timeout=30,
            ) as response:
                assert response.headers["content-type"].startswith("text/event-stream")
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        arrivals.append(time.monotonic() - started)
                        chunks.append(line)

        assert len(chunks) == 5
        # The child sleeps 50ms between chunks. Had the supervisor buffered the
        # response, every chunk would have landed at the same moment.
        assert arrivals[-1] - arrivals[0] > 0.1

    def test_every_chunk_arrives_intact(self, supervisor_app):
        import httpx

        with serve(supervisor_app) as base_url:
            body = httpx.get(
                f"{base_url}/v1/projects/alpha/stream", timeout=30,
            ).text

        for index in range(5):
            assert f"data: chunk-{index}" in body


class TestWhenAChildCannotBeReached:
    def test_a_project_that_will_not_start_is_a_503(self, supervisor_app):
        supervisor_app.state.pool.fail_with = RuntimeError("index is corrupt")

        response = TestClient(supervisor_app).get("/v1/projects/alpha/documents")

        assert response.status_code == 503
        assert "index is corrupt" in response.json()["detail"]

    def test_an_agent_that_does_not_answer_is_a_502(self, child):
        app = FastAPI()
        pool = FakePool("http://127.0.0.1:1")  # nothing listens there

        @app.get("/v1/models")
        async def models():
            return {"served_by": "supervisor"}

        app.add_middleware(ProjectRoutingMiddleware, pool_provider=lambda: pool)

        response = TestClient(app).get("/v1/projects/alpha/documents")

        assert response.status_code == 502
        assert "did not answer" in response.json()["detail"]

    def test_the_supervisor_keeps_serving_its_own_routes(self, supervisor_app):
        supervisor_app.state.pool.fail_with = RuntimeError("nope")
        client = TestClient(supervisor_app)

        assert client.get("/v1/projects/alpha/documents").status_code == 503
        assert client.get("/v1/models").json() == {"served_by": "supervisor"}


class TestWithoutASupervisor:
    """Every unsupervised install: the middleware must be invisible."""

    def test_no_pool_means_nothing_is_routed(self, child):
        app = FastAPI()

        @app.get("/v1/projects/{project_id}/documents")
        async def documents(project_id: str):
            return {"served_by": "single-process", "project_id": project_id}

        app.add_middleware(ProjectRoutingMiddleware, pool_provider=lambda: None)

        assert TestClient(app).get("/v1/projects/alpha/documents").json() == \
            {"served_by": "single-process", "project_id": "alpha"}

    def test_a_chat_body_is_not_even_read(self, child):
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request: __import__("fastapi").Request):
            return {"body": (await request.body()).decode()}

        app.add_middleware(ProjectRoutingMiddleware, pool_provider=lambda: None)

        response = TestClient(app).post(
            "/v1/chat/completions", json={"project_id": "alpha"},
        )
        assert json.loads(response.json()["body"])["project_id"] == "alpha"
