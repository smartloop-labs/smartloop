"""Requests that name a project in the URL are bound to it for their lifetime.

The chat endpoint carries ``project_id`` in the body and binds itself; every
other router takes it in the path or query string, which the middleware can
read from the ASGI scope without touching the request stream.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from smartloop.api.middleware import ProjectScopeMiddleware, _project_from_scope
from smartloop.workspace.context import active_project_id


def _scope(path: str, query: str = "") -> dict:
    return {"type": "http", "path": path, "query_string": query.encode()}


class TestReadingTheProjectOffTheScope:
    def test_from_the_path(self):
        assert _project_from_scope(_scope("/v1/projects/abc/documents")) == "abc"

    def test_from_a_nested_path(self):
        assert _project_from_scope(_scope("/v1/projects/abc/mcp/tools/count")) == "abc"

    def test_from_the_query_string(self):
        assert _project_from_scope(_scope("/v1/feedback", "project_id=xyz")) == "xyz"

    def test_the_path_wins_over_the_query(self):
        scope = _scope("/v1/projects/abc/documents", "project_id=xyz")
        assert _project_from_scope(scope) == "abc"

    def test_no_project_named(self):
        assert _project_from_scope(_scope("/v1/models")) is None

    def test_the_collection_itself_names_no_project(self):
        assert _project_from_scope(_scope("/v1/projects")) is None

    def test_a_collection_verb_is_not_a_project(self):
        """``import`` sits where an id sits and is not one.

        Bound as a project it produced "bound project import is not in this
        workspace" the moment anything resolved it — a warning about a
        workspace member that cannot exist, in the middle of an import that
        had otherwise succeeded.
        """
        assert _project_from_scope(_scope("/v1/projects/import")) is None

    def test_the_two_layers_read_the_same_list(self):
        """Binding and routing agree on what is not a project.

        They disagreed once: the supervisor stopped forwarding ``import``
        while the middleware went on binding it.  The next verb added must
        not be able to land in one layer only.
        """
        from smartloop.api.middleware import _COLLECTION_ACTIONS
        from smartloop.supervisor import routing

        assert routing._COLLECTION_ACTIONS is _COLLECTION_ACTIONS
        for path in _COLLECTION_ACTIONS:
            assert _project_from_scope(_scope(path)) is None
            assert routing.project_to_proxy({**_scope(path), "method": "POST"}) is None

    def test_a_malformed_query_string_is_not_fatal(self):
        scope = {"type": "http", "path": "/v1/models", "query_string": b"\xff\xfe=%%"}
        assert _project_from_scope(scope) is None


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()

    @application.get("/v1/projects/{project_id}/probe")
    async def probe(project_id: str):
        return {"bound": active_project_id()}

    @application.get("/v1/unscoped")
    async def unscoped():
        return {"bound": active_project_id()}

    @application.get("/v1/projects/{project_id}/stream")
    async def stream(project_id: str):
        async def body():
            # Reads while the response is streaming, i.e. after the handler
            # has already returned — the case a BaseHTTPMiddleware would miss.
            yield (active_project_id() or "none").encode()

        return StreamingResponse(body(), media_type="text/plain")

    application.add_middleware(ProjectScopeMiddleware)
    return application


class TestTheBindingReachesTheHandler:
    def test_a_path_scoped_request_is_bound(self, app):
        assert TestClient(app).get("/v1/projects/abc/probe").json() == {"bound": "abc"}

    def test_an_unscoped_request_is_not(self, app):
        assert TestClient(app).get("/v1/unscoped").json() == {"bound": None}

    def test_the_binding_survives_into_a_streaming_body(self, app):
        assert TestClient(app).get("/v1/projects/abc/stream").text == "abc"

    def test_the_binding_does_not_leak_between_requests(self, app):
        client = TestClient(app)
        assert client.get("/v1/projects/abc/probe").json() == {"bound": "abc"}
        assert client.get("/v1/unscoped").json() == {"bound": None}
        assert active_project_id() is None
