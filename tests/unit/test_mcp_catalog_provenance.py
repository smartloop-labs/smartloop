"""Curated provenance has to survive the whole install, not just the request.

``catalog_slug`` is the record of which catalog entry a connection came from,
and two things downstream depend on it:

* :meth:`ModelService._enrich_mcp_server` skips the LLM for a curated server.
  Without the slug, a 93-tool server like monday.com goes to the small model
  and comes back as unparseable JSON — two inference calls spent to produce
  nothing, and the curated description overwritten if it had succeeded.
* :func:`~smartloop.mcp.definitions.definition_for` checks the slug before it
  falls back to matching substrings against the server's name and URL, which is
  how a deleted connection's dropped-in file is found and removed.

The install is not one call: register resolves the catalog, the OAuth grant
leaves and comes back, and only then is the record written and saved.  The
existing tests cover the ends of that chain — the request model has the field,
the service methods accept it, ``MCPServer`` declares it, enrichment honours it
— and a connection installed from the catalog can still land with the slug
missing, because nothing covers the hops in between.  These do.
"""

import asyncio
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smartloop.api.model.mcp import MCPOAuthCallbackRequest, MCPRegisterRequest
from smartloop.api.routers import mcp_servers as router_mod
from smartloop.mcp.client import MCPConnectResult
from smartloop.mcp.oauth import MCPOAuthService, OAuthMetadata, ProtectedResourceMetadata

SLUG = "monday"
CATALOG_URL = "https://mcp.monday.com/mcp"
CATALOG_NAME = "monday.com"
CATALOG_DESC = "Query and update monday.com boards"
CATALOG_TAGS = ["project-management", "boards", "workflows"]

# What the vendor's own handshake says it is called.  Deliberately different
# from the catalog's display name and empty where the catalog has copy: that
# difference is what tells a curated write apart from a generic one.
HANDSHAKE = SimpleNamespace(
    name="monday.com MCP Server", title=None, version="1.0",
    description=None, icons=None,
)


def _catalog_entry():
    return SimpleNamespace(
        slug=SLUG,
        display_name=CATALOG_NAME,
        description=CATALOG_DESC,
        tags=list(CATALOG_TAGS),
        server_type="remote",
        remote_config=SimpleNamespace(server_url=CATALOG_URL, auth_type="oauth"),
        local_config=None,
    )


class _FakeOAuthRegister:
    """OAuth discovery that lands, so registration takes the OAuth branch."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def discover_protected_resource_metadata(self, url, metadata_url=None):
        return ProtectedResourceMetadata(
            resource=CATALOG_URL, authorization_servers=["https://auth.monday.com/mcp"],
        )

    def discover_oauth_metadata(self, url):
        return OAuthMetadata(
            issuer="https://auth.monday.com/mcp",
            authorization_endpoint="https://auth.monday.com/oauth2/authorize",
            token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
            registration_endpoint="https://auth.monday.com/oauth_ms/oauth/register",
        )

    def register_client(self, endpoint, redirect_uri):
        return MagicMock(client_id="cid", client_secret="csec")

    def build_authorization_url(self, **kwargs):
        return "https://auth.monday.com/oauth2/authorize?stub"

    canonical_resource = staticmethod(MCPOAuthService.canonical_resource)
    generate_pkce = staticmethod(MCPOAuthService.generate_pkce)
    generate_state = staticmethod(MCPOAuthService.generate_state)


class _FakeOAuthCallback:
    """Code exchange that lands, for the callback half."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def exchange_code_for_tokens(self, **kwargs):
        return MagicMock(
            access_token="at", token_type="Bearer", refresh_token="rt", expires_in=3600,
        )


def _service_stub():
    """A model service whose add_remote_mcp_server returns a persistable record.

    The client-driven completion serializes what it wrote back to the caller, so
    the stub has to answer with the shape ``_mcp_server_to_dict`` produces.
    """
    service = MagicMock()

    def _record(**kwargs):
        return {
            "id": "s1",
            "name": kwargs.get("name"),
            "description": kwargs.get("description"),
            "tags": kwargs.get("tags") or [],
            "catalog_slug": kwargs.get("catalog_slug"),
            "server_type": "remote",
            "server_url": kwargs.get("server_url"),
            "enabled": True,
            "tools": [],
        }

    service.add_remote_mcp_server.side_effect = _record
    return service


def _register(catalog_hit=True, slug=SLUG):
    """Run the real register endpoint and return the stored OAuth session."""
    entry = _catalog_entry() if catalog_hit else None

    # The direct, unauthenticated connect has to fail for OAuth to be reached.
    failing = MagicMock()
    failing.__enter__ = MagicMock(side_effect=RuntimeError("401"))
    failing.__exit__ = MagicMock(return_value=False)

    router_mod._pending_oauth_sessions.clear()
    with patch.object(router_mod, "resolve_catalog_entry", return_value=entry), \
         patch("smartloop.mcp.MCPOAuthService", _FakeOAuthRegister), \
         patch("smartloop.mcp.MCPClient", return_value=failing), \
         patch.object(
             router_mod, "_try_register_smartloop_mcp_server",
             new=AsyncMock(return_value=None),
         ):
        response = asyncio.run(
            router_mod.register_mcp_server(
                project_id="p1",
                request=MCPRegisterRequest(
                    server_url=CATALOG_URL, server_type="remote", catalog_slug=slug,
                ),
                model_service=MagicMock(),
            )
        )
    session = next(iter(router_mod._pending_oauth_sessions.values()))
    return response, session


def _connect_result():
    tool = SimpleNamespace(
        name="get_board_items_page", description="items", input_schema={"type": "object"},
    )
    return MCPConnectResult(success=True, tools=[tool], server_info=HANDSHAKE)


def _run_callback(session, model_service):
    """Drive the browser-redirect callback over a stored session."""
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.connect_and_discover_tools = MagicMock(return_value=_connect_result())

    router_mod._pending_oauth_sessions.clear()
    router_mod._completed_oauth_responses.clear()
    router_mod._pending_oauth_sessions[session["state"]] = session
    with patch("smartloop.mcp.MCPOAuthService", _FakeOAuthCallback), \
         patch("smartloop.mcp.MCPClient", return_value=client):
        asyncio.run(
            router_mod.mcp_oauth_callback(
                code="c", state=session["state"], model_service=model_service,
            )
        )
    return model_service.add_remote_mcp_server.call_args


class TestRegistrationCarriesTheSlug:
    def test_the_oauth_session_holds_every_curated_field(self):
        """The record is written in the callback, long after this request has
        returned, so anything the catalog said has to be in the session."""
        _, session = _register()
        assert session["catalog_slug"] == SLUG
        assert session["tags"] == CATALOG_TAGS
        assert session["description"] == CATALOG_DESC
        assert session["name"] == CATALOG_NAME

    def test_a_catalog_miss_still_records_what_the_client_asked_for(self):
        """`resolve_catalog_entry` never raises — a stale cache, an expired
        platform token or an outage all read as "not curated".  Provenance is
        the client's claim about where it installed from, so losing the lookup
        must not lose the slug: without it the connection silently pays for LLM
        enrichment and leans on name matching to find its definition file."""
        _, session = _register(catalog_hit=False)
        assert session["catalog_slug"] == SLUG
        # The metadata genuinely is unknown here, unlike the provenance.
        assert session["description"] is None
        assert session["tags"] is None


class TestTheCallbackWritesTheSlug:
    def test_the_record_keeps_the_curated_copy_over_the_handshake(self):
        _, session = _register()
        service = MagicMock()
        kwargs = _run_callback(session, service).kwargs

        assert kwargs["catalog_slug"] == SLUG
        assert kwargs["tags"] == CATALOG_TAGS
        assert kwargs["description"] == CATALOG_DESC
        # The vendor calls itself "monday.com MCP Server"; the catalog's display
        # name is what the user picked in the panel.
        assert kwargs["name"] == CATALOG_NAME


class TestTheClientDrivenCompletion:
    """The Electron client has its own `completeOAuth`, which posts the grant
    back instead of letting the browser redirect land on the GET callback.  It
    reached `add_remote_mcp_server` with neither tags nor catalog_slug, and with
    the vendor's handshake name — so the same install, completed the other way,
    produced a connection with no provenance.
    """

    def test_posting_the_grant_back_keeps_provenance(self):
        _, session = _register()
        service = _service_stub()

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.connect_and_discover_tools = MagicMock(return_value=_connect_result())

        router_mod._pending_oauth_sessions.clear()
        router_mod._pending_oauth_sessions[session["state"]] = session
        with patch("smartloop.mcp.MCPOAuthService", _FakeOAuthCallback), \
             patch("smartloop.mcp.MCPClient", return_value=client):
            asyncio.run(
                router_mod.complete_mcp_oauth(
                    project_id="p1",
                    request=MCPOAuthCallbackRequest(
                        server_url=CATALOG_URL,
                        code="c",
                        state=session["state"],
                        code_verifier=session["code_verifier"],
                        client_id="cid",
                        client_secret="csec",
                        token_endpoint=session["token_endpoint"],
                        authorization_endpoint=session["authorization_endpoint"],
                    ),
                    model_service=service,
                )
            )

        kwargs = service.add_remote_mcp_server.call_args.kwargs
        assert kwargs["catalog_slug"] == SLUG
        assert kwargs["tags"] == CATALOG_TAGS
        assert kwargs["name"] == CATALOG_NAME
        assert kwargs["description"] == CATALOG_DESC

    def test_a_grant_with_no_session_still_installs(self):
        """A session expires after an hour, and a completion arriving without
        one is still a valid grant: it installs on the handshake's own terms
        rather than failing."""
        service = _service_stub()

        client = MagicMock()
        client.__enter__ = MagicMock(return_value=client)
        client.__exit__ = MagicMock(return_value=False)
        client.connect_and_discover_tools = MagicMock(return_value=_connect_result())

        router_mod._pending_oauth_sessions.clear()
        with patch("smartloop.mcp.MCPOAuthService", _FakeOAuthCallback), \
             patch("smartloop.mcp.MCPClient", return_value=client):
            asyncio.run(
                router_mod.complete_mcp_oauth(
                    project_id="p1",
                    request=MCPOAuthCallbackRequest(
                        server_url=CATALOG_URL, code="c", state="gone",
                        code_verifier="cv", client_id="cid",
                        token_endpoint="https://auth.monday.com/oauth_ms/oauth/token",
                        authorization_endpoint="https://auth.monday.com/oauth2/authorize",
                    ),
                    model_service=service,
                )
            )

        kwargs = service.add_remote_mcp_server.call_args.kwargs
        assert kwargs["catalog_slug"] is None
        assert kwargs["name"] == HANDSHAKE.name


class TestProvenanceSurvivesThePersistence:
    """The last hop.  A connection installed at 15:40 read back its slug from
    the API and had lost it on disk by the next process start, so the round trip
    through config.json is part of the chain too.
    """

    def test_the_slug_and_tags_come_back_after_a_save_and_reload(self, tmp_path):
        from smartloop.workspace import workspace as ws_mod
        from smartloop.workspace.workspace import (
            MCPServer, MCPServerType, Project, RemoteMCPServerConfig, Workspace,
        )

        server = MCPServer(
            id="s1",
            name=CATALOG_NAME,
            description=CATALOG_DESC,
            tags=list(CATALOG_TAGS),
            catalog_slug=SLUG,
            server_type=MCPServerType.REMOTE,
            remote_config=RemoteMCPServerConfig(server_url=CATALOG_URL),
        )
        home = str(tmp_path)
        workspace = Workspace(
            id="w1",
            workspace_dir=os.path.join(home, "w1"),
            models_dir=os.path.join(home, "w1", "models"),
            projects=[Project(
                id="p1", name="proj", current=True,
                project_dir=os.path.join(home, "w1", "p1"),
                mcp_servers=[server],
            )],
        )

        # save() writes to the class-level config path, and load() re-resolves
        # the workspace directory against the settings' home — both are pointed
        # at the temp home rather than the real one.
        config_file = os.path.join(home, "config.json")
        with patch.object(Workspace, "config_file", config_file), \
             patch.object(ws_mod, "_settings", SimpleNamespace(home_dir=home)):
            workspace.save()
            reloaded = Workspace.load()

        assert reloaded is not None
        restored = reloaded.projects[0].mcp_servers[0]
        assert restored.catalog_slug == SLUG
        assert restored.tags == CATALOG_TAGS
        assert restored.description == CATALOG_DESC

        # And the file itself, since a client reading the persisted state
        # directly is what first showed the field going missing.  Per-project
        # state lives in the project's own directory now; config.json keeps a
        # thin index that says where.
        with open(config_file) as fh:
            index = json.load(fh)["workspace"]["projects"][0]
        assert "mcp_servers" not in index  # the bulk stays out of config.json
        with open(os.path.join(index["project_dir"], "project.json")) as fh:
            persisted = json.load(fh)["project"]["mcp_servers"][0]
        assert persisted["catalog_slug"] == SLUG
