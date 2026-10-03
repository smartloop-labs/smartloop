"""RFC 9207: the redirect has to come from the server we sent the user to.

A client that talks to more than one authorization server can be walked into a
mix-up. The attacker starts a flow against a server it controls, hands a code
issued there to this callback, and the code is redeemed at the *honest*
server's token endpoint — the one named in the pending session. The ``iss``
parameter is what tells the two apart, and Clerk already sends it.

Absence matters as much as mismatch: a server that advertises the parameter and
then omits it has either been impersonated or had the parameter stripped, so
treating "no value" as "nothing to check" would hand over the bypass for free.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from smartloop.mcp.client import MCPConnectResult
from smartloop.mcp.oauth import MCPOAuthService, OAuthMetadata

ISSUER = "https://clerk.getclera.com"


def _session(issuer=ISSUER, advertised=True, **overrides):
    base = {
        "created_at": time.time(),
        "project_id": "p1",
        "server_url": "https://mcp.getclera.com/",
        "state": "st",
        "code_verifier": "cv",
        "code_challenge": "chal",
        "scope": "profile email offline_access",
        "client_id": "cid",
        "client_secret": None,
        "token_endpoint": f"{ISSUER}/oauth/token",
        "authorization_endpoint": f"{ISSUER}/oauth/authorize",
        "issuer": issuer,
        "issuer_parameter_supported": advertised,
        "resource": "https://mcp.getclera.com",
        "callback_url": "http://localhost:38540/v1/mcp/oauth/callback",
        "name": "Clera",
        "return_url": None,
        "description": None,
        "tags": None,
        "catalog_slug": None,
    }
    base.update(overrides)
    return base


def _callback(session=None, **params):
    """Deliver a redirect and report both the page and whether the code was spent."""
    from smartloop.api.routers import mcp_servers as router_mod

    exchanged = []

    class _FakeOAuth:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def exchange_code_for_tokens(self, **kwargs):
            exchanged.append(kwargs)
            return MagicMock(
                access_token="at", token_type="Bearer",
                refresh_token="rt", expires_in=3600, scope="profile",
            )

        # The scope retry reissues an authorization through the real builder.
        build_authorization_url = MCPOAuthService.build_authorization_url

    tool = MagicMock(name="t", description="d", input_schema={})
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.connect_and_discover_tools = MagicMock(
        return_value=MCPConnectResult(success=True, tools=[tool])
    )

    model_service = MagicMock()
    model_service.add_remote_mcp_server = MagicMock(
        return_value={"id": "s1", "name": "Clera", "tools": []}
    )

    router_mod._pending_oauth_sessions.clear()
    router_mod._completed_oauth_responses.clear()
    if session is not None:
        router_mod._pending_oauth_sessions["st"] = session

    with patch("smartloop.mcp.MCPOAuthService", _FakeOAuth), patch(
        "smartloop.mcp.MCPClient", return_value=client
    ):
        resp = asyncio.run(
            router_mod.mcp_oauth_callback(model_service=model_service, **params)
        )
    return resp, exchanged


class TestTheIssuerIsChecked:
    def test_a_matching_issuer_proceeds(self):
        resp, exchanged = _callback(_session(), code="c", state="st", iss=ISSUER)
        assert resp.status_code == 200
        assert len(exchanged) == 1

    def test_a_mismatched_issuer_never_spends_the_code(self):
        """The whole point: an attacker's code must not reach the honest server."""
        resp, exchanged = _callback(
            _session(), code="c", state="st", iss="https://evil.example.com"
        )
        assert resp.status_code == 400
        assert exchanged == [], "the authorization code was presented anyway"

    def test_the_rejection_names_both_issuers(self):
        resp, _ = _callback(
            _session(), code="c", state="st", iss="https://evil.example.com"
        )
        body = resp.body.decode()
        assert "clerk.getclera.com" in body
        assert "evil.example.com" in body

    def test_a_missing_issuer_is_refused_when_the_server_promised_one(self):
        resp, exchanged = _callback(_session(advertised=True), code="c", state="st")
        assert resp.status_code == 400
        assert exchanged == []

    def test_a_missing_issuer_is_fine_when_none_was_promised(self):
        resp, exchanged = _callback(_session(advertised=False), code="c", state="st")
        assert resp.status_code == 200
        assert len(exchanged) == 1

    def test_metadata_with_no_issuer_at_all_does_not_block_the_flow(self):
        """Pre-RFC-9207 servers stay reachable."""
        resp, exchanged = _callback(
            _session(issuer=None, advertised=False), code="c", state="st"
        )
        assert resp.status_code == 200
        assert len(exchanged) == 1

    def test_a_server_that_promises_an_issuer_but_publishes_none_is_refused(self):
        resp, exchanged = _callback(
            _session(issuer=None, advertised=True), code="c", state="st", iss=ISSUER
        )
        assert resp.status_code == 400
        assert exchanged == []


class TestItRunsBeforeEverythingElse:
    def test_an_error_response_is_checked_too(self):
        """Otherwise the invalid_scope retry reissues on an attacker's say-so."""
        resp, _ = _callback(
            _session(), state="st", error="invalid_scope",
            iss="https://evil.example.com",
        )
        assert resp.status_code == 400
        # Not the 302 the retry would have produced.
        assert "did not come from the authorization server" in resp.body.decode()

    def test_a_genuine_error_response_still_retries(self):
        resp, _ = _callback(
            _session(), state="st", error="invalid_scope", iss=ISSUER
        )
        assert resp.status_code == 302

    def test_an_unknown_state_is_left_to_the_session_check(self):
        resp, exchanged = _callback(None, code="c", state="st", iss=ISSUER)
        assert resp.status_code == 400
        assert "Session expired" in resp.body.decode()
        assert exchanged == []


class TestMetadataParsing:
    def test_the_flag_is_read_from_the_document(self):
        svc = MCPOAuthService()
        svc._http_client = MagicMock()
        svc._http_client.get = MagicMock(
            return_value=_ok({
                "issuer": ISSUER,
                "authorization_endpoint": f"{ISSUER}/oauth/authorize",
                "token_endpoint": f"{ISSUER}/oauth/token",
                "authorization_response_iss_parameter_supported": True,
            })
        )
        meta = svc.discover_oauth_metadata(ISSUER)
        assert meta.authorization_response_iss_parameter_supported is True
        assert meta.issuer == ISSUER

    def test_it_defaults_to_false_when_absent(self):
        assert (
            OAuthMetadata().authorization_response_iss_parameter_supported is False
        )


def _ok(payload):
    resp = MagicMock()
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock()
    return resp
