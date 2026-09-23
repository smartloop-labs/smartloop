"""A server registered from a URL carrying its own token authenticates its calls.

Registering ``http://host/v1/mcp?token=abc`` splits the credential off the URL
and stores it on the remote config; there are no OAuth credentials because none
were ever negotiated.  Discovery has always read that field, so such a server
lists its tools and looks healthy — and then every call to one of those tools
went out with no Authorization header and came back 401.
"""

from types import SimpleNamespace
from urllib.parse import urlsplit

from smartloop.agents.mcp.mcp_tool_caller import MCPToolCaller


def _server(*, token=None, oauth_token=None):
    return SimpleNamespace(
        name="local-61185",
        remote_config=SimpleNamespace(
            server_url="http://127.0.0.1:61185/v1/mcp", token=token,
        ),
        oauth_credentials=(
            SimpleNamespace(access_token=oauth_token) if oauth_token else None
        ),
    )


def test_static_token_on_remote_config_is_used():
    assert MCPToolCaller.get_access_token(_server(token="abc")) == "abc"


def test_oauth_credentials_still_win():
    server = _server(token="static", oauth_token="fresh")
    assert MCPToolCaller.get_access_token(server) == "fresh"


def test_no_credential_anywhere_is_none():
    assert MCPToolCaller.get_access_token(_server()) is None


def test_info_probe_keeps_the_query_string(monkeypatch):
    """"/info" belongs on the path — appended raw it lands inside the query."""
    import asyncio
    import httpx

    from smartloop.api.routers import mcp_servers

    seen = []

    def _fake_get(url, **kwargs):
        seen.append(url)
        raise httpx.ConnectError("probe not served")

    monkeypatch.setattr(httpx, "get", _fake_get)

    result = asyncio.run(
        mcp_servers._try_register_smartloop_mcp_server(
            "http://127.0.0.1:61185/v1/mcp?token=abc",
            model_service=None,
            project_id="p",
        )
    )

    # An unreachable probe is "not a Smartloop server", not an error.
    assert result is None
    assert seen == ["http://127.0.0.1:61185/v1/mcp/info?token=abc"]
