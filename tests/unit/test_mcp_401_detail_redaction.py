"""The 401 detail must not carry the token that was refused.

It embedded ``Authorization sent: <first 30 characters>``.  "Bearer " is seven
of those, so twenty-three characters of a live bearer token travelled in a
string that is logged at error level *and* stored as ``server.last_error`` —
which ``GET /v1/projects/{id}/mcp`` returns to the client and the UI renders.
A short token went out whole.
"""

import httpx
import pytest

from smartloop.mcp.client import MCPClient, MCPConnectionError

_TOKEN = "sl_live_0123456789abcdef"


def _client_answering_401(body=""):
    client = MCPClient()

    def _post(url, **kwargs):
        return httpx.Response(
            401, text=body, request=httpx.Request("POST", url),
        )

    client._client = type("_C", (), {"post": staticmethod(_post)})()
    return client


def _detail(body="", token=_TOKEN):
    client = _client_answering_401(body)
    with pytest.raises(MCPConnectionError) as exc:
        client._send_request(
            "https://example/mcp", "tools/call", {}, access_token=token,
        )
    return str(exc.value)


class TestTheRefusedTokenIsNotEchoed:
    def test_no_part_of_the_token_appears(self):
        detail = _detail()
        assert _TOKEN not in detail
        # The prefix the old code would have emitted, too.
        assert _TOKEN[:23] not in detail
        assert "0123456789" not in detail

    def test_it_still_says_a_credential_was_sent(self):
        """The diagnostic value is presence and shape, not the secret."""
        detail = _detail()
        assert "Bearer" in detail
        assert f"{len(_TOKEN)} chars" in detail

    def test_it_says_when_none_was_sent(self):
        assert "Authorization sent: (none)" in _detail(token=None)

    def test_an_empty_body_does_not_widen_the_budget(self):
        """_friendly_discovery_error truncates at 180 chars; with an empty body
        the whole preview used to fit, which is how the fragment reached the API.
        """
        detail = _detail(body="")
        assert _TOKEN[:10] not in detail[:180]
