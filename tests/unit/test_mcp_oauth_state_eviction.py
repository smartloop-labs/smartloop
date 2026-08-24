"""The router's in-memory OAuth state must not grow without bound.

``_pending_oauth_sessions`` is popped by the callback, but an abandoned flow
(the user closes the browser tab, the provider errors out before redirecting)
never produces one — and the entry holds a ``client_secret`` and
``code_verifier``.  ``_completed_oauth_responses`` was never evicted at all; it
retains a full ``HTMLResponse`` per registration purely to absorb the browser's
duplicate GET of the redirect, which arrives within seconds.
"""

import time

import pytest
from fastapi.responses import HTMLResponse

from smartloop.api.routers import mcp_servers as m


@pytest.fixture(autouse=True)
def _clean_state():
    """These are process globals — don't let one test leak into the next."""
    m._pending_oauth_sessions.clear()
    m._completed_oauth_responses.clear()
    yield
    m._pending_oauth_sessions.clear()
    m._completed_oauth_responses.clear()


class TestPendingSessionEviction:
    def test_abandoned_session_expires(self):
        now = time.time()
        m._pending_oauth_sessions["stale"] = {
            "created_at": now - m._OAUTH_SESSION_TTL_S - 1, "client_secret": "s3cret",
        }
        m._pending_oauth_sessions["fresh"] = {"created_at": now, "client_secret": "s3cret"}

        m._purge_oauth_state(now=now)

        assert "stale" not in m._pending_oauth_sessions
        assert "fresh" in m._pending_oauth_sessions

    def test_untimestamped_session_is_stamped_not_dropped(self):
        """An entry with no ``created_at`` must not read as infinitely old and
        vanish on the very next sweep — it gets stamped and expires one TTL on."""
        now = time.time()
        m._pending_oauth_sessions["legacy"] = {"client_secret": "s3cret"}

        m._purge_oauth_state(now=now)
        assert "legacy" in m._pending_oauth_sessions
        assert m._pending_oauth_sessions["legacy"]["created_at"] == now

        m._purge_oauth_state(now=now + m._OAUTH_SESSION_TTL_S + 1)
        assert "legacy" not in m._pending_oauth_sessions


class TestCompletedResponseEviction:
    def test_cached_response_expires(self):
        now = time.time()
        m._completed_oauth_responses["old"] = (now - 1, HTMLResponse("old"))
        m._completed_oauth_responses["new"] = (now + 60, HTMLResponse("new"))

        m._purge_oauth_state(now=now)

        assert "old" not in m._completed_oauth_responses
        assert "new" in m._completed_oauth_responses

    def test_cache_is_size_capped_oldest_first(self):
        now = time.time()
        for i in range(m._OAUTH_RESPONSE_MAX + 10):
            m._completed_oauth_responses[f"s{i}"] = (now + 3600, HTMLResponse(str(i)))

        m._purge_oauth_state(now=now)

        assert len(m._completed_oauth_responses) == m._OAUTH_RESPONSE_MAX
        assert "s0" not in m._completed_oauth_responses          # oldest evicted
        assert f"s{m._OAUTH_RESPONSE_MAX + 9}" in m._completed_oauth_responses
