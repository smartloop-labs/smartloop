"""``POST /v1/models/load`` answers the status each failure actually is.

It mapped every ``PermissionError`` to 401, and ``PermissionError`` is four
different things here: a platform 401 (sign in), a plan refusal
(``ModelAccessError``, 403 — it subclasses ``PermissionError``), and — because
``PermissionError`` is an ``OSError`` subclass — any local filesystem refusal.
So a full disk and an unentitled model both told the desktop client to show a
sign-in prompt, for a session that was already valid.
"""

from unittest.mock import MagicMock

import pytest

from smartloop.api.main import app
from smartloop.api.services import get_model_service
from smartloop.platform import ModelAccessError


@pytest.fixture
def model_service():
    mock = MagicMock()
    mock.get_current_project_name = MagicMock(return_value="default")
    app.dependency_overrides[get_model_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_model_service, None)


def _platform_401() -> PermissionError:
    exc = PermissionError("Authentication failed. Token may be expired or invalid.")
    exc.status_code = 401
    return exc


def _plan_403() -> ModelAccessError:
    exc = ModelAccessError("Model 'gemma3-12b' is not available on your current plan.")
    exc.status_code = 403
    return exc


class TestLoadModelFailureStatuses:
    @pytest.mark.parametrize("exc, expected", [
        (_platform_401(), 401),
        (_plan_403(), 403),
        # No status_code and not a plan refusal — a local disk problem, which
        # signing in cannot fix and upgrading cannot fix.
        (PermissionError("models dir is not writable"), 500),
        (ConnectionError("platform unreachable"), 502),
    ])
    def test_the_status_matches_the_failure(self, client, model_service, exc, expected):
        model_service.load_model = MagicMock(side_effect=exc)
        resp = client.post("/v1/models/load", json={})
        assert resp.status_code == expected

    def test_a_plan_refusal_is_not_a_login_prompt(self, client, model_service):
        """The regression in one line: 403 must not read as 401."""
        model_service.load_model = MagicMock(side_effect=_plan_403())
        resp = client.post("/v1/models/load", json={})
        assert resp.status_code == 403
        assert "plan" in resp.json()["detail"]

    def test_a_successful_load_still_answers_200(self, client, model_service):
        model_service.load_model = MagicMock(return_value=None)
        resp = client.post("/v1/models/load", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
