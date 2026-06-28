"""API test fixtures."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from smartloop.api.main import app
from smartloop.platform import get_platform_service


@pytest.fixture
def platform_service():
    """A MagicMock PlatformService bound via FastAPI dependency override.

    Tests can configure return values or side_effects on
    ``platform_service.get_current_user`` / ``update_current_user`` /
    ``list_models`` to drive router behavior. No real DB or HTTP I/O.
    """
    mock = MagicMock()
    app.dependency_overrides[get_platform_service] = lambda: mock
    yield mock
    app.dependency_overrides.pop(get_platform_service, None)


@pytest.fixture
def client(platform_service):
    """FastAPI test client — in-process, no real server socket."""
    return TestClient(app)
