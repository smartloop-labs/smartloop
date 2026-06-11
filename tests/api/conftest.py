"""API test fixtures."""

import pytest
from fastapi.testclient import TestClient

from smartloop.api.main import app


@pytest.fixture
def client():
    """FastAPI test client — in-process, no real server socket."""
    return TestClient(app)
