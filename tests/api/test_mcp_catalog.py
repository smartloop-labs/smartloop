"""Tests for the /v1/mcp-catalog API router."""

from unittest.mock import patch

_LIST_CATALOG = "smartloop.platform.client.SLPPlatformClient.list_mcp_catalog"


class TestListMCPCatalog:
    def test_returns_catalog_list(self, client):
        mock_data = [
            {
                "slug": "linear",
                "display_name": "Linear",
                "description": "Project management",
                "server_type": "remote",
                "hosted_by": "external",
                "remote_config": {
                    "server_url": "https://mcp.linear.app/sse",
                    "auth_type": "oauth",
                },
                "icon_url": "https://linear.app/favicon.ico",
                "category": "Project Management",
                "tags": ["project-management"],
                "sort_order": 0,
                "catalog_version": 1,
            },
            {
                "slug": "yahoo-finance",
                "display_name": "Yahoo Finance",
                "description": "Stock data",
                "server_type": "local",
                "hosted_by": "local",
                "local_config": {
                    "command": "uvx",
                    "args": ["yahoo-finance-mcp"],
                    "env": {},
                },
                "category": "Finance",
                "tags": ["finance"],
                "sort_order": 10,
                "catalog_version": 1,
            },
        ]
        with patch(_LIST_CATALOG, return_value=mock_data):
            resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["slug"] == "linear"
        assert data[0]["server_type"] == "remote"
        assert data[0]["hosted_by"] == "external"
        assert data[0]["remote_config"]["server_url"] == "https://mcp.linear.app/sse"
        assert data[1]["slug"] == "yahoo-finance"
        assert data[1]["hosted_by"] == "local"
        assert data[1]["local_config"]["command"] == "uvx"

    def test_returns_empty_list(self, client):
        with patch(_LIST_CATALOG, return_value=[]):
            resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_502_when_platform_unavailable(self, client):
        with patch(_LIST_CATALOG, side_effect=ConnectionError("Connection refused")):
            resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"

    def test_401_when_unauthorized(self, client):
        with patch(_LIST_CATALOG, side_effect=PermissionError("Authentication failed.")):
            resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 401
        assert "Authentication failed" in resp.json()["detail"]
