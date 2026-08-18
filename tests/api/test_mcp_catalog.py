"""Tests for the /v1/mcp-catalog API router.

The router reads through ``PlatformService``, so the catalog stays available
when the platform API is down. These drive the shared ``platform_service``
dependency override rather than the raw client.
"""


class TestListMCPCatalog:
    def test_returns_catalog_list(self, client, platform_service):
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
        platform_service.list_mcp_catalog.return_value = mock_data
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

    def test_derives_server_type_when_platform_omits_it(self, client, platform_service):
        """The platform catalog does not send server_type; the router derives it
        from hosted_by / config presence rather than 500-ing on validation."""
        mock_data = [
            {
                "slug": "notion",
                "display_name": "Notion",
                "hosted_by": "external",
                "remote_config": {
                    "server_url": "https://mcp.notion.com/mcp",
                    "auth_type": "oauth",
                },
                "sort_order": 0,
                "catalog_version": 1,
            },
            {
                "slug": "yahoo-finance",
                "display_name": "Yahoo Finance",
                "hosted_by": "local",
                "local_config": {"command": "uvx", "args": ["yahoo-finance-mcp"]},
                "sort_order": 10,
                "catalog_version": 1,
            },
        ]
        platform_service.list_mcp_catalog.return_value = mock_data
        resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["server_type"] == "remote"
        assert data[1]["server_type"] == "local"

    def test_returns_empty_list(self, client, platform_service):
        platform_service.list_mcp_catalog.return_value = []
        resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 200
        assert resp.json() == []

    def test_502_when_platform_unavailable(self, client, platform_service):
        platform_service.list_mcp_catalog.side_effect = ConnectionError("Connection refused")
        resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 502
        assert resp.json()["detail"] == "Platform API unavailable"

    def test_401_when_unauthorized(self, client, platform_service):
        platform_service.list_mcp_catalog.side_effect = PermissionError("Authentication failed.")
        resp = client.get("/v1/mcp-catalog")

        assert resp.status_code == 401
        assert "Authentication failed" in resp.json()["detail"]
