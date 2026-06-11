"""In-process MCP servers exposed as FastMCP HTTP endpoints."""

from smartloop.api.config import api_settings


def get_api_base_url() -> str:
    """Return the base URL of the running API server."""
    port = api_settings.api_port

    host = api_settings.api_host
    # 0.0.0.0 isn't reachable from the client side — use localhost instead
    if host == "0.0.0.0":
        host = "127.0.0.1"

    return f"http://{host}:{port}"
