"""Authentication bridge for Smartloop platform calls.

Currently delegates to credential_store. Will later be swapped to
call the local server's /auth endpoints when clients become stateless.
"""

from smartloop.auth import credential_store


def auth_headers() -> dict[str, str]:
    """Return HTTP auth headers for Smartloop API calls."""
    return credential_store.auth_headers()


def has_credentials() -> bool:
    """Return True if valid credentials are available."""
    return credential_store.has_credentials()
