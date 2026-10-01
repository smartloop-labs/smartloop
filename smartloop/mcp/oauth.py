"""Sync OAuth service for MCP server authentication."""
import secrets
import hashlib
import base64
from typing import Optional, Tuple
from urllib.parse import urljoin, urlencode, urlparse
from dataclasses import dataclass

import httpx


@dataclass
class ProtectedResourceMetadata:
    """Protected Resource Metadata (RFC 9728)"""
    resource: Optional[str] = None
    authorization_servers: list[str] = None
    scopes_supported: list[str] = None

    def __post_init__(self):
        if self.authorization_servers is None:
            self.authorization_servers = []
        if self.scopes_supported is None:
            self.scopes_supported = []


@dataclass
class OAuthMetadata:
    """OAuth 2.0 Authorization Server Metadata"""
    issuer: Optional[str] = None
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    registration_endpoint: Optional[str] = None
    scopes_supported: list[str] = None
    response_types_supported: list[str] = None
    grant_types_supported: list[str] = None
    code_challenge_methods_supported: list[str] = None
    resource: Optional[str] = None
    #: RFC 9207 §3.  When the server says it sends ``iss`` on the authorization
    #: response, a response arriving *without* one must be rejected — otherwise
    #: an attacker strips the parameter to get back the behaviour it defends
    #: against.
    authorization_response_iss_parameter_supported: bool = False

    def __post_init__(self):
        if self.scopes_supported is None:
            self.scopes_supported = []
        if self.response_types_supported is None:
            self.response_types_supported = []
        if self.grant_types_supported is None:
            self.grant_types_supported = []
        if self.code_challenge_methods_supported is None:
            self.code_challenge_methods_supported = []


@dataclass
class ClientRegistrationResponse:
    client_id: str
    client_secret: Optional[str] = None
    client_id_issued_at: Optional[int] = None
    client_secret_expires_at: Optional[int] = None


@dataclass
class TokenResponse:
    access_token: str
    token_type: str
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class MCPOAuthError(Exception):
    pass


class MetadataDiscoveryError(MCPOAuthError):
    pass


class ResourceMetadataDiscoveryError(MCPOAuthError):
    pass


class ClientRegistrationError(MCPOAuthError):
    pass


class TokenExchangeError(MCPOAuthError):
    """A token endpoint refused a grant, or could not be reached.

    ``error_code`` is the OAuth 2.0 error identifier from the response body
    (RFC 6749 §5.2) when the server sent one, and ``None`` for a transport
    failure.  It is what separates a grant that is dead — ``invalid_grant``,
    ``invalid_client`` — from a blip: only the first kind means re-minting the
    same refresh token can never work again, and only the first kind should
    cost a user their connection.
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code

    #: Error identifiers that mean the grant itself is finished.  A refresh
    #: token the server has retired, or client credentials it no longer
    #: recognises, cannot be recovered by trying again.
    DEAD_GRANT_CODES = frozenset({"invalid_grant", "invalid_client"})

    @property
    def grant_is_dead(self) -> bool:
        return self.error_code in self.DEAD_GRANT_CODES


def _token_error(
    exc: httpx.HTTPStatusError, label: str
) -> "TokenExchangeError":
    """Build a :class:`TokenExchangeError` from an RFC 6749 §5.2 error body.

    All three fields matter and for different reasons.  ``error`` is the fixed
    enum the retry logic switches on (see ``DEAD_GRANT_CODES``).
    ``error_description`` is the only part that names *what* was wrong — which
    scope, which client — and ``error_uri`` points at the provider's own page
    about it.  A body that is not JSON, or not an OAuth error at all, leaves the
    status line as the whole story rather than failing a second time in here.
    """
    error_code = None
    detail = ""
    try:
        data = exc.response.json()
        error_code = data.get("error")
        description = data.get("error_description") or error_code or ""
        uri = data.get("error_uri")
        if description:
            detail = f": {description}"
        if uri:
            detail += f" (see {uri})"
    except Exception:
        pass
    return TokenExchangeError(
        f"{label}: HTTP {exc.response.status_code}{detail}",
        error_code=error_code,
        status_code=exc.response.status_code,
    )


class MCPOAuthService:
    """Sync service for handling OAuth flows with remote MCP servers."""

    WELL_KNOWN_PATH = "/.well-known/oauth-authorization-server"
    WELL_KNOWN_OIDC_PATH = "/.well-known/openid-configuration"
    WELL_KNOWN_PRM_PATH = "/.well-known/oauth-protected-resource"

    def __init__(self, http_client: Optional[httpx.Client] = None):
        self._http_client = http_client
        self._owns_client = http_client is None

    def __enter__(self):
        if self._owns_client:
            self._http_client = httpx.Client(timeout=30.0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_client and self._http_client:
            self._http_client.close()

    def _get_base_url(self, server_url: str) -> str:
        parsed = urlparse(server_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_path_suffix(self, server_url: str) -> str:
        parsed = urlparse(server_url)
        return parsed.path.lstrip("/") if parsed.path else ""

    def _get_metadata_urls_to_try(self, server_url: str) -> list[str]:
        base_url = self._get_base_url(server_url)
        path_suffix = self._get_path_suffix(server_url)

        bare = f"{base_url}{self.WELL_KNOWN_PATH}"
        if not path_suffix:
            return [bare]

        # For a path-scoped issuer, try the path-bearing well-known forms first. The bare
        # (path-stripped) URL frequently describes a *different* root authorization server
        # that lacks a registration_endpoint (e.g. Atlassian/Trello), so demote it to a
        # last resort rather than letting it shadow the correct tenant metadata.
        issuer = f"{base_url}/{path_suffix}"
        return [
            f"{base_url}{self.WELL_KNOWN_PATH}/{path_suffix}",  # RFC 8414 well-known-inserted
            f"{issuer}{self.WELL_KNOWN_PATH}",                  # OIDC-style path-append (oauth)
            f"{issuer}{self.WELL_KNOWN_OIDC_PATH}",             # OIDC-style path-append (openid)
            bare,                                               # last-resort fallback
        ]

    def _get_prm_urls_to_try(self, server_url: str) -> list[str]:
        base_url = self._get_base_url(server_url)
        path_suffix = self._get_path_suffix(server_url)

        urls = [f"{base_url}{self.WELL_KNOWN_PRM_PATH}"]
        if path_suffix:
            urls.append(f"{base_url}{self.WELL_KNOWN_PRM_PATH}/{path_suffix}")
        return urls

    def _fetch_first_metadata(self, urls: list[str], error_cls, label: str) -> dict:
        """GET each URL in order, returning the first that yields a valid JSON body.

        Falls through to the next candidate on ANY error (HTTP status such as 503,
        connection/transport errors, or invalid JSON) — not just 404 — so a transient
        failure on an earlier candidate never aborts discovery. Raises ``error_cls``
        only once every candidate has been exhausted.
        """
        last_error = None
        for url in urls:
            try:
                response = self._http_client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = f"{url}: HTTP {e.response.status_code}"
            except httpx.RequestError as e:
                last_error = f"{url}: {str(e)}"
            except ValueError as e:
                last_error = f"{url}: invalid JSON ({str(e)})"

        raise error_cls(
            f"Failed to discover {label} from any endpoint. Last error: {last_error}"
        )

    def discover_protected_resource_metadata(
        self, server_url: str, metadata_url: Optional[str] = None
    ) -> ProtectedResourceMetadata:
        # RFC 9728 §5.1: honor the metadata URL advertised in the 401 WWW-Authenticate
        # header when present, then fall back to guessing the well-known locations.
        urls_to_try = self._get_prm_urls_to_try(server_url)
        if metadata_url and metadata_url not in urls_to_try:
            urls_to_try = [metadata_url, *urls_to_try]

        data = self._fetch_first_metadata(
            urls_to_try, ResourceMetadataDiscoveryError, "PRM"
        )
        return ProtectedResourceMetadata(
            resource=data.get("resource"),
            authorization_servers=data.get("authorization_servers", []),
            scopes_supported=data.get("scopes_supported", []),
        )

    def discover_oauth_metadata(self, server_url: str) -> OAuthMetadata:
        urls_to_try = self._get_metadata_urls_to_try(server_url)
        data = self._fetch_first_metadata(
            urls_to_try, MetadataDiscoveryError, "OAuth metadata"
        )
        return OAuthMetadata(
            issuer=data.get("issuer"),
            authorization_endpoint=data.get("authorization_endpoint", ""),
            token_endpoint=data.get("token_endpoint", ""),
            registration_endpoint=data.get("registration_endpoint"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", []),
            grant_types_supported=data.get("grant_types_supported", []),
            code_challenge_methods_supported=data.get("code_challenge_methods_supported", []),
            resource=data.get("resource"),
            authorization_response_iss_parameter_supported=bool(
                data.get("authorization_response_iss_parameter_supported", False)
            ),
        )

    def register_client(
        self,
        registration_endpoint: str,
        redirect_uri: str,
        client_name: str = "Smartloop"
    ) -> ClientRegistrationResponse:
        registration_data = {
            "client_name": client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        }

        try:
            response = self._http_client.post(
                registration_endpoint,
                json=registration_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()

            return ClientRegistrationResponse(
                client_id=data["client_id"],
                client_secret=data.get("client_secret"),
                client_id_issued_at=data.get("client_id_issued_at"),
                client_secret_expires_at=data.get("client_secret_expires_at"),
            )
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = f": {error_data.get('error_description', error_data.get('error', ''))}"
            except Exception:
                pass
            raise ClientRegistrationError(
                f"Client registration failed: HTTP {e.response.status_code}{error_detail}"
            ) from e
        except httpx.RequestError as e:
            raise ClientRegistrationError(
                f"Failed to connect to {registration_endpoint}: {str(e)}"
            ) from e
        except KeyError as e:
            raise ClientRegistrationError(
                f"Invalid registration response - missing {str(e)}"
            ) from e

    @staticmethod
    def canonical_resource(server_url: str) -> str:
        """Canonical resource indicator (RFC 8707 §2) for an MCP server URL.

        Used when the server publishes no PRM ``resource`` of its own. Lowercases
        scheme/host and drops the fragment, which RFC 8707 forbids; the path and
        query are significant (``/mcp`` vs ``/sse`` are distinct resources) so
        they are preserved verbatim.
        """
        parsed = urlparse(server_url.strip())
        path = "" if parsed.path == "/" else parsed.path
        canonical = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
        return f"{canonical}?{parsed.query}" if parsed.query else canonical

    @staticmethod
    def generate_pkce() -> Tuple[str, str]:
        code_verifier = secrets.token_urlsafe(32)
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        return code_verifier, code_challenge

    @staticmethod
    def generate_state() -> str:
        return secrets.token_urlsafe(32)

    def build_authorization_url(
        self,
        authorization_endpoint: str,
        client_id: str,
        redirect_uri: str,
        state: str,
        code_challenge: str,
        scope: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> str:
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        if scope:
            params["scope"] = scope
        if resource:
            params["resource"] = resource
        return f"{authorization_endpoint}?{urlencode(params)}"

    def exchange_code_for_tokens(
        self,
        token_endpoint: str,
        code: str,
        redirect_uri: str,
        client_id: str,
        code_verifier: str,
        client_secret: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> TokenResponse:
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
        if client_secret:
            token_data["client_secret"] = client_secret
        if resource:
            token_data["resource"] = resource

        try:
            response = self._http_client.post(
                token_endpoint,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            data = response.json()

            return TokenResponse(
                access_token=data["access_token"],
                token_type=data.get("token_type", "Bearer"),
                expires_in=data.get("expires_in"),
                refresh_token=data.get("refresh_token"),
                scope=data.get("scope"),
            )
        except httpx.HTTPStatusError as e:
            raise _token_error(e, "Token exchange failed") from e
        except httpx.RequestError as e:
            raise TokenExchangeError(
                f"Failed to connect to {token_endpoint}: {str(e)}"
            ) from e
        except KeyError as e:
            raise TokenExchangeError(
                f"Invalid token response - missing {str(e)}"
            ) from e

    def refresh_access_token(
        self,
        token_endpoint: str,
        refresh_token: str,
        client_id: str,
        client_secret: Optional[str] = None,
        resource: Optional[str] = None,
    ) -> TokenResponse:
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
        if client_secret:
            token_data["client_secret"] = client_secret
        if resource:
            token_data["resource"] = resource

        try:
            response = self._http_client.post(
                token_endpoint,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()
            data = response.json()

            return TokenResponse(
                access_token=data["access_token"],
                token_type=data.get("token_type", "Bearer"),
                expires_in=data.get("expires_in"),
                refresh_token=data.get("refresh_token"),
                scope=data.get("scope"),
            )
        except httpx.HTTPStatusError as e:
            raise _token_error(e, "Token refresh failed") from e
        except httpx.RequestError as e:
            raise TokenExchangeError(
                f"Failed to connect to {token_endpoint}: {str(e)}"
            ) from e
        except KeyError as e:
            raise TokenExchangeError(
                f"Invalid token response - missing {str(e)}"
            ) from e
