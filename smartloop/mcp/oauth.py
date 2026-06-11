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
    pass


class MCPOAuthService:
    """Sync service for handling OAuth flows with remote MCP servers."""

    WELL_KNOWN_PATH = "/.well-known/oauth-authorization-server"
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

        urls = [f"{base_url}{self.WELL_KNOWN_PATH}"]
        if path_suffix:
            urls.append(f"{base_url}{self.WELL_KNOWN_PATH}/{path_suffix}")
        return urls

    def _get_prm_urls_to_try(self, server_url: str) -> list[str]:
        base_url = self._get_base_url(server_url)
        path_suffix = self._get_path_suffix(server_url)

        urls = [f"{base_url}{self.WELL_KNOWN_PRM_PATH}"]
        if path_suffix:
            urls.append(f"{base_url}{self.WELL_KNOWN_PRM_PATH}/{path_suffix}")
        return urls

    def discover_protected_resource_metadata(self, server_url: str) -> ProtectedResourceMetadata:
        urls_to_try = self._get_prm_urls_to_try(server_url)
        last_error = None

        for prm_url in urls_to_try:
            try:
                response = self._http_client.get(prm_url)
                response.raise_for_status()
                data = response.json()

                return ProtectedResourceMetadata(
                    resource=data.get("resource"),
                    authorization_servers=data.get("authorization_servers", []),
                    scopes_supported=data.get("scopes_supported", []),
                )
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404 and prm_url != urls_to_try[-1]:
                    last_error = e
                    continue
                raise ResourceMetadataDiscoveryError(
                    f"Failed to fetch PRM from {prm_url}: HTTP {e.response.status_code}"
                ) from e
            except httpx.RequestError as e:
                if prm_url != urls_to_try[-1]:
                    last_error = e
                    continue
                raise ResourceMetadataDiscoveryError(
                    f"Failed to connect to {prm_url}: {str(e)}"
                ) from e
            except (KeyError, ValueError) as e:
                raise ResourceMetadataDiscoveryError(
                    f"Invalid PRM format from {prm_url}: {str(e)}"
                ) from e

        raise ResourceMetadataDiscoveryError(
            f"Failed to discover PRM from any endpoint. Last error: {last_error}"
        )

    def discover_oauth_metadata(self, server_url: str) -> OAuthMetadata:
        urls_to_try = self._get_metadata_urls_to_try(server_url)
        last_error = None

        for metadata_url in urls_to_try:
            try:
                response = self._http_client.get(metadata_url)
                response.raise_for_status()
                data = response.json()

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
                )
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404 and metadata_url != urls_to_try[-1]:
                    last_error = e
                    continue
                raise MetadataDiscoveryError(
                    f"Failed to fetch OAuth metadata from {metadata_url}: HTTP {status_code}"
                ) from e
            except httpx.RequestError as e:
                if metadata_url != urls_to_try[-1]:
                    last_error = e
                    continue
                raise MetadataDiscoveryError(
                    f"Failed to connect to {metadata_url}: {str(e)}"
                ) from e
            except (KeyError, ValueError) as e:
                raise MetadataDiscoveryError(
                    f"Invalid OAuth metadata from {metadata_url}: {str(e)}"
                ) from e

        raise MetadataDiscoveryError(
            f"Failed to discover OAuth metadata from any endpoint. Last error: {last_error}"
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
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = f": {error_data.get('error_description', error_data.get('error', ''))}"
            except Exception:
                pass
            raise TokenExchangeError(
                f"Token exchange failed: HTTP {e.response.status_code}{error_detail}"
            ) from e
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
        client_secret: Optional[str] = None
    ) -> TokenResponse:
        token_data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }
        if client_secret:
            token_data["client_secret"] = client_secret

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
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = f": {error_data.get('error_description', error_data.get('error', ''))}"
            except Exception:
                pass
            raise TokenExchangeError(
                f"Token refresh failed: HTTP {e.response.status_code}{error_detail}"
            ) from e
        except httpx.RequestError as e:
            raise TokenExchangeError(
                f"Failed to connect to {token_endpoint}: {str(e)}"
            ) from e
        except KeyError as e:
            raise TokenExchangeError(
                f"Invalid token response - missing {str(e)}"
            ) from e
