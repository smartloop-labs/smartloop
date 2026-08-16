"""Tests for the pluggable web search providers."""

import httpx
import pytest

from smartloop.auth import credential_store
from smartloop.search import providers


class TestBraveSearchProvider:
    def _client(self, status=200, payload=None):
        class FakeClient:
            def __init__(self):
                self.get_calls = []

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None, params=None):
                self.get_calls.append((url, headers, params))
                return httpx.Response(status, json=payload or {}, request=httpx.Request("GET", url))

        return FakeClient()

    def test_returns_result_urls(self, monkeypatch):
        fake = self._client(payload={"web": {"results": [
            {"url": "https://a.com", "title": "A"},
            {"url": "https://b.com", "title": "B"},
            {"url": "", "title": "no url"},
        ]}})
        monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)
        prov = providers.BraveSearchProvider("sk-brave")
        assert prov.search("canyon bikes", 5) == ["https://a.com", "https://b.com"]
        url, headers, params = fake.get_calls[0]
        assert headers["X-Subscription-Token"] == "sk-brave"
        assert params["q"] == "canyon bikes"
        assert params["count"] == 5

    def test_clamps_count_to_brave_max(self, monkeypatch):
        fake = self._client(payload={"web": {"results": []}})
        monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)
        prov = providers.BraveSearchProvider("sk-brave")
        prov.search("q", 50)
        assert fake.get_calls[0][2]["count"] == 20

    def test_rejected_key_raises_auth_error(self, monkeypatch):
        fake = self._client(status=401)
        monkeypatch.setattr(httpx, "Client", lambda *a, **k: fake)
        prov = providers.BraveSearchProvider("sk-bad")
        with pytest.raises(providers.SearchAuthError):
            prov.search("q", 5)


class TestResolution:
    def test_none_when_not_configured(self, monkeypatch):
        monkeypatch.delenv("SLP_SEARCH_PROVIDER", raising=False)
        monkeypatch.delenv("SLP_BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("SLP_SEARCH_PROVIDER_TOKEN", raising=False)
        assert providers.get_search_provider() is None

    def test_env_brave(self, monkeypatch):
        monkeypatch.setenv("SLP_SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("SLP_BRAVE_API_KEY", "sk-env")
        prov = providers.get_search_provider()
        assert prov is not None
        assert prov.name == "brave"
        assert isinstance(prov, providers.BraveSearchProvider)

    def test_stored_brave_falls_back_to_hosted_when_cleared(self, monkeypatch):
        credential_store.set_search_provider_config("brave", "sk-store")
        prov = providers.get_search_provider()
        assert prov is not None and prov.name == "brave"
        credential_store.clear_search_provider_config()
        assert providers.get_search_provider() is None

    def test_env_overrides_stored(self, monkeypatch):
        credential_store.set_search_provider_config("brave", "sk-store")
        monkeypatch.setenv("SLP_SEARCH_PROVIDER", "brave")
        monkeypatch.setenv("SLP_BRAVE_API_KEY", "sk-env")
        prov = providers.get_search_provider()
        assert prov._api_key == "sk-env"

    def test_unknown_provider_falls_back_to_hosted(self, monkeypatch):
        monkeypatch.setenv("SLP_SEARCH_PROVIDER", "not-a-provider")
        monkeypatch.setenv("SLP_SEARCH_PROVIDER_TOKEN", "sk-any")
        assert providers.get_search_provider() is None

    def test_resolve_provider_defaults_to_hosted(self):
        provider = providers.resolve_provider()
        assert provider.name == "hosted"


class TestHostedProvider:
    def test_no_credentials_raises(self, monkeypatch):
        credential_store.clear_search_provider_config()
        monkeypatch.delenv("SLP_DEVELOPER_TOKEN", raising=False)
        credential_store.clear_access_token()
        with pytest.raises(providers.NoSearchCredentials):
            providers.HostedSearchProvider().search("q", 5)

    def test_rejects_without_token_on_401(self, monkeypatch):
        credential_store.set_token("sk-dev")

        class FakeClient:
            def __init__(self):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def get(self, url, headers=None, params=None):
                return httpx.Response(403, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "Client", lambda *a, **k: FakeClient())
        with pytest.raises(providers.SearchAuthError):
            providers.HostedSearchProvider().search("q", 5)
