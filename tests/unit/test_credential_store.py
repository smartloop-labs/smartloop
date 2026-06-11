"""Tests for smartloop.auth.credential_store."""

import os

from smartloop.auth import credential_store


class TestDeveloperToken:
    def test_set_and_get(self):
        credential_store.set_token("sk-test-123")
        assert credential_store.get_token() == "sk-test-123"

    def test_clear(self):
        credential_store.set_token("sk-test-123")
        credential_store.clear_token()
        assert credential_store.get_token() is None

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-from-env")
        assert credential_store.get_token() == "sk-from-env"

    def test_keychain_takes_precedence_over_env(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-from-env")
        credential_store.set_token("sk-from-store")
        assert credential_store.get_token() == "sk-from-store"


class TestAccessToken:
    def test_set_and_get(self):
        credential_store.set_access_token("jwt-abc")
        assert credential_store.get_access_token() == "jwt-abc"

    def test_clear(self):
        credential_store.set_access_token("jwt-abc")
        credential_store.clear_access_token()
        assert credential_store.get_access_token() is None


class TestRefreshToken:
    def test_set_and_get(self):
        credential_store.set_refresh_token("rt-xyz")
        assert credential_store.get_refresh_token() == "rt-xyz"

    def test_clear(self):
        credential_store.set_refresh_token("rt-xyz")
        credential_store.clear_refresh_token()
        assert credential_store.get_refresh_token() is None


class TestActiveToken:
    def test_prefers_access_over_developer(self):
        credential_store.set_token("sk-dev")
        credential_store.set_access_token("jwt-abc")
        assert credential_store.get_active_token() == "jwt-abc"

    def test_falls_back_to_developer(self):
        credential_store.set_token("sk-dev")
        assert credential_store.get_active_token() == "sk-dev"

    def test_none_when_empty(self):
        assert credential_store.get_active_token() is None


class TestHasCredentials:
    def test_false_when_empty(self):
        assert credential_store.has_credentials() is False

    def test_true_with_developer_token(self):
        credential_store.set_token("sk-dev")
        assert credential_store.has_credentials() is True

    def test_true_with_access_token(self):
        credential_store.set_access_token("jwt-abc")
        assert credential_store.has_credentials() is True


class TestAuthHeaders:
    def test_bearer_when_access_token(self):
        credential_store.set_access_token("jwt-abc")
        assert credential_store.auth_headers() == {"Authorization": "Bearer jwt-abc"}

    def test_api_key_when_developer_token(self):
        credential_store.set_token("sk-dev")
        assert credential_store.auth_headers() == {"X-Api-Key": "sk-dev"}

    def test_bearer_preferred_over_api_key(self):
        credential_store.set_token("sk-dev")
        credential_store.set_access_token("jwt-abc")
        assert credential_store.auth_headers() == {"Authorization": "Bearer jwt-abc"}

    def test_empty_when_no_tokens(self):
        assert credential_store.auth_headers() == {}


class TestFileBackendPersistence:
    def test_credentials_file_created(self, tmp_path):
        credential_store.set_token("sk-test")
        assert os.path.exists(os.path.join(str(tmp_path), ".credentials"))
        assert os.path.exists(os.path.join(str(tmp_path), ".key"))

    def test_credentials_file_permissions(self, tmp_path):
        credential_store.set_token("sk-test")
        key_stat = os.stat(os.path.join(str(tmp_path), ".key"))
        assert key_stat.st_mode & 0o777 == 0o600
