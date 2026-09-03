"""Tests for smartloop.auth.credential_store."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

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


class TestTheEnvVarIsARealCredential:
    """``SLP_DEVELOPER_TOKEN`` is step 3 of the documented resolution order.

    Only ``get_token`` honoured it.  ``auth_headers`` and ``get_active_token``
    read the file store directly, so an env-var-only user was reported as signed
    in by ``/v1/auth/status`` — which asks ``get_token`` — while every outbound
    request carried no credentials at all, and the web-search gate (which asks
    ``has_credentials``) turned itself off.
    """

    def test_it_produces_an_api_key_header(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-env")
        assert credential_store.auth_headers() == {"X-Api-Key": "sk-env"}
        assert credential_store.has_credentials() is True

    def test_it_is_the_active_token(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-env")
        assert credential_store.get_active_token() == "sk-env"

    def test_a_stored_token_still_wins(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-env")
        credential_store.set_token("sk-stored")
        assert credential_store.auth_headers() == {"X-Api-Key": "sk-stored"}

    def test_an_access_token_still_wins(self, monkeypatch):
        monkeypatch.setenv("SLP_DEVELOPER_TOKEN", "sk-env")
        credential_store.set_access_token("jwt-abc")
        assert credential_store.auth_headers() == {"Authorization": "Bearer jwt-abc"}


class TestConcurrentWritesKeepBoth:
    """Changing one key rewrites the whole file.

    A refresh stores an access token and a rotated refresh token back to back,
    and the background loop can be doing the same thing on another thread.
    Unsynchronized, each writer saved the file as it looked before the other's
    change — so one of the two tokens vanished, and a half-written Fernet blob
    reads as no credentials at all.
    """

    def test_keys_written_at_once_all_survive(self):
        writers = [
            (credential_store.set_access_token, "jwt-abc"),
            (credential_store.set_refresh_token, "rt-xyz"),
            (credential_store.set_token, "sk-dev"),
        ]
        barrier = threading.Barrier(len(writers))

        def write(fn, value):
            barrier.wait()
            for _ in range(25):
                fn(value)

        with ThreadPoolExecutor(max_workers=len(writers)) as pool:
            for f in [pool.submit(write, fn, v) for fn, v in writers]:
                f.result()

        assert credential_store.get_access_token() == "jwt-abc"
        assert credential_store.get_refresh_token() == "rt-xyz"
        assert credential_store.get_token() == "sk-dev"

    def test_a_reader_never_sees_a_half_written_file(self):
        credential_store.set_access_token("jwt-original")
        stop = threading.Event()
        seen = []

        def read():
            while not stop.is_set():
                seen.append(credential_store.get_access_token())

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            for i in range(50):
                credential_store.set_access_token(f"jwt-{i}")
        finally:
            stop.set()
            reader.join(timeout=5)

        # Every read landed on one whole version or another — never on ``None``,
        # which is what a torn file decrypts to.
        assert seen and all(v is not None for v in seen)
