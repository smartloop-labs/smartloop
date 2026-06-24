"""Tests for smartloop.model.downloader.artifact_source."""

import pytest

from smartloop.model.downloader.artifact_source import (
    SLPModelSource,
    SLPBootstrapSource,
)


class FakePlatformClient:
    """Stub for SLPPlatformClient — no network calls."""

    def __init__(self, artifacts=None, bootstrap_models=None):
        self._artifacts = artifacts or {}
        self._bootstrap_models = bootstrap_models or []

    def get_model_artifacts(self, model_name):
        if model_name not in self._artifacts:
            raise PermissionError(f"Model '{model_name}' not found on server.")
        return self._artifacts[model_name]

    def list_bootstrap_models(self):
        if not self._bootstrap_models:
            raise ValueError("Bootstrap API returned no models")
        return self._bootstrap_models


class TestSLPModelSource:
    def test_get_artifacts(self):
        client = FakePlatformClient(artifacts={
            "slp-orchestra-mini": {"model.gguf": "https://storage/model.gguf"},
        })
        source = SLPModelSource(client)
        result = source.get_artifacts("slp-orchestra-mini")
        assert result == {"model.gguf": "https://storage/model.gguf"}

    def test_requires_auth(self):
        source = SLPModelSource(FakePlatformClient())
        assert source.requires_auth() is True

    def test_unknown_model_raises(self):
        source = SLPModelSource(FakePlatformClient(artifacts={}))
        with pytest.raises(PermissionError):
            source.get_artifacts("nonexistent")


class TestSLPBootstrapSource:
    def _make_source(self, models=None):
        return SLPBootstrapSource(FakePlatformClient(bootstrap_models=models or []))

    def test_list_models(self):
        models = [{"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}}]
        source = self._make_source(models)
        assert source.list_models() == models

    def test_list_models_caches(self):
        models = [{"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}}]
        source = self._make_source(models)
        first = source.list_models()
        second = source.list_models()
        assert first is second

    def test_requires_auth_false(self):
        source = self._make_source()
        assert source.requires_auth() is False

    def test_get_artifacts(self):
        models = [{"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "https://url"}}]
        source = self._make_source(models)
        assert source.get_artifacts("slp-orchestra-mini") == {"m.gguf": "https://url"}

    def test_get_artifacts_unknown_model(self):
        models = [{"name": "slp-orchestra-mini", "artifacts": {"m.gguf": "url"}}]
        source = self._make_source(models)
        with pytest.raises(ValueError, match="not found in bootstrap"):
            source.get_artifacts("nonexistent")

    def test_get_artifacts_empty_artifacts(self):
        models = [{"name": "slp-orchestra-mini", "artifacts": {}}]
        source = self._make_source(models)
        with pytest.raises(ValueError, match="No artifacts"):
            source.get_artifacts("slp-orchestra-mini")

    def test_empty_bootstrap_raises(self):
        source = self._make_source([])
        with pytest.raises(ValueError):
            source.list_models()
