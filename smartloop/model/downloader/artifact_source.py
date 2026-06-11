"""Artifact sources provide download URLs for model files.

An ArtifactSource knows how to resolve a model identifier into a
``{filename: url}`` map. The actual downloading is handled separately
by the downloader.
"""

from abc import ABC, abstractmethod

from smartloop.platform import SLPPlatformClient
from smartloop.platform import auth as platform_auth


class ArtifactSource(ABC):
    """Base class for artifact URL providers."""

    @abstractmethod
    def get_artifacts(self, model_name: str) -> dict[str, str]:
        """Return ``{filename: signed_url}`` for the given model."""

    def requires_auth(self) -> bool:
        """Whether this source requires authentication."""
        return False


class SLPModelSource(ArtifactSource):
    """Authenticated artifact source using the /v1/models/{name}/download endpoint."""

    def __init__(self, platform_client: SLPPlatformClient):
        self._platform_client = platform_client

    def get_artifacts(self, model_name: str) -> dict[str, str]:
        return self._platform_client.get_model_artifacts(model_name)

    def requires_auth(self) -> bool:
        return True


class SLPBootstrapSource(ArtifactSource):
    """Unauthenticated artifact source using the /v1/bootstrap/models endpoint."""

    def __init__(self, platform_client: SLPPlatformClient):
        self._platform_client = platform_client
        self._models: list[dict] | None = None

    def list_models(self) -> list[dict]:
        """Fetch and cache the bootstrap model list."""
        if self._models is None:
            self._models = self._platform_client.list_bootstrap_models()
        return self._models

    def get_artifacts(self, model_name: str) -> dict[str, str]:
        models = self.list_models()
        target = next((m for m in models if m.get("name") == model_name), None)
        if not target:
            available = ", ".join(m.get("name", "?") for m in models)
            raise ValueError(
                f"Model '{model_name}' not found in bootstrap. "
                f"Available: {available}"
            )
        artifacts = target.get("artifacts", {})
        if not artifacts:
            raise ValueError(f"No artifacts returned for bootstrap model '{model_name}'")
        return artifacts
