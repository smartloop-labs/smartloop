"""Tests for the sl-mini rename and its backward-compatibility layer.

The orchestrator model's short name changed from ``slp-orchestra-mini`` to
``sl-mini``.  Two things must not move with it: the ``model_id`` (it names the
on-disk model directory, so renaming it would orphan downloaded weights) and
the name used against the platform API (``sl-mini`` isn't published upstream
yet, so artifacts are still fetched under the old name).
"""

import pytest

from smartloop.config import SUPPORTED_MODELS
from smartloop.constants import (
    BOOTSTRAP_MODEL,
    canonical_model_name,
    display_model_name,
    platform_model_name,
)

LEGACY = "slp-orchestra-mini"


class TestCanonicalModelName:
    def test_legacy_name_maps_to_new_name(self):
        assert canonical_model_name(LEGACY) == "sl-mini"

    def test_new_name_is_stable(self):
        assert canonical_model_name("sl-mini") == "sl-mini"

    def test_bootstrap_constant_is_the_new_name(self):
        assert BOOTSTRAP_MODEL == "sl-mini"

    def test_other_models_pass_through(self):
        assert canonical_model_name("gemma3-4b") == "gemma3-4b"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_empty_passes_through(self, empty):
        assert canonical_model_name(empty) == empty


class TestPlatformModelName:
    def test_new_name_still_downloads_under_the_old_one(self):
        # sl-mini does not exist upstream yet.
        assert platform_model_name("sl-mini") == LEGACY

    def test_legacy_name_also_maps_to_the_published_name(self):
        assert platform_model_name(LEGACY) == LEGACY

    def test_unmapped_models_use_their_own_name(self):
        assert platform_model_name("gemma3-4b") == "gemma3-4b"


class TestSupportedModels:
    def test_new_name_resolves(self):
        assert SUPPORTED_MODELS.get("sl-mini") is not None

    def test_legacy_name_resolves_to_the_same_class(self):
        assert SUPPORTED_MODELS.get(LEGACY) is SUPPORTED_MODELS.get("sl-mini")

    def test_legacy_name_is_a_member(self):
        assert LEGACY in SUPPORTED_MODELS
        assert SUPPORTED_MODELS[LEGACY] is SUPPORTED_MODELS["sl-mini"]

    def test_alias_is_not_listed_as_a_separate_model(self):
        # Iteration must not show the same model twice.
        assert LEGACY not in list(SUPPORTED_MODELS.keys())
        assert "sl-mini" in list(SUPPORTED_MODELS.keys())

    def test_model_id_is_unchanged_by_the_rename(self):
        # The model_id names the on-disk directory and the upstream bucket path.
        assert SUPPORTED_MODELS["sl-mini"].model_id() == "smartloop-ai/slp-orchestra-mini"


class TestDisplayName:
    def test_legacy_record_still_displays_as_sl_mini(self):
        assert display_model_name(LEGACY) == "sl-mini"

    def test_unknown_model_passes_through(self):
        assert display_model_name("qwen3-8b") == "qwen3-8b"


class TestAppSettings:
    def test_defaults_use_the_new_name(self):
        from smartloop.config import AppSettings

        settings = AppSettings()
        assert settings.default_model == "sl-mini"
        assert settings.model_name == "sl-mini"

    def test_legacy_env_override_is_normalized(self, monkeypatch):
        # An existing .env may still pin the old name.
        from smartloop.config import AppSettings

        monkeypatch.setenv("SLP_API_BASE_MODEL_NAME", LEGACY)
        monkeypatch.setenv("SLP_BASE_MODEL_NAME", LEGACY)
        settings = AppSettings()
        assert settings.default_model == "sl-mini"
        assert settings.model_name == "sl-mini"


class TestWorkspaceModelRecords:
    def test_legacy_record_is_migrated_on_load(self):
        from smartloop.workspace import WorkspaceModel

        model = WorkspaceModel.model_validate({
            "model_name": LEGACY,
            "model_id": "smartloop-ai/slp-orchestra-mini",
            "downloaded": True,
        })
        assert model.model_name == "sl-mini"
        # model_id is the on-disk directory — it must survive untouched.
        assert model.model_id == "smartloop-ai/slp-orchestra-mini"

    def test_lookup_finds_a_legacy_record_by_the_new_name(self):
        from smartloop.workspace import Workspace, WorkspaceModel

        ws = Workspace.model_construct(
            id="ws", workspace_dir="/tmp/ws", models_dir="/tmp/ws/models",
            projects=[],
            models=[WorkspaceModel(model_name=LEGACY,
                                   model_id="smartloop-ai/slp-orchestra-mini")],
        )
        assert ws.get_model("sl-mini") is not None
        assert ws.get_model(LEGACY) is not None

    def test_add_model_does_not_duplicate_a_legacy_record(self):
        from smartloop.workspace import Workspace, WorkspaceModel

        ws = Workspace.model_construct(
            id="ws", workspace_dir="/tmp/ws", models_dir="/tmp/ws/models",
            projects=[],
            models=[WorkspaceModel(model_name=LEGACY,
                                   model_id="smartloop-ai/slp-orchestra-mini")],
        )
        ws.add_model("sl-mini", "smartloop-ai/slp-orchestra-mini")
        assert len(ws.models) == 1

    def test_add_model_stores_the_canonical_name(self):
        from smartloop.workspace import Workspace

        ws = Workspace.model_construct(
            id="ws", workspace_dir="/tmp/ws", models_dir="/tmp/ws/models",
            projects=[], models=[],
        )
        entry = ws.add_model(LEGACY, "smartloop-ai/slp-orchestra-mini")
        assert entry.model_name == "sl-mini"


class TestDownloadUsesThePublishedName:
    """The downloader must ask the platform for ``slp-orchestra-mini``.

    ``sl-mini`` is not published upstream yet, so a request under the new name
    would 404 — the artifacts still live under the old one.
    """

    def _downloader(self, monkeypatch, tmp_path, asked):
        from smartloop.model.downloader import artifact_downloader as ad

        monkeypatch.setattr(ad.platform_auth, "has_credentials", lambda: True)

        class FakeSource:
            def get_artifacts(self, model_name):
                asked.append(model_name)
                return {}  # no GGUFs -> download() returns early

        downloader = ad.SLPArtifactDownloader.__new__(ad.SLPArtifactDownloader)
        downloader.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
        downloader.logger = __import__("logging").getLogger("test")
        downloader._model_source = FakeSource()
        return downloader

    def test_download_requests_the_legacy_name(self, monkeypatch, tmp_path):
        asked = []
        downloader = self._downloader(monkeypatch, tmp_path, asked)
        downloader.download("smartloop-ai/slp-orchestra-mini")
        assert asked == [LEGACY]

    def test_bootstrap_matches_the_published_name(self, monkeypatch, tmp_path):
        from smartloop.model.downloader import artifact_downloader as ad

        published = [{"name": LEGACY, "artifacts": {"m.gguf": "https://u/m.gguf"}}]
        asked = []

        class FakeBootstrapSource:
            def list_models(self):
                return published

            def get_artifacts(self, model_name):
                asked.append(model_name)
                return {}

        downloader = ad.SLPArtifactDownloader.__new__(ad.SLPArtifactDownloader)
        downloader.workspace = type("WS", (), {"models_dir": str(tmp_path)})()
        downloader.logger = __import__("logging").getLogger("test")
        downloader._bootstrap_source = FakeBootstrapSource()

        # Asking for the new name must find the entry published under the old one.
        with pytest.raises(ValueError, match="No GGUF artifact"):
            downloader.bootstrap_download("sl-mini")
        assert asked == [LEGACY]
