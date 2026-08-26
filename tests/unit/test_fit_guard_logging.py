"""The fit-guard warnings have to be readable by whoever hits them.

``llm_large_n_ctx_cap`` uses 0 as the sentinel for *no cap* — the factory
auto-sizes the context from live memory — so printing it raw read as a context
window of zero tokens.  And the remedy the warning offered ("lower the cap")
cannot help when the cap is already unset: uncapped, the KV cache is estimated
at zero, so the figure quoted is bare weights plus overhead.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartloop.api.services.model_service import ModelService, _n_ctx_desc


class TestNCtxDescription:
    def test_zero_is_described_not_printed(self):
        assert _n_ctx_desc(0) == "an auto-sized context"

    def test_a_real_cap_is_printed(self):
        assert _n_ctx_desc(8192) == "n_ctx=8192"


class TestTheRemedyMatchesTheSetting:
    def _warn(self, cap, caplog):
        svc = object.__new__(ModelService)
        svc._device_free_total_gb = MagicMock(return_value=(4.4, 24.0))
        with patch("os.path.getsize", return_value=int(3.2 * 1024**3)):
            with caplog.at_level("WARNING"):
                fits = svc._large_model_fits("/models/e2b.gguf", cap, "gemma4-e2b")
        return fits, caplog.text

    def test_uncapped_does_not_offer_the_cap(self, caplog):
        """Lowering a cap that is already 0 cannot change the estimate."""
        fits, text = self._warn(0, caplog)

        assert fits is False
        assert "Lower SLP_LLM_LARGE_N_CTX_CAP" not in text
        assert "is unset" in text
        assert "Free memory" in text

    def test_capped_still_offers_the_cap(self, caplog):
        fits, text = self._warn(8192, caplog)

        assert fits is False
        assert "Lower SLP_LLM_LARGE_N_CTX_CAP" in text

    def test_the_numbers_are_still_reported(self, caplog):
        _fits, text = self._warn(0, caplog)

        assert "4.4GB free of 24.0GB" in text
        assert "gemma4-e2b" in text

    def test_an_uncapped_estimate_excludes_the_kv_cache(self, caplog):
        """Weights + overhead only — the window auto-sizes to whatever fits."""
        _fits, text = self._warn(0, caplog)

        assert "~3.9GB" in text

    def test_a_capped_estimate_includes_the_kv_cache(self, caplog):
        _fits, text = self._warn(8192, caplog)

        assert "~4.9GB" in text


class TestCapabilitySkipMessage:
    def test_an_uncapped_skip_does_not_say_n_ctx_zero(self, caplog):
        svc = object.__new__(ModelService)
        svc._model = MagicMock(workspace=MagicMock())
        svc._model.workspace.model_dir_for.return_value = "/models"
        svc._capability_skip = None
        svc._find_model_ggufs = MagicMock(return_value=("/models/e2b.gguf", None))
        svc._large_model_fits = MagicMock(return_value=False)
        svc._evict_capability_models = MagicMock(return_value=False)
        svc.get_project_id = MagicMock(return_value="p1")
        ws = MagicMock()
        ws.model_name = "gemma4-e2b"
        ws.model_id = "google/gemma4-e2b"

        with patch("smartloop.config.app_settings.AppSettings") as settings:
            settings.return_value.llm_large_n_ctx_cap = 0
            with caplog.at_level("WARNING"):
                assert svc._load_workspace_llm(ws, "capability 'general'") is None

        assert "n_ctx=0" not in caplog.text
        assert "an auto-sized context" in caplog.text
