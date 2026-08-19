"""
Regression tests for the llama.cpp embedding backend.

Dense embeddings run on llama.cpp (GGUF), not PyTorch/sentence-transformers, so
the embedding hot path never touches torch's MPS shader JIT — the exact code that
segfaults inside ``libtorch_cpu.dylib`` in the packaged (PyInstaller) build.

These tests exercise the device→offload mapping and the ``_get_model`` wiring
without loading a real GGUF (the ``Llama`` constructor is stubbed and the GGUF
is pre-placed in the cache dir).
"""
import pytest

import smartloop.db.embeddings as embeddings
import smartloop.db.llama_embeddings as llama_embeddings


class TestOffloadMapping:
    def test_cpu_uses_no_gpu_layers(self):
        assert llama_embeddings._n_gpu_layers_for("cpu") == 0

    def test_cpu_is_case_insensitive(self):
        assert llama_embeddings._n_gpu_layers_for("  CPU ") == 0

    def test_non_cpu_offloads_fully(self):
        # "auto"/"cuda"/anything non-cpu → full Metal/Vulkan offload (-1).
        assert llama_embeddings._n_gpu_layers_for("auto") == -1
        assert llama_embeddings._n_gpu_layers_for("cuda") == -1


class TestLoadEmbedder:
    def test_builds_llama_with_embedding_and_mean_pooling(self, monkeypatch, tmp_path):
        captured = {}

        class _FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        # Inject a fake llama_cpp module so no native model loads.
        import types
        fake_mod = types.ModuleType("llama_cpp")
        fake_mod.Llama = _FakeLlama
        monkeypatch.setitem(__import__("sys").modules, "llama_cpp", fake_mod)
        # Already-cached GGUF: resolution short-circuits before any network.
        (tmp_path / "bge-m3-Q8_0.gguf").write_bytes(b"gguf")

        emb = llama_embeddings.load_embedder(
            str(tmp_path), "gpustack/bge-m3-GGUF", "bge-m3-Q8_0.gguf",
            device="cpu", base_url="https://dl.smartloop.ai/embeddings",
        )

        assert emb is not None
        assert captured["embedding"] is True
        assert captured["pooling_type"] == llama_embeddings._POOLING_TYPE_UNSPECIFIED
        assert captured["n_gpu_layers"] == 0  # cpu
        assert captured["model_path"] == str(tmp_path / "bge-m3-Q8_0.gguf")
        assert emb.identity == "llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf"

    def test_unavailable_gguf_returns_none(self, monkeypatch):
        import types
        fake_mod = types.ModuleType("llama_cpp")
        fake_mod.Llama = object
        monkeypatch.setitem(__import__("sys").modules, "llama_cpp", fake_mod)

        # No cache dir to download into, and the Hub is deliberately not
        # consulted — None so callers fall back to TF-IDF rather than crashing.
        assert llama_embeddings.load_embedder(
            None, "r", "f.gguf", device="cpu",
        ) is None


class TestBundledPath:
    def test_no_meipass_returns_none(self, monkeypatch):
        monkeypatch.delattr("sys._MEIPASS", raising=False)
        assert llama_embeddings._bundled_gguf_path("bge-m3-Q4_K_M.gguf") is None

    def test_bundled_file_is_preferred_over_download(self, tmp_path, monkeypatch):
        # Simulate a frozen build whose bundle contains the GGUF.
        bundle = tmp_path / "embeddings"
        bundle.mkdir()
        gguf = bundle / "bge-m3-Q4_K_M.gguf"
        gguf.write_bytes(b"gguf")
        monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

        captured = {}

        class _FakeLlama:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import types
        fake_mod = types.ModuleType("llama_cpp")
        fake_mod.Llama = _FakeLlama
        monkeypatch.setitem(__import__("sys").modules, "llama_cpp", fake_mod)

        def _boom(**kwargs):
            raise AssertionError("must not download when bundled")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _boom, raising=False)

        emb = llama_embeddings.load_embedder(
            None, "gpustack/bge-m3-GGUF", "bge-m3-Q4_K_M.gguf", device="cpu",
        )
        assert emb is not None
        assert captured["model_path"] == str(gguf)


class TestEncodeNormalizes:
    def test_encode_l2_normalizes(self):
        import numpy as np

        class _M:
            def embed(self, batch):
                return [[3.0, 4.0]]  # norm 5

        emb = llama_embeddings.LlamaEmbedder(_M(), "id")
        out = emb.encode("hello")
        assert pytest.approx(float(np.linalg.norm(out)), rel=1e-5) == 1.0


class TestGetModelWiring:
    def test_get_model_uses_llama_backend(self, monkeypatch):
        calls = {}

        def _fake_loader(cache_dir, repo, filename, device, base_url):
            calls["repo"] = repo
            calls["filename"] = filename
            calls["device"] = device
            calls["base_url"] = base_url
            return object()

        monkeypatch.setattr(llama_embeddings, "load_embedder", _fake_loader)
        # embeddings._get_model imports load_embedder from the module at call time.
        monkeypatch.setattr(embeddings, "_model", None)
        monkeypatch.setattr(embeddings, "_model_load_attempted", False)

        model = embeddings._get_model(base_dir=None)

        assert model is not None
        assert calls["repo"] == "gpustack/bge-m3-GGUF"
        assert calls["filename"] == "bge-m3-Q4_K_M.gguf"
        # The GGUF comes from our bucket, not the token-gated Hub.
        assert calls["base_url"] == "https://dl.smartloop.ai/embeddings"


class TestActiveIdentity:
    def test_identity_encodes_engine_and_model(self, monkeypatch):
        monkeypatch.delenv("SLP_EMBEDDING_GGUF_REPO", raising=False)
        monkeypatch.delenv("SLP_EMBEDDING_GGUF_FILE", raising=False)
        ident = embeddings.active_embedding_identity()
        assert ident == "llama-cpp:gpustack/bge-m3-GGUF/bge-m3-Q4_K_M.gguf"
