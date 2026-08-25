"""Embedding memory behaviour: bounded encode batches, and unload after a crawl.

Adding an MCP server used to push memory over: the crawl embedded every chunk
in a single llama.cpp call, and the model then stayed resident with nothing to
do. These cover both halves.
"""
import numpy as np
import pytest


class _FakeModel:
    """Records the size of each embed() call."""

    def __init__(self):
        self.calls = []
        self.closed = False

    def embed(self, batch):
        self.calls.append(len(batch))
        return [[float(len(t)), 1.0] for t in batch]

    def close(self):
        self.closed = True


class TestEncodeBatching:
    def test_large_input_is_sliced_into_bounded_calls(self):
        from smartloop.db.llama_embeddings import LlamaEmbedder, _MAX_ENCODE_BATCH

        model = _FakeModel()
        embedder = LlamaEmbedder(model, "test")

        n = _MAX_ENCODE_BATCH * 2 + 5
        out = embedder.encode([f"chunk {i}" for i in range(n)])

        assert max(model.calls) <= _MAX_ENCODE_BATCH
        assert sum(model.calls) == n
        assert out.shape == (n, 2)

    def test_slicing_does_not_change_the_result(self):
        """Vectors must come back in input order, one per text, normalized."""
        from smartloop.db.llama_embeddings import LlamaEmbedder, _MAX_ENCODE_BATCH

        texts = [f"{'x' * i}" for i in range(1, _MAX_ENCODE_BATCH * 2)]
        embedder = LlamaEmbedder(_FakeModel(), "test")

        out = embedder.encode(texts)

        assert out.shape[0] == len(texts)
        assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
        # The fake encodes length in the first component, so order is checkable
        # after normalization by comparing the ratio it implies.
        assert out[0][0] < out[-1][0]

    def test_single_string_still_returns_one_vector(self):
        from smartloop.db.llama_embeddings import LlamaEmbedder

        out = LlamaEmbedder(_FakeModel(), "test").encode("hello")
        assert out.ndim == 1

    def test_empty_input_does_not_call_the_model(self):
        from smartloop.db.llama_embeddings import LlamaEmbedder

        model = _FakeModel()
        out = LlamaEmbedder(model, "test").encode([])

        assert model.calls == []
        assert out.shape[0] == 0


class TestUnload:
    @pytest.fixture(autouse=True)
    def _restore(self):
        """Never leave a fake in the module singleton."""
        from smartloop.db import embeddings

        before = (embeddings._model, embeddings._model_load_attempted)
        yield
        embeddings._model, embeddings._model_load_attempted = before

    def test_unload_closes_and_clears_the_singleton(self):
        from smartloop.db import embeddings

        model = _FakeModel()
        embeddings._model = model
        embeddings._model_load_attempted = True

        embeddings.unload()

        assert model.closed is True
        assert embeddings._model is None

    def test_unload_allows_a_later_reload(self):
        """The 'already tried' flag must be cleared, or unloading is permanent
        and everything afterwards silently falls back to TF-IDF."""
        from smartloop.db import embeddings

        embeddings._model = _FakeModel()
        embeddings._model_load_attempted = True

        embeddings.unload()

        assert embeddings._model_load_attempted is False

    def test_unload_is_safe_when_nothing_is_loaded(self):
        from smartloop.db import embeddings

        embeddings._model = None
        embeddings._model_load_attempted = False

        embeddings.unload()  # must not raise

    def test_unload_survives_a_failing_close(self):
        from smartloop.db import embeddings

        class Broken:
            def close(self):
                raise RuntimeError("metal teardown blew up")

        embeddings._model = Broken()
        embeddings.unload()

        assert embeddings._model is None


class TestCrawlUnloadsEmbeddings:
    def test_crawl_server_unloads_even_when_the_crawl_raises(self, monkeypatch):
        """A crawl that failed has still loaded the model."""
        from smartloop.db import mcp_document_indexer as mod

        calls = []
        monkeypatch.setattr(
            "smartloop.db.embeddings.unload", lambda: calls.append("unload")
        )

        indexer = object.__new__(mod.MCPDocumentIndexer)
        import logging
        import threading

        indexer._lock = threading.RLock()
        indexer._crawling = set()
        indexer._logger = logging.getLogger("test")

        def boom(*a, **k):
            raise RuntimeError("crawl failed")

        monkeypatch.setattr(indexer, "_crawl_server", boom, raising=False)

        with pytest.raises(RuntimeError):
            indexer.crawl_server(object())

        assert calls == ["unload"]


class TestInputClamp:
    """No sequence may exceed n_ubatch — llama.cpp ggml_abort()s rather than
    degrading, which takes the process with it."""

    def test_oversized_input_is_truncated_before_reaching_the_model(self):
        from smartloop.db.llama_embeddings import (
            LlamaEmbedder,
            _MAX_INPUT_CHARS,
        )

        model = _FakeModel()
        seen = []
        model.embed = lambda b: (seen.extend(len(t) for t in b),
                                 [[1.0, 0.0] for _ in b])[1]

        LlamaEmbedder(model, "test").encode(["x" * (_MAX_INPUT_CHARS * 3), "short"])

        assert max(seen) <= _MAX_INPUT_CHARS

    def test_context_is_large_enough_for_a_full_chunk(self):
        """The clamp and the indexers' CHUNK_SIZE must stay consistent: a
        chunk that fits the indexer must fit the embedder untruncated."""
        from smartloop.db.llama_embeddings import _EMBED_N_CTX, _MAX_INPUT_CHARS
        from smartloop.db.workspace_indexer import CHUNK_SIZE

        assert _MAX_INPUT_CHARS >= CHUNK_SIZE
        # Worst case is roughly one token per character; the context must still
        # cover it or llama.cpp aborts.
        assert _EMBED_N_CTX >= _MAX_INPUT_CHARS


class TestUnloadAfterResponse:
    def test_auxiliary_unload_also_drops_the_embedder(self, monkeypatch):
        """Retrieval finishes long before the response does, so the embedder is
        idle from then until the next question — same as the auxiliary LLMs."""
        from smartloop.api.services.model_service import ModelService

        calls = []
        monkeypatch.setattr("smartloop.db.embeddings.unload", lambda: calls.append("unload"))

        svc = object.__new__(ModelService)
        monkeypatch.setattr(svc, "unload_vision_model", lambda: None, raising=False)
        monkeypatch.setattr(svc, "unload_large_model", lambda: None, raising=False)
        monkeypatch.setattr(svc, "_evict_capability_models", lambda: None, raising=False)

        svc.unload_auxiliary_models()

        assert calls == ["unload"]

    def test_a_failing_unload_does_not_break_the_response(self, monkeypatch):
        from smartloop.api.services.model_service import ModelService

        def _boom():
            raise RuntimeError("teardown blew up")

        monkeypatch.setattr("smartloop.db.embeddings.unload", _boom)

        svc = object.__new__(ModelService)
        monkeypatch.setattr(svc, "unload_vision_model", lambda: None, raising=False)
        monkeypatch.setattr(svc, "unload_large_model", lambda: None, raising=False)
        monkeypatch.setattr(svc, "_evict_capability_models", lambda: None, raising=False)

        svc.unload_auxiliary_models()  # must not raise
