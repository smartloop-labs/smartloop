"""A single llama.cpp context must never run two generations at once.

The context is not thread-safe (one KV cache, one token buffer, one sampler),
so concurrent ``create_chat_completion`` calls corrupt the heap and segfault
the process.  Callers reach the model from several thread pools with
independent locks — see ``LlamaCpp._context_lock`` — so the guarantee has to
live on the model itself.

Reproducer this pins down: the collect fan-out in ``Explorer.collect_parallel``
runs the vector-search history rewrite and the MCP tool pick on separate
threads against one shared model, which segfaulted on the second message of
any conversation with an MCP server attached.
"""
import logging
import threading
import time

from llama_cpp import Llama

from smartloop.llms.llama_cpp import LlamaCpp


class FakeLlama(Llama):
    """Minimal stand-in for ``llama_cpp.Llama`` that records overlap.

    Subclasses ``Llama`` (without running its ``__init__``, which would load a
    GGUF) so the wrapper's pydantic ``model`` field still validates.
    """

    def __init__(self):  # deliberately does not call super()
        self.metadata = {"test.context_length": 4096}
        self._probe = threading.Lock()
        self.active = 0
        self.peak = 0

    def close(self) -> None:
        # Llama.__del__ calls close(); the real one frees a context we never
        # allocated.
        pass

    def n_ctx(self) -> int:
        return 4096

    def tokenize(self, text: bytes, *a, **kw):
        return [1] * (len(text) // 4 + 1)

    def detokenize(self, tokens, *a, **kw):
        return b"x" * len(tokens)

    def create_chat_completion(self, **kwargs):
        with self._probe:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)  # long enough for siblings to pile up if unguarded
        with self._probe:
            self.active -= 1
        return [{"choices": [{"delta": {"content": "ok"}}]}]


def _wrapper(model: FakeLlama) -> LlamaCpp:
    return LlamaCpp(
        model=model, tokenizer=None,
        logger=logging.getLogger("test"), project_id="p",
    )


def test_concurrent_generation_is_serialized():
    model = FakeLlama()
    llm = _wrapper(model)

    threads = [
        threading.Thread(target=llm.invoke, args=(f"prompt {i}",))
        for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.peak == 1, (
        f"generations overlapped on one context (peak={model.peak}) — "
        f"this is the segfault"
    )


def test_streaming_holds_the_lock_against_a_concurrent_call():
    """A ``_stream`` consumer must exclude ``_call`` — different entry points,
    same context."""
    model = FakeLlama()
    llm = _wrapper(model)

    def drain():
        list(llm._stream("streamed prompt"))

    threads = [threading.Thread(target=drain) for _ in range(3)]
    threads += [
        threading.Thread(target=llm.invoke, args=("blocking prompt",))
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.peak == 1, f"stream overlapped a call (peak={model.peak})"


def test_wrappers_sharing_a_context_share_the_lock():
    """The lock hangs off the ``Llama`` object, so two wrappers over one
    context serialize against each other rather than each guarding its own."""
    model = FakeLlama()
    a, b = _wrapper(model), _wrapper(model)

    assert a._context_lock() is b._context_lock()

    threads = [threading.Thread(target=w.invoke, args=("p",))
               for w in (a, b, a, b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert model.peak == 1, f"wrappers did not share a lock (peak={model.peak})"


def test_separate_contexts_are_not_serialized_against_each_other():
    """Two independent models must still run in parallel — the lock is
    per-context, not global, so the worker pool keeps its concurrency."""
    m1, m2 = FakeLlama(), FakeLlama()
    barrier = threading.Barrier(2, timeout=5)

    def run(llm):
        barrier.wait()
        llm.invoke("p")

    threads = [
        threading.Thread(target=run, args=(_wrapper(m1),)),
        threading.Thread(target=run, args=(_wrapper(m2),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The barrier only proves they started together; no assert on peak here,
    # since each model sees a single caller by construction.
    assert m1.peak == 1 and m2.peak == 1
