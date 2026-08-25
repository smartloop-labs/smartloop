"""
LLM worker-pool resilience:

* ``init_worker`` retries the model load with a shrinking context window and
  must NEVER raise — a raised ProcessPoolExecutor initializer marks the whole
  pool broken, silently degrading every later ``submit_llm`` caller.
* ``HybridExecutor._resolve_worker_count`` sizes the pool to live free memory,
  capping at the requested count and scaling down (never below 1) when fewer
  copies fit.
"""
from unittest import mock

import pytest

import smartloop.agents.llm_worker as worker
from smartloop.agents.hybrid_executor import LLMExecutor


@pytest.fixture(autouse=True)
def _reset_worker_global():
    """The worker model lives in a module global — reset it around each test."""
    worker._llm = None
    yield
    worker._llm = None


def _patch_settings(worker_ctx=8192, main_n_ctx_cap=4096):
    s = mock.patch("smartloop.config.app_settings.AppSettings")
    m = s.start()
    m.return_value.llm_worker_n_ctx = worker_ctx
    m.return_value.llm_main_n_ctx_cap = main_n_ctx_cap
    return s


# --------------------------------------------------------------------------
# init_worker retry ladder
# --------------------------------------------------------------------------

def test_init_worker_retries_with_smaller_ctx():
    """First two context sizes fail; the worker loads at the third."""
    attempts = []

    def fake_get(model_path, logger, project_id, probe, max_n_ctx):
        attempts.append(max_n_ctx)
        if max_n_ctx > 2048:
            raise ValueError("Failed to create llama_context")
        return f"LLM@{max_n_ctx}"

    s = _patch_settings(worker_ctx=8192)
    try:
        with mock.patch(
            "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance",
            side_effect=fake_get,
        ):
            worker.init_worker("/model.gguf", "proj")
    finally:
        s.stop()

    # Ladder halves from worker_ctx down to the 2048 floor.
    assert attempts == [8192, 4096, 2048]
    assert worker._llm == "LLM@2048"


def test_init_worker_does_not_raise_when_all_attempts_fail():
    """Total failure leaves _llm None and must not raise (keeps pool alive)."""
    s = _patch_settings(worker_ctx=8192)
    try:
        with mock.patch(
            "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance",
            side_effect=RuntimeError("OOM"),
        ):
            # No exception escapes the initializer.
            worker.init_worker("/model.gguf", "proj")
    finally:
        s.stop()

    assert worker._llm is None


def test_init_worker_loads_first_try_without_extra_attempts():
    attempts = []

    def fake_get(model_path, logger, project_id, probe, max_n_ctx):
        attempts.append(max_n_ctx)
        return "LLM"

    s = _patch_settings(worker_ctx=8192)
    try:
        with mock.patch(
            "smartloop.llms.llm_factory.LlmFactory.get_llama_cpp_instance",
            side_effect=fake_get,
        ):
            worker.init_worker("/model.gguf", "proj")
    finally:
        s.stop()

    assert attempts == [8192]  # stopped at first success
    assert worker._llm == "LLM"


# --------------------------------------------------------------------------
# run_inference
# --------------------------------------------------------------------------

def test_run_inference_raises_clean_error_when_uninitialized():
    worker._llm = None
    with pytest.raises(RuntimeError, match="no model loaded"):
        worker.run_inference("hi")


def test_run_inference_invokes_loaded_model():
    worker._llm = mock.Mock()
    worker._llm.invoke.return_value = "answer"
    assert worker.run_inference("question") == "answer"
    worker._llm.invoke.assert_called_once_with("question")


# --------------------------------------------------------------------------
# HybridExecutor._resolve_worker_count
# --------------------------------------------------------------------------

def _patch_device(free_gb, total_gb=48.0, gpu=False, mps=False):
    """Patch the device seams used by ``_resolve_worker_count``.

    VRAM is no longer probed: worker sizing keys off llama.cpp GPU offload
    (capped at one device) and system RAM.  ``gpu``/``mps`` select the branch;
    ``free_gb``/``total_gb`` drive the RAM budget.
    """
    gb = 1024 ** 3
    vm = mock.Mock()
    vm.total = int(total_gb * gb)
    vm.available = int(free_gb * gb)
    patches = [
        mock.patch("smartloop.agents.hybrid_executor.llama_gpu_offload", return_value=gpu or mps),
        mock.patch("smartloop.agents.hybrid_executor.is_mps", return_value=mps),
        mock.patch("psutil.virtual_memory", return_value=vm),
    ]
    for p in patches:
        p.start()
    return patches


def _resolve(free_gb, requested, model_gb=2.0, worker_ctx=8192,
             total_gb=48.0, main_n_ctx_cap=4096, gpu=False, mps=False):
    s = _patch_settings(worker_ctx=worker_ctx, main_n_ctx_cap=main_n_ctx_cap)
    dev = _patch_device(free_gb, total_gb=total_gb, gpu=gpu, mps=mps)
    size = mock.patch("os.path.getsize", return_value=int(model_gb * 1024 ** 3))
    size.start()
    try:
        return LLMExecutor._resolve_worker_count("/model.gguf", requested)
    finally:
        size.stop()
        for p in dev:
            p.stop()
        s.stop()


def test_resolve_gpu_caps_at_one_device():
    # A GPU offload backend (Vulkan/Metal) isn't enumerable, so worker copies
    # cap at a single device regardless of requested count or free memory.
    assert _resolve(free_gb=40.0, requested=4, gpu=True) == 1


def test_resolve_caps_at_requested_when_plenty_fits():
    # CPU: 40GB free, ~3.7GB/copy → ~10 fit, but requested is the ceiling.
    assert _resolve(free_gb=40.0, requested=2) == 2


def test_resolve_scales_down_when_memory_tight():
    # CPU: 7GB free - 2.0 reserve = 5.0 usable; per copy ~3.7 → only 1 fits.
    assert _resolve(free_gb=7.0, requested=4) == 1


def test_resolve_keeps_at_least_one_when_nothing_fits():
    # Free below the reserve → 0 fit, but the pool was enabled so keep 1.
    assert _resolve(free_gb=1.0, requested=4) == 1


def test_resolve_reserve_prevents_packing_ram():
    # CPU reserve (2.0GB) keeps a copy from packing all RAM.
    # usable = 12.0 - 2.0 = 10.0; per copy 3.7 → 2 fit; cap at requested 2.
    assert _resolve(free_gb=12.0, requested=2) == 2
    # Drop free so usable is under 2 copies: 9.0 - 2.0 = 7.0; per copy 3.7 → 1.
    assert _resolve(free_gb=9.0, requested=2) == 1


def test_resolve_reserves_main_model_when_not_yet_resident():
    # Executor built BEFORE the main model loads: total ≈ free, so the main
    # model's footprint is not yet reflected in usage and must be reserved.
    # model 0.8GB, main cap 4096 → main_fp = 0.8 + 0.5 + 0.7 = 2.0GB;
    # per copy = 0.8 + 1.0 + 0.7 = 2.5GB.
    # usable = 7.8 - 2.0 reserve - 2.0 main = 3.8 → 1 copy fits.
    assert _resolve(
        free_gb=7.8, total_gb=8.0, requested=2, model_gb=0.8
    ) == 1
    # Same box AFTER the main model is resident (free dropped by ~2GB): the
    # footprint is already in `used`, so we don't double-charge it.
    # usable = 5.8 - 2.0 - 0 = 3.8 → still 1 (not 0); reserve isn't doubled.
    assert _resolve(
        free_gb=5.8, total_gb=8.0, requested=2, model_gb=0.8
    ) == 1


# --------------------------------------------------------------------------
# Worker log queue: what actually gets handed to ProcessPoolExecutor
# --------------------------------------------------------------------------

def _idle_executor(log_queue=None):
    """An executor with no pool — enough to exercise _worker_log_queue()."""
    return LLMExecutor(max_io_workers=1, n_llm_workers=0, log_queue=log_queue)


def test_worker_log_queue_bridges_a_thread_queue():
    """A thread queue must never reach ProcessPoolExecutor(initargs=...).

    ``queue.Queue`` isn't picklable, so passing one through would fail to
    pickle and kill the pool initializer — which would take every worker with
    it.  The shared log queue *is* a thread queue, so this is the normal path,
    not an edge case.
    """
    import multiprocessing.queues
    import queue as _queue

    shared = _queue.Queue(-1)
    ex = _idle_executor(log_queue=shared)
    try:
        worker_queue = ex._worker_log_queue()
        assert not isinstance(worker_queue, _queue.Queue)
        assert isinstance(worker_queue, multiprocessing.queues.Queue)
        assert ex._log_bridge is not None
    finally:
        ex.shutdown()


def test_worker_log_queue_passes_an_mp_queue_straight_through():
    """An explicit mp queue already crosses processes — don't wrap it."""
    import multiprocessing

    mp_queue = multiprocessing.Queue(-1)
    ex = _idle_executor(log_queue=mp_queue)
    try:
        assert ex._worker_log_queue() is mp_queue
        assert ex._log_bridge is None  # no second queue, no extra semaphores
    finally:
        ex.shutdown()
        from smartloop.utils.logging_setup import _release_mp_queue
        _release_mp_queue(mp_queue)


def test_worker_log_queue_is_none_without_logging_setup(monkeypatch):
    """No shared queue → workers get None and log to their own stderr."""
    import smartloop.utils.logging_setup as ls

    monkeypatch.setattr(ls, "_LOG_QUEUE", None)
    ex = _idle_executor()
    try:
        assert ex._worker_log_queue() is None
        assert ex._log_bridge is None
    finally:
        ex.shutdown()


def test_shutdown_closes_the_log_bridge():
    """The bridge's mp queue must die with the pool it was built to feed."""
    import queue as _queue

    ex = _idle_executor(log_queue=_queue.Queue(-1))
    ex._worker_log_queue()
    bridge = ex._log_bridge
    assert bridge is not None

    ex.shutdown()

    assert ex._log_bridge is None
    assert bridge.queue._closed
    assert not bridge._thread.is_alive()
