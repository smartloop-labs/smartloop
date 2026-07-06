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

def _patch_cuda(free_gb, total_gb=48.0, device_count=8):
    """Patch torch onto a CUDA device with the given free/total VRAM.

    ``device_count`` defaults high because ``_resolve_worker_count`` also
    caps at one worker copy per GPU (same-GPU processes serialize without
    CUDA MPS).  These tests exercise the memory-cap logic; a low
    device_count would cap ``requested`` before the memory check runs.
    """
    gb = 1024 ** 3
    patches = [
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.device_count", return_value=device_count),
        mock.patch(
            "torch.cuda.mem_get_info",
            return_value=(int(free_gb * gb), int(total_gb * gb)),
        ),
        mock.patch("torch.cuda.get_device_name", return_value="TestGPU"),
    ]
    for p in patches:
        p.start()
    return patches


def _resolve(free_gb, requested, model_gb=2.0, worker_ctx=8192,
             total_gb=48.0, main_n_ctx_cap=4096):
    s = _patch_settings(worker_ctx=worker_ctx, main_n_ctx_cap=main_n_ctx_cap)
    cuda = _patch_cuda(free_gb, total_gb=total_gb)
    size = mock.patch("os.path.getsize", return_value=int(model_gb * 1024 ** 3))
    size.start()
    try:
        return LLMExecutor._resolve_worker_count("/model.gguf", requested)
    finally:
        size.stop()
        for p in cuda:
            p.stop()
        s.stop()


def test_resolve_caps_at_requested_when_plenty_fits():
    # 40GB free, ~3.7GB/copy → ~10 fit, but requested is the ceiling.
    assert _resolve(free_gb=40.0, requested=2) == 2


def test_resolve_scales_down_when_memory_tight():
    # 8GB free - 1.5 reserve = 6.5 usable; per copy ~3.7 → only 1 fits.
    assert _resolve(free_gb=8.0, requested=4) == 1


def test_resolve_keeps_at_least_one_when_nothing_fits():
    # Free below the reserve → 0 fit, but the pool was enabled so keep 1.
    assert _resolve(free_gb=1.0, requested=4) == 1


def test_resolve_reserve_prevents_packing_all_vram():
    # Exactly enough for 2 copies (7.4GB) only AFTER ignoring reserve; with the
    # 1.5GB reserve the second copy no longer fits → scales to 1.
    # usable = 9.0 - 1.5 = 7.5; per copy 3.7 → 2 fit; cap at requested 2.
    assert _resolve(free_gb=9.0, requested=2) == 2
    # Drop free so usable is just under 2 copies: 8.8 - 1.5 = 7.3 → 1 fits.
    assert _resolve(free_gb=8.8, requested=2) == 1


def test_resolve_reserves_main_model_when_not_yet_resident():
    # Executor built BEFORE the main model loads: total ≈ free, so the main
    # model's footprint is not yet reflected in usage and must be reserved.
    # model 0.8GB, main cap 4096 → main_fp = 0.8 + 0.5 + 0.7 = 2.0GB;
    # per copy = 0.8 + 1.0 + 0.7 = 2.5GB on an 8GB card.
    # usable = 7.8 - 1.5 reserve - 2.0 main = 4.3 → 1 copy fits.
    assert _resolve(
        free_gb=7.8, total_gb=8.0, requested=2, model_gb=0.8
    ) == 1
    # Same card AFTER the main model is resident (free dropped by ~2GB): the
    # footprint is already in `used`, so we don't double-charge it.
    # usable = 5.8 - 1.5 - 0 = 4.3 → still 1 (not 0); reserve isn't doubled.
    assert _resolve(
        free_gb=5.8, total_gb=8.0, requested=2, model_gb=0.8
    ) == 1
