"""LLMExecutor — threads for I/O, processes for LLM.

Dual-pool resource manager that dispatches I/O-bound work (search, MCP
calls) to threads and LLM inference to separate processes (each with its
own model instance).
"""
import logging
import os
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from smartloop.agents.llm_worker import init_worker, run_inference

logger = logging.getLogger(__name__)


class LLMExecutor:
    """Dual-pool executor: threads for I/O, processes for LLM inference.

    Parameters
    ----------
    max_io_workers:
        Size of the thread pool used for search, MCP calls, and other
        I/O-bound work.
    n_llm_workers:
        Number of worker processes for LLM inference.  Each process loads
        its own copy of the model, so RAM/VRAM usage scales linearly.
        Set to 0 (default) to disable — ``submit_llm`` falls back to the
        thread pool, which serializes LLM calls as before.
    model_path:
        Path to the GGUF model file.  Required when *n_llm_workers* > 0.
    project_id:
        Project identifier passed to the worker initializer.
    """

    def __init__(
        self,
        max_io_workers: int = 6,
        n_llm_workers: int = 0,
        model_path: str | None = None,
        project_id: str | None = None,
    ):
        self._io_pool = ThreadPoolExecutor(max_workers=max_io_workers, thread_name_prefix="io")
        self._llm_pool: ProcessPoolExecutor | None = None

        if n_llm_workers > 0 and model_path:
            # ``n_llm_workers`` is the on/off switch + preferred count; the
            # actual number of worker copies is sized dynamically to what fits
            # in free VRAM (see _resolve_worker_count) so we neither crash by
            # over-allocating nor leave the GPU underutilized.
            effective_workers = self._resolve_worker_count(model_path, n_llm_workers)

            # Use the "spawn" start method, not the Linux default "fork".
            # The parent process has already initialized CUDA (torch), and a
            # forked child cannot re-initialize CUDA ("Cannot re-initialize
            # CUDA in forked subprocess"), which crashes the worker. Spawn
            # starts each worker as a fresh process with no inherited CUDA
            # state. (On macOS/Windows spawn is already the default.)
            mp_context = multiprocessing.get_context("spawn")
            logger.info(
                f"Creating ProcessPoolExecutor with {effective_workers} LLM worker(s) "
                f"(model={model_path}, start_method=spawn)"
            )
            self._llm_pool = ProcessPoolExecutor(
                max_workers=effective_workers,
                mp_context=mp_context,
                initializer=init_worker,
                initargs=(model_path, project_id or ""),
            )

    @staticmethod
    def _resolve_worker_count(model_path: str, requested: int) -> int:
        """Return how many LLM worker copies to spawn, sized to available memory.

        Each worker holds its own model copy (weights + a KV cache that scales
        with ``slp_llm_worker_n_ctx``).  ``requested`` is the upper bound; the
        count is then scaled *down* to what fits in the live free memory of the
        active device — CUDA VRAM, Apple MPS unified memory, or CPU RAM — so
        workers don't crash on load.  Extra free memory never raises the count
        above ``requested``.  The estimate is deliberately conservative since
        under-counting a worker's footprint crashes the last copy to load.
        """
        try:
            model_gb = os.path.getsize(model_path) / (1024 ** 3)

            from smartloop.config.app_settings import AppSettings
            worker_ctx = AppSettings().llm_worker_n_ctx
            # Per-copy footprint: weights + KV cache + per-process overhead.
            # The estimate is deliberately conservative — under-counting any
            # term makes ``fits`` round up to a worker that then fails
            # ``llama_context`` allocation.  Overhead must cover the CUDA
            # context that every spawned process allocates (~0.6GB before any
            # weights) plus compute/scratch buffers, so we budget 0.7GB.  The
            # KV estimate scales with worker_ctx (~0.5GB / 4k tokens).
            kv_gb = 0.5 * (worker_ctx / 4096)
            per_copy_gb = model_gb + kv_gb + 0.7

            # Resolve (device_label, free_gb, total_gb, reserve_gb) for whichever
            # accelerator is active.  reserve keeps memory for the main model,
            # fragmentation headroom and the OS so workers don't claim
            # everything and crash the last copy that tries to load.
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                # mem_get_info() is a single snapshot taken before any worker
                # loads; workers spawn concurrently, so keep a VRAM reserve
                # (was 0.0) to absorb that drift + allocator fragmentation.
                device, free_gb, total_gb, reserve_gb = (
                    torch.cuda.get_device_name(0), free_b / 1024**3, total_b / 1024**3, 1.5,
                )
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # Apple Silicon: unified memory shared with the OS/main model.
                import psutil
                vm = psutil.virtual_memory()
                device, total_gb = "Apple MPS (unified)", vm.total / 1024**3
                free_gb, reserve_gb = vm.available / 1024**3, 3.0  # leave headroom
            else:
                import psutil
                vm = psutil.virtual_memory()
                device, total_gb = "CPU", vm.total / 1024**3
                free_gb, reserve_gb = vm.available / 1024**3, 2.0

            usable_gb = max(0.0, free_gb - reserve_gb)
            fits = int(usable_gb // per_copy_gb) if per_copy_gb > 0 else 0
            # Never spawn more than requested: free VRAM is not a reason to
            # exceed the configured parallelism, and capping here removes the
            # surprise over-allocation path.  Scale down when fewer fit, but
            # keep ≥1 since the pool was explicitly enabled.
            effective = max(1, min(requested, fits))

            logger.info(
                f"[device] {device}: {free_gb:.1f}GB free / {total_gb:.1f}GB total "
                f"(reserve {reserve_gb:.1f}GB). Per worker copy ~{per_copy_gb:.2f}GB "
                f"(weights {model_gb:.2f} + KV {kv_gb:.2f} @ n_ctx<={worker_ctx} + 0.7). "
                f"Fits ~{fits} copy(ies); requested {requested} → using {effective}."
            )
            if effective != requested:
                logger.warning(
                    f"[device] Adjusted LLM workers {requested} → {effective} based on "
                    f"{free_gb:.1f}GB free memory. Tune with SLP_LLM_WORKER_N_CTX "
                    f"(smaller = more copies) or free memory."
                )
            return effective
        except Exception as exc:
            logger.info(
                f"[device] Could not estimate memory capacity ({exc}); "
                f"using requested {requested} worker(s)"
            )
            return requested

    # -- Public dispatch API ------------------------------------------------

    @property
    def has_llm_pool(self) -> bool:
        """True when a dedicated LLM worker process pool is available."""
        return self._llm_pool is not None

    def submit_io(self, fn, *args, **kwargs):
        """Submit I/O-bound work (search, MCP calls) to the thread pool."""
        return self._io_pool.submit(fn, *args, **kwargs)

    def submit_llm(self, prompt: str):
        """Submit LLM inference to the process pool.

        Falls back to the thread pool when the process pool is disabled
        (``n_llm_workers=0``).
        """
        if self._llm_pool is not None:
            return self._llm_pool.submit(run_inference, prompt)
        return self._io_pool.submit(run_inference, prompt)

    # -- Lifecycle ----------------------------------------------------------

    def shutdown(self, wait: bool = False) -> None:
        """Shut down both pools."""
        self._io_pool.shutdown(wait=wait)
        if self._llm_pool is not None:
            self._llm_pool.shutdown(wait=wait)
