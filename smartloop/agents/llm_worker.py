"""Process-level LLM worker for parallel inference.

Each worker process holds its own Llama model instance in a module-global
variable, initialized once via ``init_worker`` (called by
``ProcessPoolExecutor(initializer=...)``).
"""
import logging
import sys
from typing import Optional

_llm = None  # Module-global: one Llama model per process

# Floor for the retry ladder — below this a context is too small to be useful
# for document extraction, so we stop shrinking and disable the worker instead.
_MIN_WORKER_N_CTX = 2048


def _configure_worker_logging(log_queue=None) -> None:
    """Configure logging inside a spawn worker process.

    When ``log_queue`` is provided, route records through it back to the
    parent's QueueListener — single source of truth for output.  When ``None``
    (tests, ad-hoc usage), fall back to a stderr handler so the worker is
    still observable.
    """
    from smartloop.utils.logging_setup import worker_setup_logging
    worker_setup_logging(log_queue)


def init_worker(model_path: str, project_id: str, log_queue=None) -> None:
    """Called once per process by ProcessPoolExecutor(initializer=...).

    Never raises: a raised initializer marks the whole ProcessPoolExecutor
    broken, so every later ``submit_llm`` would fail with BrokenProcessPool and
    silently degrade all callers.  If the model can't be loaded even at the
    smallest context, we log and leave ``_llm = None`` — only this one worker
    is lost, and ``run_inference`` surfaces a clean error its caller can catch.
    """
    global _llm
    _configure_worker_logging(log_queue)

    import multiprocessing
    proc = multiprocessing.current_process()
    proc.name = f"smartloop-agent-{proc.pid}"

    from smartloop.llms.llm_factory import LlmFactory
    from smartloop.config.app_settings import AppSettings
    logger = logging.getLogger("llm_worker")
    worker_ctx = AppSettings().llm_worker_n_ctx

    # Retry the load with a progressively smaller context window.  The KV cache
    # is the main per-copy VRAM lever, so a worker that can't fit its full
    # context alongside sibling copies (or the main model) may still load at a
    # smaller one.  This is additive to the factory's own GPU→CPU fallback:
    # each attempt caps the auto-sized context lower via max_n_ctx.
    ctx_attempts: list[int] = []
    ctx = worker_ctx
    while ctx > _MIN_WORKER_N_CTX:
        ctx_attempts.append(ctx)
        ctx //= 2
    ctx_attempts.append(_MIN_WORKER_N_CTX)

    # probe=False: this worker is already a disposable subprocess, so a nested
    # probe load only adds GPU contention.
    last_err: Optional[Exception] = None
    for attempt_ctx in ctx_attempts:
        logger.info(
            f"Initializing LLM worker process (model={model_path}, "
            f"n_ctx<={attempt_ctx})"
        )
        try:
            _llm = LlmFactory.get_llama_cpp_instance(
                model_path, logger, project_id, probe=False, max_n_ctx=attempt_ctx,
            )
            if attempt_ctx != worker_ctx:
                logger.warning(
                    f"LLM worker loaded at reduced n_ctx<={attempt_ctx} "
                    f"(requested {worker_ctx}); tune SLP_LLM_WORKER_N_CTX or "
                    f"reduce SLP_N_LLM_WORKERS to fit more headroom."
                )
            return
        except Exception as e:
            last_err = e
            logger.warning(
                f"LLM worker load failed at n_ctx<={attempt_ctx}: {e}"
            )

    _llm = None
    logger.error(
        f"LLM worker could not load model after {len(ctx_attempts)} attempt(s) "
        f"down to n_ctx<={_MIN_WORKER_N_CTX}; this worker is disabled. "
        f"Last error: {last_err}"
    )


def run_inference(prompt: str) -> str:
    """Run inference on the process-local model. Returns text."""
    if _llm is None:
        raise RuntimeError(
            "LLM worker has no model loaded (initialization failed); "
            "caller should fall back to in-thread inference"
        )
    return str(_llm.invoke(prompt))
