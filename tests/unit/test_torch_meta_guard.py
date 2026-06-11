"""Tests for the accelerate meta-init leak guard.

A leaked ``init_empty_weights`` context patches the global
``nn.Module.register_parameter`` to create parameters on the ``meta`` device,
which breaks every subsequent model load.  The guard detects and repairs it.
"""
import torch.nn as nn

from smartloop.utils.torch_meta_guard import repair_meta_init, safe_model_load


def test_repair_restores_leaked_register_parameter():
    from accelerate import init_empty_weights

    cm = init_empty_weights()
    cm.__enter__()  # leak: entered but never exited
    try:
        assert "Module.register_parameter" != nn.Module.register_parameter.__qualname__
        assert repair_meta_init() is True
        assert nn.Module.register_parameter.__qualname__ == "Module.register_parameter"
    finally:
        # Context already repaired; swallow the generator's restore attempt.
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass


def test_repair_is_noop_when_not_patched():
    # Pristine state — nothing to repair.
    assert nn.Module.register_parameter.__qualname__ == "Module.register_parameter"
    assert repair_meta_init() is False


def test_safe_model_load_repairs_before_yielding():
    from accelerate import init_empty_weights

    cm = init_empty_weights()
    cm.__enter__()
    try:
        with safe_model_load():
            # Inside the guarded block the patch is already repaired.
            assert nn.Module.register_parameter.__qualname__ == "Module.register_parameter"
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception:
            pass
