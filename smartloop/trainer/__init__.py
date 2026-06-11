"""Re-export from smartloop_trainer package (optional dependency).

Install with: make install-trainer
"""

__all__ = [
    "Trainer",
    "TrainingProgress",
    "Builder",
    "BuildProgress",
    "TrainTask",
    "BuildTask",
    "LoraGGUFConverter",
]

_IMPORT_MAP = {
    "Trainer": ("smartloop_trainer", "Trainer"),
    "TrainingProgress": ("smartloop_trainer", "TrainingProgress"),
    "Builder": ("smartloop_trainer", "Builder"),
    "BuildProgress": ("smartloop_trainer", "BuildProgress"),
    "TrainTask": ("smartloop_trainer", "TrainTask"),
    "BuildTask": ("smartloop_trainer", "BuildTask"),
    "LoraGGUFConverter": ("smartloop_trainer.lora_converter", "LoraGGUFConverter"),
}


def __getattr__(name):
    if name in _IMPORT_MAP:
        module_path, attr = _IMPORT_MAP[name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            return getattr(mod, attr)
        except ImportError:
            raise ImportError(
                f"smartloop-trainer is not installed. "
                f"Install with: make install-trainer"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
