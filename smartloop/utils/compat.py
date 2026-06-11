"""
Compatibility shims for version mismatches between project dependencies.
Import this module early, before loading any HuggingFace models.
"""
from typing import TypedDict
import transformers.utils as _transformers_utils

# `LossKwargs` was removed from transformers.utils in 4.52+.
# Dynamic model code (e.g. Phi-4-mini-instruct/modeling_phi3.py) that targets
# older versions still imports it. Inject a no-op stub if missing.
if not hasattr(_transformers_utils, "LossKwargs"):
    class LossKwargs(TypedDict, total=False):
        """Compatibility stub for transformers >= 4.52 where LossKwargs was removed."""

    _transformers_utils.LossKwargs = LossKwargs  # type: ignore[attr-defined]
