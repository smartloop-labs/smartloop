"""Standalone entry point that just loads a GGUF model and exits.

Spawned by :meth:`LlmFactory._probe_model_load` as a disposable subprocess so
a hard crash (SIGABRT/segfault) or a hang in llama.cpp's native init — GPU
driver init included — costs a bounded, throwaway process instead of the one
serving the project.

Runnable both as ``python -m smartloop.llms.model_probe <config-json>`` and,
via the frozen build's own ``-m`` re-exec in ``main.py``, as
``<slp-binary> -m smartloop.llms.model_probe <config-json>`` — the same
mechanism already used to spawn the project agent and the model host from a
PyInstaller build, so a probe is no longer skipped there (see the note in
``LlmFactory._probe_model_load`` this replaces).
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    from llama_cpp import Llama

    config = json.loads(sys.argv[1])
    llm = Llama(**config)
    del llm


if __name__ == "__main__":
    main()
