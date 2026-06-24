from collections import OrderedDict

from smartloop.model import GemmaItTextModel
from smartloop.model import GemmaItModel
from smartloop.model import Llama3_2_1bModel
from smartloop.model import Llama3_2_3bModel
from smartloop.model import Phi4MiniModel
from smartloop.model import Gemma4_2bModel
from smartloop.model import Qwen2VL_2bModel

# Backward-compatibility aliases: deprecated/old model name -> canonical name.
# The bootstrap model was renamed from "gemma3-1b" to "slp-orchestra-mini";
# existing configs, env vars (SLP_BASE_MODEL_NAME) and workspace.json files may
# still reference the old name, so both must resolve to the same model.
MODEL_ALIASES = {
    "gemma3-1b": "slp-orchestra-mini",
}


def canonical_model_name(name):
    """Map a (possibly aliased) model name to its canonical name."""
    if name is None:
        return name
    return MODEL_ALIASES.get(name, name)


class _SupportedModels(OrderedDict):
    """OrderedDict that resolves backward-compatibility aliases on lookup.

    Iteration still yields only the canonical names, so model listings and
    reverse name<->id resolution stay clean while ``get``/``[]``/``in`` accept
    the old aliases transparently.
    """

    def __getitem__(self, key):
        return super().__getitem__(canonical_model_name(key))

    def get(self, key, default=None):
        return super().get(canonical_model_name(key), default)

    def __contains__(self, key):
        return super().__contains__(canonical_model_name(key))


SUPPORTED_MODELS = _SupportedModels({
    "slp-orchestra-mini": GemmaItTextModel,
    "gemma3-4b": GemmaItModel,
    "llama3-1b": Llama3_2_1bModel,
    "llama3-3b": Llama3_2_3bModel,
    "phi4-mini": Phi4MiniModel,
    "gemma4-2b": Gemma4_2bModel,
    "qwen2-vl-2b": Qwen2VL_2bModel,
})
