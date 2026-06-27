from collections import OrderedDict

from smartloop.model import GemmaItTextModel
from smartloop.model import GemmaItModel
from smartloop.model import Gemma4_2bModel
from smartloop.model import Qwen2VL_2bModel

SUPPORTED_MODELS = OrderedDict({
    "slp-orchestra-mini": GemmaItTextModel,
    "gemma3-4b": GemmaItModel,
    "gemma4-2b": Gemma4_2bModel,
    "qwen2-vl-2b": Qwen2VL_2bModel,
})
