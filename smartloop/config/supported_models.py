from collections import OrderedDict

from smartloop.model import GemmaItTextModel
from smartloop.model import GemmaItModel
from smartloop.model import Llama3_2_1bModel
from smartloop.model import Llama3_2_3bModel
from smartloop.model import Phi4MiniModel
from smartloop.model import Gemma4_2bModel
from smartloop.model import Qwen2_5VL_3bModel
from smartloop.model import Qwen2VL_2bModel

SUPPORTED_MODELS = OrderedDict({
    "gemma3-1b": GemmaItTextModel,
    "gemma3-4b": GemmaItModel,
    "llama3-1b": Llama3_2_1bModel,
    "llama3-3b": Llama3_2_3bModel,
    "phi4-mini": Phi4MiniModel,
    "gemma4-2b": Gemma4_2bModel,
    "qwen2-vl-2b": Qwen2VL_2bModel,
    "qwen2.5-vl-3b": Qwen2_5VL_3bModel,
})
