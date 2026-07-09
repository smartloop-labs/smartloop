from collections import OrderedDict

from smartloop.model import GemmaItTextModel
from smartloop.model import GemmaItModel
from smartloop.model import Gemma4_2bModel
from smartloop.model import Qwen2VL_2bModel

# Suffix identifying the OpenVINO INT8 IR variant of a bootstrap model.
# On an Intel host with the OpenVINO runtime the downloader prefers the
# "<name>-ov-int8" bootstrap entry (which ships an OpenVINO IR export instead
# of a GGUF) and stores it in its own model folder.
OV_INT8_SUFFIX = "-ov-int8"


def base_model_name(name: str) -> str:
    """Strip the OpenVINO IR variant suffix so an "<name>-ov-int8" entry maps
    back to the same short model name as its GGUF sibling."""
    return name[: -len(OV_INT8_SUFFIX)] if name.endswith(OV_INT8_SUFFIX) else name

SUPPORTED_MODELS = OrderedDict({
    "slp-orchestra-mini": GemmaItTextModel,
    "gemma3-4b": GemmaItModel,
    "gemma4-2b": Gemma4_2bModel,
    "qwen2-vl-2b": Qwen2VL_2bModel,
})
