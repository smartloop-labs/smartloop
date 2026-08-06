from collections import OrderedDict


class _LazyModelDict:
    """Loads model classes on first access to avoid circular imports at module load time."""
    _data = None

    def _load(self):
        if self._data is None:
            from smartloop.model.gemma_it_text import GemmaItTextModel
            from smartloop.model.gemma_it import GemmaItModel
            from smartloop.model.gemma4_2b import Gemma4_2bModel
            from smartloop.model.qwen2_vl_2b import Qwen2VL_2bModel
            from smartloop.model.ministral_3b import Ministral3bModel
            self._data = OrderedDict({
                "slp-orchestra-mini": GemmaItTextModel,
                "gemma3-4b": GemmaItModel,
                "gemma4-e2b": Gemma4_2bModel,
                "qwen2-vl-2b": Qwen2VL_2bModel,
                "ministral3-3b": Ministral3bModel,
            })
        return self._data

    def get(self, key, default=None):
        return self._load().get(key, default)

    def __getitem__(self, key):
        return self._load()[key]

    def __contains__(self, key):
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def items(self):
        return self._load().items()

    def values(self):
        return self._load().values()

    def keys(self):
        return self._load().keys()


SUPPORTED_MODELS = _LazyModelDict()
