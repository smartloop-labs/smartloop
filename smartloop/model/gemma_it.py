import logging

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelType
from smartloop.workspace import Workspace

class GemmaItModel(GenericModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def default_capabilities() -> list[str]:
        # gemma-3-4b is a 4B model: large enough to drive skills (step
        # execution) and analysis (the reduce/synthesis task).  Advertise both
        # in the catalog (mirrors Qwen2VL_2bModel's "vision") so the capabilities
        # show before the model is registered, matching what
        # effective_capabilities infers once it is.
        return ["skills", "analysis"]

    @staticmethod
    def model_id() -> str:
        return "google/gemma-3-4b-it"