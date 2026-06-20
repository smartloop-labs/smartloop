import logging

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelType
from smartloop.workspace import Workspace


class Qwen2_5VL_3bModel(GenericModel):
    """Qwen2.5-VL 3B — small multimodal model used for document OCR / vision.

    Runs via llama-cpp-python's ``Qwen25VLChatHandler`` (present in 0.3.x),
    unlike gemma-3-4b which has no Gemma3 vision handler there.
    """

    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def model_id() -> str:
        return "Qwen/Qwen2.5-VL-3B-Instruct"
