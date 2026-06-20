import logging

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelType
from smartloop.workspace import Workspace


class Qwen2VL_2bModel(GenericModel):
    """Qwen2-VL 2B — a lighter multimodal model for document OCR / vision.

    Smaller than Qwen2.5-VL-3B so it fits on GPU alongside the orchestrator and
    on-demand agent models on memory-constrained devices.  Runs via the same
    ``Qwen25VLChatHandler`` (the ``qwen2vl`` architecture is handled by it).
    """

    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def model_id() -> str:
        return "Qwen/Qwen2-VL-2B-Instruct"
