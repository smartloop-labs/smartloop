from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelCapability, ModelType

if TYPE_CHECKING:
    from smartloop.workspace import Workspace

class GemmaItModel(GenericModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @staticmethod
    def model_version() -> int:
        return 3

    @staticmethod
    def param_size_in_b() -> int:
        return 4

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def default_capabilities() -> list[ModelCapability]:
        return [
            ModelCapability.GENERAL,
            ModelCapability.SKILLS,
            ModelCapability.DP,
        ]

    @staticmethod
    def vision_chat_handler() -> str:
        # llama-cpp-python has no Gemma3-specific handler; its Gemma4ChatHandler
        # is just an empty subclass of the generic mtmd handler. Gemma 3's mmproj
        # loads through that same generic path (projector auto-detected from the
        # GGUF), so declare the base MTMDChatHandler directly.
        return "MTMDChatHandler"

    @staticmethod
    def model_id() -> str:
        return "google/gemma-3-4b-it"