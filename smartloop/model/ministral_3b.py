from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelType, ModelCapability

if TYPE_CHECKING:
    from smartloop.workspace import Workspace

class Ministral3bModel(GenericModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def model_version() -> int:
        return 3

    @staticmethod
    def param_size_in_b() -> int:
        return 3
    
    @staticmethod
    def default_capabilities() -> list[ModelCapability]:
        return [
            ModelCapability.GENERAL,
            ModelCapability.SKILLS,
            ModelCapability.VISION
        ]

    @staticmethod
    def vision_chat_handler() -> str:
        # llama-cpp-python has no Pixtral/mistral3-specific handler. Ministral-3B's
        # vision (arch 'mistral3') decodes through llama.cpp's generic mtmd path
        # with the projector auto-detected from the mmproj GGUF — the same path
        # Gemma 3 uses — so declare the base MTMDChatHandler directly.
        return "MTMDChatHandler"

    @staticmethod
    def model_id() -> str:
        return "mistralai/Ministral-3-3B-Instruct-2512"
