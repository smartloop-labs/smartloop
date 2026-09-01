from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelCapability, ModelType

if TYPE_CHECKING:
    from smartloop.workspace import Workspace

class Gemma4_2bModel(GenericModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def model_version() -> int:
        return 4

    @staticmethod
    def param_size_in_b() -> int:
        return 2

    @staticmethod
    def default_capabilities() -> list[ModelCapability]:
        return [
            ModelCapability.GENERAL,
            ModelCapability.SKILLS,
            ModelCapability.DP,
            ModelCapability.VISION,
        ]

    @staticmethod
    def vision_chat_handler() -> str:
        return "Gemma4ChatHandler"

    @staticmethod
    def model_id() -> str:
        return "google/gemma-4-e2b-it"
