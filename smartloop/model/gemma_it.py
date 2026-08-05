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
        return [ModelCapability.KNOWLEDGE, ModelCapability.SKILLS, ModelCapability.DP]

    @staticmethod
    def model_id() -> str:
        return "google/gemma-3-4b-it"