import logging

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelType
from smartloop.workspace import Workspace


class Ministral3bModel(GenericModel):
    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @property
    def model_type(self) -> ModelType:
        return ModelType.TEXT

    @staticmethod
    def model_id() -> str:
        return "mistralai/Ministral-3-3B-Instruct-2512"
