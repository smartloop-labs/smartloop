import logging
import os
from smartloop.config import ModelConfig
from smartloop.config import SUPPORTED_MODELS
from smartloop.config.app_settings import AppSettings
from smartloop.utils.log_utils import ConsoleLogger
from smartloop.model.model_manager import Model, ModelManager
from smartloop.workspace import Workspace

from rich.console import Console

_settings = AppSettings()

class ModelFactory:

    @staticmethod
    def get_model(config: ModelConfig,
        logger: logging.Logger = ConsoleLogger(Console())
    ) -> Model:
        try:
            model_type = SUPPORTED_MODELS[config.model_name]
        except KeyError:
            raise ValueError(f"Unknown model: {config.model_name}")

        os.makedirs(_settings.home_dir, exist_ok=True)

        workspace = Workspace.init(config=config)

        return ModelManager.get_model(
            model_type,
            workspace,
            logger,
            config=config,
        )
