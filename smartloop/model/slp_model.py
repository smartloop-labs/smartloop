import logging

from abc import ABC, abstractmethod
from pathlib import Path

from typing import Iterator, Union

from smartloop.types import ModelType
from smartloop.llms.base_llm import BaseLLM
from smartloop.workspace import Workspace

class SLPModel(ABC):
    """Abstract base class for Smartloop model"""

    def __init__(self, workspace: Workspace, logger: logging.Logger):
        self.workspace = workspace
        self.logger = logger

    @abstractmethod
    def init(self):
        """Initialize the model"""
        pass

    @abstractmethod
    def add_document(self, source: Union[str, Path]):
        """Add a new source file or URL to the model"""
        pass

    @abstractmethod
    def build(self):
        """Build the model"""
        pass

    @abstractmethod
    def train(self, progress_callback=None):
        """Train the model"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        pass

    @staticmethod
    @abstractmethod
    def model_id() -> str:
        pass

    @abstractmethod
    def get_instance(self, debug: bool = False) -> BaseLLM:
        pass
