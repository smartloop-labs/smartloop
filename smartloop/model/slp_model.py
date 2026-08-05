from __future__ import annotations

import logging

from abc import ABC, abstractmethod
from pathlib import Path

from typing import TYPE_CHECKING, Iterator, Optional, Union

from smartloop.types import ModelType
from smartloop.llms.base_llm import BaseLLM

if TYPE_CHECKING:
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

    @staticmethod
    def model_version() -> int:
        """Model family version (e.g. Gemma 3 → 3).

        Used for selection: a higher version always wins over a lower one,
        even when its parameter count is smaller. Override per model.
        """
        return 0

    @staticmethod
    def param_size_in_b() -> float:
        """Parameter count in billions (e.g. 4 for a 4B model).

        Override per model. Returns 0 when unknown, in which case callers
        fall back to parsing the size from the model name/id.
        """
        return 0.0

    @staticmethod
    def vision_chat_handler() -> Optional[str]:
        """llama-cpp chat-handler class name for this model's image input.

        Declared alongside ``ModelCapability.VISION`` so a vision model owns
        the handler it needs (e.g. ``Gemma4ChatHandler``). The loader uses
        this to decode image pixels; using the wrong handler for an
        architecture crashes the native clip code, so a model that declares
        VISION must also declare the handler here. Returns ``None`` for
        non-vision models (loaded text-only)."""
        return None

    @abstractmethod
    def get_instance(self, debug: bool = False) -> BaseLLM:
        pass
