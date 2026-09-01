from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smartloop.model.downloader import SLPDownloader
from smartloop.model.generic import GenericModel
from smartloop.types import ModelCapability, ModelType

if TYPE_CHECKING:
    from smartloop.workspace import Workspace


class Qwen3_5_2bModel(GenericModel):
    """Qwen3.5-2B — a lighter multimodal model for document OCR / vision.

    A hybrid-reasoning model: it can think in a ``<think>`` block before
    answering, declared through ``ModelCapability.THINKING`` and left off by
    default (the inherited :meth:`thinking_default`) — reasoning costs tokens
    and latency on every call, so it is opt-in from the model picker.  Its GGUF
    architecture is ``qwen35``, which no ``_VISION_HANDLER_BY_ARCH`` entry
    matches — so the handler declared below is what makes its vision work at
    all, not merely an override.
    """

    def __init__(self, workspace: Workspace, downloader: SLPDownloader, logger: logging.Logger):
        super().__init__(workspace, downloader, logger)

    @staticmethod
    def model_version() -> int:
        return 3

    @staticmethod
    def param_size_in_b() -> int:
        return 2

    @property
    def model_type(self) -> ModelType:
        return ModelType.MULTI_MODAL

    @staticmethod
    def default_capabilities() -> list[ModelCapability]:
        return [
            ModelCapability.GENERAL,
            ModelCapability.SKILLS,
            ModelCapability.DP,
            ModelCapability.VISION,
            ModelCapability.THINKING
        ]

    @staticmethod
    def vision_chat_handler() -> str:
        # llama-cpp-python has no Qwen3.5-specific handler.  The generic
        # MTMDChatHandler is the right one and NOT a fallback: it renders the
        # model's own embedded chat template (swapping image parts for
        # llama.cpp's media marker, which mtmd expands into the model's
        # vision tokens) and initializes the projector from the mmproj GGUF.
        # Qwen25VLChatHandler — what arch-sniffing would pick for a qwen*vl
        # architecture — would instead impose a hardcoded ChatML prompt that
        # predates Qwen3.5 and has no <think> control, so declaring the
        # handler here (declared beats sniffed) is what keeps both the image
        # tokens and the reasoning switch correct.
        return "MTMDChatHandler"

    @staticmethod
    def model_id() -> str:
        return "Qwen/Qwen3.5-2B"
