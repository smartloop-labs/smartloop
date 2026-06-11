from abc import abstractmethod
import logging
from typing import Any

class Worker:
    def __init__(self, model: Any, tokenizer: Any, logger: logging.Logger):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger

    @abstractmethod
    def run(self, **kwargs) -> Any | None:
        pass
