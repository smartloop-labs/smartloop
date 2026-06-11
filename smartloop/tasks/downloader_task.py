"""
Downloader Task — downloads model artifacts in a background thread.

auto=True: starts automatically when submitted.
Downloads GGUF files first for fastest time-to-usable-model, then
downloads remaining artifacts (safetensors, tokenizer, etc.).
"""

import logging
import os
from typing import Optional

from smartloop.tasks.base_task import BaseTask, TaskResult
from smartloop.model.downloader import SLPDownloader, DownloadProgress, ProgressCallback
from smartloop.workspace import Workspace


class DownloaderTask(BaseTask):
    """Background task that downloads model artifacts.

    Downloads GGUF first so the model can be loaded while the remaining
    training artifacts are still downloading.
    """

    def __init__(
        self,
        workspace: Workspace,
        downloader: SLPDownloader,
        logger: logging.Logger,
        model_id: str,
        progress_callback: Optional[ProgressCallback] = None,
        bootstrap: bool = False,
        **kwargs,
    ):
        super().__init__(name="downloader", logger=logger, auto=True, **kwargs)
        self.workspace = workspace
        self.downloader = downloader
        self.model_id = model_id
        self.progress_callback = progress_callback
        self.bootstrap = bootstrap
        self._gguf_ready = False

    @property
    def gguf_ready(self) -> bool:
        """True once the GGUF file has been downloaded."""
        return self._gguf_ready

    def _check_gguf_exists(self) -> bool:
        """Check whether a GGUF file already exists on disk."""
        model_dir = os.path.join(self.workspace.models_dir, self.model_id)
        if os.path.isdir(model_dir):
            return any(f.endswith(".gguf") for f in os.listdir(model_dir))
        return False

    def should_run(self) -> bool:
        """Skip if the GGUF already exists."""
        return not self._check_gguf_exists()

    def execute(self, **kwargs) -> TaskResult:
        """Download GGUF model artifacts."""
        try:
            self.logger.info(f"[downloader] Downloading GGUF for {self.model_id}")

            if not self._check_gguf_exists():
                if self.bootstrap:
                    self.downloader.bootstrap_download(
                        progress_callback=self.progress_callback,
                    )
                else:
                    self.downloader.download(
                        model_id=self.model_id,
                        progress_callback=self.progress_callback,
                    )

            self._gguf_ready = True
            self.logger.info(f"[downloader] GGUF ready for {self.model_id}")

            # Mark model as downloaded in workspace
            ws_model = self.workspace.get_model()
            if ws_model and not ws_model.downloaded:
                ws_model.downloaded = True

            self.logger.info(f"[downloader] All artifacts downloaded for {self.model_id}")
            return TaskResult(success=True, data={"model_id": self.model_id})

        except PermissionError:
            # Non-retryable — re-raise so BaseTask skips retries
            raise
        except Exception as e:
            self.logger.error(f"[downloader] Download failed: {e}")
            return TaskResult(success=False, error=str(e))
