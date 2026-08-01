import os
import logging
import threading
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars

from smartloop.model.downloader.slp_downloader import SLPDownloader, DownloadProgress, ProgressCallback, AggregateProgress
from smartloop.workspace import Workspace

MAX_WORKERS = min(4, os.cpu_count() or 4)
POLL_INTERVAL = 0.5


class HFDownloader(SLPDownloader):
    def __init__(self, workspace: Workspace, logger: logging.Logger):
        super().__init__(workspace, logger)
        self.logger = logger

    def download(self, model_id: str, progress_callback: Optional[ProgressCallback] = None) -> Path:
        """Download model from HuggingFace Hub with byte-level progress."""
        # Disable HF's tqdm bars — we report progress via our own callback
        disable_progress_bars()
        self.logger.info(f"Downloading model: {model_id} from HuggingFace Hub")
        local_dir = os.path.join(self.workspace.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        ws_model = self.workspace.get_model()
        token = (ws_model.config.hf_token
                 if ws_model and ws_model.config else None)

        api = HfApi()
        repo_files = list(api.list_repo_tree(model_id, recursive=True, token=token))
        all_files = [f for f in repo_files if hasattr(f, 'size') and f.size is not None]

        pending = []
        skipped = 0
        for f in all_files:
            local_path = os.path.join(local_dir, f.path)
            if os.path.isfile(local_path) and os.path.getsize(local_path) == f.size:
                skipped += 1
            else:
                pending.append(f)

        if skipped:
            self.logger.info(f"[ok] {skipped} file(s) already up-to-date, skipped")

        if not pending:
            self.logger.info(f"[ok] All files already downloaded at: {local_dir}")
            return Path(local_dir)

        total_bytes = sum(f.size for f in pending)
        total_mb = total_bytes / (1024 * 1024)
        self.logger.info(f"Downloading {len(pending)} file(s), {total_mb:.1f} MB total")

        # Roll the parallel per-file progress up into a single 0->100% bar so
        # the client sees one download, not the files fighting over the meter.
        aggregate = AggregateProgress(total_bytes, progress_callback, model_id)

        done = threading.Event()

        def _poll_progress():
            last_reported = {}
            while not done.is_set():
                for f in pending:
                    local_path = os.path.join(local_dir, f.path)
                    incomplete = local_path + ".incomplete"
                    file_downloaded = 0
                    for p in (local_path, incomplete):
                        try:
                            file_downloaded = min(os.path.getsize(p), f.size)
                            break
                        except OSError:
                            continue
                    prev = last_reported.get(f.path, 0)
                    if progress_callback and file_downloaded > 0 and file_downloaded != prev:
                        last_reported[f.path] = file_downloaded
                        aggregate(DownloadProgress(
                            filename=os.path.basename(f.path),
                            downloaded=file_downloaded,
                            total=f.size,
                            status="downloading",
                        ))
                done.wait(POLL_INTERVAL)

        monitor = threading.Thread(target=_poll_progress, daemon=True)
        monitor.start()

        try:
            failed = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {
                    pool.submit(
                        hf_hub_download,
                        repo_id=model_id,
                        filename=f.path,
                        local_dir=str(local_dir),
                        token=token,
                    ): f.path
                    for f in pending
                }
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"[error] {filename}: {e}")
                        failed.append(filename)

            if failed:
                raise RuntimeError(
                    f"Failed to download {len(failed)} file(s): {', '.join(failed)}"
                )

            aggregate.finish()

            self.logger.info(f"[ok] Model downloaded to: {local_dir}")
            return Path(local_dir)
        except (OSError, ValueError) as e:
            self.logger.error(f"[error] Error downloading model: {e}")
            raise e
        finally:
            done.set()
            monitor.join()
