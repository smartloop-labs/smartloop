import hashlib
import os
import logging
import threading

import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from smartloop.platform import SLPPlatformClient
from smartloop.platform import auth as platform_auth
from smartloop.config import AppSettings
from smartloop.model.downloader.artifact_source import SLPModelSource, SLPBootstrapSource
from smartloop.model.downloader.slp_downloader import SLPDownloader, DownloadProgress, ProgressCallback
from smartloop.workspace import Workspace

_CPU_COUNT = os.cpu_count() or 4
_IO_THREAD_BUDGET = min(32, _CPU_COUNT * 2)  # total concurrent connections (capped at 32)

def _is_mmproj(path: str) -> bool:
    """True when ``path`` is a vision mmproj / CLIP projector GGUF."""
    name = os.path.basename(path).lower()
    return "mmproj" in name or "clip" in name


def _select_gguf_artifacts(artifacts: dict) -> dict:
    """Pick which GGUFs to download from a model's artifact map.

    Keeps the quantized main weights (e.g. ``*-Q4_K_M.gguf``, preferred over the
    raw F16) plus the mmproj / CLIP projector (vision models need it).  The
    mmproj is never counted as the main weights, and the main weights are never
    dropped when the model ships no quantized variant (the bug being fixed:
    ``{**quantized, **mmproj}`` collapsed to mmproj-only for F16-only models,
    leaving no loadable main GGUF).
    """
    gguf = {k: v for k, v in artifacts.items()
            if os.path.basename(k).endswith(".gguf")}
    mmproj = {k: v for k, v in gguf.items() if _is_mmproj(k)}
    main = {k: v for k, v in gguf.items() if not _is_mmproj(k)}
    quantized = {k: v for k, v in main.items()
                 if "-Q" in os.path.basename(k).upper()}
    return {**(quantized or main), **mmproj}


MAX_WORKERS = min(4, _CPU_COUNT)                      # parallel file downloads (capped: I/O-bound)
CHUNK_SIZE = 8 * 1024 * 1024                         # 8 MB read buffer for simple downloads
RANGE_READ_SIZE = 1 * 1024 * 1024                    # 1 MB read buffer within each chunk
CHUNKED_THRESHOLD = 100 * 1024 * 1024                # 100 MB — files larger than this use chunked download
CHUNK_WORKERS = max(2, _IO_THREAD_BUDGET // MAX_WORKERS)  # parallel chunk downloads per file

class SLPArtifactDownloader(SLPDownloader):
    """Download model artifacts from URLs provided by an ArtifactSource.

    Fetches signed URLs via artifact sources, then downloads all
    files in parallel with MD5 verification and progress reporting.
    """

    def __init__(self, workspace: Workspace, logger: logging.Logger):
        super().__init__(workspace, logger)
        self.settings = AppSettings()
        platform_client = SLPPlatformClient(self.settings.slp_api_base)
        self._model_source = SLPModelSource(platform_client)
        self._bootstrap_source = SLPBootstrapSource(platform_client)

    @staticmethod
    def _compute_local_md5(filepath: str) -> str:
        """Return the hex-encoded MD5 digest of a local file."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def _parse_remote_md5(headers) -> str | None:
        """Extract hex-encoded MD5 from the ``ETag`` response header.

        For non-composite GCS objects the ETag is the hex-encoded MD5
        wrapped in quotes (32 hex chars).
        """
        etag = headers.get("ETag", "").strip('" ')
        if len(etag) == 32:
            try:
                int(etag, 16)  # validate hex
                return etag.lower()
            except ValueError:
                pass
        return None

    def _get_remote_file_info(self, url: str) -> tuple[int, str | None, bool]:
        """HEAD request to get file size, MD5 hash, and Range support."""
        try:
            resp = httpx.head(url, follow_redirects=True, timeout=30)
            cl = resp.headers.get("Content-Length")
            size = int(cl) if cl else 0
            md5 = self._parse_remote_md5(resp.headers)
            accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
            return size, md5, accept_ranges
        except Exception:
            return 0, None, False

    def _download_artifacts(self, artifacts: dict[str, str], local_dir: str,
                             progress_callback: ProgressCallback | None,
                             log_prefix: str, log_dir: str) -> Path:
        """Check, download, and verify a set of artifacts. Shared by bootstrap and download."""
        pending: dict[str, str] = {}
        pending_sizes: dict[str, int] = {}
        pending_range: dict[str, bool] = {}
        skipped = 0

        for filename, url in artifacts.items():
            flat_name = os.path.basename(filename)
            local_path = os.path.join(local_dir, flat_name)
            size, remote_md5, accept_ranges = self._get_remote_file_info(url)

            if os.path.isfile(local_path):
                if remote_md5:
                    local_md5 = self._compute_local_md5(local_path)
                    if local_md5 == remote_md5:
                        skipped += 1
                        continue
                elif size and os.path.getsize(local_path) == size:
                    skipped += 1
                    continue

            pending[filename] = url
            pending_sizes[filename] = size
            pending_range[filename] = accept_ranges

        if skipped:
            self.logger.info(
                f"[ok] {skipped} file(s) already up-to-date, skipped"
            )

        if not pending:
            self.logger.info(f"[ok] All files already downloaded at: {log_dir}")
            return Path(local_dir)

        total_bytes = sum(s for s in pending_sizes.values() if s > 0)
        total_mb = total_bytes / (1024 * 1024)
        self.logger.info(
            f"Downloading {len(pending)} file(s), {total_mb:.1f} MB total"
        )

        def _per_file_callback(progress: DownloadProgress) -> None:
            if not progress_callback:
                return
            progress_callback(DownloadProgress(
                filename=progress.filename,
                downloaded=progress.downloaded,
                total=progress.total,
                status=progress.status,
            ))

        failed = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    self._download_file, filename, url, local_dir,
                    pending_sizes.get(filename, 0), position, _per_file_callback,
                    pending_range.get(filename, False),
                ): filename
                for position, (filename, url) in enumerate(pending.items())
            }

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                except PermissionError:
                    raise
                except Exception as e:
                    self.logger.error(f"[error] {filename}: {e}")
                    failed.append(filename)

        if failed:
            raise RuntimeError(
                f"Failed to download {len(failed)} file(s): {', '.join(failed)}"
            )

        self.logger.info(f"[ok] {log_prefix} downloaded to: {log_dir}")
        return Path(local_dir)

    def _download_file(self, filename: str, url: str, local_dir: str, size: int, position: int,
                        progress_callback: ProgressCallback | None = None,
                        supports_range: bool = False) -> str:
        """Download a single file, using chunked parallel Range requests for large files."""
        flat_name = os.path.basename(filename)
        dest = os.path.join(local_dir, flat_name)
        size_mb = size / (1024 * 1024)

        if supports_range and size >= CHUNKED_THRESHOLD:
            chunk_size_mb = size / CHUNK_WORKERS / (1024 * 1024)
            self.logger.info(
                f"Downloading {flat_name} ({size_mb:.1f} MB) using "
                f"{CHUNK_WORKERS} parallel chunks of {chunk_size_mb:.1f} MB each"
            )
            self._download_file_chunked(flat_name, url, dest, size, progress_callback)
        else:
            reason = "no Range support" if not supports_range else "below threshold"
            self.logger.info(f"Downloading {flat_name} ({size_mb:.1f} MB) single-stream ({reason})")
            self._download_file_simple(flat_name, url, dest, size, progress_callback)

        if progress_callback:
            progress_callback(DownloadProgress(
                filename=flat_name, downloaded=size, total=size, status="completed",
            ))

        return filename

    def _download_file_simple(self, flat_name: str, url: str, dest: str, size: int,
                               progress_callback: ProgressCallback | None = None) -> None:
        """Download a file in a single request (small files)."""
        downloaded = 0
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            if resp.status_code == 401:
                raise PermissionError(
                    "Download rejected (401 Unauthorized). "
                    "The download URL may have expired or the resource requires authentication."
                )
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(DownloadProgress(
                            filename=flat_name,
                            downloaded=downloaded,
                            total=size,
                            status="downloading",
                        ))

    def _download_chunk(self, url: str, dest: str, start: int, end: int,
                        flat_name: str, total_size: int,
                        downloaded_counter: list, lock: threading.Lock,
                        progress_callback: ProgressCallback | None = None) -> None:
        """Download a byte range and write it to the correct file offset."""
        headers = {"Range": f"bytes={start}-{end}"}
        with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                          timeout=300) as resp:
            if resp.status_code == 401:
                raise PermissionError(
                    "Download rejected (401 Unauthorized). "
                    "The download URL may have expired or the resource requires authentication."
                )
            resp.raise_for_status()
            with open(dest, "r+b") as f:
                f.seek(start)
                for chunk in resp.iter_bytes(chunk_size=RANGE_READ_SIZE):
                    f.write(chunk)
                    if progress_callback:
                        with lock:
                            downloaded_counter[0] += len(chunk)
                            current = downloaded_counter[0]
                        progress_callback(DownloadProgress(
                            filename=flat_name,
                            downloaded=current,
                            total=total_size,
                            status="downloading",
                        ))

    def _download_file_chunked(self, flat_name: str, url: str, dest: str, size: int,
                                progress_callback: ProgressCallback | None = None) -> None:
        """Download a large file using parallel Range requests."""
        # Pre-allocate the file
        with open(dest, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")

        # Split the file into exactly CHUNK_WORKERS equal segments
        chunk_size = -(-size // CHUNK_WORKERS)  # ceiling division
        chunks = []
        offset = 0
        while offset < size:
            end = min(offset + chunk_size - 1, size - 1)
            chunks.append((offset, end))
            offset = end + 1

        downloaded_counter = [0]  # mutable counter shared across threads
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=CHUNK_WORKERS) as pool:
            futures = {
                pool.submit(
                    self._download_chunk, url, dest, start, end,
                    flat_name, size, downloaded_counter, lock, progress_callback,
                ): (start, end)
                for start, end in chunks
            }

            for future in as_completed(futures):
                chunk_range = futures[future]
                try:
                    future.result()
                except Exception as e:
                    raise RuntimeError(
                        f"Chunk download failed for {flat_name} "
                        f"(bytes {chunk_range[0]}-{chunk_range[1]}): {e}"
                    ) from e

    def bootstrap_download(self, model_name: str | None = None,
                           progress_callback: ProgressCallback | None = None) -> Path:
        """Download GGUF model via the unauthenticated /v1/bootstrap/models endpoint.

        Intended for first-run scenarios (Electron installer, brew install)
        where the user has no developer token yet. Downloads only the GGUF
        file so the model is immediately ready for inference without building.

        If *model_name* is provided, only that model's artifacts are downloaded.
        Otherwise downloads the first model returned by the endpoint.
        """
        models = self._bootstrap_source.list_models()
        if model_name:
            target = next((m for m in models if m.get("name") == model_name), None)
            if not target:
                available = ", ".join(m.get("name", "?") for m in models)
                raise ValueError(
                    f"Model '{model_name}' not found in bootstrap. "
                    f"Available: {available}"
                )
        else:
            target = models[0]

        resolved_name = target.get("name", "unknown")
        model_id = self._resolve_model_id(resolved_name)
        artifacts = self._bootstrap_source.get_artifacts(resolved_name)

        artifacts = _select_gguf_artifacts(artifacts)

        if not artifacts:
            raise ValueError(f"No GGUF artifact found for bootstrap model '{resolved_name}'")

        local_dir = os.path.join(self.workspace.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        self.logger.info(f"Bootstrap downloading model: {resolved_name}")

        return self._download_artifacts(artifacts, local_dir, progress_callback,
                                        log_prefix="Bootstrap model", log_dir=local_dir)

    @staticmethod
    def _resolve_model_id(model_name: str) -> str:
        """Lookup the HuggingFace model_id from a short model name.

        E.g. "slp-orchestra-mini" → "google/gemma-3-1b-it".
        Falls back to model_name if no mapping is found.
        """
        from smartloop.config import SUPPORTED_MODELS
        model_type = SUPPORTED_MODELS.get(model_name)
        if model_type:
            return model_type.model_id()
        return model_name

    @staticmethod
    def _resolve_model_name(model_id: str) -> str:
        """Reverse-lookup the short model name from a HuggingFace model_id.

        E.g. "google/gemma-3-1b-it" → "slp-orchestra-mini".
        Falls back to model_id if no mapping is found.
        """
        from smartloop.config import SUPPORTED_MODELS
        for name, model_type in SUPPORTED_MODELS.items():
            if model_type.model_id() == model_id:
                return name
        return model_id

    def download(self, model_id: str,
                 progress_callback: ProgressCallback | None = None) -> Path:
        """Download GGUF model via the /v1/models/{model_name}/download endpoint.

        Accepts the HuggingFace-style *model_id* (e.g. "google/gemma-3-1b-it")
        to match the base-class interface, resolves it to the short model name
        for the API call, and stores files under the model_id path.
        """
        if not platform_auth.has_credentials():
            raise EnvironmentError(
                "Authentication required for GCS download. "
                "Set a developer token via 'slp token set' or sign in through the Smartloop app."
            )

        model_name = self._resolve_model_name(model_id)
        local_dir = os.path.join(self.workspace.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        self.logger.info(f"Downloading model: {model_name}")
        print()

        artifacts = self._model_source.get_artifacts(model_name)

        artifacts = _select_gguf_artifacts(artifacts)

        if not artifacts:
            self.logger.info(f"[ok] No GGUF files to download")
            return Path(local_dir)

        return self._download_artifacts(artifacts, local_dir, progress_callback,
                                        log_prefix="Model", log_dir=local_dir)
