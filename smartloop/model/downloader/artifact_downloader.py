import hashlib
import os
import logging
import random
import threading
import time

import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, TypeVar

from smartloop.platform import SLPPlatformClient
from smartloop.config import AppSettings
from smartloop.constants import canonical_model_name, display_model_name, platform_model_name
from smartloop.model.downloader.artifact_source import SLPModelSource, SLPBootstrapSource
from smartloop.model.downloader.slp_downloader import SLPDownloader, DownloadProgress, ProgressCallback, AggregateProgress
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

# Retry policy for transient network failures during download. A dropped
# connection mid-stream on a multi-GB GGUF would otherwise fail the whole
# download; instead we retry with exponential backoff + jitter and resume
# from where we left off. The attempt count is sized so the cumulative
# backoff window (~1+2+4+8+16+30+30 ≈ 90s) bridges a typical 1–2 minute
# network blip — during an outage, connection attempts fail almost instantly,
# so it's the backoff, not the attempt count alone, that spans the gap.
MAX_DOWNLOAD_ATTEMPTS = 8
_RETRY_BASE_DELAY = 1.0                               # seconds; first backoff
_RETRY_MAX_DELAY = 30.0                               # seconds; backoff cap
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}   # transient HTTP statuses

# Per-operation timeout. The read timeout is deliberately short so a stalled
# stream (e.g. Wi-Fi dropped mid-download) is detected in ~1 minute and retried,
# instead of blocking on a dead socket for the whole read window. A large file
# is still allowed to take arbitrarily long overall — this bounds the idle gap
# between received bytes, not the total transfer time.
_STREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
# How many times to re-download a file whose bytes fail MD5 verification
# (silent corruption that completed without a network error).
_MAX_VERIFY_ATTEMPTS = 3

# Statuses that mean the signed URL went stale, not that the caller is unentitled.
# Entitlement is settled before the transfer starts (see ``resolve_artifacts``),
# so an auth refusal *here* is about the URL rather than the account: GCS answers
# 403 ``<Code>ExpiredToken</Code>`` for a signature past ``X-Goog-Expires`` and
# 401 for one it cannot parse. Telling the user to sign in would be a dead end —
# they already are, or did not need to be.
_STALE_URL_STATUS = {401, 403}

_UNAUTHORIZED_MSG = (
    "Download rejected (401 Unauthorized). "
    "The download URL may have expired or the resource requires authentication."
)

_T = TypeVar("_T")


class StaleDownloadURLError(ConnectionError):
    """A signed URL was rejected mid-transfer, so a fresh one may well work.

    A ``ConnectionError`` rather than a ``PermissionError`` because that is the
    action it implies: the account was already cleared before the stream opened,
    and the fix is to ask for new URLs, not new credentials.
    """


def _is_retryable_error(exc: Exception) -> bool:
    """True for transient failures worth retrying (network blips, 5xx, 429)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_STATUS
    # httpx.TransportError covers connect/read/write timeouts, connection resets,
    # and RemoteProtocolError (server closed the stream / incomplete read).
    return isinstance(exc, httpx.TransportError)


def _is_transient_error(exc: Exception) -> bool:
    """True when trying again — now or later — could plausibly succeed.

    Wider than :func:`_is_retryable_error`, which decides whether *this* call
    retries immediately. This decides how the failure is described to whoever
    asked for the download, and a stale signed URL belongs here even though
    retrying the same URL is pointless: re-requesting the download works.
    """
    if _is_retryable_error(exc):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _STALE_URL_STATUS
    return isinstance(exc, ConnectionError)


def _classified(message: str, transient: bool) -> Exception:
    """An error that reads as *message* but still says whether to try again.

    The API router turns whatever escapes the download onto an HTTP status for
    its own client (see ``_status_for``), and the only thing that status has to
    convey is whether trying again could help. Re-raising as a bare
    ``RuntimeError`` — which every layer here used to do — answered "500,
    something broke" for a dropped socket and a corrupt file alike, so a client
    could not tell "your connection died, retry" from "this will never work".
    """
    return ConnectionError(message) if transient else RuntimeError(message)


def _aggregate_error(failed: "list[tuple[str, Exception]]") -> Exception:
    """Collapse per-file failures into one exception that keeps their kind.

    Transient only when *every* file failed transiently: one unfixable file
    means the download stays broken, so promising a retry would be a lie.
    """
    message = (f"Failed to download {len(failed)} file(s): "
               f"{', '.join(name for name, _ in failed)}")
    return _classified(message, all(_is_transient_error(exc) for _, exc in failed))


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with 50–100% jitter for a 1-based attempt number."""
    delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
    return delay * (0.5 + random.random() * 0.5)

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

    def _run_with_retries(self, description: str, attempt_fn: "Callable[[int], _T]") -> "_T":
        """Call ``attempt_fn(attempt_number)`` with retries on transient errors.

        ``attempt_fn`` must be resumable/idempotent: it is re-invoked from scratch
        on each retry and is expected to pick up where the previous attempt left
        off (e.g. by inspecting bytes already written). ``PermissionError`` (401)
        and non-transient errors propagate immediately.
        """
        for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                return attempt_fn(attempt)
            except PermissionError:
                raise
            except Exception as exc:
                if not _is_retryable_error(exc) or attempt == MAX_DOWNLOAD_ATTEMPTS:
                    raise
                delay = _backoff_delay(attempt)
                self.logger.warning(
                    f"[retry] {description} failed "
                    f"(attempt {attempt}/{MAX_DOWNLOAD_ATTEMPTS}): {exc}. "
                    f"Retrying in {delay:.1f}s"
                )
                time.sleep(delay)
        # Unreachable: the loop either returns or raises on the final attempt.
        raise RuntimeError(f"{description} exhausted retries")

    def _get_remote_file_info(self, url: str) -> tuple[int, str | None, bool]:
        """HEAD request to get file size, MD5 hash, and Range support.

        Retries transient failures; falls back to ``(0, None, False)`` when the
        probe ultimately fails (callers then download single-stream, unverified).
        """
        def attempt(_attempt_no: int) -> tuple[int, str | None, bool]:
            resp = httpx.head(url, follow_redirects=True, timeout=30)
            cl = resp.headers.get("Content-Length")
            size = int(cl) if cl else 0
            md5 = self._parse_remote_md5(resp.headers)
            accept_ranges = resp.headers.get("Accept-Ranges", "").lower() == "bytes"
            return size, md5, accept_ranges

        try:
            return self._run_with_retries("HEAD probe", attempt)
        except Exception:
            return 0, None, False

    def _download_artifacts(self, artifacts: dict[str, str], local_dir: str,
                             progress_callback: ProgressCallback | None,
                             log_prefix: str, log_dir: str, label: str) -> Path:
        """Check, download, and verify a set of artifacts. Shared by bootstrap and download."""
        pending: dict[str, str] = {}
        pending_sizes: dict[str, int] = {}
        pending_range: dict[str, bool] = {}
        pending_md5: dict[str, str | None] = {}
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
            pending_md5[filename] = remote_md5

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

        # Roll the parallel per-file progress up into a single 0->100% bar so
        # the client sees one download, not the files fighting over the meter.
        aggregate = AggregateProgress(total_bytes, progress_callback, label)

        failed: "list[tuple[str, Exception]]" = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    self._download_file, filename, url, local_dir,
                    pending_sizes.get(filename, 0), position, aggregate,
                    pending_range.get(filename, False), pending_md5.get(filename),
                ): filename
                for position, (filename, url) in enumerate(pending.items())
            }

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    future.result()
                except PermissionError:
                    # A local filesystem refusal (unwritable models dir, a
                    # sandbox denial) — nothing about it improves by letting the
                    # other files finish first.
                    raise
                except Exception as e:
                    self.logger.error(f"[error] {filename}: {e}")
                    failed.append((filename, e))

        if failed:
            raise _aggregate_error(failed)

        aggregate.finish()
        self.logger.info(f"[ok] {log_prefix} downloaded to: {log_dir}")
        return Path(local_dir)

    def _download_file(self, filename: str, url: str, local_dir: str, size: int, position: int,
                        progress_callback: ProgressCallback | None = None,
                        supports_range: bool = False, remote_md5: str | None = None) -> str:
        """Download a single file, then verify and atomically publish it.

        Bytes are written to a ``.part`` sidecar so the final path only ever
        appears once the download is complete (and MD5-verified, when the server
        provides a digest). A corrupt download — bytes that finished without a
        network error but don't match the expected MD5 — is re-downloaded from
        scratch up to ``_MAX_VERIFY_ATTEMPTS`` times before giving up.
        """
        flat_name = os.path.basename(filename)
        dest = os.path.join(local_dir, flat_name)
        part = dest + ".part"
        size_mb = size / (1024 * 1024)

        for verify_attempt in range(1, _MAX_VERIFY_ATTEMPTS + 1):
            # Start each verification attempt from a clean slate so we never
            # resume onto bytes a previous corrupt attempt wrote.
            if os.path.exists(part):
                os.remove(part)

            if supports_range and size >= CHUNKED_THRESHOLD:
                chunk_size_mb = size / CHUNK_WORKERS / (1024 * 1024)
                self.logger.info(
                    f"Downloading {flat_name} ({size_mb:.1f} MB) using "
                    f"{CHUNK_WORKERS} parallel chunks of {chunk_size_mb:.1f} MB each"
                )
                self._download_file_chunked(flat_name, url, part, size, progress_callback)
            else:
                reason = "no Range support" if not supports_range else "below threshold"
                self.logger.info(f"Downloading {flat_name} ({size_mb:.1f} MB) single-stream ({reason})")
                self._download_file_simple(flat_name, url, part, size, progress_callback, supports_range)

            # No server-provided digest → can't verify; the atomic rename below
            # still guarantees we never publish a truncated file.
            if not remote_md5:
                break

            local_md5 = self._compute_local_md5(part)
            if local_md5 == remote_md5:
                break

            self.logger.warning(
                f"[verify] {flat_name} MD5 mismatch "
                f"(attempt {verify_attempt}/{_MAX_VERIFY_ATTEMPTS}): "
                f"expected {remote_md5}, got {local_md5}"
            )
            if verify_attempt == _MAX_VERIFY_ATTEMPTS:
                if os.path.exists(part):
                    os.remove(part)
                raise RuntimeError(
                    f"{flat_name}: MD5 verification failed after {_MAX_VERIFY_ATTEMPTS} attempts"
                )

        # Atomically publish: the final path appears only now, fully written.
        os.replace(part, dest)

        if progress_callback:
            progress_callback(DownloadProgress(
                filename=flat_name, downloaded=size, total=size, status="completed",
            ))

        return filename

    def _download_file_simple(self, flat_name: str, url: str, dest: str, size: int,
                               progress_callback: ProgressCallback | None = None,
                               supports_range: bool = False) -> None:
        """Download a file in a single request, retrying transient failures.

        When the server supports Range requests, a retry resumes from the bytes
        already written rather than restarting from zero. Otherwise each retry
        re-downloads the whole file.
        """
        def attempt(attempt_no: int) -> None:
            # Resume only from bytes written by a PRIOR attempt in this call — the
            # first attempt always starts clean, so a pre-existing corrupt file
            # (which is why it's being re-downloaded) is overwritten, not trusted.
            resume_from = 0
            if attempt_no > 1 and supports_range and os.path.exists(dest):
                resume_from = os.path.getsize(dest)
                if size and resume_from >= size:
                    return  # this attempt's predecessor already finished the file
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=_STREAM_TIMEOUT) as resp:
                if resp.status_code == 401:
                    raise StaleDownloadURLError(_UNAUTHORIZED_MSG)
                # If we asked to resume but the server ignored it (sent 200, not
                # 206), it's streaming the whole file — rewrite from the start.
                if resume_from and resp.status_code != 206:
                    resume_from = 0
                resp.raise_for_status()
                downloaded = resume_from
                with open(dest, "ab" if resume_from else "wb") as f:
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

        self._run_with_retries(f"download {flat_name}", attempt)

    def _download_chunk(self, url: str, dest: str, start: int, end: int,
                        flat_name: str, total_size: int,
                        downloaded_counter: list, lock: threading.Lock,
                        progress_callback: ProgressCallback | None = None) -> None:
        """Download a byte range to its file offset, retrying transient failures.

        A retry resumes from within the range: the file is pre-allocated (so its
        on-disk size can't tell us how far we got), so we track the bytes written
        for this chunk locally and resume from ``start + written``. The shared
        progress counter only ever advances by newly written bytes, so a retry
        never double-counts.
        """
        written = 0  # bytes of THIS chunk already flushed to disk

        def attempt(_attempt_no: int) -> None:
            nonlocal written
            cur_start = start + written
            if cur_start > end:
                return  # chunk already fully written by a prior attempt
            headers = {"Range": f"bytes={cur_start}-{end}"}
            with httpx.stream("GET", url, headers=headers, follow_redirects=True,
                              timeout=_STREAM_TIMEOUT) as resp:
                if resp.status_code == 401:
                    raise StaleDownloadURLError(_UNAUTHORIZED_MSG)
                resp.raise_for_status()
                with open(dest, "r+b") as f:
                    f.seek(cur_start)
                    for chunk in resp.iter_bytes(chunk_size=RANGE_READ_SIZE):
                        f.write(chunk)
                        written += len(chunk)
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

        self._run_with_retries(f"chunk {start}-{end} of {flat_name}", attempt)

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
                    # Keep the cause's kind: every model worth downloading is
                    # over CHUNKED_THRESHOLD, so this is the path a real dropped
                    # connection takes, and flattening it to RuntimeError here
                    # would undo the classification the outer layers preserve.
                    raise _classified(
                        f"Chunk download failed for {flat_name} "
                        f"(bytes {chunk_range[0]}-{chunk_range[1]}): {e}",
                        _is_transient_error(e),
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
            # The bootstrap endpoint publishes upstream names, which can lag a
            # local rename — match on the published name for the request.
            wanted = platform_model_name(model_name)
            target = next((m for m in models if m.get("name") == wanted), None)
            if not target:
                available = ", ".join(m.get("name", "?") for m in models)
                raise ValueError(
                    f"Model '{display_model_name(model_name)}' not found in bootstrap. "
                    f"Available: {available}"
                )
        else:
            target = models[0]

        published_name = target.get("name", "unknown")
        resolved_name = canonical_model_name(published_name)
        model_id = self._resolve_model_id(resolved_name)
        artifacts = self._bootstrap_source.get_artifacts(published_name)

        artifacts = _select_gguf_artifacts(artifacts)

        if not artifacts:
            raise ValueError(f"No GGUF artifact found for bootstrap model '{resolved_name}'")

        local_dir = os.path.join(self.workspace.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        self.logger.info(f"Bootstrap downloading model: {display_model_name(resolved_name)}")

        return self._download_artifacts(artifacts, local_dir, progress_callback,
                                        log_prefix="Bootstrap model", log_dir=local_dir,
                                        label=display_model_name(resolved_name))

    @staticmethod
    def _resolve_model_id(model_name: str) -> str:
        """Lookup the HuggingFace model_id from a short model name.

        E.g. "sl-mini" → "smartloop-ai/slp-orchestra-mini".  Legacy short names
        resolve too (``SUPPORTED_MODELS`` canonicalizes its keys).  Falls back to
        model_name if no mapping is found.
        """
        from smartloop.config import SUPPORTED_MODELS
        model_type = SUPPORTED_MODELS.get(model_name)
        if model_type:
            return model_type.model_id()
        return model_name

    @staticmethod
    def _resolve_model_name(model_id: str) -> str:
        """Reverse-lookup the canonical short model name from a HuggingFace model_id.

        E.g. "smartloop-ai/slp-orchestra-mini" → "sl-mini".  Returns the
        canonical name, NOT the name the platform publishes — callers hitting the
        API must pass it through :func:`platform_model_name` first.  Falls back
        to model_id if no mapping is found.
        """
        from smartloop.config import SUPPORTED_MODELS
        for name, model_type in SUPPORTED_MODELS.items():
            if model_type.model_id() == model_id:
                return name
        return model_id

    def resolve_artifacts(self, model_id: str) -> dict[str, str]:
        """Ask the platform for this model's signed GGUF URLs.

        This is the step that decides entitlement, and it is separated from
        :meth:`download` so a caller can settle "may this account have these
        weights?" *before* committing to a long-running transfer — the API
        router runs it ahead of opening its progress stream, so a refusal is an
        HTTP status rather than an error buried in a stream that already
        returned 200.

        The URLs are minted per call and the route is rate-limited, so the
        result is passed on to :meth:`download` rather than being requested
        twice.  Raises ``PermissionError`` when the account is not entitled and
        ``ConnectionError`` for transient failures (see
        :meth:`SLPPlatformClient.get_model_artifacts`).
        """
        model_name = self._resolve_model_name(model_id)
        # The catalog is keyed by the name the platform publishes, which can lag
        # a local rename (sl-mini is still served as slp-orchestra-mini).
        artifacts = self._model_source.get_artifacts(platform_model_name(model_name))
        return _select_gguf_artifacts(artifacts)

    def download(self, model_id: str,
                 progress_callback: ProgressCallback | None = None,
                 artifacts: dict[str, str] | None = None) -> Path:
        """Download GGUF model via the /v1/models/{model_name}/download endpoint.

        Accepts the HuggingFace-style *model_id* (e.g. "google/gemma-3-1b-it")
        to match the base-class interface, resolves it to the short model name
        for the API call, and stores files under the model_id path.

        *artifacts* accepts an already-resolved map from
        :meth:`resolve_artifacts`; callers that have one pass it so the signed
        URLs are minted once.  Omitting it resolves them here, which is what the
        CLI and TUI do.

        No credential check runs here: free models are downloadable anonymously,
        so gating on a token locally would deny a download the platform would
        have served.  The platform is the only authority on access — it answers
        403 (``ModelAccessError``) for a model the caller's plan doesn't grant.
        """
        model_name = self._resolve_model_name(model_id)

        self.logger.info(f"Downloading model: {display_model_name(model_name)}")
        print()

        if artifacts is None:
            artifacts = self.resolve_artifacts(model_id)

        # After the entitlement call, not before: a refused download should not
        # leave an empty model directory behind for the workspace to trip over.
        local_dir = os.path.join(self.workspace.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        if not artifacts:
            self.logger.info(f"[ok] No GGUF files to download")
            return Path(local_dir)

        return self._download_artifacts(artifacts, local_dir, progress_callback,
                                        log_prefix="Model", log_dir=local_dir,
                                        label=display_model_name(model_name))
