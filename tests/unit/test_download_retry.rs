//! Port of `tests/unit/test_download_retry.py`.
//!
//! Tests for retry/resume behavior in the artifact downloader.  Backoff is
//! instant (the `_no_sleep` fixture: `CancelToken::set_instant_wait`), so the
//! ~90s retry window costs nothing while a cancelled token still reports
//! cancelled.

use std::sync::Arc;
use std::time::Duration;

use md5::{Digest, Md5};
use parking_lot::Mutex;
use smartloop::model::downloader::artifact_downloader::{
    self as ad, DownloadError, HttpError, MAX_DOWNLOAD_ATTEMPTS, STREAM_TIMEOUT, StaleDownloadURLError, TransportErrorKind,
    backoff_delay, is_retryable_error,
};
use smartloop::model::downloader::slp_downloader::ProgressCallback;

use crate::shared::model_fakes::{
    FakeResp, ScriptedHttp, connect_error, downloader, range_header, read_error, read_timeout, remote_protocol_error,
    status_error,
};

fn md5_hex(data: &[u8]) -> String {
    hex::encode(Md5::digest(data))
}

// ------------------------------------------------------------------ Pure helpers

/// `TestRetryClassification`
mod test_retry_classification {
    use super::*;

    #[test]
    fn test_transport_errors_are_retryable() {
        assert!(is_retryable_error(&read_error("boom").into()));
        assert!(is_retryable_error(&connect_error("boom").into()));
        assert!(is_retryable_error(&remote_protocol_error("boom").into()));
        assert!(is_retryable_error(&read_timeout("boom").into()));
    }

    #[test]
    fn test_transient_statuses_are_retryable() {
        for code in [408, 429, 500, 502, 503, 504] {
            assert!(is_retryable_error(&status_error(code).into()), "{code}");
        }
    }

    #[test]
    fn test_client_errors_are_not_retryable() {
        for code in [400, 403, 404] {
            assert!(!is_retryable_error(&status_error(code).into()), "{code}");
        }
        assert!(!is_retryable_error(&DownloadError::Value("nope".to_string())));
    }

    #[test]
    fn test_backoff_grows_and_is_capped_and_jittered() {
        // Jitter keeps each delay within (0.5, 1.0] of the exponential base.
        for attempt in 1..8u32 {
            let base = (ad::RETRY_BASE_DELAY * 2f64.powi(attempt as i32 - 1)).min(ad::RETRY_MAX_DELAY);
            let d = backoff_delay(attempt);
            assert!(0.5 * base <= d && d <= base, "attempt {attempt}: {d} not in [{}, {base}]", 0.5 * base);
        }
        assert!(backoff_delay(20) <= ad::RETRY_MAX_DELAY);
    }
}

/// `TestRunWithRetries`
mod test_run_with_retries {
    use super::*;

    #[test]
    fn test_returns_on_first_success() {
        let dl = downloader(None, None);
        let mut calls = Vec::new();
        let result = dl
            .run_with_retries("op", &mut |n| {
                calls.push(n);
                Ok("ok")
            })
            .unwrap();
        assert_eq!(result, "ok");
        assert_eq!(calls, vec![1]);
    }

    #[test]
    fn test_retries_transient_then_succeeds() {
        let dl = downloader(None, None);
        let mut calls = Vec::new();
        let result = dl
            .run_with_retries("op", &mut |n| {
                calls.push(n);
                if n < 3 {
                    return Err(read_error("drop").into());
                }
                Ok("done")
            })
            .unwrap();
        assert_eq!(result, "done");
        assert_eq!(calls, vec![1, 2, 3]);
    }

    #[test]
    fn test_non_retryable_raises_immediately() {
        let dl = downloader(None, None);
        let mut calls = Vec::new();
        let err = dl
            .run_with_retries::<()>("op", &mut |n| {
                calls.push(n);
                Err(DownloadError::Value("fatal".to_string()))
            })
            .unwrap_err();
        assert!(matches!(err, DownloadError::Value(_)));
        assert_eq!(calls, vec![1]);
    }

    #[test]
    fn test_permission_error_not_retried() {
        let dl = downloader(None, None);
        let mut calls = Vec::new();
        let err = dl
            .run_with_retries::<()>("op", &mut |n| {
                calls.push(n);
                Err(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "401").into())
            })
            .unwrap_err();
        assert!(err.is_permission_error());
        assert_eq!(calls, vec![1]);
    }

    #[test]
    fn test_exhausts_attempts_then_raises() {
        let dl = downloader(None, None);
        let mut calls = Vec::new();
        let err = dl
            .run_with_retries::<()>("op", &mut |n| {
                calls.push(n);
                Err(connect_error("down").into())
            })
            .unwrap_err();
        assert!(matches!(err, DownloadError::Http(HttpError::Transport { kind: TransportErrorKind::Connect, .. })));
        assert_eq!(calls, (1..=MAX_DOWNLOAD_ATTEMPTS).collect::<Vec<_>>());
    }
}

// ------------------------------------------------- Simple (single-stream) download

/// `TestSimpleResume`
mod test_simple_resume {
    use super::*;

    #[test]
    fn test_resume_after_midstream_drop() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        // First attempt: 200, writes b"ABCD" then the stream drops.
        let first = FakeResp::new(200, &[b"ABCD", b"NEVER"]).failing_at(1, read_error("drop"));
        // Second attempt: 206 (Range honored), delivers the remainder.
        let second = FakeResp::new(206, &[b"EFGH"]);
        let http = ScriptedHttp::new(vec![first, second]);

        let dl = downloader(None, Some(Arc::clone(&http)));
        dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, true).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
        // The retry asked to resume from byte 4.
        let calls = http.header_calls.lock();
        assert!(calls[0].is_empty());
        assert_eq!(calls[1], range_header("bytes=4-"));
    }

    #[test]
    fn test_server_ignores_range_restarts_clean() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        let first = FakeResp::new(200, &[b"ABCD", b"NEVER"]).failing_at(1, read_error("drop"));
        // Server ignores Range on retry (200, full body) -> must overwrite, not append.
        let second = FakeResp::new(200, &[b"ABCDEFGH"]);
        let http = ScriptedHttp::new(vec![first, second]);

        let dl = downloader(None, Some(http));
        dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, true).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
    }

    /// A 401 on the transfer is a stale URL, not a credentials problem.
    ///
    /// Entitlement was settled before the transfer began, so this cannot mean
    /// "you are not allowed" — the signed URL went past its expiry.  It is a
    /// connection-class error so the caller offers a retry rather than a
    /// sign-in prompt, and it still short-circuits the retry loop, because
    /// re-fetching the same dead URL eight times helps nobody.
    #[test]
    fn test_401_raises_without_retry() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        let http = ScriptedHttp::new(vec![FakeResp::new(401, &[])]);

        let dl = downloader(None, Some(Arc::clone(&http)));
        let err = dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, true).unwrap_err();
        assert!(matches!(err, DownloadError::StaleUrl(StaleDownloadURLError(_))));
        assert!(err.is_connection_error());
        assert!(!err.is_permission_error());
        assert_eq!(http.calls(), 1);
    }

    /// A stalled stream (Wi-Fi drop) surfaces as ReadTimeout and is retried.
    #[test]
    fn test_recovers_from_midstream_read_timeout() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        let first = FakeResp::new(200, &[b"ABCD", b"NEVER"]).failing_at(1, read_timeout("stalled"));
        let second = FakeResp::new(206, &[b"EFGH"]);
        let http = ScriptedHttp::new(vec![first, second]);

        let dl = downloader(None, Some(http));
        dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, true).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
    }

    /// Streaming uses a bounded read timeout so stalls are detected quickly.
    #[test]
    fn test_uses_short_read_timeout() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        let http = ScriptedHttp::new(vec![FakeResp::new(200, &[b"ABCDEFGH"])]);

        let dl = downloader(None, Some(Arc::clone(&http)));
        dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, false).unwrap();
        let timeout = http.timeout_calls.lock()[0];
        assert_eq!(timeout, STREAM_TIMEOUT);
        assert!(timeout.read <= Duration::from_secs(120));
    }
}

// --------------------------------------------- Chunked (parallel Range) download

/// `TestChunkResume`
mod test_chunk_resume {
    use super::*;

    #[test]
    fn test_chunk_resumes_and_counts_once() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("big.bin");
        std::fs::write(&dest, [0u8; 8]).unwrap(); // pre-allocated, as the chunked path does

        let first = FakeResp::new(206, &[b"ABCD", b"NEVER"]).failing_at(1, read_error("drop"));
        let second = FakeResp::new(206, &[b"EFGH"]);
        let http = ScriptedHttp::new(vec![first, second]);

        let counter = Mutex::new(0u64);
        let noop: ProgressCallback = Arc::new(|_| {});
        let dl = downloader(None, Some(Arc::clone(&http)));
        dl.download_chunk("http://x/file", &dest, 0, 7, "big.bin", 8, &counter, Some(noop)).unwrap();

        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
        // Progress counted only the 8 real bytes, not the dropped retry bytes.
        assert_eq!(*counter.lock(), 8);
        // Retry resumed from the correct offset within the chunk.
        let calls = http.header_calls.lock();
        assert_eq!(calls[0], range_header("bytes=0-7"));
        assert_eq!(calls[1], range_header("bytes=4-7"));
    }
}

/// A range response is used only when it is exactly the bytes asked for.
mod test_range_responses_are_checked {
    use super::*;
    use smartloop::model::downloader::artifact_downloader::{SourceChanged, is_transient_error, parse_content_range};

    fn preallocated(dir: &std::path::Path) -> std::path::PathBuf {
        let dest = dir.join("big.bin");
        std::fs::write(&dest, [0u8; 8]).unwrap();
        dest
    }

    fn fetch(dest: &std::path::Path, responses: Vec<FakeResp>) -> (Arc<ScriptedHttp>, Result<(), DownloadError>) {
        let http = ScriptedHttp::new(responses);
        let dl = downloader(None, Some(Arc::clone(&http)));
        let counter = Mutex::new(0u64);
        let result = dl.download_chunk("http://x/file", dest, 0, 7, "big.bin", 8, &counter, None);
        (http, result)
    }

    #[test]
    fn test_a_whole_object_answer_is_rejected_before_any_write() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let (http, result) = fetch(&dest, vec![FakeResp::new(200, &[b"ABCDEFGH"])]);
        let err = result.unwrap_err();
        assert!(matches!(err, DownloadError::SourceChanged(SourceChanged(_))));
        assert!(is_transient_error(&err));
        assert_eq!(http.calls(), 1, "not retried in place");
        assert_eq!(std::fs::read(&dest).unwrap(), [0u8; 8]);
    }

    #[test]
    fn test_a_content_range_for_other_bytes_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let resp = FakeResp::new(206, &[b"ABCDEFGH"]).with_header("Content-Range", "bytes 8-15/16");
        let (_http, result) = fetch(&dest, vec![resp]);
        assert!(matches!(result.unwrap_err(), DownloadError::SourceChanged(_)));
        assert_eq!(std::fs::read(&dest).unwrap(), [0u8; 8]);
    }

    #[test]
    fn test_a_content_range_for_a_different_size_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let resp = FakeResp::new(206, &[b"ABCDEFGH"]).with_header("Content-Range", "bytes 0-7/9");
        let (_http, result) = fetch(&dest, vec![resp]);
        assert!(matches!(result.unwrap_err(), DownloadError::SourceChanged(_)));
    }

    #[test]
    fn test_a_matching_content_range_is_accepted() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let resp = FakeResp::new(206, &[b"ABCDEFGH"]).with_header("content-range", "bytes 0-7/8");
        let (_http, result) = fetch(&dest, vec![resp]);
        result.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
    }

    #[test]
    fn test_a_short_body_resumes_where_it_stopped() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let (http, result) =
            fetch(&dest, vec![FakeResp::new(206, &[b"ABCD"]), FakeResp::new(206, &[b"EFGH"])]);
        result.unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
        let calls = http.header_calls.lock();
        assert_eq!(calls[1], range_header("bytes=4-7"));
    }

    #[test]
    fn test_a_short_body_that_never_completes_fails_retryably() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = preallocated(tmp.path());
        let short: Vec<FakeResp> = (0..MAX_DOWNLOAD_ATTEMPTS).map(|_| FakeResp::new(206, &[])).collect();
        let (_http, result) = fetch(&dest, short);
        assert!(is_retryable_error(&result.unwrap_err()));
    }

    #[test]
    fn test_bytes_past_the_range_are_not_written() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("big.bin");
        std::fs::write(&dest, [0u8; 12]).unwrap();
        let http = ScriptedHttp::new(vec![FakeResp::new(206, &[b"ABCDEFGHIJKL"])]);
        let dl = downloader(None, Some(http));
        let counter = Mutex::new(0u64);
        dl.download_chunk("http://x/file", &dest, 0, 7, "big.bin", 12, &counter, None).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH\0\0\0\0");
    }

    #[test]
    fn test_a_simple_download_that_ends_short_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let dest = tmp.path().join("file.bin");
        let http = ScriptedHttp::new(vec![FakeResp::new(200, &[b"ABCD"]), FakeResp::new(206, &[b"EFGH"])]);
        let dl = downloader(None, Some(Arc::clone(&http)));
        dl.download_file_simple("file.bin", "http://x/file", &dest, 8, None, true).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), b"ABCDEFGH");
        assert_eq!(http.header_calls.lock()[1], range_header("bytes=4-"));
    }

    #[test]
    fn test_content_range_parsing() {
        assert_eq!(parse_content_range("bytes 0-7/8"), Some((0, 7, Some(8))));
        assert_eq!(parse_content_range("bytes 5-9/*"), Some((5, 9, None)));
        assert_eq!(parse_content_range("bytes */8"), None);
        assert_eq!(parse_content_range("bytes 9-5/10"), None);
    }
}

// ------------------- MD5 verification + atomic (.part) publish, via download_file

/// `TestVerifyAndAtomicPublish`
mod test_verify_and_atomic_publish {
    use super::*;

    fn run(dir: &std::path::Path, responses: Vec<FakeResp>, remote_md5: Option<&str>) -> (Arc<ScriptedHttp>, Result<String, DownloadError>) {
        let http = ScriptedHttp::new(responses);
        let dl = downloader(None, Some(Arc::clone(&http)));
        // size below CHUNKED_THRESHOLD + supports_range=false -> simple path,
        // one stream call per verification attempt.
        let result = dl.download_file("model.gguf", "http://x/f", dir, 8, 0, None, false, remote_md5);
        (http, result)
    }

    #[test]
    fn test_verified_download_published_atomically() {
        let tmp = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGH";
        let (http, result) = run(tmp.path(), vec![FakeResp::new(200, &[data])], Some(&md5_hex(data)));
        result.unwrap();
        assert_eq!(std::fs::read(tmp.path().join("model.gguf")).unwrap(), data);
        assert!(!tmp.path().join("model.gguf.part").exists());
        assert_eq!(http.calls(), 1); // no re-download needed
    }

    #[test]
    fn test_mismatch_redownloads_then_succeeds() {
        let tmp = tempfile::tempdir().unwrap();
        let good = b"ABCDEFGH";
        let (http, result) = run(
            tmp.path(),
            vec![
                FakeResp::new(200, &[b"XXXXXXXX"]), // corrupt first
                FakeResp::new(200, &[good]),        // clean on re-download
            ],
            Some(&md5_hex(good)),
        );
        result.unwrap();
        assert_eq!(std::fs::read(tmp.path().join("model.gguf")).unwrap(), good);
        assert!(!tmp.path().join("model.gguf.part").exists());
        assert_eq!(http.calls(), 2);
    }

    #[test]
    fn test_persistent_mismatch_raises_and_leaves_nothing() {
        let tmp = tempfile::tempdir().unwrap();
        let corrupt: Vec<FakeResp> =
            (0..ad::MAX_VERIFY_ATTEMPTS).map(|_| FakeResp::new(200, &[b"XXXXXXXX"])).collect();
        let (_http, result) = run(tmp.path(), corrupt, Some(&md5_hex(b"ABCDEFGH")));
        let err = result.unwrap_err();
        assert!(matches!(err, DownloadError::Runtime { .. }));
        assert!(err.to_string().contains("MD5 verification failed"));
        // Neither the final file nor the partial sidecar survive a hard failure.
        assert!(!tmp.path().join("model.gguf").exists());
        assert!(!tmp.path().join("model.gguf.part").exists());
    }

    #[test]
    fn test_no_remote_md5_skips_verify_but_still_atomic() {
        let tmp = tempfile::tempdir().unwrap();
        let data = b"ABCDEFGH";
        let (http, result) = run(tmp.path(), vec![FakeResp::new(200, &[data])], None);
        result.unwrap();
        assert_eq!(std::fs::read(tmp.path().join("model.gguf")).unwrap(), data);
        assert!(!tmp.path().join("model.gguf.part").exists());
        assert_eq!(http.calls(), 1); // verification skipped, no re-download
    }
}

/// `TestChunkedFailureKeepsItsKind`
///
/// The chunked path is the one production actually takes.  CHUNKED_THRESHOLD
/// is 100 MB and every shipped model is larger, so a real dropped connection
/// always arrives through here.  It used to re-raise every chunk failure as a
/// bare runtime error, which threw away the classification the outer layers
/// were careful to preserve.
mod test_chunked_failure_keeps_its_kind {
    use super::*;

    fn chunked(exc: fn() -> DownloadError) -> DownloadError {
        let tmp = tempfile::tempdir().unwrap();
        let dl = downloader(None, None);
        dl.set_chunk_workers(Some(1));
        dl.set_download_chunk_hook(Some(Arc::new(move |_start, _end| Err(exc()))));
        dl.download_file_chunked("file.bin", "http://x/file", &tmp.path().join("file.bin"), 1024, None)
            .unwrap_err()
    }

    #[test]
    fn test_dropped_connection_stays_a_connection_error() {
        let err = chunked(|| connect_error("dropped").into());
        assert!(err.is_connection_error());
        assert!(err.to_string().contains("Chunk download failed"));
    }

    #[test]
    fn test_stale_signed_url_stays_a_connection_error() {
        let err = chunked(|| StaleDownloadURLError("expired".to_string()).into());
        assert!(err.is_connection_error());
    }

    #[test]
    fn test_permanent_failure_stays_a_runtime_error() {
        let err = chunked(|| DownloadError::Value("corrupt".to_string()));
        assert!(matches!(err, DownloadError::Runtime { .. }));
        assert!(!err.is_connection_error());
    }
}

/// A large file is fetched in small ranges from a shared queue, so every
/// connection stays busy to the end instead of the slowest one finishing
/// the last slice alone.
mod test_ranges_keep_every_connection_busy {
    use super::*;
    use smartloop::model::downloader::artifact_downloader::{segments, SEGMENT_SIZE};

    #[test]
    fn test_a_large_file_is_cut_into_segment_sized_ranges() {
        let size = 16_810_714_496u64; // Qwen3.8-27B Q4_K_M
        let ranges = segments(size, 5);
        assert_eq!(ranges.len() as u64, size.div_ceil(SEGMENT_SIZE));
        assert!(ranges.iter().all(|(s, e)| e - s < SEGMENT_SIZE));
        // Contiguous and complete.
        assert_eq!(ranges.first().unwrap().0, 0);
        assert_eq!(ranges.last().unwrap().1, size - 1);
        assert!(ranges.windows(2).all(|w| w[1].0 == w[0].1 + 1));
    }

    #[test]
    fn test_a_small_file_still_gives_every_connection_a_range() {
        let ranges = segments(120 * 1024 * 1024, 5);
        assert_eq!(ranges.len(), 5);
        assert_eq!(ranges.last().unwrap().1, 120 * 1024 * 1024 - 1);
    }

    #[test]
    fn test_every_range_is_fetched_once() {
        let tmp = tempfile::tempdir().unwrap();
        let dl = downloader(None, None);
        dl.set_chunk_workers(Some(3));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = seen.clone();
        dl.set_download_chunk_hook(Some(Arc::new(move |start, end| {
            record.lock().push((start, end));
            Ok(())
        })));
        let size = 3 * SEGMENT_SIZE + 10;
        dl.download_file_chunked("file.bin", "http://x/file", &tmp.path().join("file.bin"), size, None).unwrap();
        let mut seen = seen.lock().clone();
        seen.sort();
        assert_eq!(seen, segments(size, 3));
    }

    #[test]
    fn test_a_failed_range_stops_the_rest_of_the_queue() {
        let tmp = tempfile::tempdir().unwrap();
        let dl = downloader(None, None);
        dl.set_chunk_workers(Some(1));
        let calls = Arc::new(Mutex::new(0usize));
        let count = calls.clone();
        dl.set_download_chunk_hook(Some(Arc::new(move |_start, _end| {
            *count.lock() += 1;
            Err(DownloadError::Value("corrupt".to_string()))
        })));
        let size = 10 * SEGMENT_SIZE;
        assert!(dl.download_file_chunked("file.bin", "http://x/file", &tmp.path().join("file.bin"), size, None).is_err());
        assert_eq!(*calls.lock(), 1, "the other nine ranges aren't fetched");
    }
}
