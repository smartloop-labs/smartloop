//! Port of `tests/unit/test_download_cancel.py`.
//!
//! A download stops when asked, and says so in a way nothing mistakes.
//!
//! Two properties are under test throughout.  The transfer must notice the
//! flag promptly — at the next chunk boundary, not at the end of the file and
//! not after a 30s backoff.  And the cancel must survive every layer that
//! classifies failures on its way out: retried, aggregated with a sibling's
//! dropped socket, or rewrapped by the chunked path, it would reach the client
//! as a 502 for a button the user pressed.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use md5::{Digest, Md5};
use parking_lot::Mutex;
use smartloop::model::downloader::artifact_downloader::{
    self as ad, DownloadError, HttpError, MAX_DOWNLOAD_ATTEMPTS, SLPArtifactDownloader, TransportErrorKind,
    aggregate_error, is_retryable_error, is_transient_error,
};
use smartloop::model::downloader::cancellation::{CancelToken, DownloadCancelled};

use crate::shared::model_fakes::{FakeResp, ScriptedHttp, connect_error, downloader, read_error};

fn cancelled_token() -> CancelToken {
    let token = CancelToken::new();
    token.cancel();
    token
}

// ------------------------------------------------------------------ //
// Classification — a cancel must never look like a network problem
// ------------------------------------------------------------------ //

/// `TestClassification`
mod test_classification {
    use super::*;

    #[test]
    fn test_a_cancel_is_not_retryable() {
        assert!(!is_retryable_error(&DownloadCancelled::new("stop").into()));
    }

    /// Transient would have `status_for` answer 502 for a button press.
    #[test]
    fn test_a_cancel_is_not_transient() {
        assert!(!is_transient_error(&DownloadCancelled::new("stop").into()));
    }

    #[test]
    fn test_it_carries_its_own_status() {
        assert_eq!(DownloadCancelled::new("stop").status_code(), 499);
        let err: DownloadError = DownloadCancelled::new("stop").into();
        assert_eq!(err.status_code(), Some(499));
    }

    #[test]
    fn test_it_is_not_a_connection_error() {
        // is_transient_error ends in "is it a connection error", so this
        // classification choice is what keeps the one above true.
        let err: DownloadError = DownloadCancelled::new("stop").into();
        assert!(!err.is_connection_error());
        assert!(!matches!(err, DownloadError::Io(_)));
        assert!(!err.is_permission_error());
    }

    #[test]
    fn test_aggregate_surfaces_the_cancel_over_a_transient_sibling() {
        let cancel = DownloadCancelled::new("stop");
        let result = aggregate_error(vec![
            ("weights.gguf".to_string(), read_error("dropped").into()),
            ("mmproj.gguf".to_string(), cancel.clone().into()),
        ]);
        assert!(matches!(result, DownloadError::Cancelled(ref c) if *c == cancel));
    }

    #[test]
    fn test_aggregate_surfaces_the_cancel_whichever_order_it_lands_in() {
        let cancel = DownloadCancelled::new("stop");
        let result = aggregate_error(vec![
            ("mmproj.gguf".to_string(), cancel.clone().into()),
            ("weights.gguf".to_string(), read_error("dropped").into()),
        ]);
        assert!(matches!(result, DownloadError::Cancelled(ref c) if *c == cancel));
    }

    #[test]
    fn test_without_a_cancel_the_old_aggregation_stands() {
        let result = aggregate_error(vec![("weights.gguf".to_string(), read_error("dropped").into())]);
        assert!(result.is_connection_error());
        assert!(result.to_string().contains("weights.gguf"));
    }
}

// ------------------------------------------------------------------ //
// The retry loop
// ------------------------------------------------------------------ //

/// `TestTheRetryLoop`
mod test_the_retry_loop {
    use super::*;

    #[test]
    fn test_a_cancel_is_never_retried() {
        let dl = downloader(None, None);
        let mut attempts = Vec::new();

        let err = dl
            .run_with_retries::<()>("thing", &mut |n| {
                attempts.push(n);
                Err(DownloadCancelled::new("stop").into())
            })
            .unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(attempts, vec![1]);
    }

    #[test]
    fn test_a_token_already_set_stops_before_the_first_attempt() {
        let dl = downloader(Some(cancelled_token()), None);
        let mut attempts = Vec::new();

        let err = dl
            .run_with_retries::<()>("thing", &mut |n| {
                attempts.push(n);
                Ok(())
            })
            .unwrap_err();
        assert!(err.is_cancelled());
        assert!(attempts.is_empty());
    }

    /// Backoff reaches 30s, and a stop request must not wait it out.
    #[test]
    fn test_a_cancel_during_backoff_ends_it_immediately() {
        let token = CancelToken::new();
        let dl = downloader(Some(token.clone()), None);
        let mut attempts = Vec::new();

        let err = dl
            .run_with_retries::<()>("thing", &mut |n| {
                attempts.push(n);
                if n == 1 {
                    token.cancel(); // arrives while the first backoff runs
                }
                Err(read_error("dropped").into())
            })
            .unwrap_err();
        assert!(err.is_cancelled());
        // One real attempt, then the backoff returned cancelled instead of
        // sleeping and re-entering.
        assert_eq!(attempts, vec![1]);
        assert!((attempts.len() as u32) < MAX_DOWNLOAD_ATTEMPTS);
    }

    #[test]
    fn test_the_cancel_keeps_the_transient_failure_as_its_cause() {
        let token = CancelToken::new();
        let dl = downloader(Some(token.clone()), None);

        let err = dl
            .run_with_retries::<()>("thing", &mut |_n| {
                token.cancel();
                Err(read_error("dropped").into())
            })
            .unwrap_err();
        assert!(err.is_cancelled());
        assert!(matches!(
            err.cause(),
            Some(DownloadError::Http(HttpError::Transport { kind: TransportErrorKind::Read, .. }))
        ));
    }
}

// ------------------------------------------------------------------ //
// The byte loops
// ------------------------------------------------------------------ //

/// `TestTheSimplePath`
mod test_the_simple_path {
    use super::*;

    #[test]
    fn test_it_stops_at_the_next_chunk() {
        let tmp = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let resp = FakeResp::cancelling(&[&[b'a'; 8], &[b'b'; 8], &[b'c'; 8]], &token, 1);
        let dl = downloader(Some(token.clone()), Some(ScriptedHttp::new(vec![resp])));

        let dest = tmp.path().join("weights.gguf.part");
        let err = dl.download_file_simple("weights.gguf", "https://x/f", &dest, 24, None, false).unwrap_err();
        assert!(err.is_cancelled());

        // Checked before the write, so the chunk in hand when the flag went up
        // is dropped rather than flushed — the file stops at the last one that
        // was already committed.
        assert_eq!(std::fs::read(&dest).unwrap(), vec![b'a'; 8]);
    }

    /// Returning from inside the stream is what drops the connection.
    ///
    /// Leaving it open would leave it to the 60s read timeout.
    #[test]
    fn test_the_socket_is_closed_on_the_way_out() {
        let tmp = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let resp = FakeResp::cancelling(&[&[b'a'; 8], &[b'b'; 8]], &token, 0);
        let closed = resp.closed_flag();
        let dl = downloader(Some(token.clone()), Some(ScriptedHttp::new(vec![resp])));

        let err = dl
            .download_file_simple("w.gguf", "https://x/f", &tmp.path().join("w.part"), 16, None, false)
            .unwrap_err();
        assert!(err.is_cancelled());
        assert!(closed.load(Ordering::SeqCst));
    }

    #[test]
    fn test_it_does_not_reconnect_after_a_cancel() {
        let tmp = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let stream = ScriptedHttp::new(vec![FakeResp::cancelling(&[&[b'a'; 8], &[b'b'; 8]], &token, 0)]);
        let dl = downloader(Some(token.clone()), Some(Arc::clone(&stream)));

        let err = dl
            .download_file_simple("w.gguf", "https://x/f", &tmp.path().join("w.part"), 16, None, false)
            .unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(stream.calls(), 1);
    }
}

/// `TestTheChunkedPath`
mod test_the_chunked_path {
    use super::*;

    #[test]
    fn test_a_range_stops_at_the_next_read() {
        let tmp = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let dest = tmp.path().join("w.part");
        std::fs::write(&dest, [0u8; 24]).unwrap();

        let resp = FakeResp::cancelling(&[&[b'a'; 8], &[b'b'; 8], &[b'c'; 8]], &token, 1).with_status(206);
        let dl = downloader(Some(token.clone()), Some(ScriptedHttp::new(vec![resp])));

        let counter = Mutex::new(0u64);
        let err = dl.download_chunk("https://x/f", &dest, 0, 23, "w.gguf", 24, &counter, None).unwrap_err();
        assert!(err.is_cancelled());
        let mut expected = vec![b'a'; 8];
        expected.extend_from_slice(&[0u8; 16]);
        assert_eq!(std::fs::read(&dest).unwrap(), expected);
    }

    /// Every model worth downloading takes this path, so its pool is where a
    /// cancel usually lands — the workers not yet started must exit free.
    #[test]
    fn test_a_queued_range_never_opens_a_socket() {
        let tmp = tempfile::tempdir().unwrap();
        let stream = ScriptedHttp::new(Vec::new());
        let dl = downloader(Some(cancelled_token()), Some(Arc::clone(&stream)));

        let counter = Mutex::new(0u64);
        let err = dl
            .download_chunk("https://x/f", &tmp.path().join("w.part"), 0, 7, "w.gguf", 8, &counter, None)
            .unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(stream.calls(), 0);
    }

    /// `classified` would turn it into a connection error and lose 499.
    #[test]
    fn test_the_pool_wrapper_does_not_rewrap_a_cancel() {
        let tmp = tempfile::tempdir().unwrap();
        let dl = downloader(Some(cancelled_token()), Some(ScriptedHttp::new(Vec::new())));

        let err = dl.download_file_chunked("w.gguf", "https://x/f", &tmp.path().join("w.part"), 64, None).unwrap_err();
        assert!(matches!(err, DownloadError::Cancelled(_)));
    }
}

/// `TestTheFileLevel`
mod test_the_file_level {
    use super::*;

    #[test]
    fn test_a_queued_file_never_opens_a_socket() {
        let tmp = tempfile::tempdir().unwrap();
        let stream = ScriptedHttp::new(Vec::new());
        let dl = downloader(Some(cancelled_token()), Some(Arc::clone(&stream)));

        let err = dl.download_file("a/weights.gguf", "https://x/f", tmp.path(), 8, 0, None, false, None).unwrap_err();
        assert!(err.is_cancelled());
        assert_eq!(stream.calls(), 0);
    }

    #[test]
    fn test_the_part_is_not_published_on_a_cancel() {
        let tmp = tempfile::tempdir().unwrap();
        let token = CancelToken::new();
        let resp = FakeResp::cancelling(&[&[b'a'; 8], &[b'b'; 8]], &token, 1);
        let dl = downloader(Some(token.clone()), Some(ScriptedHttp::new(vec![resp])));

        let err = dl.download_file("a/weights.gguf", "https://x/f", tmp.path(), 16, 0, None, false, None).unwrap_err();
        assert!(err.is_cancelled());

        assert!(!tmp.path().join("weights.gguf").exists());
        assert!(tmp.path().join("weights.gguf.part").exists());
    }

    /// A 4GB digest is seconds of work with no network in it.
    #[test]
    fn test_the_digest_is_interruptible() {
        let tmp = tempfile::tempdir().unwrap();
        let big = tmp.path().join("w.gguf");
        std::fs::write(&big, vec![b'x'; ad::CHUNK_SIZE * 2 + 16]).unwrap();
        let token = cancelled_token();

        let err = SLPArtifactDownloader::compute_local_md5(&big, Some(&token)).unwrap_err();
        assert!(err.is_cancelled());
    }

    #[test]
    fn test_the_digest_is_unchanged_without_a_token() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("w.gguf");
        std::fs::write(&f, b"hello").unwrap();
        assert_eq!(SLPArtifactDownloader::compute_local_md5(&f, None).unwrap(), hex::encode(Md5::digest(b"hello")));
    }
}

/// `TestTheHeadProbe`
mod test_the_head_probe {
    use super::*;

    /// Swallowed, the probe answers "no size, no digest, no Range" and the
    /// file goes on to download single-stream and unverified.
    #[test]
    fn test_it_does_not_swallow_a_cancel() {
        let http = ScriptedHttp::with_head(Box::new(|_url| panic!("probe should not run")));
        let dl = downloader(Some(cancelled_token()), Some(http));

        let err = dl.get_remote_file_info("https://x/f").unwrap_err();
        assert!(err.is_cancelled());
    }

    #[test]
    fn test_an_ordinary_probe_failure_still_falls_back() {
        let http = ScriptedHttp::with_head(Box::new(|_url| Err(connect_error("no dns"))));
        let dl = downloader(None, Some(http));
        assert_eq!(dl.get_remote_file_info("https://x/f").unwrap(), (0, None, false));
    }
}
