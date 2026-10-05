//! Port of `tests/unit/test_artifact_downloader.py`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use md5::{Digest, Md5};
use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;
use smartloop::auth::credential_store;
use smartloop::model::downloader::artifact_downloader::{
    DownloadError, Headers, SLPArtifactDownloader, StaleDownloadURLError, aggregate_error, classified,
    is_retryable_error, is_transient_error,
};
use smartloop::model::downloader::artifact_source::{ArtifactMap, ArtifactSource, BootstrapArtifactSource};
use smartloop::model::downloader::inflight::{self, is_downloading};
use smartloop::model::downloader::slp_downloader::SLPDownloader;
use smartloop::platform::ModelAccessError;

use crate::shared::model_fakes::{connect_error, read_timeout, status_error, test_logger};

/// The registry is process-global, so a leaked mark would silently make a
/// later test's `is_downloading` assertion pass for the wrong reason (the
/// `_clear_inflight` autouse fixture).
struct ClearInflight;

impl ClearInflight {
    fn new() -> Self {
        inflight::reset_for_tests();
        ClearInflight
    }
}

impl Drop for ClearInflight {
    fn drop(&mut self) {
        inflight::reset_for_tests();
    }
}

fn md5_hex(data: &[u8]) -> String {
    hex::encode(Md5::digest(data))
}

fn artifact_map(pairs: &[(&str, &str)]) -> ArtifactMap {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// An artifact source answering with a closure.
struct FnSource<F>(F);

impl<F> ArtifactSource for FnSource<F>
where
    F: Fn(&str) -> anyhow::Result<ArtifactMap> + Send + Sync,
{
    fn get_artifacts(&self, model_name: &str) -> anyhow::Result<ArtifactMap> {
        (self.0)(model_name)
    }
}

/// A downloader built without `__init__`: models land under `models_dir` and
/// the platform is `source`.
fn make_downloader(models_dir: &Path, source: Arc<dyn ArtifactSource>) -> SLPArtifactDownloader {
    SLPArtifactDownloader::for_tests(Some(models_dir.to_path_buf()), test_logger(), None).with_model_source(source)
}

fn denying_source(message: &'static str) -> Arc<dyn ArtifactSource> {
    Arc::new(FnSource(move |_name: &str| -> anyhow::Result<ArtifactMap> {
        Err(ModelAccessError::new(message).into())
    }))
}

// ----------------------------------------------------------- TestComputeLocalMd5

/// `TestComputeLocalMd5`
mod test_compute_local_md5 {
    use super::*;

    #[test]
    fn test_correct_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("test.bin");
        let content = b"hello world";
        std::fs::write(&f, content).unwrap();
        assert_eq!(SLPArtifactDownloader::compute_local_md5(&f, None).unwrap(), md5_hex(content));
    }

    #[test]
    fn test_empty_file() {
        let tmp = tempfile::tempdir().unwrap();
        let f = tmp.path().join("empty.bin");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(SLPArtifactDownloader::compute_local_md5(&f, None).unwrap(), md5_hex(b""));
    }
}

// ------------------------------------------------------------ TestParseRemoteMd5

/// `TestParseRemoteMd5`
mod test_parse_remote_md5 {
    use super::*;

    fn etag(value: &str) -> Headers {
        let mut headers = Headers::new();
        headers.insert("ETag".to_string(), value.to_string());
        headers
    }

    #[test]
    fn test_valid_etag() {
        assert_eq!(
            SLPArtifactDownloader::parse_remote_md5(&etag("\"d41d8cd98f00b204e9800998ecf8427e\"")).as_deref(),
            Some("d41d8cd98f00b204e9800998ecf8427e")
        );
    }

    #[test]
    fn test_unquoted_etag() {
        assert_eq!(
            SLPArtifactDownloader::parse_remote_md5(&etag("d41d8cd98f00b204e9800998ecf8427e")).as_deref(),
            Some("d41d8cd98f00b204e9800998ecf8427e")
        );
    }

    #[test]
    fn test_non_md5_etag_returns_none() {
        // Composite GCS objects have longer ETags
        assert_eq!(SLPArtifactDownloader::parse_remote_md5(&etag("\"abc123-5\"")), None);
    }

    #[test]
    fn test_missing_etag() {
        assert_eq!(SLPArtifactDownloader::parse_remote_md5(&Headers::new()), None);
    }
}

// ------------------------------------------------------------ TestResolveModelName

/// `TestResolveModelName`
mod test_resolve_model_name {
    use super::*;

    #[test]
    fn test_known_model_roundtrip() {
        // resolve_model_id and resolve_model_name should be inverses
        let model_id = SLPArtifactDownloader::resolve_model_id("sl-mini");
        if model_id != "sl-mini" {
            // only if the model is in SUPPORTED_MODELS
            let name = SLPArtifactDownloader::resolve_model_name(&model_id);
            assert_eq!(name, "sl-mini");
        }
    }

    #[test]
    fn test_legacy_name_resolves_to_same_model_id() {
        // An existing workspace still records the pre-rename name.
        assert_eq!(
            SLPArtifactDownloader::resolve_model_id("slp-orchestra-mini"),
            SLPArtifactDownloader::resolve_model_id("sl-mini")
        );
    }

    #[test]
    fn test_reverse_lookup_returns_canonical_name() {
        let model_id = SLPArtifactDownloader::resolve_model_id("sl-mini");
        assert_eq!(SLPArtifactDownloader::resolve_model_name(&model_id), "sl-mini");
    }

    #[test]
    fn test_unknown_model_passthrough() {
        assert_eq!(SLPArtifactDownloader::resolve_model_name("unknown/model"), "unknown/model");
        assert_eq!(SLPArtifactDownloader::resolve_model_id("nonexistent"), "nonexistent");
    }
}

// ------------------------------------------------------ TestDownloadIsNotAuthGated

/// Restores an environment variable on drop.
struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvGuard {
    fn set(key: &'static str, value: Option<&str>) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }
        EnvGuard { key, previous }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: every test that touches the environment is `#[serial]`.
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// `TestDownloadIsNotAuthGated`
///
/// Free models download anonymously, so the downloader must not refuse to try
/// before it has asked the platform.  The platform is the only authority on
/// access: it serves the free tier to an anonymous caller and answers 403
/// (`ModelAccessError`) for anything the caller's plan doesn't grant.
mod test_download_is_not_auth_gated {
    use super::*;

    #[test]
    #[serial]
    fn test_reaches_the_platform_with_no_credentials() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        // The credential store is isolated to a temp dir (the conftest
        // `_isolate_credential_store` fixture), so no credentials are set.
        let home = tempfile::tempdir().unwrap();
        let _home = EnvGuard::set("SLP_HOME", Some(&home.path().to_string_lossy()));
        let _token = EnvGuard::set("SLP_DEVELOPER_TOKEN", None);
        assert!(!credential_store::has_credentials());

        let asked = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = Arc::clone(&asked);
        let source = Arc::new(FnSource(move |name: &str| -> anyhow::Result<ArtifactMap> {
            seen.lock().push(name.to_string());
            Ok(ArtifactMap::new()) // no GGUFs -> download() returns early
        }));

        let downloader = make_downloader(tmp.path(), source);
        SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap();

        assert_eq!(*asked.lock(), vec!["gemma3-4b".to_string()]);
    }

    #[test]
    #[serial]
    fn test_plan_denial_propagates() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader =
            make_downloader(tmp.path(), denying_source("Model 'gemma3-4b' is not available on your current plan."));
        let err = SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap_err();
        let denial = err.downcast_ref::<ModelAccessError>().expect("a ModelAccessError");
        assert!(denial.to_string().contains("not available on your current plan"));
    }
}

// ------------------------------------------ TestResolveArtifactsIsSeparableFromTheTransfer

/// `TestResolveArtifactsIsSeparableFromTheTransfer`
///
/// Entitlement is decided by asking for the signed URLs, so that call is split
/// out: /v1/init runs it before opening its progress stream, then hands the
/// result to download() rather than paying for the rate-limited route twice.
mod test_resolve_artifacts_is_separable_from_the_transfer {
    use super::*;

    #[test]
    #[serial]
    fn test_resolve_selects_the_gguf_artifacts() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let source = Arc::new(FnSource(|_name: &str| -> anyhow::Result<ArtifactMap> {
            Ok(artifact_map(&[
                ("model.gguf", "https://signed/model.gguf"),
                ("README.md", "https://signed/README.md"),
            ]))
        }));

        let resolved = make_downloader(tmp.path(), source).resolve_artifacts("google/gemma-3-4b-it").unwrap();

        assert_eq!(resolved, artifact_map(&[("model.gguf", "https://signed/model.gguf")]));
    }

    #[test]
    #[serial]
    fn test_passed_artifacts_skip_the_second_platform_call() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = Arc::clone(&calls);
        let source = Arc::new(FnSource(move |name: &str| -> anyhow::Result<ArtifactMap> {
            seen.lock().push(name.to_string());
            Ok(ArtifactMap::new())
        }));

        let downloader = make_downloader(tmp.path(), source);
        downloader.download_with_artifacts("google/gemma-3-4b-it", None, Some(ArtifactMap::new())).unwrap();

        assert!(calls.lock().is_empty(), "download() re-minted URLs the caller already had");
    }

    #[test]
    #[serial]
    fn test_refusal_leaves_no_model_directory_behind() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader = make_downloader(tmp.path(), denying_source("nope"));
        let err = SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap_err();
        assert!(err.downcast_ref::<ModelAccessError>().is_some());

        assert!(!tmp.path().join("google/gemma-3-4b-it").exists());
    }
}

// --------------------------------------------- TestAggregateErrorKeepsTheFailureKind

/// `TestAggregateErrorKeepsTheFailureKind`
///
/// Per-file failures collapse into one error that still says what broke.  The
/// router maps whatever escapes onto an HTTP status, so flattening everything
/// to a runtime error — as this once did — told the client "500" for a dropped
/// socket and a corrupt file alike.  They need different answers: one is
/// worth retrying, the other never will be.
mod test_aggregate_error_keeps_the_failure_kind {
    use super::*;

    fn one(exc: DownloadError) -> DownloadError {
        aggregate_error(vec![("model.gguf".to_string(), exc)])
    }

    fn disk_full() -> DownloadError {
        std::io::Error::from_raw_os_error(28).into()
    }

    #[test]
    fn test_transient_failures_aggregate_to_connection_error() {
        let cases: Vec<DownloadError> = vec![
            connect_error("dropped").into(),
            read_timeout("stalled").into(),
            StaleDownloadURLError("url expired".to_string()).into(),
        ];
        for exc in cases {
            let result = one(exc);
            assert!(matches!(result, DownloadError::Connection { .. }), "{result:?}");
            assert!(result.to_string().contains("model.gguf"));
        }
    }

    /// GCS answers 403 ExpiredToken past X-Goog-Expires.
    ///
    /// Entitlement was settled before the transfer started, so this is about
    /// the URL, not the account — and it must not surface as a plan refusal.
    #[test]
    fn test_stale_signed_url_is_transient_not_a_verdict() {
        for code in [401, 403] {
            let result = one(status_error(code).into());
            assert!(matches!(result, DownloadError::Connection { .. }), "{code}: {result:?}");
            assert!(!result.is_permission_error());
        }
    }

    #[test]
    fn test_permanent_failures_stay_runtime_errors() {
        let cases: Vec<DownloadError> = vec![
            DownloadError::runtime("model.gguf: MD5 verification failed after 3 attempts", None),
            DownloadError::Value("no artifacts".to_string()),
        ];
        for exc in cases {
            let result = one(exc);
            assert!(matches!(result, DownloadError::Runtime { .. }), "{result:?}");
            assert!(!result.is_connection_error());
        }
    }

    /// Don't promise a retry will help when part of the batch is unfixable.
    #[test]
    fn test_one_permanent_failure_makes_the_batch_permanent() {
        let result = aggregate_error(vec![
            ("a.gguf".to_string(), connect_error("dropped").into()),
            ("b.gguf".to_string(), DownloadError::runtime("MD5 verification failed", None)),
        ]);
        assert!(matches!(result, DownloadError::Runtime { .. }));
        assert!(!result.is_connection_error());
    }

    #[test]
    fn test_message_names_every_failed_file() {
        let result = aggregate_error(vec![
            ("a.gguf".to_string(), connect_error("x").into()),
            ("b.gguf".to_string(), connect_error("y").into()),
        ]);
        assert_eq!(result.to_string(), "Failed to download 2 file(s): a.gguf, b.gguf");
    }

    #[test]
    fn test_disk_full_is_transient_but_not_in_flight_retryable() {
        // Freeing some space and retrying is the canonical retry — the most
        // retryable failure there is — so the batch must classify as
        // transient for the client.  It must stay out of is_retryable_error: a
        // disk that cannot accept bytes has to fail fast, not grind through
        // backoff against a full platter.
        assert!(is_transient_error(&disk_full()));
        assert!(!is_retryable_error(&disk_full()));
        let result = one(disk_full());
        assert!(matches!(result, DownloadError::Connection { .. }));
    }

    #[test]
    fn test_disk_full_message_names_the_disk_not_just_the_file() {
        // The old message said "Failed to download 1 file(s): model.gguf",
        // which reads the same as a dropped socket and drops the one fact that
        // has an action: the disk is full.
        let result = one(disk_full());
        assert!(result.to_string().contains("no space left on device"));
        assert!(result.to_string().contains("model.gguf"));
    }

    #[test]
    fn test_disk_full_wrapped_by_the_chunked_path_is_still_seen() {
        // The chunked path preserves the cause's kind by wrapping the IO error
        // in a classified error (keeping it as its cause); both frames have to
        // be read for the disk-full to surface.
        let raw = disk_full();
        let wrapped = classified(
            "Chunk download failed for model.gguf (bytes 0-100): [Errno 28] No space left on device",
            is_transient_error(&raw),
        )
        .with_cause(raw);
        let result = one(wrapped);
        assert!(matches!(result, DownloadError::Connection { .. }));
        assert!(result.to_string().contains("no space left on device"));
    }

    #[test]
    fn test_disk_full_does_not_mask_a_permanent_failure() {
        // Freeing space fixes the disk-full file, but an MD5 mismatch is
        // unfixable — a batch with both must not promise a retry will help.
        let result = aggregate_error(vec![
            ("a.gguf".to_string(), disk_full()),
            ("b.gguf".to_string(), DownloadError::runtime("MD5 verification failed after 3 attempts", None)),
        ]);
        assert!(matches!(result, DownloadError::Runtime { .. }));
        assert!(!result.is_connection_error());
        assert!(result.to_string().contains("no space left on device"));
    }
}

// ------------------------------------------------ TestTheTransferMarksItselfInFlight

/// A bootstrap catalog with one artifact-less entry.
struct BootstrapSource;

impl ArtifactSource for BootstrapSource {
    fn get_artifacts(&self, _name: &str) -> anyhow::Result<ArtifactMap> {
        Ok(artifact_map(&[("model.gguf", "https://signed/x")]))
    }
}

impl BootstrapArtifactSource for BootstrapSource {
    fn list_models(&self) -> anyhow::Result<Arc<Vec<Value>>> {
        Ok(Arc::new(vec![json!({"name": "slp-orchestra-mini"})]))
    }
}

/// `TestTheTransferMarksItselfInFlight`
///
/// The registry only protects a download if the downloader registers one.
/// `test_inflight` proves the registry works and the storage tests prove the
/// sweep consults it — but both sides write their own `downloading()`, so
/// deleting the two marks in the downloader left the whole suite green while
/// the sweep went back to deleting live transfers.  These tests are the seam
/// between them.
mod test_the_transfer_marks_itself_in_flight {
    use super::*;

    fn in_flight_downloader(models_dir: &Path, artifacts: ArtifactMap) -> SLPArtifactDownloader {
        let source = Arc::new(FnSource(move |_name: &str| -> anyhow::Result<ArtifactMap> { Ok(artifacts.clone()) }));
        make_downloader(models_dir, source)
    }

    /// A `_download_artifacts` stand-in that records whether the directory
    /// was marked while it ran.
    fn recording(seen: Arc<Mutex<Option<bool>>>) -> smartloop::model::downloader::artifact_downloader::DownloadArtifactsHook {
        Arc::new(move |_artifacts: &ArtifactMap, local_dir: &Path| {
            *seen.lock() = Some(is_downloading(local_dir));
            Ok(local_dir.to_path_buf())
        })
    }

    #[test]
    #[serial]
    fn test_download_holds_the_mark_across_the_transfer() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader = in_flight_downloader(tmp.path(), artifact_map(&[("model.gguf", "https://signed/x")]));
        let seen = Arc::new(Mutex::new(None));

        downloader.set_download_artifacts_hook(Some(recording(Arc::clone(&seen))));
        SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap();

        assert_eq!(*seen.lock(), Some(true), "the transfer ran unmarked");
        assert!(inflight::active_snapshot().is_empty(), "the mark outlived the download");
    }

    #[test]
    #[serial]
    fn test_bootstrap_download_holds_the_mark_across_the_transfer() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader =
            in_flight_downloader(tmp.path(), ArtifactMap::new()).with_bootstrap_source(Arc::new(BootstrapSource));
        let seen = Arc::new(Mutex::new(None));

        downloader.set_download_artifacts_hook(Some(recording(Arc::clone(&seen))));
        downloader.bootstrap_download(None, None).unwrap();

        assert_eq!(*seen.lock(), Some(true), "the bootstrap transfer ran unmarked");
        assert!(inflight::active_snapshot().is_empty());
    }

    /// The property both code comments claim and neither test proved.
    ///
    /// The directory is created *inside* the marked block, so a sweep cannot
    /// land in the gap between "the tree exists" and "someone is writing to
    /// it" — on disk those two are the same thing.
    #[test]
    #[serial]
    fn test_the_mark_covers_the_makedirs() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader = in_flight_downloader(tmp.path(), artifact_map(&[("model.gguf", "https://signed/x")]));
        let local_dir: PathBuf = tmp.path().join("google/gemma-3-4b-it");
        let seen = Arc::new(Mutex::new(None));

        let expected = local_dir.clone();
        let record = Arc::clone(&seen);
        downloader.set_makedirs_hook(Some(Arc::new(move |path: &Path| {
            if path == expected {
                *record.lock() = Some(is_downloading(path));
            }
        })));
        let published = local_dir.clone();
        downloader.set_download_artifacts_hook(Some(Arc::new(move |_a: &ArtifactMap, _d: &Path| Ok(published.clone()))));
        SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap();

        assert_eq!(*seen.lock(), Some(true), "the model directory was created before the mark was taken");
    }

    #[test]
    #[serial]
    fn test_a_refused_download_leaves_the_registry_clean() {
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader = make_downloader(tmp.path(), denying_source("nope"));

        let err = SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap_err();
        assert!(err.downcast_ref::<ModelAccessError>().is_some());

        assert!(inflight::active_snapshot().is_empty());
    }

    #[test]
    #[serial]
    fn test_a_failed_transfer_releases_the_mark() {
        // The mark is held by a guard precisely so the unwinding path cannot
        // forget it; a wedged entry would keep the sweep away from that tree
        // for the life of the process.
        let _clear = ClearInflight::new();
        let tmp = tempfile::tempdir().unwrap();
        let downloader = in_flight_downloader(tmp.path(), artifact_map(&[("model.gguf", "https://signed/x")]));

        downloader.set_download_artifacts_hook(Some(Arc::new(|_a: &ArtifactMap, _d: &Path| {
            Err(DownloadError::runtime("model.gguf: MD5 verification failed", None))
        })));
        let err = SLPDownloader::download(&downloader, "google/gemma-3-4b-it", None).unwrap_err();
        assert!(matches!(err.downcast_ref::<DownloadError>(), Some(DownloadError::Runtime { .. })));

        assert!(inflight::active_snapshot().is_empty(), "a failed transfer wedged the registry");
    }
}
