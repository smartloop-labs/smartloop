//! Port of `tests/api/test_init_access.py`.
//!
//! /v1/init must answer a refused download with a status code, not a stream.
//!
//! Free models download with no token at all, so there is no credential gate
//! left on this route.  What remains is entitlement, and it is resolved
//! against the platform *before* the SSE stream opens: once the streaming
//! response begins, the request has committed to 200 and a refusal can only be
//! described inside a body that a client keying off the status will never read.
//!
//! Only failures that happen mid-transfer arrive as SSE error frames, tagged
//! with the status they would have been.
//!
//! The Python module drives several of these through the route with the
//! router's `SLPArtifactDownloader` and `_download_and_stream` monkeypatched.
//! The Rust init router builds its downloader internally and exposes no
//! substitute for either, so those cases are asserted here at the seams the
//! router does expose: the classification a pre-stream refusal is answered
//! with (`HTTPException(status_for(exc), str(exc))`) and the error frame a
//! mid-stream failure is reported with ([`init::error_event`]).  The cases
//! that are *only* about the route's orchestration around a stubbed
//! downloader are kept, ignored, with the reason.

use std::io;
use std::sync::Arc;

use serde_json::Value;
use serial_test::serial;

use smartloop::api::routers::_errors::{status_for, status_for_anyhow, HTTPException};
use smartloop::api::routers::init::{self, InitRequest};
use smartloop::api::services::ModelService;
use smartloop::model::downloader::artifact_downloader::{aggregate_error, DownloadError};
use smartloop::platform::{ModelAccessError, PlatformError};
use smartloop::workspace::workspace::{Workspace, WorkspaceModel};

use crate::common::misc_client::sse_frames;
use crate::shared::misc_isolation::IsolatedHome;

const FREE_MODEL: &str = "gemma3-4b";
const PAID_MODEL: &str = "gemma3-12b";

/// `errno.ENOSPC`.
const ENOSPC: i32 = 28;

/// The `model_service` fixture: a workspace with no weights on disk, saved
/// under an isolated `SLP_HOME`, with the paid model registered but not
/// downloaded.  Every request in this module therefore takes the "needs
/// download" path.  Returns the home guard and a cold model service.
fn model_service() -> (IsolatedHome, Arc<ModelService>) {
    let home = IsolatedHome::new();
    let root = home.dir.path();
    let mut workspace = Workspace::new("ws-test");
    workspace.workspace_dir = root.to_string_lossy().into_owned();
    workspace.models_dir = root.join("models").to_string_lossy().into_owned();
    workspace.models.push(WorkspaceModel {
        model_name: PAID_MODEL.to_string(),
        model_id: "google/gemma-3-12b-it".to_string(),
        downloaded: false,
        ..Default::default()
    });
    workspace.save().expect("save the fixture workspace");
    (home, ModelService::new_bare())
}

/// `PermissionError` / `ConnectionError` tagged with an upstream status, as the
/// platform client raises them.
fn tagged_permission(message: &str, status: u16) -> PlatformError {
    PlatformError::Permission { message: message.to_string(), status_code: Some(status) }
}

fn tagged_connection(message: &str, status: u16) -> PlatformError {
    PlatformError::Connection { message: message.to_string(), status_code: Some(status) }
}

/// What the route answers when resolving the signed URLs fails before the
/// stream opens: `HTTPException(status_code=status_for(exc), detail=str(exc))`.
fn pre_stream_refusal(exc: anyhow::Error) -> HTTPException {
    HTTPException::new(status_for_anyhow(&exc), exc.to_string())
}

/// The error frame the stream ends with when *exc* escapes mid-transfer.
fn error_frame(exc: anyhow::Error) -> Value {
    let event = init::error_event(&exc);
    let frames = sse_frames(&event);
    frames
        .into_iter()
        .find(|f| f.get("status").and_then(Value::as_str) == Some("error"))
        .unwrap_or_else(|| panic!("no error frame in stream: {event:?}"))
}

mod test_entitlement_is_settled_before_the_stream {
    use super::*;

    #[test]
    fn test_refusal_is_a_403_not_a_200_stream() {
        let exc = ModelAccessError::new(format!("Model '{PAID_MODEL}' is not available on your current plan."));

        let resp = pre_stream_refusal(exc.into());

        assert_eq!(resp.status_code, 403);
        assert!(resp.detail.as_str().unwrap_or_default().contains("current plan"));
    }

    #[test]
    fn test_rate_limit_surfaces_as_429() {
        // Not 403: retrying fixes this one, and calling it a plan problem would
        // send the user to a billing page instead of back in a minute.
        let exc = tagged_connection("Too many download requests. Please wait a moment and try again.", 429);

        let resp = pre_stream_refusal(exc.into());

        assert_eq!(resp.status_code, 429);
    }

    #[test]
    fn test_platform_unreachable_is_502() {
        let exc = PlatformError::connection("Connection error: refused");

        let resp = pre_stream_refusal(exc.into());

        assert_eq!(resp.status_code, 502);
    }

    /// The only remaining 400 on this route.  It is not fixed by signing in,
    /// so a client must be able to tell it apart from an entitlement refusal.
    #[tokio::test]
    #[serial]
    async fn test_unknown_model_is_still_400() {
        let (_home, svc) = model_service();

        let request = InitRequest { model_name: Some("no-such-model".to_string()), ..Default::default() };
        let resp = init::init_with(request, svc).await;

        let err = resp.expect_err("an unknown model is refused before the stream");
        assert_eq!(err.status_code, 400);
        assert!(err.detail.as_str().unwrap_or_default().contains("Unknown model"));
    }

    /// The point of DEV-383: the old behaviour was a 400 ("Authentication
    /// required") for any download without a token.  A free model must now get
    /// as far as asking the platform for URLs.
    #[test]
    #[ignore = "needs a substitute for the downloader the init router builds (Python patches init_router.SLPArtifactDownloader); the Rust router exposes none"]
    fn test_free_model_is_resolved_without_credentials() {
        panic!("route-level stub of SLPArtifactDownloader.resolve_artifacts for {FREE_MODEL} is not available");
    }

    /// The download route is rate-limited, so the URLs are minted once and
    /// handed to the download as `artifacts=`.
    #[test]
    #[ignore = "needs a substitute for the downloader the init router builds (Python patches init_router.SLPArtifactDownloader); the Rust router exposes none"]
    fn test_resolved_urls_are_handed_to_the_download() {
        panic!("route-level stub of SLPArtifactDownloader.download is not available");
    }
}

/// A failure after the stream opened still has to be classifiable.
///
/// The status line is 200 by then, so the code rides on the event for the
/// client to handle it the same way it handles the pre-stream status.
mod test_mid_stream_failure_carries_its_status {
    use super::*;

    #[test]
    fn test_late_plan_denial_is_tagged_403() {
        // A plan that changed between resolving the URLs and using them.
        let exc = ModelAccessError::new(format!("Model '{PAID_MODEL}' is not available on your current plan."));

        let error = error_frame(exc.into());

        assert_eq!(error["code"], 403);
        assert!(error["message"].as_str().unwrap_or_default().contains("not available on your current plan"));
    }

    #[test]
    fn test_expired_url_is_tagged_with_its_own_status() {
        let exc = tagged_permission("Download rejected (401 Unauthorized).", 401);

        let error = error_frame(exc.into());

        assert_eq!(error["code"], 401);
    }
}

/// The code has to answer "would trying again help?", not just "it broke".
///
/// Every one of these used to arrive as a bare 500 or, worse, a 403 — which
/// the desktop client renders as "Sign in to download this model" for a
/// signed-out user and "Your plan doesn't include this model" for a signed-in
/// one.  A full disk telling someone to upgrade their plan is the failure this
/// guards.
mod test_mid_stream_failure_is_classified_by_what_fixes_it {
    use super::*;

    // Aggregated by `aggregate_error` once the transfer gives up: a dropped
    // socket, an expired signed URL, an upstream 5xx.
    #[test]
    fn test_status_reflects_the_remedy_connection_error() {
        let exc = DownloadError::connection("Failed to download 1 file(s): model.gguf", None);
        assert_eq!(status_for(&exc), 502);
    }

    #[test]
    fn test_status_reflects_the_remedy_model_access_error() {
        let exc = ModelAccessError::new("not on your plan");
        assert_eq!(status_for(&exc), 403);
    }

    // An OSError subclass, and emphatically not an entitlement verdict.
    #[test]
    fn test_status_reflects_the_remedy_permission_error() {
        let exc = io::Error::new(io::ErrorKind::PermissionDenied, "[Errno 13] Permission denied: '/models'");
        assert_eq!(status_for(&exc), 500);
    }

    // Out of disk: retryable (free a little and try again) but not a network
    // failure — 507, not 502/500.
    #[test]
    fn test_status_reflects_the_remedy_disk_full() {
        let exc = io::Error::from_raw_os_error(ENOSPC);
        assert_eq!(status_for(&exc), 507);
    }

    #[test]
    fn test_status_reflects_the_remedy_runtime_error() {
        let exc = DownloadError::runtime("model.gguf: MD5 verification failed", None);
        assert_eq!(status_for(&exc), 500);
    }

    /// A platform error carries its own status and keeps it.
    #[test]
    fn test_upstream_status_still_wins() {
        assert_eq!(status_for(&tagged_connection("Too many download requests.", 429)), 429);
        assert_eq!(status_for(&tagged_permission("Sign in to download this model.", 401)), 401);
    }

    #[test]
    fn test_dropped_socket_reaches_the_client_as_502() {
        let exc = DownloadError::connection("Failed to download 1 file(s): model.gguf", None);

        let error = error_frame(exc.into());

        assert_eq!(error["code"], 502);
    }

    #[test]
    fn test_local_disk_refusal_is_not_a_sign_in_prompt() {
        let exc = io::Error::new(io::ErrorKind::PermissionDenied, "[Errno 13] Permission denied: '/models'");

        let error = error_frame(exc.into());

        assert_eq!(error["code"], 500);
    }

    /// The exact shape a real transfer produces: `aggregate_error` wraps the
    /// chunked path's ENOSPC, keeps it transient, and names the disk.  The
    /// client must see 507 (free some space and retry) and a message about the
    /// disk — not a bare filename list that could be a dropped socket.
    #[test]
    fn test_disk_full_reaches_the_client_as_507() {
        let exc = aggregate_error(vec![(
            "weights.gguf".to_string(),
            DownloadError::Io(io::Error::from_raw_os_error(ENOSPC)),
        )]);

        let error = error_frame(exc.into());

        assert_eq!(error["code"], 507);
        let message = error["message"].as_str().unwrap_or_default();
        assert!(message.contains("no space left on device"), "{message}");
        assert!(message.contains("weights.gguf"), "{message}");
    }
}

/// /v1/init must not need a prior request to have warmed the service.
///
/// `ModelService` is a process-wide singleton whose model starts as `None`;
/// `ensure_model()` is what assigns it, and it runs *after* the download.  The
/// router used to save through the model immediately after
/// `mark_model_downloaded`, so on a fresh process the weights landed on disk
/// and the client still got an error frame.
mod test_init_on_a_fresh_process {
    use super::*;

    #[test]
    #[ignore = "needs substitutes for the init router's downloader and _download_and_stream, and a mockable model service; the Rust router exposes none"]
    fn test_download_completes_without_a_warmed_model() {
        panic!("route-level stubs for the download stream are not available");
    }

    /// The model_ready branch dereferenced the same `None`.
    #[test]
    #[ignore = "the model_ready branch goes on to seed the catalog and build the model pipeline for real; without a mockable model service that reaches the network"]
    fn test_already_downloaded_also_survives_a_fresh_process() {
        panic!("a mockable model service for /v1/init is not available");
    }

    /// The flag is only useful if it is persisted, and it must not need the model.
    #[test]
    #[ignore = "needs substitutes for the init router's downloader and _download_and_stream; the Rust router exposes none"]
    fn test_the_download_flag_reaches_disk() {
        panic!("route-level stubs for the download stream are not available");
    }

    /// The flag has to land on the instance the router mutated.
    #[test]
    #[ignore = "needs substitutes for the init router's downloader and _download_and_stream, and a model service that records mark_model_downloaded; the Rust router exposes none"]
    fn test_the_router_hands_over_its_own_workspace() {
        panic!("route-level stubs for the download stream are not available");
    }

    /// A cold `ensure_workspace` builds a *fresh* workspace every call.
    ///
    /// The router registers the model on its own instance and deliberately
    /// does not save (a refused download must strand nothing in config.json),
    /// so resolving a second instance inside `mark_model_downloaded` read the
    /// model back off disk *without* that registration, skipped the flag for a
    /// model it could not find, and saved the stale copy over it — leaving
    /// config.json with no record of the model at all.
    #[test]
    #[serial]
    fn test_the_flag_survives_a_second_load() {
        /// `monkeypatch.setattr(Workspace, "config_file", ...)`, undone on drop.
        struct ConfigFile;
        impl Drop for ConfigFile {
            fn drop(&mut self) {
                Workspace::set_config_file(None);
            }
        }

        let home = IsolatedHome::new();
        let tmp = home.dir.path();
        Workspace::set_config_file(Some(tmp.join("config.json").to_string_lossy().into_owned()));
        let _config_file = ConfigFile;
        let mut seed = Workspace::new("ws-1");
        seed.workspace_dir = tmp.join("ws-1").to_string_lossy().into_owned();
        seed.models_dir = tmp.join("ws-1").join("models").to_string_lossy().into_owned();
        seed.save().expect("save the seed workspace");

        let svc = ModelService::new_bare();

        // What the router does before the stream opens.
        let routers_workspace = svc.ensure_workspace(None).expect("cold ensure_workspace");
        routers_workspace.write().add_model(PAID_MODEL, "google/gemma-3-12b-it", None, false);
        let second = svc.ensure_workspace(None).expect("cold ensure_workspace");
        assert!(
            !Arc::ptr_eq(&second, &routers_workspace),
            "a cold _ensure_workspace is supposed to return a fresh instance"
        );

        // What the stream does after the transfer.
        svc.mark_model_downloaded(PAID_MODEL, Some(routers_workspace.clone()))
            .expect("mark the model downloaded");

        let reloaded = Workspace::load().expect("config.json is on disk");
        let names: Vec<&str> = reloaded.models.iter().map(|m| m.model_name.as_str()).collect();
        assert_eq!(names, vec![PAID_MODEL], "the router's add_model was clobbered by a stale instance");
        assert!(reloaded.get_model(Some(PAID_MODEL)).expect("registered").downloaded);
    }
}
