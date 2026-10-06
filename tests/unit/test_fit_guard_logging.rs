//! Port of `tests/unit/test_fit_guard_logging.py`.
//!
//! The fit-guard warnings have to be readable by whoever hits them.
//!
//! `llm_large_n_ctx_cap` uses 0 as the sentinel for *no cap* — the factory
//! auto-sizes the context from live memory — so printing it raw read as a
//! context window of zero tokens.  And the remedy the warning offered ("lower
//! the cap") cannot help when the cap is already unset: uncapped, the KV cache
//! is estimated at zero, so the figure quoted is bare weights plus overhead.

use std::fs::File;
use std::sync::Arc;

use serial_test::serial;

use smartloop::api::services::model_service::{ModelServiceHooks, n_ctx_desc};
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{attach_workspace, bare_service, capture_logs};

#[test]
fn test_zero_is_described_not_printed() {
    assert_eq!(n_ctx_desc(0), "an auto-sized context");
}

#[test]
fn test_a_real_cap_is_printed() {
    assert_eq!(n_ctx_desc(8192), "n_ctx=8192");
}

/// Run the fit guard for a 3.2GB model with 4.4GB of 24GB free, capturing
/// the warning (`os.path.getsize` is stood in for by a sparse file of that
/// size).
fn warn(cap: i64) -> (bool, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = dir.path().join("e2b.gguf");
    File::create(&gguf)
        .and_then(|f| f.set_len((3.2 * 1024f64.powi(3)) as u64))
        .expect("sparse gguf");

    let svc = bare_service();
    svc.set_hooks(ModelServiceHooks {
        device_free_total_gb: Some(Arc::new(|| Some((4.4, 24.0)))),
        ..Default::default()
    });
    let path = gguf.to_string_lossy().into_owned();
    capture_logs(tracing::Level::WARN, || svc.large_model_fits(&path, cap, "gemma4-e2b", None))
}

/// Lowering a cap that is already 0 cannot change the estimate.
#[test]
#[serial]
fn test_uncapped_does_not_offer_the_cap() {
    let (fits, text) = warn(0);

    assert!(!fits);
    assert!(!text.contains("Lower SLP_LLM_LARGE_N_CTX_CAP"));
    assert!(text.contains("is unset"));
    assert!(text.contains("Free memory"));
}

#[test]
#[serial]
fn test_capped_still_offers_the_cap() {
    let (fits, text) = warn(8192);

    assert!(!fits);
    assert!(text.contains("Lower SLP_LLM_LARGE_N_CTX_CAP"));
}

#[test]
#[serial]
fn test_the_numbers_are_still_reported() {
    let (_fits, text) = warn(0);

    assert!(text.contains("4.4GB free of 24.0GB"));
    assert!(text.contains("gemma4-e2b"));
}

/// Weights + overhead only — the window auto-sizes to whatever fits.
#[test]
#[serial]
fn test_an_uncapped_estimate_excludes_the_kv_cache() {
    let (_fits, text) = warn(0);

    assert!(text.contains("~3.9GB"));
}

#[test]
#[serial]
fn test_a_capped_estimate_includes_the_kv_cache() {
    let (_fits, text) = warn(8192);

    assert!(text.contains("~4.9GB"));
}

#[test]
#[serial]
fn test_an_uncapped_skip_does_not_say_n_ctx_zero() {
    let svc = bare_service();
    attach_workspace(&svc, Workspace::new("ws"));
    *svc.capability_skip.lock() = None;
    svc.set_hooks(ModelServiceHooks {
        find_model_ggufs: Some(Arc::new(|_| (Some("/models/e2b.gguf".to_string()), None))),
        large_model_fits: Some(Arc::new(|_, _, _, _| false)),
        evict_capability_models: Some(Arc::new(|_| false)),
        ..Default::default()
    });
    let ws = WorkspaceModel {
        model_name: "gemma4-e2b".to_string(),
        model_id: "google/gemma4-e2b".to_string(),
        ..Default::default()
    };

    // `AppSettings().llm_large_n_ctx_cap = 0`.
    let previous = std::env::var("SLP_LLM_LARGE_N_CTX_CAP").ok();
    // SAFETY: serialized with every other env-touching test.
    unsafe { std::env::set_var("SLP_LLM_LARGE_N_CTX_CAP", "0") };
    let (loaded, text) = capture_logs(tracing::Level::WARN, || {
        svc.load_workspace_llm(&ws, "capability 'general'", None)
    });
    // SAFETY: as above.
    unsafe {
        match previous {
            Some(v) => std::env::set_var("SLP_LLM_LARGE_N_CTX_CAP", v),
            None => std::env::remove_var("SLP_LLM_LARGE_N_CTX_CAP"),
        }
    }

    assert!(loaded.is_none());
    assert!(!text.contains("n_ctx=0"));
    assert!(text.contains("an auto-sized context"));
}
