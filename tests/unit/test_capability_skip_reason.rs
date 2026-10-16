//! Port of `tests/unit/test_capability_skip_reason.py`.
//!
//! A fallback says why the selected model could not run.
//!
//! A capability model that passes selection can still fail to load — most
//! often because it does not fit in memory alongside the resident
//! orchestrator.  The step then answers on the bootstrap model, which used to
//! be reported as a bare "Selected sl-mini": indistinguishable from the user's
//! selection being ignored.  Memory is the one cause they can act on, so it is
//! named.

use std::sync::Arc;

use smartloop::agents::orchestration::model_router::{
    CapabilitySkipReason, CapabilitySkipReasonFn, ModelRouter, ModelRouterOptions, SKIP_STEP,
};
use smartloop::agents::planner::ModelTier;
use smartloop::llms::BaseLLM;
use smartloop::types::ModelCapability;

use crate::shared::orch_steps::{MetadataLlm, ProgressRecorder};

fn llm(name: &str) -> Arc<dyn BaseLLM> {
    Arc::new(MetadataLlm::named(name, ""))
}

fn router_with(skip_reason: Option<CapabilitySkipReasonFn>) -> (ModelRouter, ProgressRecorder) {
    let router = ModelRouter::new(
        llm("Slp Orchestra Mini"),
        llm("Slp Orchestra Mini"),
        ModelRouterOptions {
            capability_llm_provider: Some(Arc::new(|_cap: &str| Ok(None))),
            capability_available: Some(Arc::new(|_cap: &str| Ok(true))),
            capability_skip_reason: skip_reason,
            ..Default::default()
        },
    );
    let progress = ProgressRecorder::new();
    router.begin_request(Some(progress.callback()));
    (router, progress)
}

fn router(skip: Option<CapabilitySkipReason>) -> (ModelRouter, ProgressRecorder) {
    router_with(Some(Arc::new(move || Ok(skip.clone()))))
}

fn last_message(progress: &ProgressRecorder) -> String {
    progress.messages().last().cloned().expect("at least one progress event")
}

/// `(status, message)` of every `model_skip` event.
fn skip_events(progress: &ProgressRecorder) -> Vec<(String, String)> {
    progress.calls.lock().iter().filter(|(step, _, _)| step == SKIP_STEP).map(|(_, s, m)| (s.clone(), m.clone())).collect()
}

fn resolve_reasoning(router: &ModelRouter) {
    router.llm_for_tier(ModelTier::Large, Some(ModelCapability::Reasoning.value()), None);
}

mod the_fallback_explains_itself {
    use super::*;

    #[test]
    fn test_memory_is_named() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "memory")));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini — not enough memory for gemma4-e2b");
    }

    #[test]
    fn test_a_gpu_fault_is_also_a_memory_message() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "faulted")));

        resolve_reasoning(&router);

        assert!(last_message(&progress).contains("ran out of memory"));
    }

    #[test]
    fn test_a_missing_download_is_named() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "missing")));

        resolve_reasoning(&router);

        assert!(last_message(&progress).contains("not downloaded"));
    }

    #[test]
    fn test_the_legacy_name_is_shown_as_the_product_name() {
        let (router, progress) =
            router(Some(CapabilitySkipReason::new("slp-orchestra-mini", "memory")));

        resolve_reasoning(&router);

        assert!(last_message(&progress).contains("sl-mini"));
        assert!(!last_message(&progress).to_lowercase().contains("orchestra"));
    }

    #[test]
    fn test_no_recorded_reason_leaves_the_message_bare() {
        let (router, progress) = router(None);

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }

    #[test]
    fn test_an_unknown_cause_is_not_guessed_at() {
        let (router, progress) =
            router(Some(CapabilitySkipReason::new("gemma4-e2b", "something-new")));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }

    #[test]
    fn test_a_broken_reporter_does_not_break_the_event() {
        let (router, progress) = router_with(Some(Arc::new(|| Err(anyhow::anyhow!("boom")))));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }

    #[test]
    fn test_a_panicking_reporter_does_not_break_the_event() {
        let (router, progress) = router_with(Some(Arc::new(|| panic!("boom"))));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }

    #[test]
    fn test_no_reporter_wired_in_keeps_the_bare_message() {
        let (router, progress) = router_with(None);

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }
}

mod the_reason_is_recorded_where_it_happens {
    use std::sync::Arc;

    use parking_lot::RwLock;

    use smartloop::agents::orchestration::model_router::CapabilitySkipReason;
    use smartloop::api::services::model_service::{CachedLlm, ModelService, GPU_FAULTED};
    use smartloop::model::model_manager::Model;
    use smartloop::workspace::workspace::{Workspace, WorkspaceModel};

    use crate::shared::orch_steps::MetadataLlm;

    /// `object.__new__(ModelService)` with the seams the Python test patched.
    fn service(fits: bool, gguf: Option<&str>, faulted: bool) -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        let workspace = Arc::new(RwLock::new(Workspace::new("ws")));
        *svc.model.write() = Some(Arc::new(Model::new("m", workspace, None)));
        *svc.capability_skip.lock() = None;
        *svc.last_fit_shortfall.lock() = None;
        {
            let mut hooks = svc.hooks.write();
            let gguf = gguf.map(str::to_string);
            hooks.find_model_ggufs = Some(Arc::new(move |_dir: &str| (gguf.clone(), None)));
            hooks.large_model_fits =
                Some(Arc::new(move |_p: &str, _cap: i64, _label: &str, _reserve: Option<f64>| fits));
        }
        if faulted {
            svc.cap_llms.lock().push(("gemma4-e2b".to_string(), GPU_FAULTED));
        }
        svc
    }

    fn model() -> WorkspaceModel {
        WorkspaceModel {
            model_name: "gemma4-e2b".to_string(),
            model_id: "google/gemma4-e2b".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_a_fit_failure_records_memory() {
        let svc = service(false, Some("/models/m.gguf"), false);

        assert!(svc.load_workspace_llm(&model(), "capability 'reasoning'", None).is_none());
        assert_eq!(
            svc.last_capability_skip(),
            Some(CapabilitySkipReason { model_name: "gemma4-e2b".into(), cause: "memory".into(), detail: None })
        );
    }

    #[test]
    fn test_a_fit_failure_carries_the_measured_shortfall() {
        let svc = service(false, Some("/models/m.gguf"), false);
        *svc.last_fit_shortfall.lock() = Some((4.9, 4.4));

        assert!(svc.load_workspace_llm(&model(), "capability 'reasoning'", None).is_none());
        assert_eq!(
            svc.last_capability_skip(),
            Some(CapabilitySkipReason::with_detail("gemma4-e2b", "memory", "needs ~4.9GB, 4.4GB free"))
        );
    }

    #[test]
    fn test_a_missing_gguf_records_missing() {
        let svc = service(true, None, false);

        assert!(svc.load_workspace_llm(&model(), "capability 'reasoning'", None).is_none());
        assert_eq!(svc.last_capability_skip(), Some(CapabilitySkipReason::new("gemma4-e2b", "missing")));
    }

    /// A runtime OOM has no measurement to report — only the cause.
    #[test]
    fn test_a_faulted_slot_records_memory() {
        let svc = service(true, Some("/models/m.gguf"), true);

        assert!(svc.load_workspace_llm(&model(), "capability 'reasoning'", None).is_none());
        assert_eq!(svc.last_capability_skip(), Some(CapabilitySkipReason::new("gemma4-e2b", "memory")));
    }

    #[test]
    fn test_a_resident_model_clears_a_stale_reason() {
        let svc = service(true, Some("/models/m.gguf"), false);
        *svc.capability_skip.lock() = Some(CapabilitySkipReason::new("gemma4-e2b", "memory"));
        let resident: Arc<dyn smartloop::llms::BaseLLM> = Arc::new(MetadataLlm::named("resident", ""));
        svc.cap_llms.lock().push(("gemma4-e2b".to_string(), CachedLlm::Loaded(resident)));

        assert!(svc.load_workspace_llm(&model(), "capability 'reasoning'", None).is_some());
        assert_eq!(svc.last_capability_skip(), None);
    }
}

// ── the figures travel to the user, not just the log ──────────────────────

/// "Not enough memory" does not say how much to free.  The numbers do.
mod the_shortfall_is_reported {
    use super::*;

    #[test]
    fn test_the_figures_are_appended() {
        let (router, progress) = router(Some(CapabilitySkipReason::with_detail(
            "gemma4-e2b",
            "memory",
            "needs ~4.9GB, 4.4GB free",
        )));

        resolve_reasoning(&router);

        assert_eq!(
            last_message(&progress),
            "Selected sl-mini — not enough memory for gemma4-e2b (needs ~4.9GB, 4.4GB free)"
        );
    }

    #[test]
    fn test_a_cause_with_no_figures_reads_as_before() {
        let (router, progress) = router(Some(CapabilitySkipReason {
            model_name: "gemma4-e2b".into(),
            cause: "missing".into(),
            detail: None,
        }));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini — gemma4-e2b is not downloaded");
    }

    #[test]
    fn test_a_two_element_record_is_still_accepted() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "memory")));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini — not enough memory for gemma4-e2b");
    }

    /// Python's one-element `("gemma4-e2b",)` tuple: a record with no cause.
    #[test]
    fn test_a_malformed_record_is_ignored() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "")));

        resolve_reasoning(&router);

        assert_eq!(last_message(&progress), "Selected sl-mini");
    }
}

mod the_shortfall_is_measured_where_it_happens {
    use std::fs::File;
    use std::sync::Arc;

    use smartloop::api::services::model_service::ModelService;

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    /// `object.__new__(ModelService)` reporting `free` device memory.
    fn service(free: (f64, f64)) -> Arc<ModelService> {
        let svc = ModelService::new_bare();
        svc.hooks.write().device_free_total_gb = Some(Arc::new(move || Some(free)));
        *svc.last_fit_shortfall.lock() = None;
        svc
    }

    /// A (sparse) GGUF of `size_gb` — stands in for patching
    /// `os.path.getsize`.
    fn gguf_of_size(dir: &tempfile::TempDir, size_gb: f64) -> String {
        let path = dir.path().join("m.gguf");
        let file = File::create(&path).expect("create gguf");
        file.set_len((size_gb * GIB) as u64).expect("size gguf");
        path.to_string_lossy().into_owned()
    }

    /// Quoting the footprint alone shows a number below the free memory.
    #[test]
    fn test_the_reserve_is_folded_into_the_requirement() {
        let svc = service((4.4, 24.0));
        let dir = tempfile::tempdir().unwrap();
        let gguf = gguf_of_size(&dir, 3.2);

        assert!(!svc.large_model_fits(&gguf, 0, "gemma4-e2b", None));

        // 3.2 weights + 0.7 overhead + 1.0 reserve = 4.9
        assert_eq!(svc.fit_shortfall_detail().as_deref(), Some("needs ~4.9GB, 4.4GB free"));
    }

    #[test]
    fn test_a_successful_fit_clears_the_figures() {
        let svc = service((20.0, 24.0));
        *svc.last_fit_shortfall.lock() = Some((4.9, 4.4));
        let dir = tempfile::tempdir().unwrap();
        let gguf = gguf_of_size(&dir, 1.0);

        assert!(svc.large_model_fits(&gguf, 0, "gemma4-e2b", None));

        assert_eq!(svc.fit_shortfall_detail(), None);
    }

    #[test]
    fn test_no_measurement_yields_no_detail() {
        let svc = service((4.4, 24.0));
        assert_eq!(svc.fit_shortfall_detail(), None);
    }
}

/// The reason also goes out as its own `model_skip` step, so a client can
/// alert on it without reading the "Selected …" line.
mod the_skip_is_its_own_event {
    use super::*;

    #[test]
    fn test_a_memory_skip_is_reported_with_its_figures() {
        let (router, progress) = router(Some(CapabilitySkipReason::with_detail(
            "gemma4-e2b",
            "memory",
            "needs ~4.9GB, 4.4GB free",
        )));

        resolve_reasoning(&router);

        assert_eq!(
            skip_events(&progress),
            vec![("processing".to_string(), "Not enough memory for gemma4-e2b (needs ~4.9GB, 4.4GB free)".to_string())]
        );
    }

    #[test]
    fn test_it_comes_before_the_selection_line() {
        let (router, progress) = router(Some(CapabilitySkipReason::new("gemma4-e2b", "missing")));

        resolve_reasoning(&router);

        let calls = progress.calls.lock().clone();
        let skip = calls.iter().position(|(step, _, _)| step == SKIP_STEP).expect("a model_skip event");
        let selected = calls.iter().position(|(_, _, m)| m.starts_with("Selected ")).expect("a selection line");
        assert!(skip < selected, "{calls:?}");
    }

    #[test]
    fn test_no_reason_means_no_skip_event() {
        let (router, progress) = router(None);

        resolve_reasoning(&router);

        assert!(skip_events(&progress).is_empty());
    }
}
