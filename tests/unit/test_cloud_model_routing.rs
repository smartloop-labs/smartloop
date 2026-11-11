//! A pinned remote model answers through `CloudLLM`, bypassing the
//! load/fit/evict machinery built for local weights.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{Value, json};
use serial_test::serial;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::llms::cloud_llm::CloudLLM;
use smartloop::model::spec::MODEL_SPECS;
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{FakeLLM, attach_workspace, bare_service, same};
use crate::shared::workspace_support::EnvGuard;

/// Register `kimi-k2`'s spec (the registry is process-wide, so every test
/// here publishes the same row) and return it as a workspace model.
fn remote(name: &str) -> WorkspaceModel {
    publish_kimi();
    let mut ws = Workspace::new("ws");
    ws.add_remote_model(name, true)
}

fn publish_kimi() {
    let Value::Object(row) = json!({
        "name": "kimi-k2", "type": "remote", "url": "https://api.smartloop.ai",
        "capabilities": ["general", "coding"], "context_length": 65536,
    }) else {
        unreachable!()
    };
    MODEL_SPECS.hydrate(&[row]);
}

/// A service with a resident default model and a workspace whose only
/// selectable model is remote.
fn svc_with_remote_pin() -> Arc<ModelService> {
    let svc = bare_service();
    *svc.llm.write() = Some(FakeLLM::named("resident_orchestrator"));
    *svc.model_name.write() = Some("sl-mini".to_string());
    let mut ws = Workspace::new("ws");
    publish_kimi();
    ws.add_remote_model("kimi-k2", true);
    attach_workspace(&svc, ws);
    svc
}

#[test]
fn test_a_remote_model_is_served_without_loading_weights() {
    let svc = bare_service();
    attach_workspace(&svc, Workspace::new("ws"));
    let loaded = Arc::new(AtomicBool::new(false));
    let loaded_in_hook = loaded.clone();
    svc.set_hooks(ModelServiceHooks {
        load_llama_instance: Some(Arc::new(move |_, name: &str, _, _| {
            loaded_in_hook.store(true, Ordering::SeqCst);
            Ok(FakeLLM::named(name))
        })),
        ..Default::default()
    });

    let llm = svc.load_workspace_llm(&remote("kimi-k2"), "pinned", None).expect("a cloud model");

    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert_eq!(llm.model_label(), "kimi-k2");
    assert_eq!(llm.get_max_tokens(), 65536);
    assert!(!loaded.load(Ordering::SeqCst));
    assert!(svc.cap_llm("kimi-k2").is_none(), "a cloud model takes no cache slot");
}

#[test]
fn test_a_sole_selected_remote_model_is_pinned() {
    let svc = svc_with_remote_pin();

    let pinned = svc.pinned_llm(None).expect("the remote model is pinned");

    assert!(ModelService::is_cloud_llm(pinned.as_ref()));
    let cloud = pinned.as_any().downcast_ref::<CloudLLM>().unwrap();
    assert_eq!(cloud.url(), "https://api.smartloop.ai");
}

#[test]
fn test_the_response_uses_the_pinned_remote_model() {
    let svc = svc_with_remote_pin();

    let llm = svc.llm_for_response(None).expect("a response model");

    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert!(!same(&llm, &svc.llm.read().clone().unwrap()));
}

/// A refusal reaches the user as an error instead of a quiet answer from the
/// local default model.
#[tokio::test]
#[serial]
async fn test_a_cloud_failure_does_not_fall_back_to_the_local_model() {
    let server = MockServer::start().await;
    // A session cached by an earlier test could answer for this one.
    smartloop::llms::cloud_session::clear();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({"detail": "Unauthorized"})))
        .mount(&server)
        .await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    let cloud: Arc<dyn BaseLLM> = Arc::new(CloudLLM::with_hooks(
        "p1",
        "kimi-k2",
        &server.uri(),
        None,
        Some(Arc::new(Default::default)),
        Some(Arc::new(|| false)),
    ));

    let svc = bare_service();
    *svc.llm.write() = Some(FakeLLM::named("resident_orchestrator"));
    let pinned = cloud.clone();
    svc.set_hooks(ModelServiceHooks {
        llm_for_response: Some(Arc::new(move |_| Some(pinned.clone()))),
        ..Default::default()
    });

    let outcome = tokio::task::spawn_blocking(move || {
        let stream = svc.stream_completion("<user>hi</user>", None, None, None, None).unwrap();
        stream.collect::<anyhow::Result<Vec<_>>>().map(|chunks| chunks.len())
    })
    .await
    .unwrap();

    let err = outcome.expect_err("the failure must surface");
    assert_eq!(err.to_string(), "Sign in to use kimi-k2.");
}

#[tokio::test]
#[serial]
async fn test_a_non_streamed_cloud_failure_is_an_error_not_answer_text() {
    let server = MockServer::start().await;
    // A session cached by an earlier test could answer for this one.
    smartloop::llms::cloud_session::clear();
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429))
        .mount(&server)
        .await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    let cloud: Arc<dyn BaseLLM> = Arc::new(CloudLLM::with_hooks(
        "p1",
        "kimi-k2",
        &server.uri(),
        None,
        Some(Arc::new(Default::default)),
        Some(Arc::new(|| false)),
    ));
    let svc = bare_service();
    svc.set_hooks(ModelServiceHooks {
        llm_for_response: Some(Arc::new(move |_| Some(cloud.clone()))),
        ..Default::default()
    });

    let outcome = tokio::task::spawn_blocking(move || svc.generate_completion("<user>hi</user>", None, None, None, None))
        .await
        .unwrap();

    assert_eq!(outcome.unwrap_err().to_string(), "kimi-k2 is busy right now. Try again shortly.");
}

/// The platform records a remote answer's tokens; pushing our own estimate
/// too would count them twice.
#[test]
fn test_a_remote_answer_is_not_reported_as_local_usage() {
    let remote = svc_with_remote_pin();
    assert!(remote.answers_remotely());

    let local = bare_service();
    let mut ws = Workspace::new("ws");
    ws.add_model("gemma3-4b", "google/gemma-3-4b-it", None, true);
    ws.get_model_mut(Some("gemma3-4b")).unwrap().downloaded = true;
    attach_workspace(&local, ws);
    assert!(!local.answers_remotely());
}

/// With local and remote models both selected, the answer is remote only
/// when the model that writes it is.
#[test]
fn test_a_mixed_selection_answers_remotely_only_on_the_remote_model() {
    let mixed = |answer: Arc<dyn BaseLLM>| {
        let svc = bare_service();
        let mut ws = Workspace::new("ws");
        ws.add_model("gemma4-e4b", "gemma4-e4b", None, true);
        ws.get_model_mut(Some("gemma4-e4b")).unwrap().downloaded = true;
        publish_kimi();
        ws.add_remote_model("kimi-k2", true);
        attach_workspace(&svc, ws);
        svc.set_hooks(ModelServiceHooks {
            llm_for_response: Some(Arc::new(move |_| Some(answer.clone()))),
            ..Default::default()
        });
        svc
    };

    assert!(!mixed(FakeLLM::named("gemma")).answers_remotely());
    let cloud: Arc<dyn BaseLLM> = Arc::new(CloudLLM::new("p1", "kimi-k2", "https://api.smartloop.ai", None));
    assert!(mixed(cloud).answers_remotely());
}

/// Orchestration resolves inside a local-only scope, where a remote pin is
/// ignored and the resident model answers; outside it the pin still holds.
#[test]
fn test_a_remote_pin_is_ignored_while_orchestrating() {
    use smartloop::llms::cloud_llm::local_only_scope;

    let svc = svc_with_remote_pin();
    {
        let _scope = local_only_scope(true);
        assert!(svc.pinned_model(None).is_none());
        assert!(svc.pinned_llm(None).is_none());
        let llm = svc.llm_for_response(None).expect("the resident model");
        assert!(!ModelService::is_cloud_llm(llm.as_ref()));
    }
    let llm = svc.llm_for_response(None).expect("the pinned model");
    assert!(ModelService::is_cloud_llm(llm.as_ref()));
}

/// A local model the catalog also serves in the cloud, registered local in a
/// workspace whose models live under `models_dir`.
fn dual_local(name: &str, models_dir: &std::path::Path) -> (Workspace, WorkspaceModel) {
    let rows = [
        json!({"name": name, "type": "remote", "url": "https://api.smartloop.ai",
               "capabilities": ["text"], "context_length": 262144, "is_accessible": true}),
        json!({"name": name, "type": "local", "model_id": format!("Org/{name}"),
               "capabilities": ["text"], "param_size_b": 27.0}),
    ]
    .map(|v| v.as_object().unwrap().clone());
    MODEL_SPECS.hydrate(&rows);
    let mut ws = Workspace::new("ws");
    ws.models_dir = models_dir.to_string_lossy().to_string();
    ws.add_model(name, &format!("Org/{name}"), None, true);
    ws.get_model_mut(Some(name)).unwrap().downloaded = true;
    let model = ws.get_model(Some(name)).unwrap();
    (ws, model)
}

#[test]
#[serial]
fn test_a_local_model_without_its_weights_runs_in_the_cloud() {
    let _fallback = EnvGuard::set("SLP_CLOUD_FALLBACK", "true");
    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-missing", tmp.path());
    let svc = bare_service();
    attach_workspace(&svc, ws);

    let llm = svc.load_workspace_llm(&model, "answer", None).expect("the cloud variant");
    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert_eq!(llm.model_label(), "dual-missing");
    assert_eq!(llm.get_max_tokens(), 262144);
}

#[test]
#[serial]
fn test_a_local_model_that_does_not_fit_runs_in_the_cloud() {
    let _fallback = EnvGuard::set("SLP_CLOUD_FALLBACK", "true");
    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-tight", tmp.path());
    let dir = tmp.path().join("Org/dual-tight");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("dual-tight-Q4_K_M.gguf"), b"GGUF").unwrap();
    let svc = bare_service();
    attach_workspace(&svc, ws);
    let loaded = Arc::new(AtomicBool::new(false));
    let loaded_in_hook = loaded.clone();
    svc.set_hooks(ModelServiceHooks {
        large_model_fits: Some(Arc::new(|_, _, _, _| false)),
        load_llama_instance: Some(Arc::new(move |_, name: &str, _, _| {
            loaded_in_hook.store(true, Ordering::SeqCst);
            Ok(FakeLLM::named(name))
        })),
        ..Default::default()
    });

    let llm = svc.load_workspace_llm(&model, "answer", None).expect("the cloud variant");
    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert!(!loaded.load(Ordering::SeqCst), "the local copy never loads");
}

#[test]
#[serial]
fn test_no_cloud_fallback_while_orchestrating() {
    let _fallback = EnvGuard::set("SLP_CLOUD_FALLBACK", "true");
    use smartloop::llms::cloud_llm::local_only_scope;

    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-orchestrating", tmp.path());
    let svc = bare_service();
    attach_workspace(&svc, ws);
    let _scope = local_only_scope(true);
    assert!(svc.load_workspace_llm(&model, "decision", None).is_none());
}

#[test]
#[serial]
fn test_a_local_model_without_its_weights_stays_off_the_cloud_when_turned_off() {
    let _fallback = EnvGuard::set("SLP_CLOUD_FALLBACK", "false");
    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-paid", tmp.path());
    let svc = bare_service();
    attach_workspace(&svc, ws);
    assert!(svc.load_workspace_llm(&model, "answer", None).is_none(), "degrades instead");
}

#[test]
#[serial]
fn test_a_local_model_that_does_not_load_goes_to_the_cloud_by_default() {
    let _fallback = EnvGuard::remove("SLP_CLOUD_FALLBACK");
    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-load-fails", tmp.path());
    let dir = tmp.path().join("Org/dual-load-fails");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("dual-load-fails-Q4_K_M.gguf"), b"GGUF").unwrap();
    let svc = bare_service();
    attach_workspace(&svc, ws);
    svc.set_hooks(ModelServiceHooks {
        large_model_fits: Some(Arc::new(|_, _, _, _| true)),
        load_llama_instance: Some(Arc::new(|_, _, _, _| Err(anyhow::anyhow!("failed to load model")))),
        ..Default::default()
    });
    let llm = svc.load_workspace_llm(&model, "answer", None).expect("the cloud copy");
    assert!(ModelService::is_cloud_llm(llm.as_ref()));
}

#[test]
#[serial]
fn test_a_model_that_did_not_fit_is_not_checked_again_on_every_step() {
    let _fallback = EnvGuard::remove("SLP_CLOUD_FALLBACK");
    let tmp = tempfile::tempdir().unwrap();
    let (ws, model) = dual_local("dual-no-fit-once", tmp.path());
    let dir = tmp.path().join("Org/dual-no-fit-once");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("dual-no-fit-once-Q4_K_M.gguf"), b"GGUF").unwrap();
    let svc = bare_service();
    attach_workspace(&svc, ws);
    let checks = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let count = checks.clone();
    svc.set_hooks(ModelServiceHooks {
        large_model_fits: Some(Arc::new(move |_, _, _, _| {
            count.fetch_add(1, Ordering::SeqCst);
            false
        })),
        ..Default::default()
    });

    // Every step of a turn resolves the model; each goes to the cloud copy.
    for _ in 0..5 {
        let llm = svc.load_workspace_llm(&model, "answer", None).expect("the cloud copy");
        assert!(ModelService::is_cloud_llm(llm.as_ref()));
    }
    let after_first = checks.load(Ordering::SeqCst);
    assert!(after_first <= 2, "checked once (and once more after releasing models), not per step: {after_first}");
}
