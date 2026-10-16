//! A pinned remote model answers through `CloudLLM`, bypassing the
//! load/fit/evict machinery built for local weights.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::json;
use serial_test::serial;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, ResponseTemplate};

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::llms::base_llm::BaseLLM;
use smartloop::llms::cloud_llm::CloudLLM;
use smartloop::workspace::{Workspace, WorkspaceModel};

use crate::shared::model_service_a::{FakeLLM, attach_workspace, bare_service, same};
use crate::shared::workspace_support::EnvGuard;

fn remote(name: &str) -> WorkspaceModel {
    let mut ws = Workspace::new("ws");
    ws.add_remote_model(name, "https://api.smartloop.ai", vec!["general".into(), "coding".into()], Some(65536), true)
}

/// A service with a resident default model and a workspace whose only
/// selectable model is remote.
fn svc_with_remote_pin() -> Arc<ModelService> {
    let svc = bare_service();
    *svc.llm.write() = Some(FakeLLM::named("resident_orchestrator"));
    *svc.model_name.write() = Some("sl-mini".to_string());
    let mut ws = Workspace::new("ws");
    ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", vec!["general".into()], None, true);
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
        ws.add_remote_model("kimi-k2", "https://api.smartloop.ai", vec!["general".into()], None, true);
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
