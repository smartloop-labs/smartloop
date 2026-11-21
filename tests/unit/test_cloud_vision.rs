//! A remote model declaring vision reads images: over HTTP, through
//! `CloudLLM::create_chat_completion`, chosen by the same quality-first,
//! local-among-similar rule as every other capability.

use std::sync::Arc;

use serde_json::{Value, json};
use serial_test::serial;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use smartloop::api::services::model_service::ModelService;
use smartloop::llms::cloud_llm::{CloudLLM, DEFAULT_REASONING_BUDGET_TOKENS};
use smartloop::model::spec::MODEL_SPECS;
use smartloop::workspace::Workspace;

use crate::shared::model_service_a::{attach_workspace, bare_service};
use crate::shared::workspace_support::EnvGuard;

/// Register a remote vision model under a name only these tests use.
fn publish_cloud_vision(name: &str, total_b: f64) {
    let Value::Object(row) = json!({
        "name": name, "type": "remote", "url": "https://api.smartloop.ai",
        "capabilities": ["general", "dp", "vision"], "context_length": 65536,
        "param_size_b": total_b, "active_param_size_b": total_b / 10.0,
    }) else {
        unreachable!()
    };
    MODEL_SPECS.hydrate(&[row]);
}

/// A service whose project selects `local` (a seeded local vision model, by
/// name) and the remote vision model `remote`.
fn svc_with(local: Option<&str>, remote: &str) -> Arc<ModelService> {
    let svc = bare_service();
    let mut ws = Workspace::new("ws");
    if let Some(local) = local {
        ws.add_model(local, local, None, true);
        if let Some(m) = ws.models.iter_mut().find(|m| m.model_name == local) {
            m.downloaded = true;
        }
    }
    ws.add_remote_model(remote, true);
    attach_workspace(&svc, ws);
    svc
}

#[tokio::test]
#[serial]
async fn test_a_cloud_model_receives_the_image_parts() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(|request: &Request| {
            let body: Value = serde_json::from_slice(&request.body).unwrap();
            let part = &body["messages"][0]["content"][1];
            assert_eq!(part["type"], "image_url");
            assert_eq!(part["image_url"]["url"], "data:image/png;base64,AAAA");
            assert_eq!(body["stream"], false);
            assert_eq!(body["max_tokens"], 512 + DEFAULT_REASONING_BUDGET_TOKENS);
            ResponseTemplate::new(200)
                .set_body_json(json!({"choices": [{"message": {"content": "a red bicycle"}, "finish_reason": "stop"}]}))
        })
        .expect(1)
        .mount(&server)
        .await;
    let _env = EnvGuard::set("SLP_API_BASE", server.uri());
    crate::test_cloud_llm::mount_session(&server, &["vision-cloud"]).await;
    let llm = CloudLLM::with_hooks("p1", "vision-cloud", &server.uri(), None, None, Some(Arc::new(|| false)));

    let messages = vec![json!({"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]})];
    let response = tokio::task::spawn_blocking(move || llm.create_chat_completion(messages, Some(512), 0.0))
        .await
        .unwrap()
        .expect("a response");

    assert_eq!(response.pointer("/choices/0/message/content").and_then(Value::as_str), Some("a red bicycle"));
}

/// With no local vision model selected, OCR reads pages on the cloud model.
#[test]
#[serial]
fn test_ocr_falls_back_to_a_remote_vision_model() {
    publish_cloud_vision("vision-cloud-only", 230.0);
    let svc = svc_with(None, "vision-cloud-only");
    let picked = std::sync::Mutex::new(Vec::new());
    let on_selected = |name: &str| picked.lock().unwrap().push(name.to_string());

    let llm = svc.get_vision_llm(Some(&on_selected)).expect("a vision model");

    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert_eq!(llm.model_label(), "vision-cloud-only");
    assert_eq!(*picked.lock().unwrap(), vec!["vision-cloud-only".to_string()]);
}

/// A remote vision model far beyond the local one answers an image question
/// without the local model being loaded.
#[test]
#[serial]
fn test_a_much_larger_remote_model_answers_image_questions() {
    publish_cloud_vision("vision-cloud-large", 230.0);
    let svc = svc_with(Some("qwen3.5-2b"), "vision-cloud-large");

    let llm = svc.get_vision_answer_llm(None).expect("a vision model");

    assert!(ModelService::is_cloud_llm(llm.as_ref()));
    assert_eq!(llm.model_label(), "vision-cloud-large");
    assert!(svc.vision_llm.read().is_none(), "the local vision model must not load");
}

/// Decisions resolve on device: no remote vision model is offered to them.
#[test]
#[serial]
fn test_a_local_only_resolution_never_reaches_a_remote_vision_model() {
    publish_cloud_vision("vision-cloud-only", 230.0);
    let svc = svc_with(None, "vision-cloud-only");
    let _scope = smartloop::llms::cloud_llm::local_only_scope(true);

    assert!(svc.get_vision_answer_llm(None).is_none());
    assert!(svc.get_vision_llm(None).is_none());
}
