//! The attachment token limit follows the model that will read it.
//!
//! `upload_asset` measures the parsed markdown with the local orchestrator's
//! tokenizer.  With a cloud model selected alone, that model answers with the
//! whole attachment, so the check is against its context window — not the
//! small local model's, which rejected documents the cloud model takes easily.

use std::sync::Arc;

use serial_test::serial;

use smartloop::api::services::model_service::{ModelService, ModelServiceHooks};
use smartloop::model::spec::MODEL_SPECS;
use smartloop::types::ModelKind;
use smartloop::workspace::WorkspaceModel;

use crate::shared::agents_mcp_agent::StubLLM;
use crate::shared::workspace_support::TempHome;

const LOCAL_MAX: usize = 4096;
const DOCUMENT_TOKENS: usize = 7640;

/// A remote model whose spec declares `context_length`, under a name only
/// these tests use.
fn remote_model(name: &str, context_length: u32) -> WorkspaceModel {
    let serde_json::Value::Object(row) = serde_json::json!({
        "name": name, "type": "remote", "capabilities": ["general"],
        "context_length": context_length,
    }) else {
        unreachable!()
    };
    MODEL_SPECS.hydrate(&[row]);
    WorkspaceModel { model_name: name.to_string(), kind: ModelKind::Remote, ..Default::default() }
}

/// A service whose orchestrator holds `LOCAL_MAX` tokens and measures every
/// document at `DOCUMENT_TOKENS`, with `pinned` as the sole selection.
fn service(pinned: Option<WorkspaceModel>) -> Arc<ModelService> {
    let llm = StubLLM::new("");
    llm.set_max_tokens(LOCAL_MAX);
    llm.set_num_tokens(DOCUMENT_TOKENS);

    let svc = ModelService::new_bare();
    let mut hooks = ModelServiceHooks::default();
    let local = llm.as_llm();
    hooks.load_model = Some(Arc::new(move || Ok(local.clone())));
    hooks.pinned_model = Some(Arc::new(move |_cap: Option<&str>| pinned.clone()));
    svc.set_hooks(hooks);
    svc
}

fn upload(svc: &ModelService) -> anyhow::Result<serde_json::Value> {
    svc.upload_asset(b"a long document", "notes.txt", Some("text/plain"))
}

#[test]
#[serial]
fn test_the_local_model_limits_an_attachment_without_a_cloud_pin() {
    let _home = TempHome::new();
    let err = upload(&service(None)).expect_err("over the local context");
    let message = err.to_string();
    assert!(message.contains("7,640 tokens"), "{message}");
    assert!(message.contains("maximum is 4,096"), "{message}");
}

#[test]
#[serial]
fn test_a_pinned_cloud_model_limits_an_attachment_by_its_context() {
    let _home = TempHome::new();
    let svc = service(Some(remote_model("attach-cloud-wide", 196_608)));
    let asset = upload(&svc).expect("fits the cloud context");
    assert_eq!(asset["markdown"], true);
}

#[test]
#[serial]
fn test_a_pinned_cloud_model_still_rejects_what_it_cannot_hold() {
    let _home = TempHome::new();
    let svc = service(Some(remote_model("attach-cloud-narrow", 6000)));
    let err = upload(&svc).expect_err("over the cloud context");
    assert!(err.to_string().contains("maximum is 6,000"), "{err}");
}

#[test]
#[serial]
fn test_a_cloud_model_without_a_declared_context_keeps_the_local_limit() {
    let _home = TempHome::new();
    let no_context = WorkspaceModel {
        model_name: "attach-cloud-unsized".to_string(),
        kind: ModelKind::Remote,
        ..Default::default()
    };
    let err = upload(&service(Some(no_context))).expect_err("falls back to the local context");
    assert!(err.to_string().contains("maximum is 4,096"), "{err}");
}
