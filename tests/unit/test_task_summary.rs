//! Task summary: a client that ran a task over several turns sends the record
//! it kept to `POST /v1/chat/summary`, and the answer model turns it into a
//! few lines.  The record is capped before the model reads it, and an empty
//! one never reaches the model.

use std::sync::Arc;

use smartloop::api::model::{TaskCheck, TaskCommand, TaskFile, TaskSummaryRequest};
use smartloop::api::routers::chat_summary::{
    summarize_task_with, task_record, MAX_COMMANDS, OUTPUT_MAX_CHARS, REPLY_MAX_CHARS,
};

use crate::shared::api_chat::FakeChatService;

fn finished_task() -> TaskSummaryRequest {
    TaskSummaryRequest {
        request: "add a --verbose flag".to_string(),
        reply: "Done.".to_string(),
        files: vec![TaskFile { path: "src/main.rs".to_string(), change: "updated".to_string(), added: 12, removed: 3 }],
        commands: vec![TaskCommand { command: "cargo build".to_string(), exit_code: Some(0), output: "Finished".to_string() }],
        checks: vec![TaskCheck { command: "cargo test".to_string(), passed: true }],
        ..TaskSummaryRequest::default()
    }
}

/// The request as docent sends it: docent's tests hold the same file and
/// check its request body against it, so a field renamed on either side
/// fails a test rather than every summary.  Fields here default when
/// missing, so the request is read back out too: a field this side doesn't
/// know would otherwise be dropped without a word.
const CONTRACT: &str = include_str!("../fixtures/task-summary-request.json");

#[test]
fn test_the_request_keeps_to_the_docent_contract() {
    let request: TaskSummaryRequest = serde_json::from_str(CONTRACT).expect("the contract parses");
    assert_eq!(
        request,
        TaskSummaryRequest {
            project_id: Some("p1".to_string()),
            request: "add a --verbose flag".to_string(),
            reply: "Done.".to_string(),
            files: vec![
                TaskFile { path: "src/main.rs".to_string(), change: "updated".to_string(), added: 12, removed: 3 },
                TaskFile { path: "src/cli.rs".to_string(), change: "created".to_string(), added: 40, removed: 0 },
            ],
            commands: vec![
                TaskCommand { command: "cargo build".to_string(), exit_code: Some(0), output: "Finished".to_string() },
                TaskCommand { command: "cargo run".to_string(), exit_code: None, output: String::new() },
            ],
            checks: vec![TaskCheck { command: "cargo test".to_string(), passed: true }],
        }
    );
    let contract: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
    assert_eq!(serde_json::to_value(&request).unwrap(), contract, "every field docent sends is read");
}

#[test]
fn test_the_record_lists_what_the_task_did() {
    let record = task_record(&finished_task());
    assert!(record.starts_with("Task: add a --verbose flag\n"), "{record}");
    assert!(record.contains("- updated src/main.rs (+12 -3)"), "{record}");
    assert!(record.contains("- `cargo build` -> exit 0\n    Finished"), "{record}");
    assert!(record.contains("- `cargo test`: passed"), "{record}");
    assert!(record.contains("Last reply:\nDone."), "{record}");
}

#[test]
fn test_a_task_that_did_nothing_has_no_record() {
    let request = TaskSummaryRequest { request: "hi".to_string(), ..TaskSummaryRequest::default() };
    assert_eq!(task_record(&request), "");
}

#[test]
fn test_long_parts_are_capped() {
    let mut request = finished_task();
    request.reply = "r".repeat(REPLY_MAX_CHARS + 500);
    request.commands = (0..MAX_COMMANDS + 5)
        .map(|i| TaskCommand { command: format!("step {i}"), exit_code: Some(0), output: format!("{}END", "o".repeat(OUTPUT_MAX_CHARS)) })
        .collect();
    let record = task_record(&request);
    assert!(record.contains("(5 earlier commands omitted)"), "{record}");
    assert!(!record.contains("`step 4`"), "the oldest commands are dropped");
    assert!(record.contains(&format!("`step {}`", MAX_COMMANDS + 4)), "the latest command is kept");
    assert!(record.contains("END"), "an output keeps its end");
    assert!(!record.contains(&"o".repeat(OUTPUT_MAX_CHARS)), "an output is cut to its tail");
    assert!(!record.contains(&"r".repeat(REPLY_MAX_CHARS + 1)), "the reply is cut");
}

#[tokio::test]
async fn test_the_model_summary_comes_back() {
    let svc = Arc::new(FakeChatService::default());
    *svc.task_summary.lock() = Some("  Added a --verbose flag to src/main.rs; tests pass.\n".to_string());
    let response = summarize_task_with(finished_task(), svc.clone()).await;
    assert_eq!(response.summary.as_deref(), Some("Added a --verbose flag to src/main.rs; tests pass."));
    assert_eq!(svc.task_records.lock().as_slice(), [task_record(&finished_task())]);
}

#[tokio::test]
async fn test_no_summary_is_null() {
    let svc = Arc::new(FakeChatService::default());
    let response = summarize_task_with(finished_task(), svc.clone()).await;
    assert_eq!(response.summary, None);

    *svc.task_summary.lock() = Some("   ".to_string());
    assert_eq!(summarize_task_with(finished_task(), svc).await.summary, None, "a blank summary is no summary");
}

#[tokio::test]
async fn test_an_empty_task_never_reaches_the_model() {
    let svc = Arc::new(FakeChatService::default());
    *svc.task_summary.lock() = Some("unused".to_string());
    let request = TaskSummaryRequest { request: "hi".to_string(), ..TaskSummaryRequest::default() };
    assert_eq!(summarize_task_with(request, svc.clone()).await.summary, None);
    assert!(svc.task_records.lock().is_empty());
}
