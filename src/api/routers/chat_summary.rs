//! `POST /v1/chat/summary`: a few lines on what a finished client task did.
//!
//! A coding client (docent) runs a task over many requests: tool rounds,
//! file writes, checks and fix rounds, most of which the server never sees.
//! When the task settles the client sends the record it kept, and the
//! project's answer model turns it into a summary for the user.

use std::sync::Arc;

use axum::routing::post;
use axum::{Json, Router};

use crate::api::model::{TaskSummaryRequest, TaskSummaryResponse};
use crate::api::services::get_model_service;
use crate::workspace::context::scope_project;

use super::_errors::{HTTPException, ValidatedJson};
use super::chat::{known_project, run_on_executor, ChatService};

/// Most of the user's request kept in the record, in characters.
pub const REQUEST_MAX_CHARS: usize = 2_000;
/// Most of the final reply kept, in characters.
pub const REPLY_MAX_CHARS: usize = 4_000;
/// Most of each command's output kept (its end), in characters.
pub const OUTPUT_MAX_CHARS: usize = 800;
/// How many commands are kept; the latest win.
pub const MAX_COMMANDS: usize = 20;
/// How many files are listed; the rest are counted.
pub const MAX_FILES: usize = 40;

/// Routes under `/v1` (tag `chat`).
pub fn router() -> Router {
    Router::new().route("/v1/chat/summary", post(summarize_task))
}

/// `POST /v1/chat/summary` — summarize a finished client task.
#[utoipa::path(
    post,
    path = "/v1/chat/summary",
    tag = "chat",
    request_body = TaskSummaryRequest,
    summary = "Summarize a finished task",
    description = "A client that ran a task over several turns sends what it saw the task do: files changed, commands and checks run, the last reply. The project's answer model writes a few lines on it; `summary` is null when none could be written.",
    responses((status = 200, description = "OK", body = TaskSummaryResponse), (status = 422, description = "Invalid request"))
)]
pub async fn summarize_task(
    ValidatedJson(request): ValidatedJson<TaskSummaryRequest>,
) -> Result<Json<TaskSummaryResponse>, HTTPException> {
    Ok(Json(summarize_task_with(request, get_model_service()).await))
}

/// [`summarize_task`] against an explicit service (test seam).
pub async fn summarize_task_with(
    request: TaskSummaryRequest,
    model_service: Arc<dyn ChatService>,
) -> TaskSummaryResponse {
    let record = task_record(&request);
    if record.is_empty() {
        return TaskSummaryResponse { summary: None };
    }
    let binding = known_project(request.project_id.as_deref(), &*model_service);
    let summary = scope_project(binding, run_on_executor(move || model_service.summarize_task(&record)))
        .await
        .ok()
        .flatten()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    TaskSummaryResponse { summary }
}

/// The record the summarizer reads, each part capped; empty when the task
/// did nothing worth summarizing (no files, commands, checks or reply).
pub fn task_record(request: &TaskSummaryRequest) -> String {
    let reply = request.reply.trim();
    if request.files.is_empty() && request.commands.is_empty() && request.checks.is_empty() && reply.is_empty() {
        return String::new();
    }
    let mut out = format!("Task: {}\n", head(request.request.trim(), REQUEST_MAX_CHARS));

    if !request.files.is_empty() {
        out.push_str("\nFiles:\n");
        for file in request.files.iter().take(MAX_FILES) {
            out.push_str(&format!("- {} {} (+{} -{})\n", file.change, file.path, file.added, file.removed));
        }
        if request.files.len() > MAX_FILES {
            out.push_str(&format!("- and {} more\n", request.files.len() - MAX_FILES));
        }
    }

    if !request.commands.is_empty() {
        out.push_str("\nCommands:\n");
        let skipped = request.commands.len().saturating_sub(MAX_COMMANDS);
        if skipped > 0 {
            out.push_str(&format!("- ({skipped} earlier commands omitted)\n"));
        }
        for command in &request.commands[skipped..] {
            let status = match command.exit_code {
                Some(code) => format!("exit {code}"),
                None => "stopped".to_string(),
            };
            out.push_str(&format!("- `{}` -> {status}\n", command.command.trim()));
            let output = tail(command.output.trim(), OUTPUT_MAX_CHARS);
            if !output.is_empty() {
                for line in output.lines() {
                    out.push_str(&format!("    {line}\n"));
                }
            }
        }
    }

    if !request.checks.is_empty() {
        out.push_str("\nChecks:\n");
        for check in &request.checks {
            let result = if check.passed { "passed" } else { "failed" };
            out.push_str(&format!("- `{}`: {result}\n", check.command.trim()));
        }
    }

    if !reply.is_empty() {
        out.push_str(&format!("\nLast reply:\n{}\n", head(reply, REPLY_MAX_CHARS)));
    }
    out
}

/// The first `max` characters of `text`, marked when cut.
fn head(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", &text[..at]),
        None => text.to_string(),
    }
}

/// The last `max` characters of `text`, marked when cut.
fn tail(text: &str, max: usize) -> String {
    let count = text.chars().count();
    if count <= max {
        return text.to_string();
    }
    let at = text.char_indices().nth(count - max).map_or(0, |(i, _)| i);
    format!("…{}", &text[at..])
}
