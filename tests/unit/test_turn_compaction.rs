//! A turn's prompt is compacted to a working budget for speed: older client
//! tool output is trimmed first, and dropped conversation turns are
//! summarized rather than lost.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Value, json};

use smartloop::api::model::{ChatMessage, Role};
use smartloop::api::routers::chat::{
    ChatService, CitationDict, HistoryTurn, ImageAttachment, KEEP_RECENT_TOOL_RESULTS, OrchestrateArgs,
    OrchestrateOutcome, ProgressCallback, compact_tool_rounds, compact_turn, compaction_window,
};
use smartloop::model_host::client::ModelClient;
use smartloop::workspace::types::Skill;

/// Counts a token per four characters; summarizes on demand.
#[derive(Default)]
struct Fake {
    summary: Option<String>,
    summarized: Mutex<Vec<String>>,
}

impl ChatService for Fake {
    fn is_loaded(&self) -> bool {
        true
    }
    fn load_model(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn get_relevant_history(&self, _s: &str, _q: &str) -> Vec<HistoryTurn> {
        Vec::new()
    }
    fn get_project_skills(&self) -> Vec<Skill> {
        Vec::new()
    }
    fn get_asset_images(&self, _ids: &[String]) -> Vec<ImageAttachment> {
        Vec::new()
    }
    fn resolve_attachment_context(&self, _ids: &[String]) -> String {
        String::new()
    }
    fn orchestrate(&self, _args: OrchestrateArgs) -> anyhow::Result<OrchestrateOutcome> {
        Ok(OrchestrateOutcome::default())
    }
    fn get_project_id(&self) -> Option<String> {
        None
    }
    fn get_project_info(&self) -> Option<String> {
        None
    }
    fn count_tokens(&self, text: &str) -> usize {
        text.chars().count() / 4
    }
    fn get_max_context_tokens(&self) -> usize {
        1_000_000
    }
    fn get_response_max_context_tokens(&self, _c: Option<&str>) -> usize {
        1_000_000
    }
    fn begin_response_request(&self, _p: Option<ProgressCallback>) {}
    fn announce_response_model(&self, _c: Option<&str>) {}
    fn append_to_session(&self, _s: &str, _r: &str, _c: &str, _ci: Option<&[CitationDict]>, _m: Option<&str>) {}
    fn model_client(&self) -> Arc<dyn ModelClient> {
        unimplemented!("not used by compaction")
    }
    fn unload_auxiliary_models(&self, _e: bool) {}
    fn unload_embedding_model(&self) {}
    fn workspace_dir(&self) -> Option<String> {
        None
    }
    fn workspace_project_ids(&self) -> Option<Vec<String>> {
        None
    }
    fn current_project_dir(&self) -> Option<String> {
        None
    }
    fn push_token_usage(&self, _p: Option<&str>, _m: &str, _pt: usize, _ct: usize) {}
    fn summarize_history(&self, transcript: &str) -> Option<String> {
        self.summarized.lock().push(transcript.to_string());
        self.summary.clone()
    }
}

fn msg(role: Role, content: &str) -> ChatMessage {
    ChatMessage { role, content: Some(content.to_string()), ..Default::default() }
}

fn tool_round(id: usize, output: &str) -> Vec<Value> {
    vec![
        json!({"role": "assistant", "content": "", "tool_calls": [
            {"id": format!("call_{id}"), "type": "function", "function": {"name": "shell", "arguments": "{}"}}
        ]}),
        json!({"role": "tool", "tool_call_id": format!("call_{id}"), "name": "shell", "content": output}),
    ]
}

#[test]
fn the_working_budget_caps_a_large_model_window() {
    assert_eq!(compaction_window(1_048_576, 32_768), 32_768);
    assert_eq!(compaction_window(8_192, 32_768), 8_192);
    assert_eq!(compaction_window(1_048_576, 0), 1_048_576);
}

#[test]
fn older_tool_output_is_trimmed_and_recent_output_and_every_call_kept() {
    let svc = Fake::default();
    let big = format!("total 48\n{}", "drwxr-xr-x  file\n".repeat(400));
    let rounds: Vec<Value> = (0..5).flat_map(|i| tool_round(i, &big)).collect();

    let compacted = compact_tool_rounds(rounds.clone(), &svc, 2_000);

    let tools: Vec<&Value> = compacted.iter().filter(|m| m["role"] == "tool").collect();
    assert_eq!(tools.len(), 5);
    let (old, recent) = tools.split_at(5 - KEEP_RECENT_TOOL_RESULTS);
    for t in old {
        let content = t["content"].as_str().unwrap();
        assert!(content.starts_with("[earlier tool output trimmed"), "{content}");
        assert!(content.contains("total 48"), "the stub names what the output began with");
    }
    for t in recent {
        assert_eq!(t["content"], big.as_str(), "the latest results stay whole");
    }
    // Every call the model made survives, so it knows what it has done.
    assert_eq!(compacted.iter().filter(|m| m["role"] == "assistant").count(), 5);
}

#[test]
fn a_prompt_within_budget_is_left_alone() {
    let svc = Fake::default();
    let messages = vec![msg(Role::User, "write my_shell.sh")];
    let rounds = tool_round(0, "ok");
    let (m, r) = compact_turn(messages.clone(), rounds.clone(), &svc, 2048, 0, 32_768);
    assert_eq!(m, messages);
    assert_eq!(r, rounds);
    assert!(svc.summarized.lock().is_empty());
}

#[test]
fn dropped_turns_are_summarized_into_the_first_message() {
    let svc = Fake { summary: Some("- user wants a starship setup script".into()), ..Default::default() };
    let filler = "x".repeat(8_000);
    let mut messages = vec![msg(Role::User, "Implement the spec")];
    for _ in 0..6 {
        messages.push(msg(Role::Assistant, &filler));
        messages.push(msg(Role::User, &filler));
    }
    messages.push(msg(Role::Assistant, "done"));
    messages.push(msg(Role::User, "now configure kitty"));

    let (m, _) = compact_turn(messages.clone(), Vec::new(), &svc, 512, 0, 6_000);

    assert!(m.len() < messages.len());
    assert_eq!(m.last().unwrap().content.as_deref(), Some("now configure kitty"));
    let first = m[0].content.as_deref().unwrap();
    assert!(first.starts_with("Implement the spec"));
    assert!(first.contains("starship setup script"), "{first}");
    assert_eq!(svc.summarized.lock().len(), 1);
}
