//! Port of `tests/unit/test_planner.py`.
//!
//! Unit tests for the Planner module (classify → plan pipeline).

use std::any::Any;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Map, Value};

use smartloop::agents::planner::{
    extract_paths, ClassificationResult, ClassifyOptions, ExecutionPlan, IntentType, MCPToolContext,
    MatchedTool, ModelTier, PlanContext, Planner, StepType, BASH_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, GREP_TOOL,
    SYSTEM_TOOLS_SERVER_ID,
};
use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationStream};
use smartloop::llms::BaseLLM;

// ------------------------------------------------------------------ //
// Fixtures                                                            //
// ------------------------------------------------------------------ //

/// `MagicMock` LLM: `invoke` returns a settable value (default `"[]"`),
/// `get_max_tokens` 4096, `get_num_tokens` 100.
struct MockLlm {
    invoke_return: Mutex<String>,
    invoke_calls: AtomicUsize,
}

impl MockLlm {
    fn set_invoke_return(&self, value: &str) {
        *self.invoke_return.lock() = value.to_string();
    }

    fn invoke_called(&self) -> bool {
        self.invoke_calls.load(Ordering::SeqCst) > 0
    }
}

impl BaseLLM for MockLlm {
    fn project_id(&self) -> &str {
        ""
    }
    fn get_num_tokens(&self, _text: &str) -> usize {
        100
    }
    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }
    fn get_max_tokens(&self) -> usize {
        4096
    }
    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        Ok(self.invoke_return.lock().clone())
    }
    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }
    fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        self.invoke_calls.fetch_add(1, Ordering::SeqCst);
        self.invoke_return.lock().clone()
    }
    fn cancelled(&self) -> Option<CancelEvent> {
        None
    }
    fn set_cancelled(&self, _event: Option<CancelEvent>) {}
    fn llm_type(&self) -> &str {
        "mock"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn mock_llm() -> Arc<MockLlm> {
    Arc::new(MockLlm { invoke_return: Mutex::new("[]".to_string()), invoke_calls: AtomicUsize::new(0) })
}

fn planner_with(llm: &Arc<MockLlm>) -> Planner {
    Planner::new(llm.clone())
}

fn planner() -> Planner {
    planner_with(&mock_llm())
}

fn sample_servers() -> Vec<Value> {
    vec![
        json!({
            "id": "srv-1",
            "name": "Fetch Server",
            "tools": [
                {"name": "fetch", "description": "Fetch a URL and return content"},
            ],
        }),
        json!({
            "id": "srv-2",
            "name": "GitHub",
            "tools": [
                {"name": "create-issue", "description": "Create a GitHub issue"},
            ],
        }),
    ]
}

fn tool(name: &str, description: &str) -> Map<String, Value> {
    json!({"name": name, "description": description}).as_object().unwrap().clone()
}

fn sample_mcp_tools() -> Vec<MCPToolContext> {
    vec![
        MCPToolContext {
            server_id: "srv-1".into(),
            server_name: "Fetch Server".into(),
            tools: vec![tool("fetch", "Fetch a URL and return content")],
        },
        MCPToolContext {
            server_id: "srv-2".into(),
            server_name: "GitHub".into(),
            tools: vec![tool("create-issue", "Create a GitHub issue")],
        },
    ]
}

fn args(pairs: &[(&str, &str)]) -> Option<BTreeMap<String, String>> {
    Some(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
}

fn ctx(user_message: &str) -> PlanContext {
    PlanContext { user_message: user_message.into(), ..Default::default() }
}

fn path_str(p: &std::path::Path) -> String {
    p.to_str().unwrap().to_string()
}

// ------------------------------------------------------------------ //
// 1. build_context                                                     //
// ------------------------------------------------------------------ //

#[test]
fn test_build_context() {
    let planner = planner();
    let ctx = planner.build_context(
        "Hello world",
        &sample_servers(),
        Some("/tmp"),
        Some(vec!["Be polite".to_string()]),
        Some("Some docs"),
    );
    assert_eq!(ctx.user_message, "Hello world");
    assert_eq!(ctx.skills, Some(vec!["Be polite".to_string()]));
    assert_eq!(ctx.document_context.as_deref(), Some("Some docs"));
    assert_eq!(ctx.cwd.as_deref(), Some("/tmp"));
    assert_eq!(ctx.mcp_tools.len(), 2);
    assert_eq!(ctx.mcp_tools[0].server_id, "srv-1");
    assert_eq!(ctx.mcp_tools[0].server_name, "Fetch Server");
}

#[test]
fn test_build_context_filters_system_tools() {
    let planner = planner();
    let servers = vec![
        json!({"id": "__system_tools__", "name": "smartloop", "tools": [{"name": "explore"}]}),
        json!({"id": "ext-1", "name": "External", "tools": [{"name": "foo"}]}),
    ];
    let ctx = planner.build_context("test", &servers, None, None, None);
    assert_eq!(ctx.mcp_tools.len(), 1);
    assert_eq!(ctx.mcp_tools[0].server_id, "ext-1");
}

#[test]
fn test_build_context_detects_urls() {
    let planner = planner();
    let ctx = planner.build_context("Fetch https://example.com/page and summarize", &[], None, None, None);
    assert!(ctx.detected_urls.contains(&"https://example.com/page".to_string()));
}

// ------------------------------------------------------------------ //
// 2. classify — deterministic (Tier 1)                                #
// ------------------------------------------------------------------ //

/// No signals → LLM fallback → default CONVERSATION.
#[test]
fn test_classify_conversation() {
    let planner = planner();
    let result = planner.classify("hello", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Conversation);
}

/// MCP tool keyword match → TOOL_CALL.
#[test]
fn test_classify_tool_call_mcp_keyword() {
    let planner = planner();
    let result = planner.classify("create an issue on GitHub", &sample_mcp_tools(), ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::ToolCall);
    assert!(!result.matched_tools.is_empty());
    assert_eq!(result.matched_tools[0].tool_name, "create-issue");
}

/// URL alone is not a signal — falls through to LLM classifier.
/// Explore step handles fetch internally.
#[test]
fn test_classify_url_no_signal() {
    let llm = mock_llm();
    let planner = planner_with(&llm);
    llm.set_invoke_return("CONVERSATION");
    let result =
        planner.classify("Summarize https://example.com/article", &sample_mcp_tools(), ClassifyOptions::default());
    assert!(result.detected_urls.contains(&"https://example.com/article".to_string()));
}

/// Bash block → ACTION(bash).
#[test]
fn test_classify_action_bash() {
    let planner = planner();
    let result = planner.classify("Run this:\n```bash\necho hello\n```", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Action);
    assert_eq!(result.action_kind.as_deref(), Some("bash"));
    assert!(result.bash_commands.contains(&"echo hello".to_string()));
}

/// File path → ACTION(read).
#[test]
fn test_classify_action_read() {
    let planner = planner();
    let tmp = tempfile::tempdir().unwrap();
    let tmp_path = tmp.path().canonicalize().unwrap();
    let test_file = tmp_path.join("test.py");
    std::fs::write(&test_file, "print('hello')").unwrap();
    let result = planner.classify(
        &format!("Read {}", test_file.display()),
        &[],
        ClassifyOptions { cwd: Some(path_str(&tmp_path)), ..Default::default() },
    );
    assert_eq!(result.intent, IntentType::Action);
    assert_eq!(result.action_kind.as_deref(), Some("read"));
}

/// Directory path → EXPLORE.
#[test]
fn test_classify_explore_directory() {
    let planner = planner();
    let tmp = tempfile::tempdir().unwrap();
    let tmp_path = tmp.path().canonicalize().unwrap();
    let sub = tmp_path.join("src");
    std::fs::create_dir(&sub).unwrap();
    let result = planner.classify(
        &format!("Explore {}", sub.display()),
        &[],
        ClassifyOptions { cwd: Some(path_str(&tmp_path)), ..Default::default() },
    );
    assert_eq!(result.intent, IntentType::Explore);
}

/// Grep keywords → ACTION(grep).
#[test]
fn test_classify_action_grep() {
    let planner = planner();
    let result = planner.classify("search for TODO in the codebase", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Action);
    assert_eq!(result.action_kind.as_deref(), Some("grep"));
}

/// Document context available + no action signals → EXPLORE with document flag.
#[test]
fn test_classify_document_context() {
    let planner = planner();
    let result = planner.classify(
        "What does the policy say about vacation days?",
        &[],
        ClassifyOptions {
            document_context: Some("Company policy: employees get 20 vacation days per year.".into()),
            ..Default::default()
        },
    );
    assert_eq!(result.intent, IntentType::Explore);
    assert!(result.has_document_context);
}

/// Document context + MCP tool match + URL → MULTI_STEP (fetch + compare).
#[test]
fn test_classify_document_context_with_tool_signal() {
    let planner = planner();
    let result = planner.classify(
        "Fetch https://example.com and compare to our docs",
        &sample_mcp_tools(),
        ClassifyOptions {
            document_context: Some("Some relevant document content here.".into()),
            ..Default::default()
        },
    );
    assert_eq!(result.intent, IntentType::MultiStep);
    assert!(result.has_document_context);
}

/// Bare URL → deterministic TOOL_CALL (no LLM needed).
#[test]
fn test_classify_url_only_deterministic_fetch() {
    let planner = planner();
    let result = planner.classify("https://example.com/some-page", &sample_mcp_tools(), ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::ToolCall);
    assert_eq!(result.detected_urls, vec!["https://example.com/some-page".to_string()]);
}

/// URL with surrounding text → deterministic TOOL_CALL.
#[test]
fn test_classify_url_with_instruction_deterministic_fetch() {
    let planner = planner();
    let result = planner.classify("summarize https://example.com/article", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::ToolCall);
    assert_eq!(result.detected_urls.len(), 1);
}

/// URL + write intent → MULTI_STEP (fetch then save).
#[test]
fn test_classify_multi_step_write_intent() {
    let planner = planner();
    let result = planner.classify(
        "Fetch https://example.com and save it to output.md",
        &sample_mcp_tools(),
        ClassifyOptions::default(),
    );
    assert!(result.has_write_intent);
    assert!(!result.detected_urls.is_empty());
}

/// Write intent alone (no paths/URLs) → MULTI_STEP.
#[test]
fn test_classify_multi_step_write_only() {
    let planner = planner();
    let result = planner.classify(
        "Write a file with a summary of the project to output.md",
        &[],
        ClassifyOptions::default(),
    );
    assert_eq!(result.intent, IntentType::MultiStep);
    assert!(result.has_write_intent);
}

/// No deterministic signals → LLM classify called.
#[test]
fn test_classify_llm_fallback() {
    let llm = mock_llm();
    let planner = planner_with(&llm);
    llm.set_invoke_return("TOOL_CALL");
    let result = planner.classify(
        "do the thing with the API",
        &[],
        ClassifyOptions { skills: Some(vec!["Use the API server for API calls".into()]), ..Default::default() },
    );
    assert_eq!(result.intent, IntentType::ToolCall);
    assert!(llm.invoke_called());
}

/// LLM returns garbage → default to CONVERSATION.
#[test]
fn test_classify_llm_fallback_unparseable() {
    let llm = mock_llm();
    let planner = planner_with(&llm);
    llm.set_invoke_return("I don't know what to do");
    let result = planner.classify("something ambiguous", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Conversation);
}

/// In a code project, a request to change something or about the code is
/// coding work; a question about a document there, or the same request
/// outside a project, is not.
#[test]
fn test_detect_signals_coding_intent_from_the_folder_and_the_ask() {
    let planner = planner();
    let dir = std::env::temp_dir().join(format!("slp-planner-code-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let here = dir.to_string_lossy().into_owned();
    let coding = |msg: &str| planner.detect_signals(msg, &[], Some(&here), None, false).has_coding_intent;
    // A plain folder: only outright coding phrases count.
    assert!(!smartloop::agents::planner::is_code_project(&here));
    assert!(!coding("add a greeting and call it from main"));
    assert!(coding("write a function that parses ISO dates"));
    std::fs::write(dir.join("Cargo.toml"), "[package]\nname = \"t\"\n").unwrap();
    assert!(smartloop::agents::planner::is_code_project(&here));
    for msg in [
        "add a greeting and call it from main",
        "why does it panic on startup?",
        "rename the config flag to --verbose",
        "the build fails with a warning",
        "update the readme",
    ] {
        assert!(coding(msg), "expected coding intent in a code project: {msg}");
    }
    for msg in [
        "what does the contract say about renewals?",
        "summarize the quarterly report",
        "who is the author of this proposal?",
    ] {
        assert!(!coding(msg), "unexpected coding intent in a code project: {msg}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Programming requests raise the coding signal; look-alikes don't.
#[test]
fn test_detect_signals_coding_intent() {
    let planner = planner();
    for msg in [
        "refactor this function to be async",
        "why does main.rs panic on startup?",
        "write a function that parses ISO dates",
        "here is the error:\n```\nTypeError: x is undefined\n```",
        "explain this code to me",
        "fix the bug in the login flow",
        "how do I do this in python",
        "write a SQL query for monthly revenue",
        "create a rust project",
        "scaffold a React app with routing",
        "set up a node.js server",
    ] {
        let result = planner.detect_signals(msg, &[], None, None, false);
        assert!(result.has_coding_intent, "expected coding intent: {msg}");
    }
    for msg in [
        "what is the zip code for Seattle?",
        "compile a list of hotels in Lisbon",
        "explain the function of the liver",
        "write a script for my wedding toast",
        "summarize the code of conduct",
        "what does this will say about the house?",
        "create a project plan for the launch",
        "rust removal tips for my bike",
    ] {
        let result = planner.detect_signals(msg, &[], None, None, false);
        assert!(!result.has_coding_intent, "unexpected coding intent: {msg}");
    }
}

/// Coding work goes to the document-processing model, ahead of a skill,
/// but not of attachments, and only when a selected model declares `dp`.
#[test]
fn test_match_capability_coding_goes_to_document_processing() {
    let all: Vec<String> = ["general", "reasoning", "dp", "vision"].iter().map(|s| s.to_string()).collect();
    assert_eq!(Planner::match_capability(&all, false, true, true, true).as_deref(), Some("dp"));
    assert_eq!(Planner::match_capability(&all, true, true, true, true).as_deref(), Some("vision"));
    assert_eq!(Planner::match_capability(&all, false, false, true, true).as_deref(), Some("reasoning"));

    let no_dp: Vec<String> = ["general", "reasoning"].iter().map(|s| s.to_string()).collect();
    assert_eq!(Planner::match_capability(&no_dp, false, true, true, false).as_deref(), Some("reasoning"));
    assert_eq!(Planner::match_capability(&no_dp, false, true, false, false).as_deref(), Some("general"));
}

/// A coding conversation plans one LARGE step on the document-processing
/// model.
#[test]
fn test_classify_coding_conversation_routes_to_document_processing() {
    let planner = planner();
    let result = planner.classify(
        "refactor this function to use iterators",
        &[],
        ClassifyOptions {
            available_capabilities: Some(vec!["general".into(), "dp".into()]),
            ..Default::default()
        },
    );
    assert!(result.has_coding_intent);
    assert_eq!(result.capability.as_deref(), Some("dp"));
    let plan = planner.create_plan(&result, &ctx("refactor this function to use iterators"), None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].model_tier, ModelTier::Large);
    assert_eq!(plan.steps[0].capability.as_deref(), Some("dp"));
}

// ------------------------------------------------------------------ //
// 3. create_plan with ClassificationResult                            //
// ------------------------------------------------------------------ //

#[test]
fn test_create_plan_conversation_without_capability() {
    // CONVERSATION intent with no capability matched → empty plan so the
    // router can stream via the small model directly.
    let planner = planner();
    let classification = ClassificationResult { intent: IntentType::Conversation, ..Default::default() };
    let plan: ExecutionPlan = planner.create_plan(&classification, &ctx("hello"), None);
    assert_eq!(plan.steps.len(), 0);
    assert!(plan.reasoning.starts_with("conversation"));
}

#[test]
fn test_create_plan_conversation_with_knowledge_capability() {
    // CONVERSATION intent + knowledge capability → one CONVERSATIONAL step
    // routed to the knowledge-capable model (e.g. gemma-3-4b).
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Conversation,
        capability: Some("knowledge".into()),
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("what are some places to visit in madrid"), None);
    assert_eq!(plan.steps.len(), 1);
    let step = &plan.steps[0];
    assert_eq!(step.step_type, StepType::Conversational);
    assert_eq!(step.capability.as_deref(), Some("knowledge"));
    assert_eq!(step.instruction, "what are some places to visit in madrid");
}

#[test]
fn test_create_plan_tool_call() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::ToolCall,
        matched_tools: vec![MatchedTool {
            server_id: "srv-2".into(),
            server_name: "GitHub".into(),
            tool_name: "create-issue".into(),
        }],
        ..Default::default()
    };
    let ctx = planner.build_context("create issue", &sample_servers(), None, None, None);
    let plan = planner.create_plan(&classification, &ctx, None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some("create-issue"));
    assert_eq!(plan.steps[0].server_id.as_deref(), Some("srv-2"));
}

#[test]
fn test_create_plan_tool_call_url_fetch() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::ToolCall,
        detected_urls: vec!["https://example.com/page".into()],
        ..Default::default()
    };
    let ctx = planner.build_context("Fetch https://example.com/page", &sample_servers(), None, None, None);
    let plan = planner.create_plan(&classification, &ctx, None);
    assert_eq!(plan.steps.len(), 1);
    // URL fetching is now a dedicated FETCH step run by the FetchAgent, not
    // an MCP tool call.
    assert_eq!(plan.steps[0].step_type, StepType::Fetch);
    assert_eq!(plan.steps[0].default_args, args(&[("url", "https://example.com/page")]));
}

#[test]
fn test_create_plan_action_bash() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("bash".into()),
        bash_commands: vec!["echo hello".into()],
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("Run:\n```bash\necho hello\n```"), None);
    assert_eq!(plan.steps.len(), 1);
    // System tools now run as TOOL steps against the smartloop MCP server.
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some(BASH_TOOL));
    assert_eq!(plan.steps[0].server_id.as_deref(), Some(SYSTEM_TOOLS_SERVER_ID));
    assert_eq!(plan.steps[0].default_args, args(&[("command", "echo hello")]));
    assert!(plan.steps[0].instruction.contains("echo hello"));
}

#[test]
fn test_create_plan_action_read() {
    let planner = planner();
    let tmp = tempfile::tempdir().unwrap();
    let test_file = tmp.path().join("test.py");
    std::fs::write(&test_file, "print('hello')").unwrap();
    let test_file = path_str(&test_file);
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("read".into()),
        detected_paths: vec![test_file.clone()],
        ..Default::default()
    };
    let ctx = PlanContext {
        user_message: format!("Read {test_file}"),
        cwd: Some(path_str(tmp.path())),
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx, None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some(FILE_READ_TOOL));
    assert_eq!(plan.steps[0].server_id.as_deref(), Some(SYSTEM_TOOLS_SERVER_ID));
    assert_eq!(plan.steps[0].default_args, args(&[("path", &test_file)]));
}

#[test]
fn test_create_plan_action_grep() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("grep".into()),
        has_grep_intent: true,
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("search for TODO"), None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some(GREP_TOOL));
    assert_eq!(plan.steps[0].server_id.as_deref(), Some(SYSTEM_TOOLS_SERVER_ID));
}

#[test]
fn test_create_plan_explore() {
    let planner = planner();
    let tmp = tempfile::tempdir().unwrap();
    let sub = tmp.path().join("src");
    std::fs::create_dir(&sub).unwrap();
    let sub = path_str(&sub);
    let classification =
        ClassificationResult { intent: IntentType::Explore, detected_paths: vec![sub.clone()], ..Default::default() };
    let ctx = PlanContext {
        user_message: format!("Explore {sub}"),
        cwd: Some(path_str(tmp.path())),
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx, None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Explore);
}

/// MULTI_STEP with LLM returning empty → falls back to signal-based steps.
#[test]
fn test_create_plan_multi_step_from_signals() {
    let llm = mock_llm();
    let planner = planner_with(&llm);
    llm.set_invoke_return("[]");
    let classification = ClassificationResult {
        intent: IntentType::MultiStep,
        detected_urls: vec!["https://example.com".into()],
        has_write_intent: true,
        ..Default::default()
    };
    let ctx =
        planner.build_context("Fetch https://example.com and save to output.md", &sample_servers(), None, None, None);
    let plan = planner.create_plan(&classification, &ctx, None);
    // Should have fetch + conversational + write from signal fallback
    assert!(plan.steps.len() >= 2);
    let types: Vec<StepType> = plan.steps.iter().map(|s| s.step_type).collect();
    assert!(types.contains(&StepType::Tool)); // fetch
}

/// MULTI_STEP with LLM returning valid steps.
#[test]
fn test_create_plan_multi_step_from_llm() {
    let llm = mock_llm();
    let planner = planner_with(&llm);
    llm.set_invoke_return(
        &json!([
            {
                "step": 1,
                "step_type": "tool",
                "tool_name": "fetch",
                "server_id": "srv-1",
                "server_name": "Fetch Server",
                "instruction": "Fetch the URL",
            },
            {
                "step": 2,
                "step_type": "action",
                "action_kind": "write",
                "tool_name": "file-write",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": "Write to output.md",
            },
        ])
        .to_string(),
    );
    let classification = ClassificationResult {
        intent: IntentType::MultiStep,
        detected_urls: vec!["https://example.com".into()],
        has_write_intent: true,
        ..Default::default()
    };
    let ctx =
        planner.build_context("Fetch https://example.com and save to output.md", &sample_servers(), None, None, None);
    let plan = planner.create_plan(&classification, &ctx, None);
    assert_eq!(plan.steps.len(), 2);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    // file-write now runs as a TOOL step against the smartloop MCP server.
    assert_eq!(plan.steps[1].step_type, StepType::Tool);
    assert_eq!(plan.steps[1].tool_name.as_deref(), Some(FILE_WRITE_TOOL));
}

// ------------------------------------------------------------------ //
// 4. _parse_plan_response                                             //
// ------------------------------------------------------------------ //

#[test]
fn test_parse_plan_response_malformed() {
    let planner = planner();
    let plan = planner.parse_plan_response("not valid json", "test", None);
    assert_eq!(plan.steps.len(), 0);
    assert!(plan.reasoning.contains("Parse error"));
}

#[test]
fn test_parse_plan_response_not_list() {
    let planner = planner();
    let plan = planner.parse_plan_response(r#"{"key": "value"}"#, "test", None);
    assert_eq!(plan.steps.len(), 0);
    assert!(plan.reasoning.contains("not a list"));
}

#[test]
fn test_parse_plan_response_markdown_fences() {
    let planner = planner();
    let raw = "```json\n[{\"step\":1,\"step_type\":\"explore\",\"tool_name\":\"explore\",\"instruction\":\"walk\"}]\n```";
    let plan = planner.parse_plan_response(raw, "test", None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Explore);
}

#[test]
fn test_parse_plan_response_caps_at_5_steps() {
    let planner = planner();
    let steps_data: Vec<Value> = (1..8)
        .map(|i| json!({"step": i, "step_type": "conversational", "tool_name": "convo", "instruction": format!("step {i}")}))
        .collect();
    let plan = planner.parse_plan_response(&Value::Array(steps_data).to_string(), "test", None);
    assert_eq!(plan.steps.len(), 5);
}

#[test]
fn test_backward_compat_tool_call_maps_to_tool() {
    let planner = planner();
    let plan = planner.parse_plan_response(
        &json!([{
            "step": 1, "step_type": "tool_call", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch", "instruction": "Fetch URL",
        }])
        .to_string(),
        "fetch",
        None,
    );
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
}

#[test]
fn test_backward_compat_mcp_maps_to_tool() {
    let planner = planner();
    let plan = planner.parse_plan_response(
        &json!([{
            "step": 1, "step_type": "mcp", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch", "instruction": "Fetch URL",
        }])
        .to_string(),
        "fetch",
        None,
    );
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
}

#[test]
fn test_action_step_type_corrected_to_tool_for_external_server() {
    let planner = planner();
    let plan = planner.parse_plan_response(
        &json!([{
            "step": 1, "step_type": "action", "tool_name": "fetch",
            "server_id": "srv-1", "server_name": "Fetch MCP Server",
            "instruction": "Fetch https://example.com and extract content.",
        }])
        .to_string(),
        "fetch",
        None,
    );
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].server_id.as_deref(), Some("srv-1"));
}

/// Legacy `read`/`write`/`grep`/`bash` step_types now map to TOOL
/// steps pinned to the smartloop MCP server.
#[test]
fn test_system_tool_step_type_is_tool() {
    let planner = planner();
    let plan = planner.parse_plan_response(
        &json!([{
            "step": 1, "step_type": "read", "tool_name": "file-read",
            "server_id": "__system_tools__", "server_name": "smartloop",
            "instruction": "Read main.py",
        }])
        .to_string(),
        "show me main.py",
        None,
    );
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some(FILE_READ_TOOL));
    assert_eq!(plan.steps[0].server_id.as_deref(), Some(SYSTEM_TOOLS_SERVER_ID));
}

// ------------------------------------------------------------------ //
// 5. _extract_paths                                                    //
// ------------------------------------------------------------------ //

#[test]
fn test_extract_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let tmp_path = tmp.path().canonicalize().unwrap();
    let test_file = tmp_path.join("test.py");
    std::fs::write(&test_file, "x").unwrap();
    let found = extract_paths(&format!("look at {}", test_file.display()), &path_str(&tmp_path));
    assert!(found.contains(&path_str(&test_file)));
}

#[test]
fn test_extract_paths_no_match() {
    let tmp = tempfile::tempdir().unwrap();
    let found = extract_paths("no paths here", &path_str(tmp.path()));
    assert!(found.is_empty());
}
