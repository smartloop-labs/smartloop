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
    MatchedTool, ModelTier, PlanContext, Planner, StepType, BASH_TOOL, EXPLORE_TOOL, FILE_WRITE_TOOL,
};
use smartloop::llms::base_llm::{CallOptions, CancelEvent, ChatMessage, GenerationStream};
use smartloop::llms::BaseLLM;
use smartloop::types::ANSWER_TIER;

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
fn test_build_context_filters_legacy_fetch_server() {
    let planner = planner();
    let servers = vec![
        json!({"id": "__fetch__", "name": "fetch", "tools": [{"name": "fetch"}]}),
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

/// Bash block → answered by the model; the harness runs it.
#[test]
fn test_classify_bash_is_a_conversation() {
    let planner = planner();
    let result = planner.classify("Run this:\n```bash\necho hello\n```", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Conversation);
    assert_eq!(result.action_kind, None);
    assert!(result.bash_commands.contains(&"echo hello".to_string()));
}

/// File path → EXPLORE (the file is read in-process).
#[test]
fn test_classify_file_path_explores() {
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
    assert_eq!(result.intent, IntentType::Explore);
    assert_eq!(result.action_kind, None);
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

/// Search keywords → EXPLORE.
#[test]
fn test_classify_search_explores() {
    let planner = planner();
    let result = planner.classify("search for TODO in the codebase", &[], ClassifyOptions::default());
    assert_eq!(result.intent, IntentType::Explore);
    assert_eq!(result.action_kind, None);
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

/// Write intent alone (no paths/URLs) → the model writes the content; the
/// harness saves it.
#[test]
fn test_classify_write_only_is_a_conversation() {
    let planner = planner();
    let result = planner.classify(
        "Write a file with a summary of the project to output.md",
        &[],
        ClassifyOptions::default(),
    );
    assert_eq!(result.intent, IntentType::Conversation);
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

/// Coding work that brings its files, as a client that read the project
/// sends it, is answered outright: no plan, no reads, no extraction.
#[test]
fn test_coding_work_with_its_files_in_hand_is_answered_directly() {
    let planner = planner();
    let mut message = String::from("I'm working in the Rust project at /tmp/app. Fix the panic in main.\n\nIts files:\n- src/main.rs\n- src/lib.rs\n\nTheir contents now:\n");
    for name in ["src/main.rs", "src/lib.rs"] {
        message.push_str(&format!("\n`{name}`\n```rs\n{}\n```\n", "fn f() {}\n".repeat(300)));
    }
    let mut result = planner.detect_signals(&message, &[], None, None, false);
    assert!(result.has_coding_intent && result.carries_files);
    assert_eq!(planner.resolve_intent(&mut result), Some(IntentType::Conversation));
    // The same request without the files plans as before.
    let mut bare = planner.detect_signals("Fix the panic in src/main.rs", &[], None, None, false);
    assert!(bare.has_coding_intent && !bare.carries_files);
    assert_ne!(planner.resolve_intent(&mut bare), Some(IntentType::Conversation));
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

/// What a turn is about is not a capability: a coding conversation is
/// still read as coding work, but it names no capability and plans one
/// LARGE step on the answer tier, where the router steps up to the largest
/// model that fits.
#[test]
fn test_classify_coding_conversation_plans_on_the_answer_tier() {
    let planner = planner();
    let result = planner.classify(
        "refactor this function to use iterators",
        &[],
        ClassifyOptions {
            available_capabilities: Some(vec!["reasoning".into(), "vision".into()]),
            ..Default::default()
        },
    );
    assert!(result.has_coding_intent);
    assert_eq!(result.capability.as_deref(), Some(ANSWER_TIER));
    let plan = planner.create_plan(&result, &ctx("refactor this function to use iterators"), None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].model_tier, ModelTier::Large);
    assert_eq!(plan.steps[0].capability.as_deref(), Some(ANSWER_TIER));
}

/// Attachments are the one signal that names a capability a model has to
/// declare: the conversation step asks for vision when a selected model
/// has it, and takes the answer tier otherwise.  With no model selected
/// nothing routes, and nothing is planned.
#[test]
fn test_classify_with_attachments_asks_for_vision() {
    let planner = planner();
    let with = |caps: Option<Vec<&str>>| {
        planner.classify(
            "what does this picture show",
            &[],
            ClassifyOptions {
                has_attachments: true,
                available_capabilities: caps.map(|c| c.into_iter().map(str::to_string).collect()),
                ..Default::default()
            },
        )
    };
    let mut result = with(Some(vec!["vision"]));
    assert_eq!(result.capability.as_deref(), Some("vision"));
    result.intent = IntentType::Conversation;
    let plan = planner.create_plan(&result, &ctx("what does this picture show"), None);
    assert_eq!(plan.steps[0].capability.as_deref(), Some("vision"));

    assert_eq!(with(Some(vec![])).capability.as_deref(), Some(ANSWER_TIER));
    let mut none = with(None);
    assert_eq!(none.capability, None);
    none.intent = IntentType::Conversation;
    assert!(planner.create_plan(&none, &ctx("what does this picture show"), None).steps.is_empty());
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
fn test_create_plan_action_bash_is_answered_not_run() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("bash".into()),
        bash_commands: vec!["echo hello".into()],
        capability: Some(ANSWER_TIER.into()),
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("Run:\n```bash\necho hello\n```"), None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Conversational);
    assert_ne!(plan.steps[0].tool_name.as_deref(), Some(BASH_TOOL));
    assert_eq!(plan.steps[0].server_id, None);
}

#[test]
fn test_create_plan_action_write_without_a_model_plans_nothing() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("write".into()),
        has_write_intent: true,
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("write a file with a poem"), None);
    assert!(plan.steps.is_empty());
}

#[test]
fn test_create_plan_action_read_explores_the_file() {
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
    assert_eq!(plan.steps[0].step_type, StepType::Explore);
    assert_eq!(plan.steps[0].tool_name.as_deref(), Some(EXPLORE_TOOL));
    assert_eq!(plan.steps[0].server_id, None);
    assert_eq!(plan.steps[0].default_args, args(&[("path", &test_file)]));
}

#[test]
fn test_create_plan_action_grep_explores() {
    let planner = planner();
    let classification = ClassificationResult {
        intent: IntentType::Action,
        action_kind: Some("grep".into()),
        has_grep_intent: true,
        ..Default::default()
    };
    let plan = planner.create_plan(&classification, &ctx("search for TODO"), None);
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Explore);
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
    // Fetch, then the model writes the content; no file-write step — the
    // harness saves the answer.
    let types: Vec<StepType> = plan.steps.iter().map(|s| s.step_type).collect();
    assert_eq!(types, vec![StepType::Fetch, StepType::Conversational]);
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
    // The file-write step is dropped: the harness writes files.
    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert!(plan.steps.iter().all(|s| s.tool_name.as_deref() != Some(FILE_WRITE_TOOL)));
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

/// Legacy `read`/`grep` steps are in-process explores; `write`/`bash` steps
/// are dropped — the harness acts on the answer.
#[test]
fn test_legacy_system_tool_steps() {
    let planner = planner();
    let plan = planner.parse_plan_response(
        &json!([
            {
                "step": 1, "step_type": "read", "tool_name": "file-read",
                "server_id": "__system_tools__", "server_name": "smartloop",
                "instruction": "Read main.py",
            },
            {
                "step": 2, "step_type": "action", "action_kind": "write", "tool_name": "file-write",
                "server_id": "__system_tools__", "instruction": "Write out.md",
            },
            {"step": 3, "step_type": "tool", "tool_name": "bash", "instruction": "git status"},
            {"step": 4, "step_type": "action", "action_kind": "grep", "instruction": "find TODO"},
        ])
        .to_string(),
        "show me main.py",
        None,
    );
    let types: Vec<StepType> = plan.steps.iter().map(|s| s.step_type).collect();
    assert_eq!(types, vec![StepType::Explore, StepType::Explore]);
    assert!(plan.steps.iter().all(|s| s.tool_name.as_deref() == Some(EXPLORE_TOOL)));
    assert!(plan.steps.iter().all(|s| s.server_id.is_none()));
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

// ------------------------------------------------------------------ //
// Promoting a write tool the collect pass resolved                    //
// ------------------------------------------------------------------ //

mod promote_deferred {
    use smartloop::agents::planner::{
        promote_deferred_tool, promoted_instruction, ClassificationResult, IntentType, MatchedTool,
        PROMOTED_HISTORY_CHARS,
    };

    fn linear_save_issue() -> Vec<MatchedTool> {
        vec![MatchedTool {
            server_id: "linear-1".into(),
            server_name: "Linear".into(),
            tool_name: "save_issue".into(),
        }]
    }

    /// "create the issue on linear": the router called it an ACTION (a
    /// `file-read` with no path); the collect pass picked the Linear tool.
    #[test]
    fn a_misclassified_action_becomes_the_tool_call() {
        let mut c = ClassificationResult {
            intent: IntentType::Action,
            action_kind: Some("read".into()),
            ..Default::default()
        };
        assert!(promote_deferred_tool(&mut c, &linear_save_issue(), &[]));
        assert_eq!(c.intent, IntentType::ToolCall);
        assert_eq!(c.matched_tools, linear_save_issue());
        assert_eq!(c.action_kind, None);
    }

    /// The router's answer did not parse and defaulted to conversation, then
    /// became explore because the lookup returned data; the write the model
    /// planned still wins.
    #[test]
    fn an_explore_is_promoted_by_a_deferred_write() {
        let mut c = ClassificationResult { intent: IntentType::Explore, ..Default::default() };
        assert!(promote_deferred_tool(&mut c, &linear_save_issue(), &linear_list_users()));
        assert_eq!(c.intent, IntentType::ToolCall);
        assert_eq!(c.matched_tools, linear_save_issue());
    }

    #[test]
    fn a_conversation_is_promoted_too() {
        let mut c = ClassificationResult { intent: IntentType::Conversation, ..Default::default() };
        assert!(promote_deferred_tool(&mut c, &linear_save_issue(), &[]));
        assert_eq!(c.intent, IntentType::ToolCall);
    }

    #[test]
    fn nothing_deferred_changes_nothing() {
        let mut c = ClassificationResult { intent: IntentType::Action, ..Default::default() };
        assert!(!promote_deferred_tool(&mut c, &[], &[]));
        assert_eq!(c.intent, IntentType::Action);
    }

    /// A concrete target in the message, a keyword match or a multi-step plan
    /// is the message speaking for itself.
    #[test]
    fn an_explicit_target_keeps_its_classification() {
        let cases = [
            ClassificationResult { intent: IntentType::Action, detected_paths: vec!["/tmp/a.txt".into()], ..Default::default() },
            ClassificationResult { intent: IntentType::Action, bash_commands: vec!["ls".into()], ..Default::default() },
            ClassificationResult { intent: IntentType::ToolCall, detected_urls: vec!["https://x.dev".into()], ..Default::default() },
            ClassificationResult { intent: IntentType::ToolCall, matched_tools: linear_save_issue(), ..Default::default() },
            ClassificationResult { intent: IntentType::MultiStep, ..Default::default() },
        ];
        for mut c in cases {
            let before = c.clone();
            assert!(!promote_deferred_tool(&mut c, &linear_save_issue(), &[]));
            assert_eq!(c, before);
        }
    }

    fn linear_list_users() -> Vec<MatchedTool> {
        vec![MatchedTool {
            server_id: "linear-1".into(),
            server_name: "Linear".into(),
            tool_name: "list_users".into(),
        }]
    }

    /// "create an issue for this for chris on linear": the collect pass looked
    /// the user up first (a read, so it ran) and deferred nothing; the router
    /// called it an ACTION with nothing to act on.
    #[test]
    fn a_targetless_action_goes_to_the_server_the_collect_pass_read() {
        let mut c = ClassificationResult { intent: IntentType::Action, ..Default::default() };
        assert!(promote_deferred_tool(&mut c, &[], &linear_list_users()));
        assert_eq!(c.intent, IntentType::ToolCall);
        assert_eq!(c.matched_tools[0].server_name, "Linear");
    }

    /// A read that already ran answers a question; only the doomed ACTION
    /// plan is replaced, and an ACTION with a target of its own is kept.
    #[test]
    fn a_read_does_not_override_other_intents() {
        let cases = [
            ClassificationResult { intent: IntentType::Conversation, ..Default::default() },
            ClassificationResult { intent: IntentType::Explore, ..Default::default() },
            ClassificationResult { intent: IntentType::Action, has_grep_intent: true, ..Default::default() },
            ClassificationResult { intent: IntentType::Action, has_write_intent: true, ..Default::default() },
        ];
        for mut c in cases {
            let before = c.clone();
            assert!(!promote_deferred_tool(&mut c, &[], &linear_list_users()));
            assert_eq!(c, before);
        }
    }

    /// A deferred write outranks a read on another server.
    #[test]
    fn a_deferred_write_outranks_a_read() {
        let mut c = ClassificationResult { intent: IntentType::Action, ..Default::default() };
        assert!(promote_deferred_tool(&mut c, &linear_save_issue(), &linear_list_users()));
        assert_eq!(c.matched_tools, linear_save_issue());
    }

    #[test]
    fn the_instruction_carries_what_was_already_looked_up() {
        let text = promoted_instruction("create it for chirs", "", r#"MCP TOOL (Linear.list_users): {"users":[]}"#);
        assert!(text.contains("Already looked up for this request"));
        assert!(text.contains(r#"{"users":[]}"#));
        assert!(!text.contains("Earlier in this conversation"));
    }

    #[test]
    fn the_instruction_carries_the_recent_conversation() {
        assert_eq!(promoted_instruction("create the issue", "  ", ""), "create the issue");

        let history = "user: pages render slowly\nassistant: consolidate the database calls";
        let text = promoted_instruction("create the issue on linear", history, "");
        assert!(text.starts_with("create the issue on linear"));
        assert!(text.contains("consolidate the database calls"));

        let long = format!("{}END", "x".repeat(PROMOTED_HISTORY_CHARS * 2));
        let text = promoted_instruction("create it", &long, "");
        assert!(text.ends_with("END"), "the most recent turns are kept");
        assert!(text.chars().count() < PROMOTED_HISTORY_CHARS + 200);
    }
}

// ------------------------------------------------------------------ //
// The best model routes; the on-device model is the fallback          //
// ------------------------------------------------------------------ //

/// An ambiguous message is classified by the model passed in, not the
/// planner's own router model.
#[test]
fn test_the_preferred_model_classifies() {
    let router = mock_llm();
    router.set_invoke_return("ACTION");
    let big = mock_llm();
    big.set_invoke_return("TOOL_CALL");
    let planner = planner_with(&router);

    let result = planner.classify(
        "file that slowness problem for chris",
        &sample_mcp_tools(),
        ClassifyOptions { llm: Some(big.clone()), ..Default::default() },
    );

    assert_eq!(result.intent, IntentType::ToolCall);
    assert!(big.invoke_called());
    assert!(!router.invoke_called(), "the router is only the fallback");
}

/// A preferred model that fails or answers with no intent word hands the
/// classification to the on-device model.
#[test]
fn test_the_on_device_model_is_the_fallback() {
    let router = mock_llm();
    router.set_invoke_return("TOOL_CALL");
    let big = mock_llm();
    big.set_invoke_return("Error generating response: 429 nearly out");
    let planner = planner_with(&router);

    let result = planner.classify(
        "file that slowness problem for chris",
        &sample_mcp_tools(),
        ClassifyOptions { llm: Some(big.clone()), ..Default::default() },
    );

    assert_eq!(result.intent, IntentType::ToolCall);
    assert!(big.invoke_called() && router.invoke_called());
    assert!(result.reasoning.contains("fallback model"));
}

/// A tool call no keyword matched is one open TOOL step, resolved at the
/// step over every connected server — not an empty plan.
#[test]
fn test_an_unmatched_tool_call_plans_an_open_tool_step() {
    let planner = planner();
    let classification = ClassificationResult { intent: IntentType::ToolCall, ..Default::default() };
    let context = PlanContext {
        user_message: "create the issue on linear".into(),
        mcp_tools: sample_mcp_tools(),
        ..Default::default()
    };

    let plan = planner.create_plan(&classification, &context, None);

    assert_eq!(plan.steps.len(), 1);
    assert_eq!(plan.steps[0].step_type, StepType::Tool);
    assert_eq!(plan.steps[0].tool_name, None);
    assert_eq!(plan.steps[0].server_id, None);

    let none = planner.create_plan(&classification, &ctx("create the issue on linear"), None);
    assert!(none.steps.is_empty(), "no connected servers, nothing to call");
}

/// With no MCP tool, skill or document to route to, every intent but
/// CONVERSATION is impossible: the planner answers without a model call,
/// which on a cloud model was seconds of "Planning the response…".
#[test]
fn test_classify_with_nothing_to_route_to_skips_the_llm() {
    let llm = mock_llm();
    llm.set_invoke_return("TOOL_CALL");
    let result = planner_with(&llm).classify(
        "covert this shell script to configure kitty",
        &[],
        ClassifyOptions::default(),
    );
    assert_eq!(result.intent, IntentType::Conversation);
    assert!(!llm.invoke_called(), "no model call when nothing can be routed to");
}

/// A skill to follow is something to route to: the model still classifies.
#[test]
fn test_classify_with_a_skill_still_asks_the_llm() {
    let llm = mock_llm();
    llm.set_invoke_return("MULTI_STEP");
    let result = planner_with(&llm).classify(
        "covert this shell script to configure kitty",
        &[],
        ClassifyOptions { skills: Some(vec!["---\nname: kitty\n---\nConfigure kitty".into()]), ..Default::default() },
    );
    assert!(llm.invoke_called());
    assert_eq!(result.intent, IntentType::MultiStep);
}
