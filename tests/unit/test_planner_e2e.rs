//! Port of `tests/unit/test_planner_e2e.py`.
//!
//! End-to-end tests for the classify → plan → execute pipeline.
//!
//! Each test exercises `SLPOrchestrator::execute()` with realistic scenarios.
//! The LLM and external MCP calls are faked; explore steps read real temp
//! files in-process.  Smartloop writes no files and runs no commands: a plan
//! step for either is dropped and the harness acts on the answer.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};
use serial_test::serial;

use smartloop::agents::fetch_agent::set_fetch_override;
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestrator::{ExecuteOptions, SLPOrchestrator, SharedCell, SharedSearchContext};
use smartloop::mcp::MCPToolCallResult;
use smartloop::workspace::workspace::MCPServer;

use crate::shared::agents_core::{
    failed_result, fetch_server, orchestrator_over, text_result, FakeLlm,
};

/// `call_tool` replacement: fake the (remote) fetch tool, run everything else
/// through a real, unhooked client.
fn delegating_call_tool(orchestrator: &SLPOrchestrator, llm: &Arc<FakeLlm>, fetch_result: MCPToolCallResult) {
    let real = Arc::new(MCPToolCaller::new(
        Arc::new(MCPArgumentExtractor::new(llm.shared())),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    ));
    orchestrator.tool_caller.set_call_tool_hook(Some(Arc::new(move |server, tool_name, arguments, token, cwd| {
        if tool_name == "fetch" {
            return fetch_result.clone();
        }
        real.call_tool(server, tool_name, arguments, token, cwd)
    })));
}

/// `patch.object(orchestrator.tool_caller, "call_tool", return_value=result)`.
fn stub_call_tool(orchestrator: &SLPOrchestrator, result: MCPToolCallResult) {
    orchestrator.tool_caller.set_call_tool_hook(Some(Arc::new(move |_, _, _, _, _| result.clone())));
}

/// `patch.object(orchestrator.explorer, "collect_parallel", return_value=ctx)`.
fn stub_collect(orchestrator: &SLPOrchestrator, ctx: SearchContext) {
    let shared: SharedSearchContext = SharedCell::wrap(ctx);
    orchestrator.set_collect_override(Some(Arc::new(move |_| shared.clone())));
}

// ------------------------------------------------------------------ #
// Helpers                                                              #
// ------------------------------------------------------------------ #

const RESUME_PDF_TEXT: &str = "# John Doe — Software Engineer\n\n\
## Experience\n\
- 5 years Python, FastAPI, Django\n\
- 3 years React, TypeScript\n\
- Cloud: AWS (EC2, Lambda, S3)\n\
- ML: scikit-learn, PyTorch basics\n\n\
## Education\n\
B.S. Computer Science, MIT 2018\n";

const JOB_DESCRIPTION_HTML: &str = "# Senior Backend Engineer — Anthropic\n\n\
## Requirements\n\
- 5+ years backend development (Python preferred)\n\
- Experience with large-scale distributed systems\n\
- Strong understanding of ML/AI concepts\n\
- Experience with cloud infrastructure (AWS/GCP)\n\
- Excellent communication skills\n\n\
## Nice to Have\n\
- Rust or Go experience\n\
- Contributions to open-source projects\n";

const GENERATED_REQUIREMENTS: &str = "# Tailored Requirements for John Doe\n\n\
## Strong Matches\n\
- 5 years Python experience (meets 5+ years requirement)\n\
- AWS cloud infrastructure experience\n\
- ML basics with scikit-learn and PyTorch\n\n\
## Gaps to Address\n\
- No distributed systems experience mentioned\n\
- No Rust or Go experience\n\
- Should highlight open-source contributions if any\n";

/// `_make_fetch_server()`.
fn make_fetch_server() -> MCPServer {
    fetch_server("fetch-srv", "Fetch MCP Server")
}

/// A model answering by prompt (`_make_llm_side_effect`).
///
/// In the classify → plan flow, the LLM is called for: classification (Tier 2
/// fallback, if deterministic detection finds nothing), multi-step planning
/// (only for MULTI_STEP intent), arg extraction for built-in tools, and
/// conversational / summarize steps.
fn make_llm(read_args: Value, write_args: Value, generated_text: &str, plan_json: &str) -> Arc<FakeLlm> {
    let llm = FakeLlm::new("");
    let generated_text = generated_text.to_string();
    let plan_json = plan_json.to_string();
    llm.set_responder(move |prompt| {
        let lowered = prompt.to_lowercase();

        // Classification prompt (contains "Classify the user's intent")
        if lowered.contains("classify the user") {
            return Ok("MULTI_STEP".to_string());
        }

        // Planner prompt (contains "task planner" or "JSON array")
        if lowered.contains("task planner") || lowered.contains("json array") {
            return Ok(plan_json.clone());
        }

        // Arg extraction prompts. The old direct path used "Reply with ONLY a
        // JSON object"; the unified MCP path uses the MCPArgumentExtractor
        // prompt ("argument extraction assistant" / "Extract arguments for
        // tool '<name>'").
        if lowered.contains("reply with only a json object")
            || lowered.contains("argument extraction assistant")
            || lowered.contains("extract arguments for tool")
        {
            if lowered.contains("file-write") || lowered.contains("content") {
                return Ok(write_args.to_string());
            }
            return Ok(read_args.to_string());
        }

        // Conversational / summarize / enrich — return generated text
        Ok(generated_text.clone())
    });
    llm
}

/// The `_patch_fetch_agent` autouse fixture: URL fetching runs through the
/// in-process FetchAgent — patch it so tests never hit the network.
struct PatchFetchAgent;

impl PatchFetchAgent {
    fn returning(text: &'static str) -> Self {
        set_fetch_override(Some(Arc::new(move |_, _, _, _| text.to_string())));
        Self
    }
}

impl Drop for PatchFetchAgent {
    fn drop(&mut self) {
        set_fetch_override(None);
    }
}

/// `tmp_workspace` fixture: a temp dir holding `resume.txt`.
fn tmp_workspace() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("resume.txt"), RESUME_PDF_TEXT).unwrap();
    let dir = tmp.path().to_string_lossy().to_string();
    (tmp, dir)
}

fn opts(cwd: Option<&str>) -> ExecuteOptions {
    ExecuteOptions { cwd: cwd.map(str::to_string), ..Default::default() }
}

// ------------------------------------------------------------------ #
// E2E Tests                                                            #
// ------------------------------------------------------------------ #

/// Full pipeline: classify → plan → execute → return results.
mod classify_plan_execute_pipeline {
    use super::*;

    /// File path + URL + write → MULTI_STEP → read + fetch; the write step is
    /// dropped — the answer carries the content and the harness saves it.
    #[test]
    #[serial]
    fn test_multi_step_read_fetch_write() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        let (_tmp, dir) = tmp_workspace();
        let resume_path = format!("{dir}/resume.txt");
        let output_path = format!("{dir}/requirements.md");

        let plan_response = json!([
            {
                "step": 1,
                "step_type": "action",
                "action_kind": "read",
                "tool_name": "file-read",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": format!("Read the resume file at {resume_path}"),
            },
            {
                "step": 2,
                "step_type": "tool",
                "tool_name": "fetch",
                "server_id": "fetch-srv",
                "server_name": "Fetch MCP Server",
                "instruction": "Fetch https://job-boards.greenhouse.io/anthropic/jobs/123 and extract the job description.",
            },
            {
                "step": 3,
                "step_type": "action",
                "action_kind": "write",
                "tool_name": "file-write",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": format!("Write the tailored requirements to {output_path}"),
            },
        ])
        .to_string();

        let llm = make_llm(
            json!({"path": resume_path, "query": ""}),
            json!({"path": output_path, "content": GENERATED_REQUIREMENTS}),
            GENERATED_REQUIREMENTS,
            &plan_response,
        );
        let orchestrator = orchestrator_over(&llm);
        delegating_call_tool(&orchestrator, &llm, text_result(JOB_DESCRIPTION_HTML));

        let result = orchestrator.execute(
            &format!(
                "Read my resume at {resume_path}, look at the job description at \
                 https://job-boards.greenhouse.io/anthropic/jobs/123, and write out the tailored requirements \
                 to {output_path}"
            ),
            &[make_fetch_server()],
            opts(Some(&dir)),
        );

        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()), "{:?}", result.tool_names);
        assert!(result.tool_names.contains(&"fetch".to_string()), "{:?}", result.tool_names);
        assert!(!result.tool_names.contains(&"file-write".to_string()));
        let content = result.content.unwrap_or_default();
        assert!(content.contains("John Doe"), "{content}");

        // Smartloop writes no files: that is the harness's job.
        assert!(!std::path::Path::new(&output_path).exists());
    }

    /// URL → deterministic TOOL_CALL via DAG: resolve → execute → reason.
    ///
    /// Verifies:
    /// 1. The full DAG pipeline runs (plan → run → reason).
    /// 2. The fetch tool is called with the URL argument.
    /// 3. Pre-plan search context (vector_context, citations) is suppressed
    ///    from the final result when live tool output exists.
    #[test]
    #[serial]
    fn test_single_tool_call_fetch() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        let (_tmp, dir) = tmp_workspace();
        let llm = FakeLlm::new("Here is a summary of the job posting with key details extracted.");
        let orchestrator = orchestrator_over(&llm);

        // Pre-plan search returns document context + citations — these must
        // NOT appear in the final result when tool output exists.
        let job_ctx = SearchContext {
            vector_context: "Unrelated indexed document content".to_string(),
            citations: Some(vec![json!({"document_id": "doc-1", "document_name": "old.pdf"})
                .as_object()
                .cloned()
                .unwrap()]),
            ..Default::default()
        };
        stub_collect(&orchestrator, job_ctx);
        stub_call_tool(&orchestrator, text_result(JOB_DESCRIPTION_HTML));

        let result = orchestrator.execute(
            "Fetch https://example.com/job and summarize",
            &[make_fetch_server()],
            opts(Some(&dir)),
        );

        assert!(result.handled);
        assert!(result.tool_names.contains(&"fetch".to_string()));
        assert_eq!(result.citations, None);
        assert!(!result.content.unwrap_or_default().contains("Unrelated indexed document"));
    }

    /// File path → EXPLORE → the file is read in-process (deterministic).
    #[test]
    #[serial]
    fn test_single_action_read() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        let (_tmp, dir) = tmp_workspace();
        let resume_path = format!("{dir}/resume.txt");

        let llm = make_llm(json!({"path": resume_path, "query": ""}), json!({}), "Key skills: Python, FastAPI.", "[]");
        let orchestrator = orchestrator_over(&llm);

        let result = orchestrator.execute(
            &format!("Read {resume_path} and summarize the key skills"),
            &[],
            ExecuteOptions {
                skills: Some(vec!["Be concise".to_string()]),
                cwd: Some(dir.clone()),
                ..Default::default()
            },
        );

        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()), "{:?}", result.tool_names);
        assert!(result.content.unwrap_or_default().contains("John Doe"));
    }

    /// When fetch fails, pipeline returns partial result (read step).
    #[test]
    #[serial]
    fn test_fetch_failure_returns_partial() {
        let _fetch = PatchFetchAgent::returning("");
        let (_tmp, dir) = tmp_workspace();
        let resume_path = format!("{dir}/resume.txt");

        let llm = make_llm(json!({"path": resume_path, "query": ""}), json!({}), "fallback", "[]");
        let orchestrator = orchestrator_over(&llm);
        delegating_call_tool(&orchestrator, &llm, failed_result("Connection timeout"));

        let result = orchestrator.execute(
            &format!("Read {resume_path} and fetch https://example.com/job"),
            &[make_fetch_server()],
            opts(Some(&dir)),
        );

        // Should return partial result (just the explore read)
        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()), "{:?}", result.tool_names);
        assert!(result.content.unwrap_or_default().contains("John Doe"));
    }

    /// With no MCP servers and no skills, execute() returns unhandled.
    #[test]
    #[serial]
    fn test_no_servers_no_skills_returns_unhandled() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        // A bare `MagicMock()`: every call answers a mock object's repr.
        let llm = FakeLlm::new("<MagicMock name='mock.invoke()' id='4400000000'>");
        let orchestrator = orchestrator_over(&llm);

        let result = orchestrator.execute("Read my resume and fetch a job description", &[], opts(None));
        assert!(!result.handled);
    }

    /// A pure greeting → CONVERSATION → handled=False.
    #[test]
    #[serial]
    fn test_conversational_returns_unhandled() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        let llm = FakeLlm::new("CONVERSATION");
        let orchestrator = orchestrator_over(&llm);

        let result = orchestrator.execute("Hello, how are you?", &[make_fetch_server()], opts(None));
        assert!(!result.handled);
    }

    /// Document context is passed through and available in results.
    #[test]
    #[serial]
    fn test_document_context_with_tool() {
        let _fetch = PatchFetchAgent::returning(JOB_DESCRIPTION_HTML);
        let (_tmp, dir) = tmp_workspace();
        let doc_context = "Internal policy: all candidates must have 3+ years Python.";

        let llm = FakeLlm::new("Summary based on context.");
        let orchestrator = orchestrator_over(&llm);

        stub_collect(&orchestrator, SearchContext { vector_context: doc_context.to_string(), ..Default::default() });
        stub_call_tool(&orchestrator, text_result(JOB_DESCRIPTION_HTML));

        let result = orchestrator.execute(
            "Check the job at https://example.com/job",
            &[make_fetch_server()],
            ExecuteOptions {
                cwd: Some(dir.clone()),
                document_context: Some(doc_context.to_string()),
                ..Default::default()
            },
        );

        assert!(result.handled);
        assert!(result.tool_names.contains(&"fetch".to_string()));
    }
}
