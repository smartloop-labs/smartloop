//! Port of `tests/unit/test_invoice_pricing_e2e.py`.
//!
//! End-to-end test: Explore invoice PDF → Fetch pricing page → Write report.
//!
//! Simulates a user asking:
//!   "Explore the invoice in data/, check the pricing at
//!    https://lovable.dev/pricing, confirm it matches, and write a detailed
//!    feature report."
//!
//! The planner produces a 3-step plan (explore → fetch → write) from the user
//! prompt.  Built-in handlers run against a real copy of the invoice; the
//! fetch MCP call is faked with the actual lovable.dev pricing data.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::json;
use serial_test::serial;

use smartloop::agents::fetch_agent::set_fetch_override;
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;
use smartloop::agents::mcp::explorer::SearchContext;
use smartloop::agents::mcp::mcp_tool_caller::MCPToolCaller;
use smartloop::agents::mcp::result_mapper::MCPResultMapper;
use smartloop::agents::orchestrator::{
    ExecuteOptions, MCPExecutionResult, SLPOrchestrator, SharedCell, SharedSearchContext,
};
use smartloop::mcp::MCPToolCallResult;
use smartloop::workspace::workspace::MCPServer;

use crate::shared::agents_core::{
    failed_result, fetch_server, obj, orchestrator_over, system_tools_server, text_result, tool, FakeLlm,
};

/// The local smartloop system-tools MCP server (real stdio subprocess).
fn system_server() -> MCPServer {
    system_tools_server(vec![
        tool(
            "file-write",
            "file-write",
            json!({"path": {"type": "string"}, "content": {"type": "string"}}),
            &["path", "content"],
        ),
        tool("file-read", "file-read", json!({"path": {"type": "string"}}), &["path"]),
        tool("grep", "grep", json!({"pattern": {"type": "string"}}), &["pattern"]),
    ])
}

/// `call_tool` replacement: fake the remote fetch tool, run the local system
/// tools for real against temp files.
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
// Fixtures & constants                                                 #
// ------------------------------------------------------------------ #

const INVOICE_TEXT: &str = "Invoice\n\
Invoice number 1PTUKWWH-0015\n\
Date of issue February 18, 2026\n\
Date due February 18, 2026\n\n\
Lovable\n\
1111b South Governors Avenue\n\
Dover, Delaware 19904\n\n\
Bill to\nMehfuz\n\
49207 Periwinkle Ter\n\
Fremont, California 94539-7524\n\n\
Description Qty Unit price Amount\n\
Pro 1\nFeb 18–Mar 18, 2026\n\
1 $25.00 $25.00\n\n\
Subtotal $25.00\n\
Total $25.00\n\
Amount due $25.00 USD\n";

fn make_invoice_search_ctx() -> SearchContext {
    SearchContext {
        vector_context: INVOICE_TEXT.to_string(),
        citations: Some(vec![obj(json!({"document_id": "inv-1", "document_name": "Invoice-1PTUKWWH-0015.txt"}))]),
        ..Default::default()
    }
}

/// Pricing data scraped from https://lovable.dev/pricing
const LOVABLE_PRICING_PAGE: &str = "# Lovable Pricing\n\n\
## Free\n\
- Price: $0/month\n\
- 5 credits/day, 30 credits/month\n\
- Public projects only\n\n\
## Pro\n\
- Price: $25/month\n\
- 100 monthly credits + 5 daily credits (up to 150/month)\n\
- Credit rollovers\n\
- On-demand credit top-ups\n\
- Unlimited lovable.app domains\n\
- Custom domains\n\
- Remove Lovable badge\n\
- User roles & permissions\n\
- Private projects\n\n\
## Business\n\
- Price: $50/month\n\
- 100 monthly credits\n\
- All Pro features plus:\n\
- Internal publish, SSO, Team workspace\n\
- Personal projects, Design templates\n\
- Role-based access, Security center\n\n\
## Enterprise\n\
- Custom pricing\n\
- Volume-based credit pricing\n\
- Dedicated support, Onboarding services\n\
- Design systems, SCIM, Custom connectors\n\
- Publishing & sharing controls, Audit logs\n";

const EXPECTED_REPORT: &str = "# Lovable Invoice & Pricing Verification Report\n\n\
## Invoice Summary\n\
- **Invoice Number:** 1PTUKWWH-0015\n\
- **Plan:** Pro 1\n\
- **Amount:** $25.00 USD\n\
- **Billing Period:** Feb 18 – Mar 18, 2026\n\
- **Vendor:** Lovable\n\
- **Billed To:** Mehfuz\n\n\
## Pricing Verification\n\
- **Pro plan listed price:** $25/month\n\
- **Invoice amount:** $25.00\n\
- **Match:** YES — invoice matches the published Pro plan price.\n\n\
## Pro Plan Features\n\
- 100 monthly credits + 5 daily credits (up to 150/month)\n\
- Credit rollovers\n\
- On-demand credit top-ups\n\
- Unlimited lovable.app domains\n\
- Custom domains\n\
- Remove Lovable badge\n\
- User roles & permissions\n\
- Private projects\n\n\
## All Plans Overview\n\
| Plan | Price | Credits |\n\
| --- | --- | --- |\n\
| Free | $0/month | 5/day, 30/month |\n\
| Pro | $25/month | 100/month + 5/day |\n\
| Business | $50/month | 100/month |\n\
| Enterprise | Custom | Volume-based |\n";

fn make_fetch_server() -> MCPServer {
    fetch_server("fetch-srv", "Fetch MCP Server")
}

/// The `_patch_fetch_agent` autouse fixture: URL fetching runs through the
/// in-process FetchAgent (not an MCP tool); every fetch answers `text`, and
/// the calls are counted (`_patch_fetch_agent.called`).
struct PatchFetchAgent {
    calls: Arc<AtomicUsize>,
}

impl PatchFetchAgent {
    fn returning(text: &'static str) -> Self {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        set_fetch_override(Some(Arc::new(move |_, _, _, _| {
            counter.fetch_add(1, Ordering::SeqCst);
            text.to_string()
        })));
        Self { calls }
    }

    fn called(&self) -> bool {
        self.calls.load(Ordering::SeqCst) > 0
    }
}

impl Drop for PatchFetchAgent {
    fn drop(&mut self) {
        set_fetch_override(None);
    }
}

// ------------------------------------------------------------------ #
// Tests                                                                #
// ------------------------------------------------------------------ #

/// E2E: user prompt → planner builds explore/fetch/write plan → orchestrator
/// executes each step against real files and faked MCP.
mod invoice_pricing_pipeline {
    use super::*;

    fn user_prompt(data_dir: &str, output_path: &str) -> String {
        format!(
            "Explore the invoice PDF in {data_dir}, check the pricing at https://lovable.dev/pricing to confirm it \
             matches, and write a detailed report of the features to {output_path}"
        )
    }

    /// The `workspace` fixture: a text representation of the invoice in a
    /// temp workspace `data/` dir.  Uses a .txt file rather than the real PDF
    /// because the explore step does not support .pdf files.
    struct Workspace {
        _tmp: tempfile::TempDir,
        root: String,
        data_dir: String,
        output_path: String,
    }

    fn workspace() -> Workspace {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_string_lossy().to_string();
        let data_dir = format!("{root}/data");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(format!("{data_dir}/Invoice-1PTUKWWH-0015.txt"), INVOICE_TEXT).unwrap();
        let output_path = format!("{root}/report.md");
        Workspace { _tmp: tmp, root, data_dir, output_path }
    }

    /// The JSON plan the LLM planner would produce.
    fn build_plan_response(data_dir: &str, output_path: &str) -> String {
        json!([
            {
                "step": 1,
                "step_type": "explore",
                "tool_name": "explore",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": format!("Explore the directory {data_dir} and read the invoice PDF to extract invoice details (plan name, amount, billing period, vendor)."),
            },
            {
                "step": 2,
                "step_type": "tool",
                "tool_name": "fetch",
                "server_id": "fetch-srv",
                "server_name": "Fetch MCP Server",
                "instruction": "Fetch https://lovable.dev/pricing and extract all plan names, prices, and features.",
            },
            {
                "step": 3,
                "step_type": "action",
                "action_kind": "write",
                "tool_name": "file-write",
                "server_id": "__system_tools__",
                "server_name": "smartloop",
                "instruction": format!("Write a detailed verification report to {output_path} comparing the invoice to the published pricing and listing all Pro plan features."),
            },
        ])
        .to_string()
    }

    /// A model returning context-aware responses for every call the
    /// orchestrator makes (`_make_invoke_side_effect`).
    ///
    /// The explore step calls the LLM twice: once to pick files, once to
    /// enrich.  The write step calls it once for arg extraction.
    fn make_llm(plan_json: String, data_dir: String, output_path: String) -> Arc<FakeLlm> {
        let llm = FakeLlm::new("");
        llm.set_max_tokens(8192);
        llm.set_responder(move |prompt| {
            let p_lower = prompt.to_lowercase();

            // 1. Planner prompt
            if p_lower.contains("tool planner") || p_lower.contains("json array") {
                return Ok(plan_json.clone());
            }

            // 2. Explore: LLM picks which files to read
            if p_lower.contains("which files are most relevant") {
                // Return the invoice file (whatever extension it is)
                let files: Vec<String> = std::fs::read_dir(&data_dir)
                    .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().to_string()).collect())
                    .unwrap_or_default();
                let invoice = files
                    .iter()
                    .find(|f| f.to_lowercase().contains("invoice"))
                    .or(files.first())
                    .cloned()
                    .unwrap_or_default();
                return Ok(invoice);
            }

            // 3. Explore: LLM enriches / summarizes file contents
            if p_lower.contains("based on the file tree") || p_lower.contains("provide a clear, structured") {
                let mut base = String::from(
                    "## Invoice Details\n\n\
                     - **Invoice Number:** 1PTUKWWH-0015\n\
                     - **Plan:** Pro 1\n\
                     - **Amount:** $25.00 USD\n\
                     - **Billing Period:** Feb 18 – Mar 18, 2026\n\
                     - **Vendor:** Lovable (Dover, Delaware)\n\
                     - **Billed To:** Mehfuz (Fremont, CA)\n\
                     - **Date of Issue:** February 18, 2026\n",
                );
                // Include pricing data when the fetched URL content is in the prompt
                if p_lower.contains("fetched url content") && p_lower.contains("lovable pricing") {
                    base.push_str(
                        "\n## Pricing Tiers\n\n\
                         - **Free:** $0/month — 5 credits/day, 30 credits/month\n\
                         - **Pro:** $25/month — 100 monthly credits + 5 daily credits\n\
                         - **Business:** $50/month — 100 monthly credits + all Pro features\n\
                         - **Enterprise:** Custom pricing — Volume-based credits\n",
                    );
                } else if p_lower.contains("could not fetch") {
                    base.push_str("\nCould not fetch content from the pricing page.");
                }
                return Ok(base);
            }

            // 4. Arg extraction for file-write (unified MCP path uses the
            // MCPArgumentExtractor prompt).
            if p_lower.contains("reply with only a json object")
                || p_lower.contains("argument extraction assistant")
                || p_lower.contains("extract arguments for tool")
            {
                return Ok(json!({"path": output_path, "content": EXPECTED_REPORT}).to_string());
            }

            // Fallback
            Ok("OK".to_string())
        });
        llm
    }

    fn orchestrate(ws: &Workspace) -> (SLPOrchestrator, Arc<FakeLlm>) {
        let plan_json = build_plan_response(&ws.data_dir, &ws.output_path);
        let llm = make_llm(plan_json, ws.data_dir.clone(), ws.output_path.clone());
        (orchestrator_over(&llm), llm)
    }

    fn execute(
        orchestrator: &SLPOrchestrator,
        ws: &Workspace,
        servers: &[MCPServer],
        document_context: Option<&str>,
    ) -> MCPExecutionResult {
        orchestrator.execute(
            &user_prompt(&ws.data_dir, &ws.output_path),
            servers,
            ExecuteOptions {
                cwd: Some(ws.root.clone()),
                document_context: document_context.map(str::to_string),
                ..Default::default()
            },
        )
    }

    // ------------------------------------------------------------------ #
    // Test: full explore → fetch → write pipeline                        #
    // ------------------------------------------------------------------ #

    /// Complete 3-step flow driven by a user prompt: explore invoice → fetch
    /// pricing → write report.
    #[test]
    #[serial]
    fn test_explore_fetch_write_pipeline() {
        let _fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let (orchestrator, llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        delegating_call_tool(&orchestrator, &llm, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server(), system_server()], None);

        // Pipeline completed
        assert!(result.handled);
        assert_eq!(result.tool_names.len(), 3);

        // Step types executed in order
        assert_eq!(result.tool_names[0], "explore"); // explore invoice
        assert_eq!(result.tool_names[1], "fetch"); // fetch pricing
        assert_eq!(result.tool_names[2], "file-write"); // write report

        // Report was written to disk
        assert!(std::path::Path::new(&ws.output_path).is_file(), "Report file was not created");
        let report = std::fs::read_to_string(&ws.output_path).unwrap();

        // Report contains invoice data
        assert!(report.contains("1PTUKWWH-0015"));
        assert!(report.contains("$25.00"));
        assert!(report.contains("Pro"));

        // Report contains pricing verification
        assert!(report.contains("Match") || report.to_lowercase().contains("match"));

        // Report contains Pro plan features
        assert!(report.to_lowercase().contains("credit"));
        assert!(report.contains("Custom domains") || report.to_lowercase().contains("custom domains"));
    }

    // ------------------------------------------------------------------ #
    // Test: pricing matches invoice amount                               #
    // ------------------------------------------------------------------ #

    /// Verify the explore step fetches the URL and returns pricing data.
    #[test]
    #[serial]
    fn test_invoice_amount_matches_pro_pricing() {
        let _fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let (orchestrator, _llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        stub_call_tool(&orchestrator, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], None);

        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()));

        // The pricing data should be in the output (fetched via explore)
        let content = result.content.unwrap_or_default();
        assert!(content.contains("$25"));
        assert!(content.contains("Pro"));
    }

    // ------------------------------------------------------------------ #
    // Test: explore step reads the invoice file                          #
    // ------------------------------------------------------------------ #

    /// The explore step fetches the URL; pricing content appears in the result.
    #[test]
    #[serial]
    fn test_explore_reads_invoice_from_data_dir() {
        let _fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let (orchestrator, _llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        stub_call_tool(&orchestrator, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], None);

        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()));
        let content = result.content.unwrap_or_default();
        assert!(content.contains("Lovable") || content.contains("Pro"));
    }

    // ------------------------------------------------------------------ #
    // Test: fetch step gets pricing data                                 #
    // ------------------------------------------------------------------ #

    /// The FetchAgent runs for the URL step and pricing content appears in
    /// the output.
    #[test]
    #[serial]
    fn test_fetch_retrieves_pricing_page() {
        let fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let (orchestrator, _llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        stub_call_tool(&orchestrator, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], None);

        assert!(result.handled);

        // The FetchAgent was invoked for the URL step
        assert!(fetch.called());

        // Pricing data appears in the combined output
        let content = result.content.unwrap_or_default();
        assert!(content.contains("Business") || content.contains("$50"));
        assert!(content.contains("Enterprise") || content.contains("Custom"));
    }

    // ------------------------------------------------------------------ #
    // Test: write step creates the report file                           #
    // ------------------------------------------------------------------ #

    /// Full 3-step pipeline produces output with all pricing tiers.
    #[test]
    #[serial]
    fn test_write_creates_report_with_all_sections() {
        let _fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let (orchestrator, _llm) = orchestrate(&ws);

        let pricing_ctx = SearchContext {
            vector_context: format!("{INVOICE_TEXT}\n\n{LOVABLE_PRICING_PAGE}"),
            citations: Some(vec![obj(json!({"document_id": "inv-1", "document_name": "Invoice-1PTUKWWH-0015.txt"}))]),
            ..Default::default()
        };
        stub_collect(&orchestrator, pricing_ctx);
        stub_call_tool(&orchestrator, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], None);

        assert!(result.handled);

        // All plan tiers from the pricing page should appear in the output
        let content = result.content.unwrap_or_default();
        assert!(content.contains("Pro"));
        assert!(content.contains("Free"));
        assert!(content.contains("Business"));
        assert!(content.contains("Enterprise"));
    }

    // ------------------------------------------------------------------ #
    // Test: fetch failure — explore result still returned                 #
    // ------------------------------------------------------------------ #

    /// If the fetch tool fails, explore results (invoice data) should still
    /// be in the output.
    #[test]
    #[serial]
    fn test_fetch_failure_returns_explore_result() {
        let _fetch = PatchFetchAgent::returning("");
        let ws = workspace();
        let (orchestrator, _llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        stub_call_tool(&orchestrator, failed_result("Connection refused"));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], None);

        assert!(result.handled);
        assert!(result.tool_names.contains(&"explore".to_string()));
        let content = result.content.unwrap_or_default();
        assert!(content.contains("1PTUKWWH-0015") || content.contains("Invoice"));
    }

    // ------------------------------------------------------------------ #
    // Test: document context supplements the plan                        #
    // ------------------------------------------------------------------ #

    /// When document_context is provided, it is included in the explore
    /// result alongside the fetched URL content.
    #[test]
    #[serial]
    fn test_document_context_passed_through() {
        let _fetch = PatchFetchAgent::returning(LOVABLE_PRICING_PAGE);
        let ws = workspace();
        let doc_context = "Internal policy: all SaaS subscriptions must be Pro tier or above.";
        let (orchestrator, _llm) = orchestrate(&ws);
        stub_collect(&orchestrator, make_invoice_search_ctx());
        stub_call_tool(&orchestrator, text_result(LOVABLE_PRICING_PAGE));

        let result = execute(&orchestrator, &ws, &[make_fetch_server()], Some(doc_context));

        assert!(result.handled);
        assert!(result.content.is_some());
        assert!(result.tool_names.contains(&"explore".to_string()));
    }
}
