//! Port of `tests/unit/test_system_tools_mcp_resolution.py`.
//!
//! Regression test: the built-in "smartloop" system tools server must be
//! included in `mcp_tool_contexts` so the planner can target it.
//!
//! Background: the default skills installed at bootstrap (`file-read`,
//! `file-write`, `bash`) reference the system tools. Previously the system
//! tools server was filtered out of `mcp_tool_contexts`, so the planner could
//! never resolve those tools. This test pins the fixed behaviour: the system
//! server is present in `mcp_tool_contexts`.

use std::collections::HashSet;

use serde_json::json;

use smartloop::agents::orchestrator::{ExecuteOptions, SharedCell};
use smartloop::agents::planner::{SYSTEM_TOOLS_SERVER_ID, SYSTEM_TOOLS_SERVER_NAME};
use smartloop::workspace::workspace::{MCPServer, MCPServerConfig, MCPServerType};

use crate::shared::agents_core::{fetch_server, orchestrator_over, tool, FakeLlm};

/// Mirror the server installed by `model_service.install_system_tools_server`.
fn system_tools_server() -> MCPServer {
    MCPServer {
        id: SYSTEM_TOOLS_SERVER_ID.to_string(),
        name: SYSTEM_TOOLS_SERVER_NAME.to_string(),
        description: Some("Built-in tools for file operations, search, and shell commands".to_string()),
        server_type: MCPServerType::Local,
        config: Some(MCPServerConfig {
            command: "python".to_string(),
            args: vec!["-m".to_string(), "smartloop.api.mcp.system_server".to_string()],
            ..Default::default()
        }),
        enabled: true,
        system: true,
        tools: vec![tool(
            "file-read",
            "Read a file and return its content",
            json!({"path": {"type": "string"}}),
            &["path"],
        )],
        ..Default::default()
    }
}

/// The system tools server must appear in mcp_tool_contexts alongside
/// third-party servers — not be filtered out (the original bug).
#[test]
fn test_build_context_includes_system_tools_server() {
    let llm = FakeLlm::new("");
    let orch = orchestrator_over(&llm);

    let ctx = orch
        .build_context(
            "read /tmp/foo.txt",
            &[system_tools_server(), fetch_server("fetch-srv", "fetch")],
            &ExecuteOptions { cwd: Some("/tmp".to_string()), ..Default::default() },
        )
        .unwrap();

    let server_ids: HashSet<String> =
        ctx.with_ref(|c| c.mcp_tool_contexts.iter().map(|t| t.server_id.clone()).collect());
    assert!(
        server_ids.contains(SYSTEM_TOOLS_SERVER_ID),
        "system tools server was filtered out of mcp_tool_contexts"
    );
    assert!(server_ids.contains("fetch-srv"));
}
