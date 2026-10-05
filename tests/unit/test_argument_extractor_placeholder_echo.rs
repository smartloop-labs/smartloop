//! Port of `tests/unit/test_argument_extractor_placeholder_echo.py`.
//!
//! Regression: small models sometimes emit a parameter's own name/description
//! as a placeholder value (e.g. path := "file_path", url := "your_url") instead
//! of a real value.  Such placeholders must be dropped so the param is treated
//! as missing — otherwise the tool runs with a bogus value (e.g. file-read tries
//! to open a file literally named "file_path").

use std::sync::Arc;

use serde_json::{json, Value};
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;

use crate::shared::agents_mcp_args::ScriptedLLM;

fn file_read_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to read."},
            "query": {
                "type": "string",
                "description": "Optional search query to filter matching lines.",
            },
        },
        "required": ["path"],
    })
}

fn extractor(llm_response: &str) -> MCPArgumentExtractor {
    MCPArgumentExtractor::new(Arc::new(ScriptedLLM::new(llm_response)))
}

fn missing_params(args: &serde_json::Map<String, Value>) -> Vec<String> {
    args.get("_missing_params")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
        .unwrap_or_default()
}

#[test]
fn test_placeholder_path_is_dropped_and_flagged_missing() {
    // LLM echoes the field name/description as the value ("what is my pitch").
    let resp = json!({"path": "file_path"}).to_string();
    let args = extractor(&resp).invoke("what is my pitch", "file-read", &file_read_schema(), None, None);
    assert_ne!(args.get("path"), Some(&json!("file_path")));
    assert!(missing_params(&args).contains(&"path".to_string()));
}

#[test]
fn test_real_path_value_is_kept() {
    let resp = json!({"path": "domain-specific-llms-proposal.md"}).to_string();
    let args = extractor(&resp).invoke("read the proposal file", "file-read", &file_read_schema(), None, None);
    assert_eq!(args["path"], json!("domain-specific-llms-proposal.md"));
    assert!(!args.contains_key("_missing_params"));
}

#[test]
fn test_is_placeholder_echo_helper() {
    let ex = MCPArgumentExtractor::new(Arc::new(ScriptedLLM::new("")));
    let schema = file_read_schema();
    let path_info = &schema["properties"]["path"];

    // Placeholders / name+description echoes are dropped.
    assert!(ex.is_placeholder_echo("path", &json!("file_path"), path_info));
    assert!(ex.is_placeholder_echo("path", &json!("path"), path_info));
    assert!(ex.is_placeholder_echo("url", &json!("your_url"), &json!({"description": "The URL"})));
    assert!(ex.is_placeholder_echo("content", &json!("string"), &json!({"description": "Content"})));

    // Real values survive: a filename with an extension, a path, a query term.
    assert!(!ex.is_placeholder_echo("path", &json!("about.md"), path_info));
    assert!(!ex.is_placeholder_echo("path", &json!("src/app.py"), path_info));
    assert!(!ex.is_placeholder_echo("query", &json!("pitch"), &schema["properties"]["query"]));
    assert!(!ex.is_placeholder_echo(
        "symbol",
        &json!("AAPL"),
        &json!({"description": "Stock ticker symbol"})
    ));
}
