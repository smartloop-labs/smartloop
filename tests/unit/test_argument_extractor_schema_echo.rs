//! Port of `tests/unit/test_argument_extractor_schema_echo.py`.
//!
//! Regression: small models sometimes copy a parameter's schema description
//! text into the value (e.g. description := "Content as Markdown. Do not
//! escape the string ..."). Such echoes must be dropped, not sent to the tool.

use std::sync::Arc;

use serde_json::{json, Value};
use smartloop::agents::mcp::argument_extractor::MCPArgumentExtractor;

use crate::shared::agents_mcp_args::ScriptedLLM;

const DESCRIPTION_TEXT: &str = "Content as Markdown. Do not escape the string — use literal \
                                newlines and special characters, not escape sequences.";

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Issue title"},
            "description": {
                "type": "string",
                "description": DESCRIPTION_TEXT,
            },
        },
        "required": ["title", "description"],
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
fn test_echoed_description_is_dropped_and_flagged_missing() {
    // LLM echoes the schema description as the description value.
    let resp = json!({
        "title": "find billing discrepancies",
        "description": DESCRIPTION_TEXT,
    })
    .to_string();
    let schema = schema();
    let args = extractor(&resp).invoke("find billing discrepancies", "save_issue", &schema, None, None);
    // The echoed description must not be passed through as a value.
    assert_ne!(args.get("description"), Some(&schema["properties"]["description"]["description"]));
    assert!(!args.contains_key("description") || args["description"] != json!(""));
    // title is a real value and survives.
    assert_eq!(args["title"], json!("find billing discrepancies"));
    // description is dropped → flagged missing.
    assert!(missing_params(&args).contains(&"description".to_string()));
}

#[test]
fn test_real_description_value_is_kept() {
    let resp = json!({
        "title": "Billing mismatch",
        "description": "Invoice INV-42 charged $540 but PO approved $495.",
    })
    .to_string();
    let args = extractor(&resp).invoke("log the billing mismatch", "save_issue", &schema(), None, None);
    assert_eq!(args["description"], json!("Invoice INV-42 charged $540 but PO approved $495."));
    assert!(!args.contains_key("_missing_params"));
}

#[test]
fn test_is_schema_echo_helper() {
    let ex = MCPArgumentExtractor::new(Arc::new(ScriptedLLM::new("")));
    let schema = schema();
    let info = &schema["properties"]["description"];
    assert!(ex.is_schema_echo(&info["description"], info));
    assert!(!ex.is_schema_echo(&json!("a genuine markdown report"), info));
    // short descriptions never trigger (avoid dropping legit short values)
    assert!(!ex.is_schema_echo(&json!("DEV"), &json!({"description": "team"})));
}
