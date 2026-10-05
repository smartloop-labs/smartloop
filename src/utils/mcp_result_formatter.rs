//! Port of `smartloop/utils/mcp_result_formatter.py`.

use std::sync::Arc;

use serde_json::Value;

use crate::llms::BaseLLM;

/// Formats MCP tool results into compact, readable markdown/plain text.
pub struct ResultFormatter {
    pub llm: Arc<dyn BaseLLM>,
}

impl ResultFormatter {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Render a tool's content blocks; JSON text blocks are compacted and
    /// flattened, errors are prefixed, and oversized output is truncated to
    /// 60% of the model's max tokens.
    pub fn format_tool_result(
        &self,
        tool_name: &str,
        content: Option<&[Value]>,
        is_error: bool,
    ) -> String {
        let _ = (tool_name, content, is_error);
        todo!()
    }

    /// True for values that carry no meaning for the LLM (ids, urls, hashes,
    /// timestamps, null/bool, ...).
    pub fn is_noisy_value(key: &str, value: &Value) -> bool {
        let _ = (key, value);
        todo!()
    }

    /// Recursively drop noisy keys and unwrap single-child wrapper objects.
    pub fn compact_json(data: &Value) -> Value {
        let _ = data;
        todo!()
    }

    /// Convert camelCase / snake_case keys to readable lowercase words.
    pub fn humanize_key(key: &str) -> String {
        let _ = key;
        todo!()
    }

    /// Convert a compacted value to plain `key: value` lines.
    pub fn flatten_to_text(data: &Value, prefix: &str) -> String {
        let _ = (data, prefix);
        todo!()
    }

    /// Render JSON as a markdown table (arrays of objects) or bullet list.
    pub fn json_to_markdown(data: &Value) -> String {
        let _ = data;
        todo!()
    }
}
