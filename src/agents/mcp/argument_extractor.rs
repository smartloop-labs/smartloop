//! MCP Argument Extractor Agent - Extracts tool arguments from user input based on input schema.
//!
//! Port of `smartloop/agents/mcp/argument_extractor.py` (itself ported from
//! core/core/agents/mcp/argument_extractor.py with simplified dependencies:
//! no conversation summary, no content extraction from attachments).
//!
//! LangChain's `parse_json_markdown` is hand-ported here as part of
//! [`MCPArgumentExtractor::parse_json`].

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::llms::BaseLLM;

/// Logger name used by the Python module.
pub const LOG_TARGET: &str = "smartloop.agents.mcp.argument_extractor";

/// Keywords in a parameter name that mark it as a URL-type parameter.
pub const URL_PARAM_KEYWORDS: &[&str] = &["url", "uri", "link", "href", "endpoint"];

/// Regex (case-insensitive) extracting http/https URLs from free text.
pub const URL_REGEX: &str = r#"(?i)https?://[^\s<>"{}|\\^`\[\]]+"#;

/// Patterns for identifying ID-type parameters (matched at the start, like `re.match`).
pub const ID_PARAM_PATTERNS: &[&str] = &[
    r".*[iI][dD]$",         // ends with "id" or "Id" (e.g., issueId, userId)
    r".*[iI][dD]s$",        // ends with "ids" or "Ids" (e.g., issueIds)
    r"^id$",                // exactly "id"
    r".*_id$",              // ends with "_id" (e.g., issue_id)
    r".*_ids$",             // ends with "_ids"
    r".*[uU][uU][iI][dD]$", // ends with "uuid" or "UUID"
];

/// Cursor/pagination parameters that are server-generated and must never be fabricated.
pub const CURSOR_PARAM_PATTERNS: &[&str] = &[
    r"^after$",       // GraphQL forward cursor (e.g., Linear searchIssues)
    r"^before$",      // GraphQL backward cursor
    r"^cursor$",      // generic cursor
    r".*[cC]ursor$",  // ends with "cursor" or "Cursor"
    r"^pageToken$",   // Google API page token
    r"^nextToken$",   // AWS-style next page token
    r"^continuationToken$",
];

/// Common identifier patterns in user messages.
pub const IDENTIFIER_PATTERNS: &[&str] = &[
    r"[A-Z]{2,10}-\d+",                                            // JIRA/Linear style: LIN-123, PROJ-456
    r"#\d+",                                                       // GitHub style: #123
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", // UUID with dashes
    r"[0-9a-fA-F]{24,64}",                                         // Hex IDs (MongoDB ObjectId, Linear IDs, etc.)
];

/// Generic filler tokens small models emit in place of a real value.
pub const GENERIC_PLACEHOLDERS: &[&str] = &[
    "string", "value", "example", "placeholder", "none", "null", "todo", "tbd", "your value",
    "n a",
];

/// Words that disqualify an 8-64 char token from being treated as an identifier.
const COMMON_WORDS: &[&str] = &[
    "the", "and", "for", "with", "this", "that", "create", "update", "delete", "list", "get",
    "add", "show", "find", "search", "make", "issue", "task", "comment", "team", "user",
    "project", "assign", "me", "my", "i", "you", "it", "we", "they", "all", "new", "bug", "fix",
];

// ------------------------------------------------------------------ compiled patterns

static URL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(URL_REGEX).expect("valid URL regex"));

/// `re.match` semantics: anchored at the start only.
static ID_PARAM_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    ID_PARAM_PATTERNS
        .iter()
        .map(|p| Regex::new(&format!("^(?:{p})")).expect("valid id pattern"))
        .collect()
});

static CURSOR_PARAM_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    CURSOR_PARAM_PATTERNS
        .iter()
        .map(|p| Regex::new(&format!("^(?:{p})")).expect("valid cursor pattern"))
        .collect()
});

/// `re.findall(pattern, text, re.IGNORECASE)`.
static IDENTIFIER_SEARCH_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    IDENTIFIER_PATTERNS
        .iter()
        .map(|p| Regex::new(&format!("(?i){p}")).expect("valid identifier pattern"))
        .collect()
});

/// `re.fullmatch(pattern, value, re.IGNORECASE)`.
static IDENTIFIER_FULL_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    IDENTIFIER_PATTERNS
        .iter()
        .map(|p| Regex::new(&format!("^(?i:{p})$")).expect("valid identifier pattern"))
        .collect()
});

static HEX_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[0-9a-fA-F]{16,64}$").unwrap());
static TOKEN_ID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_-]{8,64}$").unwrap());
static WHITESPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());
static FILE_EXT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.\w{1,6}$").unwrap());
static PLACEHOLDER_SEP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\s_\-]+").unwrap());
static FENCE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"```\w*\s*").unwrap());

/// Normalize a hex string to UUID format with dashes if it's 32 hex chars.
pub fn normalize_uuid(value: &str) -> String {
    let clean = value.replace('-', "").to_lowercase();
    if clean.chars().count() == 32 && clean.chars().all(|c| "0123456789abcdef".contains(c)) {
        return format!(
            "{}-{}-{}-{}-{}",
            &clean[..8],
            &clean[8..12],
            &clean[12..16],
            &clean[16..20],
            &clean[20..]
        );
    }
    value.to_string()
}

/// Agent that extracts tool arguments from user input based on the tool's input schema.
#[derive(Clone)]
pub struct MCPArgumentExtractor {
    /// Default LLM used for extraction.
    pub llm: Arc<dyn BaseLLM>,
}

impl MCPArgumentExtractor {
    /// Create an extractor backed by `llm`.
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Check if a parameter is a search/query parameter that should be filled directly from user input.
    pub fn is_query_parameter(&self, param_name: &str, param_info: &Value) -> bool {
        let name_lower = param_name.to_lowercase();
        if ["query", "search", "search_query", "q"].contains(&name_lower.as_str()) {
            return true;
        }
        let desc = str_field(param_info, "description").to_lowercase();
        ["search query", "search term", "query string"]
            .iter()
            .any(|kw| desc.contains(kw))
    }

    /// Check if a parameter expects a URL value.
    pub fn is_url_parameter(&self, param_name: &str, param_info: &Value) -> bool {
        let name_lower = param_name.to_lowercase();
        if URL_PARAM_KEYWORDS.iter().any(|kw| name_lower.contains(kw)) {
            return true;
        }
        let fmt = str_field(param_info, "format").to_lowercase();
        if ["uri", "url", "iri"].contains(&fmt.as_str()) {
            return true;
        }
        let desc = str_field(param_info, "description").to_lowercase();
        ["url", "uri", "web address", "link to"]
            .iter()
            .any(|kw| desc.contains(kw))
    }

    /// Extract all http/https URLs from user text.
    pub fn extract_urls_from_text(&self, text: &str) -> Vec<String> {
        URL_RE.find_iter(text).map(|m| m.as_str().to_string()).collect()
    }

    /// Check if a parameter is an ID-type parameter based on name and description.
    pub fn is_id_parameter(&self, param_name: &str, param_info: &Value) -> bool {
        if ID_PARAM_RES.iter().any(|re| re.is_match(param_name)) {
            return true;
        }

        let description = str_field(param_info, "description").to_lowercase();
        let id_keywords = ["identifier", "uuid", "unique id", "object id", "reference"];
        if id_keywords.iter().any(|kw| description.contains(kw)) {
            return true;
        }

        false
    }

    /// Check if a parameter is a cursor/pagination token that must not be fabricated.
    pub fn is_cursor_parameter(&self, param_name: &str) -> bool {
        CURSOR_PARAM_RES.iter().any(|re| re.is_match(param_name))
    }

    /// Extract potential identifiers from user text.
    pub fn extract_identifiers_from_text(&self, text: &str) -> Vec<String> {
        let mut identifiers = Vec::new();
        for re in IDENTIFIER_SEARCH_RES.iter() {
            identifiers.extend(re.find_iter(text).map(|m| m.as_str().to_string()));
        }
        identifiers
    }

    /// Check if a value looks like a valid identifier.
    pub fn is_valid_id_value(&self, value: &Value) -> bool {
        let Some(value) = value.as_str() else {
            return false;
        };

        let value = value.trim();
        if value.contains(' ') {
            return false;
        }

        if IDENTIFIER_FULL_RES.iter().any(|re| re.is_match(value)) {
            return true;
        }

        if HEX_ID_RE.is_match(value) {
            return true;
        }

        if TOKEN_ID_RE.is_match(value) {
            let value_lower = value.to_lowercase();
            if COMMON_WORDS.contains(&value_lower.as_str()) {
                return false;
            }
            if value.chars().count() > 20 && COMMON_WORDS.iter().any(|w| value_lower.contains(w)) {
                return false;
            }
            return true;
        }

        false
    }

    /// Lowercase, collapse whitespace, drop trailing punctuation.
    pub fn normalize_text(s: &str) -> String {
        let lowered = s.trim().to_lowercase();
        WHITESPACE_RE
            .replace_all(&lowered, " ")
            .trim_end_matches(['.', '!', '?', ' '])
            .to_string()
    }

    /// True when the extracted value just echoes the field's schema description.
    ///
    /// Small models frequently copy a parameter's description/guidance text
    /// (e.g. "Content as Markdown. Do not escape the string ...") into the
    /// value instead of producing a real value.  Such values are not real data
    /// and must be dropped so the param is treated as missing.
    ///
    /// Short values (e.g. ticker symbols, codes) are never considered echoes
    /// regardless of whether they appear in the description as examples.
    pub fn is_schema_echo(&self, value: &Value, param_info: &Value) -> bool {
        let Some(value) = value.as_str() else {
            return false;
        };
        let desc = str_field(param_info, "description");
        if desc.is_empty() {
            return false;
        }
        let nv = Self::normalize_text(value);
        let nd = Self::normalize_text(&desc);
        // Short values are almost certainly real data, not echoed descriptions.
        // A ticker like "AAPL" or a country code "US" should never be dropped.
        if nv.chars().count() < 20 {
            return false;
        }
        // Require a substantial description to avoid dropping legit short values.
        if nd.chars().count() < 20 {
            return false;
        }
        // Only treat as echo when the value matches the whole description or
        // contains most of it — not the reverse (value appearing as an example
        // in the description is fine).
        if nv == nd || nv.contains(&nd) {
            let head: String = value.chars().take(80).collect();
            tracing::warn!(
                target: LOG_TARGET,
                "Dropping echoed schema description as value: {}",
                py_repr_str(&head)
            );
            return true;
        }
        false
    }

    /// True when the value merely restates the parameter's name/description.
    ///
    /// Small models frequently emit the field's own name or description as a
    /// placeholder (e.g. path -> "file_path", url -> "your_url") instead of a
    /// real value.  These tokens are never legitimate data — dropping them lets
    /// the param be treated as missing rather than executed with a bogus value
    /// (e.g. reading a file literally named "file_path").
    pub fn is_placeholder_echo(&self, key: &str, value: &Value, param_info: &Value) -> bool {
        let Some(value) = value.as_str() else {
            return false;
        };
        let v = value.trim();
        if v.is_empty() {
            return false;
        }
        // Real data shapes are never placeholders: explicit paths, or a
        // filename with an extension (about.md, src/app.py, ./notes).
        if v.contains('/') || v.contains('\\') || FILE_EXT_RE.is_match(v) {
            return false;
        }

        let norm = |s: &str| -> String {
            let lowered = s.trim().to_lowercase();
            PLACEHOLDER_SEP_RE
                .replace_all(&lowered, " ")
                .trim()
                .to_string()
        };
        let nv = norm(v);
        if nv.is_empty() {
            return false;
        }

        if GENERIC_PLACEHOLDERS.contains(&nv.as_str()) {
            return true;
        }

        // Echo of the parameter name ("path" -> "path" / "file path").
        let nv_words: Vec<&str> = nv.split_whitespace().collect();
        let nk = norm(key);
        if !nk.is_empty() && (nv == nk || (nv_words.len() <= 2 && nv_words.contains(&nk.as_str()))) {
            return true;
        }

        // Echo of a leading phrase in the description ("file path" ⊂
        // "file path to read") for short, placeholder-like values.
        let desc = norm(&str_field(param_info, "description"));
        if !desc.is_empty() && nv_words.len() <= 3 && desc.contains(&nv) {
            return true;
        }

        false
    }

    /// Parse JSON from LLM response text, handling markdown fences.
    ///
    /// Strips fences with any language tag (`tool_code`, `json`, etc.) before
    /// parsing, since langchain's parser only handles `json`.  A response the
    /// partial parser rejects outright (mismatched brackets) yields `Null`.
    pub fn parse_json(&self, text: &str) -> anyhow::Result<Value> {
        let stripped = FENCE_RE.replace_all(text, "");
        let stripped = stripped.trim().trim_end_matches('`');
        parse_json_markdown(stripped)
    }

    /// Build a formatted context string from previous tool execution results.
    pub fn build_previous_results_context(&self, previous_tool_results: &[Value]) -> String {
        if previous_tool_results.is_empty() {
            return String::new();
        }

        let mut context_parts: Vec<String> = vec!["\nPREVIOUS TOOL RESULTS:".to_string()];

        for (idx, tool_result) in previous_tool_results.iter().enumerate() {
            let idx = idx + 1;
            let tool_name = match tool_result.get("tool_name") {
                Some(v) => py_str(v),
                None => "unknown".to_string(),
            };
            let empty = Value::Object(Map::new());
            let result_data = tool_result.get("result").unwrap_or(&empty);

            context_parts.push(format!("\n{idx}. Tool: {tool_name}"));

            match result_data {
                Value::Object(_) => {
                    let extracted_info = self.extract_key_fields(result_data);
                    if !extracted_info.is_empty() {
                        context_parts.push(format!(
                            "   Data: {}",
                            py_json_dumps_indent(&Value::Object(extracted_info))
                        ));
                    } else {
                        let result_str = py_json_dumps_indent(result_data);
                        if result_str.chars().count() < 500 {
                            context_parts.push(format!("   Result: {result_str}"));
                        } else {
                            context_parts.push(format!(
                                "   Result: (Large result - {} chars)",
                                py_str(result_data).chars().count()
                            ));
                        }
                    }
                }
                Value::Array(items) => {
                    context_parts.push(format!("   Items count: {}", items.len()));
                    for (item_idx, item) in items.iter().take(3).enumerate() {
                        let extracted = if item.is_object() {
                            Value::Object(self.extract_key_fields(item))
                        } else {
                            item.clone()
                        };
                        let rendered = if py_truthy(&extracted) {
                            py_json_dumps_indent(&extracted)
                        } else {
                            py_str(item).chars().take(100).collect()
                        };
                        context_parts.push(format!("   Item {}: {rendered}", item_idx + 1));
                    }
                }
                other => {
                    let result_str = py_str(other);
                    if result_str.chars().count() < 200 {
                        context_parts.push(format!("   Result: {result_str}"));
                    } else {
                        let head: String = result_str.chars().take(200).collect();
                        context_parts.push(format!("   Result: {head}..."));
                    }
                }
            }
        }

        context_parts.join("\n")
    }

    /// Extract key fields from a data dictionary (IDs, names, titles, etc.).
    /// Returns an empty map when `data` is not an object.
    pub fn extract_key_fields(&self, data: &Value) -> Map<String, Value> {
        let Some(data) = data.as_object() else {
            return Map::new();
        };

        let mut key_fields = Map::new();
        let priority_fields = [
            "id", "_id", "teamId", "issueId", "projectId", "userId", "commentId", "identifier",
            "key", "number", "name", "title", "label", "description",
        ];

        for field in priority_fields {
            if let Some(v) = data.get(field) {
                key_fields.insert(field.to_string(), v.clone());
            }
        }

        let empty = Value::Object(Map::new());
        for (key, value) in data {
            if !key_fields.contains_key(key) && self.is_id_parameter(key, &empty) {
                key_fields.insert(key.clone(), value.clone());
            }
        }

        key_fields
    }

    /// The params to extract: `required` when the schema declares it, else
    /// every non-ID, non-cursor property.
    ///
    /// schema.required is the source of truth.  When it is empty (common with
    /// MCP servers like Linear), fall back to non-ID properties — IDs are
    /// server-generated and the LLM should never fabricate them.
    fn accepted_params(&self, required_params: &[String], properties: &Map<String, Value>) -> Vec<String> {
        if !required_params.is_empty() {
            return required_params.to_vec();
        }
        let empty = Value::Object(Map::new());
        properties
            .keys()
            .filter(|p| {
                !self.is_id_parameter(p, properties.get(*p).unwrap_or(&empty))
                    && !self.is_cursor_parameter(p)
            })
            .cloned()
            .collect()
    }

    /// Extract tool arguments from user input based on the tool's input schema.
    ///
    /// Returns the extracted arguments. If required parameters are missing,
    /// the map contains a `_missing_params` key with the list of missing
    /// params. Never fails: errors are logged and an empty map is returned.
    pub fn invoke(
        &self,
        user_input: &str,
        tool_name: &str,
        input_schema: &Value,
        previous_tool_results: Option<&[Value]>,
        llm: Option<&dyn BaseLLM>,
    ) -> Map<String, Value> {
        let parameters_desc = self.build_parameters_description(input_schema);

        if parameters_desc.is_empty() {
            tracing::info!(target: LOG_TARGET, "Tool '{}' has no input parameters", tool_name);
            return Map::new();
        }

        let effective_llm: &dyn BaseLLM = llm.unwrap_or(self.llm.as_ref());

        // Build previous tool results context (for chained calls)
        let mut previous_results_context = String::new();
        if let Some(prev) = previous_tool_results {
            if !prev.is_empty() {
                previous_results_context = self.build_previous_results_context(prev);
            }
        }

        let mut chained_instructions = "";
        if !previous_results_context.is_empty() {
            chained_instructions = "
CHAINED TOOL CALL INSTRUCTIONS:
- Previous step results are provided — use them to infer parameter values
- Extract values from anywhere in the input: instructions, data, or original request
- If a value is clearly stated or derivable from context, use it
";
        }

        let required_params = string_list(input_schema.get("required"));
        let properties = object_field(input_schema, "properties");

        let accepted_params = self.accepted_params(&required_params, &properties);
        let allowed_keys = accepted_params
            .iter()
            .map(|p| format!("\"{p}\""))
            .collect::<Vec<_>>()
            .join(", ");
        let required_list = if required_params.is_empty() {
            "None".to_string()
        } else {
            required_params.join(", ")
        };

        let prompt = format!(
            "<system>
You are a precise argument extraction assistant.
Given a tool schema and user input, extract parameter values as a flat JSON object.
Use \"MISSING\" for any required value that cannot be determined.
Do NOT invent or fabricate values — only use what is stated or derivable from the user input.
The text after each parameter is a DESCRIPTION of the field — it is guidance, never a value. NEVER copy a parameter's description text as its value.
For parameters where no value can be derived from the user input, omit them entirely.
Respond with ONLY raw JSON. No markdown fences, no explanation.
</system>
<user>
Extract arguments for tool '{tool_name}'.

The JSON response MUST only use these keys: {allowed_keys}

Tool Parameters:
{parameters_desc}

Required: {required_list}
{chained_instructions}
Input:
{user_input}
Results from steps for context in chained calls:
{previous_results_context}
Respond with raw JSON only, no markdown.
</user>"
        );

        let prompt_tokens = effective_llm.get_num_tokens(&prompt);
        tracing::info!(
            target: LOG_TARGET,
            "Argument extraction prompt ({} tokens) for tool '{}'",
            prompt_tokens,
            tool_name
        );

        let response = effective_llm.invoke(&prompt, None, None);
        let response_text = if response.is_empty() { "{}".to_string() } else { response };

        let response_tokens = effective_llm.get_num_tokens(&response_text);
        tracing::info!(
            target: LOG_TARGET,
            "Argument extraction response ({} tokens): {}",
            response_tokens,
            response_text.chars().take(200).collect::<String>()
        );

        let arguments = match self.parse_json(&response_text) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Failed to parse arguments: {}", e);
                return Map::new();
            }
        };

        let Value::Object(arguments) = arguments else {
            tracing::warn!(
                target: LOG_TARGET,
                "Response is not a dict: {}",
                py_type_name(&arguments)
            );
            return Map::new();
        };

        // Keep only accepted parameters (required when declared, all
        // properties when required is empty). Filter out null/empty/fabricated
        // values.
        let empty = Value::Object(Map::new());
        let mut filtered_arguments = Map::new();
        for (k, v) in &arguments {
            let info = properties.get(k).unwrap_or(&empty);
            let keep = accepted_params.contains(k)
                && !v.is_null()
                && v != ""
                && v != "null"
                && v != "MISSING"
                && !v.is_object()
                && !v.is_array()
                && !self.is_schema_echo(v, info)
                && !self.is_placeholder_echo(k, v, info);
            if keep {
                filtered_arguments.insert(k.clone(), v.clone());
            }
        }

        // Only flag params as missing if they are actually required
        let missing_from_llm: Vec<String> = arguments
            .iter()
            .filter(|(k, v)| *v == "MISSING" && required_params.contains(*k))
            .map(|(k, _)| k.clone())
            .collect();

        // Ensure all required parameters are present
        let (mut filtered_arguments, missing_params) =
            self.ensure_required_params(filtered_arguments, input_schema, user_input);

        let mut all_missing: Vec<String> = Vec::new();
        for p in missing_from_llm.into_iter().chain(missing_params) {
            if !all_missing.contains(&p) {
                all_missing.push(p);
            }
        }

        if !all_missing.is_empty() {
            tracing::warn!(
                target: LOG_TARGET,
                "Missing required parameters for '{}': {}",
                tool_name,
                py_str_list(&all_missing)
            );
            filtered_arguments.insert(
                "_missing_params".to_string(),
                Value::Array(all_missing.into_iter().map(Value::String).collect()),
            );
        }

        let rendered = py_repr(&Value::Object(filtered_arguments.clone()));
        tracing::info!(
            target: LOG_TARGET,
            "Extracted arguments for '{}': {}",
            tool_name,
            rendered
        );
        filtered_arguments
    }

    /// Ensure all required parameters are present in arguments.
    ///
    /// Every required parameter is treated uniformly — schema.required is the
    /// sole source of truth.  Missing string params without a default are
    /// reported as missing; other types fall back to the schema default or a
    /// type-zero value.
    pub fn ensure_required_params(
        &self,
        mut arguments: Map<String, Value>,
        input_schema: &Value,
        user_input: &str,
    ) -> (Map<String, Value>, Vec<String>) {
        let _ = user_input;
        let required_params = string_list(input_schema.get("required"));
        let properties = object_field(input_schema, "properties");
        let mut missing_params: Vec<String> = Vec::new();
        let empty = Value::Object(Map::new());

        for param_name in &required_params {
            let param_info = properties.get(param_name).unwrap_or(&empty);
            let param_type = param_info.get("type").cloned().unwrap_or_else(|| Value::from("string"));

            let is_missing = match arguments.get(param_name) {
                None | Some(Value::Null) => true,
                Some(v) => v == "",
            };

            if is_missing {
                let mut default_value = param_info.get("default").cloned().filter(|v| !v.is_null());

                if default_value.is_none() {
                    match param_type.as_str() {
                        Some("string") => {
                            // For string params without a default, flag as
                            // missing so the caller can decide what to do.
                            missing_params.push(param_name.clone());
                            continue;
                        }
                        Some("integer") => default_value = Some(Value::from(0)),
                        Some("number") => default_value = Some(Value::from(0.0)),
                        Some("boolean") => default_value = Some(Value::Bool(false)),
                        Some("array") => default_value = Some(Value::Array(Vec::new())),
                        Some("object") => default_value = Some(Value::Object(Map::new())),
                        _ => {}
                    }
                }

                if let Some(default_value) = default_value {
                    arguments.insert(param_name.clone(), default_value);
                }
            }
        }

        (arguments, missing_params)
    }

    /// Build a human-readable description of parameters from JSON Schema.
    ///
    /// Only describes accepted parameters (required when declared, all
    /// properties when required is empty). This prevents the LLM from seeing
    /// and fabricating values for optional params like 'id'.
    pub fn build_parameters_description(&self, input_schema: &Value) -> String {
        let Some(schema) = input_schema.as_object() else {
            return String::new();
        };
        if schema.is_empty() || !schema.contains_key("properties") {
            return String::new();
        }

        let properties = object_field(input_schema, "properties");
        let required = string_list(schema.get("required"));

        if properties.is_empty() {
            return String::new();
        }

        // Only show the params we actually want extracted — matches the
        // accepted_params logic in invoke().
        let accepted_params = self.accepted_params(&required, &properties);

        let empty = Value::Object(Map::new());
        let descs: Vec<String> = accepted_params
            .iter()
            .map(|param_name| {
                let param_info = properties.get(param_name).unwrap_or(&empty);
                describe_param(param_name, param_info, "any")
            })
            .collect();

        if !required.is_empty() {
            return format!(
                "REQUIRED parameters (must always be provided):\n{}",
                descs.join("\n")
            );
        }
        format!("Parameters:\n{}", descs.join("\n"))
    }

    /// Extract arguments with pre-resolved IDs.  Only uses the LLM for the
    /// accepted params that are not already resolved.
    ///
    /// Returns the resolved ids plus the LLM-extracted content; when the
    /// schema declares required params, any still missing are listed under
    /// `_missing_params`.
    pub fn invoke_with_resolved_ids(
        &self,
        user_input: &str,
        tool_name: &str,
        input_schema: &Value,
        resolved_ids: &Map<String, Value>,
        llm: Option<&dyn BaseLLM>,
    ) -> Map<String, Value> {
        let required_params = string_list(input_schema.get("required"));
        let properties = object_field(input_schema, "properties");
        let accepted_params = self.accepted_params(&required_params, &properties);

        let mut arguments = resolved_ids.clone();
        let resolved_keys: Vec<String> = resolved_ids.keys().cloned().collect();
        tracing::info!(
            target: LOG_TARGET,
            "invoke_with_resolved_ids: Starting with pre-resolved: {}",
            py_str_list(&resolved_keys)
        );

        // Find accepted params that still need extraction (not already resolved)
        let remaining: Vec<String> = accepted_params
            .into_iter()
            .filter(|p| !resolved_ids.contains_key(p))
            .collect();

        if !remaining.is_empty() {
            tracing::info!(
                target: LOG_TARGET,
                "invoke_with_resolved_ids: Extracting remaining params via LLM: {}",
                py_str_list(&remaining)
            );
            let llm_extracted = self.extract_non_id_params(user_input, &remaining, &properties, llm);
            for (k, v) in llm_extracted {
                arguments.insert(k, v);
            }
        } else {
            tracing::info!(target: LOG_TARGET, "invoke_with_resolved_ids: All params already resolved");
        }

        // Flag missing required params (only when schema declares them)
        if !required_params.is_empty() {
            let missing_params: Vec<String> = required_params
                .iter()
                .filter(|p| match arguments.get(*p) {
                    None | Some(Value::Null) => true,
                    Some(v) => v == "",
                })
                .cloned()
                .collect();
            if !missing_params.is_empty() {
                arguments.insert(
                    "_missing_params".to_string(),
                    Value::Array(missing_params.into_iter().map(Value::String).collect()),
                );
            }
        }

        let final_keys: Vec<String> = arguments.keys().cloned().collect();
        tracing::info!(
            target: LOG_TARGET,
            "invoke_with_resolved_ids: Final arguments for '{}': {}",
            tool_name,
            py_str_list(&final_keys)
        );
        arguments
    }

    /// Extract only non-ID parameters using the LLM.
    pub fn extract_non_id_params(
        &self,
        user_input: &str,
        non_id_params: &[String],
        properties: &Map<String, Value>,
        llm: Option<&dyn BaseLLM>,
    ) -> Map<String, Value> {
        if non_id_params.is_empty() {
            return Map::new();
        }

        let empty = Value::Object(Map::new());
        let params_desc = non_id_params
            .iter()
            .map(|param_name| {
                let param_info = properties.get(param_name).unwrap_or(&empty);
                describe_param(param_name, param_info, "string")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "<system>
You are a precise parameter extraction assistant.
Extract values for the requested parameters from the user input.
Do NOT make up values that aren't present in the input.
The text after each parameter is a DESCRIPTION of the field — never copy a parameter's description text as its value.
Respond with ONLY raw JSON. No markdown fences, no explanation.
</system>
<user>
Extract values for the following parameters from the input.

Input: {user_input}

Parameters to extract:
{params_desc}

Respond with raw JSON only, no markdown.
</user>"
        );

        let effective_llm: &dyn BaseLLM = llm.unwrap_or(self.llm.as_ref());
        let prompt_tokens = effective_llm.get_num_tokens(&prompt);
        tracing::info!(target: LOG_TARGET, "Content extraction prompt ({} tokens)", prompt_tokens);

        let response = effective_llm.invoke(&prompt, None, None);
        let response_text = if response.is_empty() { "{}".to_string() } else { response };

        let response_tokens = effective_llm.get_num_tokens(&response_text);
        tracing::info!(
            target: LOG_TARGET,
            "Content extraction response ({} tokens): {}",
            response_tokens,
            response_text.chars().take(200).collect::<String>()
        );

        let extracted = match self.parse_json(&response_text) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(target: LOG_TARGET, "Non-ID param extraction failed: {}", e);
                return Map::new();
            }
        };

        let Value::Object(extracted) = extracted else {
            return Map::new();
        };

        let mut result = Map::new();
        for (k, v) in extracted {
            let info = properties.get(&k).unwrap_or(&empty);
            if non_id_params.contains(&k)
                && !v.is_null()
                && v != ""
                && !self.is_schema_echo(&v, info)
                && !self.is_placeholder_echo(&k, &v, info)
            {
                result.insert(k, v);
            }
        }

        let keys: Vec<String> = result.keys().cloned().collect();
        tracing::info!(target: LOG_TARGET, "Extracted non-ID params: {}", py_str_list(&keys));
        result
    }
}

/// `- name (type): description` plus allowed values / default lines.
fn describe_param(param_name: &str, param_info: &Value, default_type: &str) -> String {
    let param_type = match param_info.get("type") {
        Some(v) => py_str(v),
        None => default_type.to_string(),
    };
    let param_desc = match param_info.get("description") {
        Some(v) => py_str(v),
        None => "No description".to_string(),
    };

    let mut param_str = format!("- {param_name} ({param_type}): {param_desc}");

    if let Some(enum_values) = param_info.get("enum") {
        let rendered: Vec<String> = match enum_values {
            Value::Array(items) => items.iter().map(py_str).collect(),
            Value::Object(map) => map.keys().cloned().collect(),
            Value::String(s) => s.chars().map(|c| c.to_string()).collect(),
            other => vec![py_str(other)],
        };
        param_str.push_str(&format!("\n  Allowed values: {}", rendered.join(", ")));
    }
    if let Some(default) = param_info.get("default") {
        param_str.push_str(&format!("\n  Default: {}", py_str(default)));
    }

    param_str
}

// ------------------------------------------------------------------ schema access helpers

/// `info.get(key, '')` for a string field; non-strings read as `""`.
pub(crate) fn str_field(info: &Value, key: &str) -> String {
    info.get(key).and_then(Value::as_str).unwrap_or("").to_string()
}

/// `schema.get(key, {})` as a map (non-objects read as empty).
pub(crate) fn object_field(schema: &Value, key: &str) -> Map<String, Value> {
    schema.get(key).and_then(Value::as_object).cloned().unwrap_or_default()
}

/// A JSON list of names (`required`), keeping only the string entries.
pub(crate) fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

// ------------------------------------------------------------------ Python rendering helpers

/// Python truthiness of a JSON value.
pub(crate) fn py_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Python `type(x).__name__` for a JSON-derived value.
pub(crate) fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python `repr()` of a `str`.
pub(crate) fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr()` of a float.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    // Shortest round-trip digits, in scientific form, to read the exponent.
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        let plain = format!("{f}");
        if plain.contains('.') { plain } else { format!("{plain}.0") }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

/// Python `repr()` of a JSON-derived value (dict/list/str/int/float/bool/None).
pub(crate) fn py_repr(v: &Value) -> String {
    match v {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_float_repr(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => {
            format!("[{}]", items.iter().map(py_repr).collect::<Vec<_>>().join(", "))
        }
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `str()` of a JSON-derived value (a bare string renders unquoted).
pub(crate) fn py_str(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python `repr()` of a list of strings (`['a', 'b']`).
pub(crate) fn py_str_list(items: &[String]) -> String {
    format!(
        "[{}]",
        items.iter().map(|s| py_repr_str(s)).collect::<Vec<_>>().join(", ")
    )
}

/// Python `json.dumps(value, indent=2)` (ASCII-escaped, `": "` separators).
pub(crate) fn py_json_dumps_indent(v: &Value) -> String {
    let pretty = serde_json::to_string_pretty(v).unwrap_or_default();
    let mut out = String::with_capacity(pretty.len());
    for c in pretty.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for unit in c.encode_utf16(&mut buf) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

// ------------------------------------------------------------------ langchain JSON parsing

static JSON_MARKDOWN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```(json)?(.*)").unwrap());
static ACTION_INPUT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)("action_input":\s*")(.*?)(")"#).unwrap());

/// Characters `_parse_json` strips from both ends.
const JSON_STRIP_CHARS: &[char] = &[' ', '\n', '\r', '\t', '`'];

/// `json.loads(s, strict=False)`: raw control characters are allowed inside
/// strings, so they are escaped before handing the text to serde.
fn json_loads(s: &str) -> Result<Value, serde_json::Error> {
    let mut escaped = String::with_capacity(s.len());
    let mut in_string = false;
    let mut backslash = false;
    for c in s.chars() {
        if in_string {
            if backslash {
                backslash = false;
                escaped.push(c);
                continue;
            }
            match c {
                '\\' => {
                    backslash = true;
                    escaped.push(c);
                }
                '"' => {
                    in_string = false;
                    escaped.push(c);
                }
                c if (c as u32) < 0x20 => escaped.push_str(&format!("\\u{:04x}", c as u32)),
                c => escaped.push(c),
            }
        } else {
            if c == '"' {
                in_string = true;
            }
            escaped.push(c);
        }
    }
    serde_json::from_str(&escaped)
}

/// LangChain `_custom_parser`: escape raw newlines/tabs inside an
/// `"action_input"` string value.
fn custom_parser(multiline_string: &str) -> String {
    ACTION_INPUT_RE
        .replace_all(multiline_string, |caps: &regex::Captures| {
            // Group 2 is non-greedy up to the first quote, so it never holds
            // an unescaped `"`; only the whitespace escapes apply.
            let value = caps[2]
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("{}{}{}", &caps[1], value, &caps[3])
        })
        .into_owned()
}

/// LangChain `parse_partial_json`: parse JSON that may be missing closing
/// braces/quotes.  Returns `Null` when a closing bracket is mismatched.
pub fn parse_partial_json(s: &str) -> Result<Value, serde_json::Error> {
    // Attempt to parse the string as-is.
    if let Ok(v) = json_loads(s) {
        return Ok(v);
    }

    let mut new_chars: Vec<String> = Vec::new();
    let mut stack: Vec<char> = Vec::new();
    let mut is_inside_string = false;
    let mut escaped = false;

    // Process each character in the string one at a time.
    for c in s.chars() {
        let mut new_char = c.to_string();
        if is_inside_string {
            if c == '"' && !escaped {
                is_inside_string = false;
            } else if c == '\n' && !escaped {
                // Replace the newline character with the escape sequence.
                new_char = "\\n".to_string();
            } else if c == '\\' {
                escaped = !escaped;
            } else {
                escaped = false;
            }
        } else if c == '"' {
            is_inside_string = true;
            escaped = false;
        } else if c == '{' {
            stack.push('}');
        } else if c == '[' {
            stack.push(']');
        } else if c == '}' || c == ']' {
            if stack.last() == Some(&c) {
                stack.pop();
            } else {
                // Mismatched closing character; the input is malformed.
                return Ok(Value::Null);
            }
        }

        new_chars.push(new_char);
    }

    // If we're still inside a string at the end of processing, close it.
    if is_inside_string {
        if escaped {
            // Remove unterminated escape character
            new_chars.pop();
        }
        new_chars.push("\"".to_string());
    }

    // Reverse the stack to get the closing characters.
    let closing: String = stack.iter().rev().collect();

    // Try to parse mods of string until we succeed or run out of characters.
    while !new_chars.is_empty() {
        let candidate = format!("{}{}", new_chars.concat(), closing);
        if let Ok(v) = json_loads(&candidate) {
            return Ok(v);
        }
        new_chars.pop();
    }

    // Ran out of characters to remove: return the parse error for the original.
    json_loads(s)
}

/// LangChain `_parse_json`.
fn parse_json_inner(json_str: &str) -> Result<Value, serde_json::Error> {
    let json_str = json_str.trim_matches(JSON_STRIP_CHARS);
    let json_str = custom_parser(json_str);
    parse_partial_json(&json_str)
}

/// LangChain `parse_json_markdown`: parse a JSON string from a Markdown string.
pub fn parse_json_markdown(json_string: &str) -> anyhow::Result<Value> {
    match parse_json_inner(json_string) {
        Ok(v) => Ok(v),
        Err(_) => {
            // Try to find JSON string within triple backticks; if none, the
            // entire string is assumed to be JSON.
            let json_str = match JSON_MARKDOWN_RE.captures(json_string) {
                Some(caps) => caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string(),
                None => json_string.to_string(),
            };
            Ok(parse_json_inner(&json_str)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_json_markdown_handles_fences_and_partial_json() {
        assert_eq!(parse_json_markdown("{\"a\": 1}").unwrap(), json!({"a": 1}));
        assert_eq!(parse_json_markdown("```json\n{\"a\": 1}\n```").unwrap(), json!({"a": 1}));
        // Missing closers are completed, unterminated strings closed.
        assert_eq!(parse_json_markdown("{\"a\": [1, 2").unwrap(), json!({"a": [1, 2]}));
        assert_eq!(parse_json_markdown("{\"a\": \"hel").unwrap(), json!({"a": "hel"}));
        // Raw newlines inside strings are tolerated (strict=False).
        assert_eq!(parse_json_markdown("{\"a\": \"x\ny\"}").unwrap(), json!({"a": "x\ny"}));
        // A mismatched closer yields None.
        assert_eq!(parse_json_markdown("{\"a\": 1]").unwrap(), Value::Null);
        assert!(parse_json_markdown("not json").is_err());
    }

    #[test]
    fn parse_json_strips_any_fence_tag() {
        let llm: Arc<dyn BaseLLM> = Arc::new(NoLlm);
        let ex = MCPArgumentExtractor::new(llm);
        assert_eq!(ex.parse_json("```tool_code\n{\"q\": \"x\"}\n```").unwrap(), json!({"q": "x"}));
    }

    #[test]
    fn normalize_uuid_formats_32_hex_chars() {
        assert_eq!(
            normalize_uuid("0123456789ABCDEF0123456789abcdef"),
            "01234567-89ab-cdef-0123-456789abcdef"
        );
        assert_eq!(normalize_uuid("LIN-123"), "LIN-123");
    }

    #[test]
    fn python_reprs() {
        assert_eq!(
            py_repr(&json!({"a": [1, 2.5, true, null, "x'y"]})),
            "{'a': [1, 2.5, True, None, \"x'y\"]}"
        );
        assert_eq!(py_repr(&json!(1.0)), "1.0");
        assert_eq!(py_repr(&json!(1e20)), "1e+20");
        assert_eq!(py_repr(&json!(0.0001)), "0.0001");
        assert_eq!(py_repr(&json!(0.00001)), "1e-05");
        assert_eq!(py_json_dumps_indent(&json!({"k": "é"})), "{\n  \"k\": \"\\u00e9\"\n}");
    }

    #[test]
    fn parameter_classification() {
        let llm: Arc<dyn BaseLLM> = Arc::new(NoLlm);
        let ex = MCPArgumentExtractor::new(llm);
        let empty = json!({});
        assert!(ex.is_id_parameter("issueId", &empty));
        assert!(ex.is_id_parameter("team_id", &empty));
        assert!(!ex.is_id_parameter("title", &empty));
        assert!(ex.is_cursor_parameter("after"));
        assert!(ex.is_cursor_parameter("nextCursor"));
        assert!(!ex.is_cursor_parameter("afterDate"));
        assert!(ex.is_url_parameter("target", &json!({"format": "uri"})));
        assert!(ex.is_query_parameter("q", &empty));
        assert!(ex.is_valid_id_value(&json!("LIN-123")));
        assert!(!ex.is_valid_id_value(&json!("create")));
        assert!(!ex.is_valid_id_value(&json!("two words")));
        assert_eq!(
            ex.extract_urls_from_text("see https://example.com/a?b=1 and http://x.io."),
            vec!["https://example.com/a?b=1".to_string(), "http://x.io.".to_string()]
        );

        // No required params: ids and cursors are hidden from the model.
        let schema = json!({"properties": {"id": {}, "after": {}, "title": {"description": "Title"}}});
        assert_eq!(ex.build_parameters_description(&schema), "Parameters:\n- title (any): Title");
    }

    #[test]
    fn ensure_required_params_defaults_by_type() {
        let llm: Arc<dyn BaseLLM> = Arc::new(NoLlm);
        let ex = MCPArgumentExtractor::new(llm);
        let schema = json!({
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "flag": {"type": "boolean"},
                "mode": {"type": "string", "default": "fast"},
            },
            "required": ["name", "count", "flag", "mode"],
        });
        let (args, missing) = ex.ensure_required_params(Map::new(), &schema, "");
        assert_eq!(missing, vec!["name".to_string()]);
        assert_eq!(Value::Object(args), json!({"count": 0, "flag": false, "mode": "fast"}));
    }

    /// A model that must never be called.
    struct NoLlm;

    impl BaseLLM for NoLlm {
        fn project_id(&self) -> &str {
            ""
        }
        fn get_num_tokens(&self, _text: &str) -> usize {
            0
        }
        fn apply_chat_template(
            &self,
            messages: Vec<crate::llms::base_llm::ChatMessage>,
        ) -> Vec<crate::llms::base_llm::ChatMessage> {
            messages
        }
        fn get_max_tokens(&self) -> usize {
            0
        }
        fn call(
            &self,
            _prompt: &str,
            _stop: Option<&[String]>,
            _options: crate::llms::base_llm::CallOptions,
        ) -> anyhow::Result<String> {
            unreachable!("not called")
        }
        fn stream<'a>(
            &'a self,
            _prompt: &str,
            _options: crate::llms::base_llm::CallOptions,
        ) -> crate::llms::base_llm::GenerationStream<'a> {
            unreachable!("not called")
        }
        fn cancelled(&self) -> Option<crate::llms::base_llm::CancelEvent> {
            None
        }
        fn set_cancelled(&self, _event: Option<crate::llms::base_llm::CancelEvent>) {}
        fn llm_type(&self) -> &str {
            "none"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }
}
