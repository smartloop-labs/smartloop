//! MCP Dependency Analyzer Agent - Analyzes tool dependencies and determines execution order.
//!
//! Port of `smartloop/agents/mcp/dependency_analyzer.py`. Pure logic: examines
//! input schemas to identify ID dependencies and finds prerequisite list tools
//! to resolve them.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::agents::mcp::argument_extractor::{object_field, py_str, py_str_list, str_field, string_list};

/// Logger name used by the Python module.
pub const LOG_TARGET: &str = "smartloop.agents.mcp.dependency_analyzer";

/// Information about a tool dependency.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DependencyInfo {
    /// Name of the parameter that has a dependency.
    pub param_name: String,
    /// Type of the parameter.
    pub param_type: String,
    /// Description of the parameter.
    pub description: String,
    /// Name of the tool to execute first.
    #[serde(default)]
    pub prerequisite_tool: Option<String>,
    /// Field name to extract from prerequisite result.
    #[serde(default)]
    pub extraction_field: Option<String>,
    /// Reason for the dependency.
    #[serde(default)]
    pub reason: String,
}

/// Result of dependency analysis for a tool.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ToolDependencyAnalysis {
    /// Whether the tool has dependencies that need resolution.
    pub has_dependencies: bool,
    /// List of identified dependencies.
    #[serde(default)]
    pub dependencies: Vec<DependencyInfo>,
    /// Ordered list of tools to execute.
    #[serde(default)]
    pub execution_order: Vec<String>,
    /// Whether the tool can be executed with current information.
    pub can_execute: bool,
    /// Parameters that cannot be resolved.
    #[serde(default)]
    pub missing_info: Vec<String>,
}

/// Common ID parameter patterns (matched case-insensitively at the start).
pub const ID_PARAM_PATTERNS: &[&str] = &[
    r".*[iI][dD]$",
    r".*[iI][dD]s$",
    r"^id$",
    r".*_id$",
    r".*_ids$",
];

/// Compiled [`ID_PARAM_PATTERNS`]: `re.match(pattern, name, re.IGNORECASE)`.
static ID_PARAM_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    ID_PARAM_PATTERNS
        .iter()
        .map(|p| Regex::new(&format!("^(?i:{p})")).expect("valid id pattern"))
        .collect()
});

/// ID shapes searched for in the user input (case-sensitive `re.search`).
static CONTEXT_ID_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"[A-Z]{2,10}-\d+",
        r"#\d+",
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        r"[0-9a-fA-F]{24,64}",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid id pattern"))
    .collect()
});

/// Tool-name words that mark a tool as a mutation rather than a listing.
const MUTATION_WORDS: &[&str] = &["create", "update", "delete", "remove", "add", "set"];

/// The `name` of a tool dict, `""` when absent.
fn tool_name_of(tool: &Value) -> String {
    str_field(tool, "name")
}

/// `required` of a tool's `input_schema` (empty when there is no schema).
fn tool_required(tool: &Value) -> Vec<String> {
    string_list(tool.get("input_schema").and_then(|s| s.get("required")))
}

/// Analyzes MCP tool input schemas to identify dependencies and determines
/// the execution order for chained tool calls.
#[derive(Debug, Clone, Default)]
pub struct MCPDependencyAnalyzer;

impl MCPDependencyAnalyzer {
    /// Create an analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Check if a parameter indicates it needs an ID/reference value.
    pub fn is_id_parameter(&self, param_name: &str, param_info: Option<&Value>) -> bool {
        if ID_PARAM_RES.iter().any(|re| re.is_match(param_name)) {
            return true;
        }

        if let Some(info) = param_info {
            let non_empty = match info {
                Value::Object(m) => !m.is_empty(),
                Value::Null => false,
                _ => true,
            };
            if non_empty {
                let description = str_field(info, "description").to_lowercase();
                let words = [
                    "identifier", "uuid", "id of", "unique id", " id", "team id", "project id",
                ];
                if words.iter().any(|w| description.contains(w)) {
                    return true;
                }
            }
        }

        false
    }

    /// Extract the entity name from a parameter
    /// (`teamId` -> `team`, `issueIds` -> `issue`, `project_id` -> `project`).
    pub fn extract_entity_from_param(&self, param_name: &str) -> Option<String> {
        let mut entity = param_name;
        let suffixes = ["Id", "Ids", "ID", "IDs", "_id", "_ids", "id", "ids"];
        for suffix in suffixes {
            if entity.ends_with(suffix) && entity.len() > suffix.len() {
                entity = &entity[..entity.len() - suffix.len()];
                break;
            }
        }

        let entity = entity.replace(['_', '-'], "").to_lowercase();
        if entity.is_empty() {
            Some(param_name.to_lowercase())
        } else {
            Some(entity)
        }
    }

    /// Find a tool that can list/get the given entity type (deterministic pattern matching).
    pub fn find_list_tool_for_entity(&self, entity: &str, available_tools: &[Value]) -> Option<String> {
        let entity_lower = entity.to_lowercase();
        let entity_plural = if entity_lower.ends_with('s') {
            entity_lower.clone()
        } else {
            format!("{entity_lower}s")
        };

        let tool_names: Vec<String> = available_tools.iter().map(tool_name_of).collect();
        tracing::info!(
            target: LOG_TARGET,
            "Looking for list tool for entity '{}'. Available tools: {}",
            entity,
            py_str_list(&tool_names)
        );

        // Priority 1: Exact patterns
        let exact_patterns = [
            format!("list_{entity_plural}"),
            format!("list-{entity_plural}"),
            format!("list_{entity_lower}"),
            format!("list-{entity_lower}"),
            format!("get_{entity_plural}"),
            format!("get-{entity_plural}"),
            format!("list{entity_plural}"),
            format!("get{entity_plural}"),
        ];

        for tool in available_tools {
            let tool_name = tool_name_of(tool);
            if exact_patterns.contains(&tool_name.to_lowercase()) {
                tracing::info!(
                    target: LOG_TARGET,
                    "Found exact match list tool '{}' for entity '{}'",
                    tool_name,
                    entity
                );
                return Some(tool_name);
            }
        }

        // Priority 2: Tools starting with list/get that contain entity and have no required params
        for tool in available_tools {
            let tool_name = tool_name_of(tool);
            let tool_name_lower = tool_name.to_lowercase();
            let required_params = tool_required(tool);

            let has_list_prefix =
                tool_name_lower.starts_with("list") || tool_name_lower.starts_with("get");
            let contains_entity =
                tool_name_lower.contains(&entity_lower) || tool_name_lower.contains(&entity_plural);
            let no_required_params = required_params.is_empty();

            let is_mutation = MUTATION_WORDS.iter().any(|w| tool_name_lower.contains(w));

            if has_list_prefix && contains_entity && no_required_params && !is_mutation {
                tracing::info!(
                    target: LOG_TARGET,
                    "Found list tool '{}' for entity '{}' (prefix match)",
                    tool_name,
                    entity
                );
                return Some(tool_name);
            }
        }

        // Priority 3: Fallback - any tool with entity name and list-like characteristics
        for tool in available_tools {
            let tool_name = tool_name_of(tool);
            let tool_name_lower = tool_name.to_lowercase();
            let tool_desc = str_field(tool, "description").to_lowercase();
            let required_params = tool_required(tool);

            let contains_entity =
                tool_name_lower.contains(&entity_lower) || tool_name_lower.contains(&entity_plural);
            let is_mutation = MUTATION_WORDS.iter().any(|w| tool_name_lower.contains(w));
            let desc_suggests_list = ["list", "get all", "fetch all", "returns all", "retrieve"]
                .iter()
                .any(|w| tool_desc.contains(w));

            if contains_entity && !is_mutation && (desc_suggests_list || required_params.is_empty()) {
                tracing::info!(
                    target: LOG_TARGET,
                    "Found list tool '{}' for entity '{}' (fallback match)",
                    tool_name,
                    entity
                );
                return Some(tool_name);
            }
        }

        tracing::warn!(
            target: LOG_TARGET,
            "No list tool found for entity '{}' in available tools: {}",
            entity,
            py_str_list(&tool_names)
        );
        None
    }

    /// Analyze the tool's input schema to identify dependencies and determine execution order.
    pub fn invoke(
        &self,
        user_input: &str,
        tool_name: &str,
        input_schema: &Value,
        available_tools: &[Value],
    ) -> ToolDependencyAnalysis {
        let required_params = string_list(input_schema.get("required"));
        let properties: Map<String, Value> = object_field(input_schema, "properties");

        if required_params.is_empty() {
            return ToolDependencyAnalysis {
                has_dependencies: false,
                dependencies: Vec::new(),
                execution_order: vec![tool_name.to_string()],
                can_execute: true,
                missing_info: Vec::new(),
            };
        }

        let mut dependencies: Vec<DependencyInfo> = Vec::new();
        let mut prerequisite_tools: Vec<String> = Vec::new();

        tracing::info!(
            target: LOG_TARGET,
            "Checking {} required params for dependencies: {}",
            required_params.len(),
            py_str_list(&required_params)
        );

        let empty = Value::Object(Map::new());
        for param_name in &required_params {
            let param_info = properties.get(param_name).unwrap_or(&empty);
            let param_type = match param_info.get("type") {
                Some(v) => py_str(v),
                None => "string".to_string(),
            };
            let param_desc = str_field(param_info, "description");

            if !self.is_id_parameter(param_name, Some(param_info)) {
                continue;
            }
            if self.check_for_id_in_context(param_name, user_input) {
                continue;
            }

            let entity = match self.extract_entity_from_param(param_name) {
                Some(e) if !e.is_empty() => e,
                _ => continue,
            };

            match self.find_list_tool_for_entity(&entity, available_tools) {
                Some(prerequisite_tool) => {
                    dependencies.push(DependencyInfo {
                        param_name: param_name.clone(),
                        param_type,
                        description: param_desc,
                        prerequisite_tool: Some(prerequisite_tool.clone()),
                        extraction_field: Some("id".to_string()),
                        reason: format!("Need to list {entity}s to get the {param_name}"),
                    });

                    if !prerequisite_tools.contains(&prerequisite_tool) {
                        prerequisite_tools.push(prerequisite_tool);
                    }
                }
                None => {
                    dependencies.push(DependencyInfo {
                        param_name: param_name.clone(),
                        param_type,
                        description: param_desc,
                        prerequisite_tool: None,
                        extraction_field: None,
                        reason: format!("No list tool found for {entity}"),
                    });
                }
            }
        }

        let mut execution_order = prerequisite_tools;
        execution_order.push(tool_name.to_string());

        let missing_info: Vec<String> = dependencies
            .iter()
            .filter(|d| d.prerequisite_tool.is_none())
            .map(|d| d.param_name.clone())
            .collect();
        let can_execute = missing_info.is_empty();
        let has_dependencies = !dependencies.is_empty();

        tracing::info!(
            target: LOG_TARGET,
            "Dependency analysis for '{}': dependencies={}, execution_order={}, can_execute={}",
            tool_name,
            dependencies.len(),
            py_str_list(&execution_order),
            if can_execute { "True" } else { "False" }
        );

        ToolDependencyAnalysis {
            has_dependencies,
            dependencies,
            execution_order,
            can_execute,
            missing_info,
        }
    }

    /// Check if an ID value for the parameter exists in user input.
    pub fn check_for_id_in_context(&self, param_name: &str, user_input: &str) -> bool {
        for re in CONTEXT_ID_RES.iter() {
            if re.is_match(user_input) {
                tracing::info!(
                    target: LOG_TARGET,
                    "Found ID pattern in user input for {}",
                    param_name
                );
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn finds_prerequisite_list_tool() {
        let a = MCPDependencyAnalyzer::new();
        let tools = vec![
            json!({"name": "create_issue", "input_schema": {"required": ["teamId", "title"]}}),
            json!({"name": "list_teams", "input_schema": {}}),
        ];
        let schema = json!({
            "properties": {"teamId": {"type": "string"}, "title": {"type": "string"}},
            "required": ["teamId", "title"],
        });
        let analysis = a.invoke("create a bug", "create_issue", &schema, &tools);
        assert!(analysis.has_dependencies);
        assert!(analysis.can_execute);
        assert_eq!(analysis.execution_order, vec!["list_teams", "create_issue"]);
        assert_eq!(analysis.dependencies[0].reason, "Need to list teams to get the teamId");

        // An id in the input means no prerequisite is needed.
        let analysis = a.invoke("move LIN-12", "create_issue", &schema, &tools);
        assert!(!analysis.has_dependencies);
        assert_eq!(analysis.execution_order, vec!["create_issue"]);
    }

    #[test]
    fn entity_from_param() {
        let a = MCPDependencyAnalyzer::new();
        assert_eq!(a.extract_entity_from_param("teamId").as_deref(), Some("team"));
        assert_eq!(a.extract_entity_from_param("issueIds").as_deref(), Some("issue"));
        assert_eq!(a.extract_entity_from_param("project_id").as_deref(), Some("project"));
        assert_eq!(a.extract_entity_from_param("id").as_deref(), Some("id"));
    }
}
