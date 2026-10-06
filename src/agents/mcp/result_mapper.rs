//! MCP Result Mapper - Creates structured ID maps from list tool results for direct lookup.
//!
//! Port of `smartloop/agents/mcp/result_mapper.py`. Pure utility, no LLM.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::agents::mcp::argument_extractor::{normalize_uuid, py_str, py_type_name, str_field};

/// Logger name used by the Python module.
pub const LOG_TARGET: &str = "smartloop.agents.mcp.result_mapper";

/// Insertion-ordered `name/id -> id` map. Python dicts preserve insertion
/// order, which `get_id`'s user-input scan depends on; `serde_json::Map` is
/// insertion-ordered under the `preserve_order` feature. Values are strings.
pub type OrderedMap = Map<String, Value>;

/// Tool-name prefixes stripped to find the entity a tool works on (longest
/// spelling of each verb first, so `list-teams` does not become `-team`).
const TOOL_PREFIXES: &[&str] = &[
    "list-", "list_", "list", "get-", "get_", "get", "search-", "search_", "search", "find-",
    "find_", "find", "fetch-", "fetch_", "fetch", "query-", "query_", "query", "update-",
    "update_", "update", "create-", "create_", "create", "add-", "add_", "add", "delete-",
    "delete_", "delete", "remove-", "remove_", "remove", "edit-", "edit_", "edit", "modify-",
    "modify_", "modify",
];

/// `re.match` ID-parameter patterns (case-sensitive).
static ID_PARAM_RES: Lazy<Vec<Regex>> = Lazy::new(|| {
    [r".*[iI][dD]$", r".*[iI][dD]s$", r"^id$", r".*_id$", r".*_ids$"]
        .iter()
        .map(|p| Regex::new(&format!("^(?:{p})")).expect("valid id pattern"))
        .collect()
});

/// Maps results from list tools to enable direct ID lookup for dependent tools.
#[derive(Debug, Clone, Default)]
pub struct MCPResultMapper {
    /// entity type -> (name or id -> id), insertion-ordered.
    entity_maps: HashMap<String, OrderedMap>,
    /// entity type -> first id seen.
    first_ids: HashMap<String, String>,
    /// entity type -> raw items (kept for parity with the Python mapper).
    #[allow(dead_code)]
    entity_items: HashMap<String, Vec<Value>>,
}

impl MCPResultMapper {
    /// Fields checked (in order) for an item's ID.
    pub const ID_FIELDS: &'static [&'static str] = &[
        "id", "_id", "uuid", "identifier", "key", "number", "uid", "guid", "Id", "ID", "UUID",
        "resourceId", "objectId", "entityId",
    ];

    /// Fields checked (in order) for an item's name/label.
    pub const NAME_FIELDS: &'static [&'static str] = &[
        "name", "title", "label", "displayName", "display_name", "key", "identifier", "slug",
        "handle", "username", "email", "Name", "Title",
    ];

    /// Create an empty mapper.
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract the entity type from a tool name (`list-teams` -> `team`, `getProjects` -> `project`).
    pub fn extract_entity_from_tool_name(&self, tool_name: &str) -> Option<String> {
        let tool_name_lower = tool_name.to_lowercase();

        for prefix in TOOL_PREFIXES {
            if let Some(rest) = tool_name_lower.strip_prefix(prefix) {
                let mut entity = rest.to_string();
                if entity.ends_with('s') && entity.chars().count() > 1 {
                    entity.pop();
                }
                return Some(entity);
            }
        }

        None
    }

    /// Extract entity type from a parameter name (`teamId` -> `team`, `issue_id` -> `issue`).
    pub fn extract_param_entity(&self, param_name: &str) -> Option<String> {
        let param_lower = param_name.to_lowercase();
        let suffixes = ["id", "ids", "_id", "_ids", "uuid"];

        for suffix in suffixes {
            if param_lower.ends_with(suffix) && param_lower.len() > suffix.len() {
                let entity = param_lower[..param_lower.len() - suffix.len()].replace(['_', '-'], "");
                return if entity.is_empty() { None } else { Some(entity) };
            }
        }

        let entity = param_lower.replace(['_', '-'], "");
        if entity.is_empty() {
            None
        } else {
            Some(entity)
        }
    }

    /// Process a list tool result and build the entity map; returns the map
    /// of names/ids to ids for this entity type (insertion-ordered).
    pub fn process_list_result(
        &mut self,
        tool_name: &str,
        result: &Value,
        entity_override: Option<&str>,
    ) -> Map<String, Value> {
        let entity_type = match entity_override {
            Some(e) if !e.is_empty() => Some(e.to_string()),
            _ => self.extract_entity_from_tool_name(tool_name),
        };

        let entity_type = match entity_type {
            Some(e) if !e.is_empty() => e,
            _ => {
                tracing::warn!(
                    target: LOG_TARGET,
                    "Could not determine entity type from tool: {}",
                    tool_name
                );
                return Map::new();
            }
        };

        tracing::info!(
            target: LOG_TARGET,
            "Processing result for {}, result type: {}",
            tool_name,
            py_type_name(result)
        );

        let items = self.extract_items_from_result(result);

        if items.is_empty() {
            let raw: String = py_str(result).chars().take(500).collect();
            tracing::warn!(
                target: LOG_TARGET,
                "No items found in result for {}. Raw result: {}",
                tool_name,
                raw
            );
            return Map::new();
        }

        let mut entity_map: OrderedMap = Map::new();
        let mut first_id: Option<String> = None;

        for item in &items {
            let Some(item) = item.as_object() else {
                continue;
            };

            let item_id = self.extract_id(item);
            let item_name = self.extract_name(item);

            if let Some(item_id) = item_id {
                if first_id.is_none() {
                    first_id = Some(item_id.clone());
                }

                if let Some(item_name) = item_name {
                    entity_map.insert(item_name.to_lowercase(), Value::String(item_id.clone()));
                    entity_map.insert(item_name, Value::String(item_id.clone()));
                }

                entity_map.insert(item_id.clone(), Value::String(item_id));
            }
        }

        self.entity_maps.insert(entity_type.clone(), entity_map.clone());
        self.entity_items.insert(entity_type.clone(), items);

        if let Some(first) = &first_id {
            self.first_ids.insert(entity_type.clone(), first.clone());
        }

        tracing::info!(
            target: LOG_TARGET,
            "Built entity map for '{}': {} entries, first_id={}",
            entity_type,
            entity_map.len(),
            first_id.as_deref().unwrap_or("None")
        );

        entity_map
    }

    /// Items parsed out of the first MCP text block (in `blocks`) whose JSON
    /// yields any; `None` when none does.
    fn items_from_text_blocks(&self, blocks: &[Value]) -> Option<Vec<Value>> {
        for block in blocks {
            let Some(obj) = block.as_object() else {
                continue;
            };
            if obj.get("type").and_then(Value::as_str) != Some("text") {
                continue;
            }
            let text = match obj.get("text") {
                Some(Value::String(s)) => s.as_str(),
                Some(_) => continue,
                None => "{}",
            };
            let Ok(text_data) = serde_json::from_str::<Value>(text) else {
                continue;
            };
            let nested_items = self.extract_items_from_result(&text_data);
            if !nested_items.is_empty() {
                return Some(nested_items);
            }
        }
        None
    }

    /// Extract the list of items from various result formats (plain lists,
    /// MCP content blocks, wrapper objects, a single object with an id).
    pub fn extract_items_from_result(&self, result: &Value) -> Vec<Value> {
        match result {
            Value::Array(list) => {
                let Some(first) = list.first() else {
                    return Vec::new();
                };

                let is_mcp_content = first
                    .as_object()
                    .map(|f| f.get("type").and_then(Value::as_str) == Some("text") && f.contains_key("text"))
                    .unwrap_or(false);

                if is_mcp_content {
                    return self.items_from_text_blocks(list).unwrap_or_default();
                }

                if list.iter().all(Value::is_object) {
                    return list.clone();
                }

                Vec::new()
            }
            Value::Object(map) => {
                // Handle MCP content format: {"content": [{"type": "text", "text": "..."}]}
                if let Some(Value::Array(content)) = map.get("content") {
                    if let Some(items) = self.items_from_text_blocks(content) {
                        return items;
                    }
                }

                // Check all fields that might contain a list of items
                for value in map.values() {
                    if let Value::Array(list) = value {
                        if list.first().map(Value::is_object).unwrap_or(false) {
                            return list.clone();
                        }
                    }

                    if value.is_object() {
                        let nested_items = self.extract_items_from_result(value);
                        if !nested_items.is_empty() {
                            return nested_items;
                        }
                    }
                }

                if self.extract_id(map).is_some() {
                    return vec![result.clone()];
                }

                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Extract ID from an item dictionary.
    pub fn extract_id(&self, item: &Map<String, Value>) -> Option<String> {
        for field in Self::ID_FIELDS {
            if let Some(Value::String(value)) = item.get(*field) {
                if !value.is_empty() {
                    return Some(self.normalize_id(value));
                }
            }
        }

        for (key, value) in item {
            if key.ends_with("Id") || key.ends_with("_id") {
                if let Value::String(value) = value {
                    if !value.is_empty() {
                        return Some(self.normalize_id(value));
                    }
                }
            }
        }

        None
    }

    /// Extract name/label from an item dictionary.
    pub fn extract_name(&self, item: &Map<String, Value>) -> Option<String> {
        for field in Self::NAME_FIELDS {
            if let Some(Value::String(value)) = item.get(*field) {
                if !value.is_empty() {
                    return Some(value.clone());
                }
            }
        }
        None
    }

    /// Normalize an ID value — converts 32-char hex strings to UUID format.
    pub fn normalize_id(&self, value: &str) -> String {
        normalize_uuid(value)
    }

    /// Get an ID for the given entity type: by name hint, then by a name
    /// mentioned in the user input, then the first ID seen.
    pub fn get_id(&self, entity_type: &str, name_hint: Option<&str>, user_input: Option<&str>) -> Option<String> {
        let entity_type_lower = entity_type.to_lowercase();
        let empty = Map::new();
        let entity_map = self.entity_maps.get(&entity_type_lower).unwrap_or(&empty);

        if let Some(name_hint) = name_hint.filter(|h| !h.is_empty()) {
            if let Some(id) = entity_map.get(name_hint) {
                return Some(py_str(id));
            }
            if let Some(id) = entity_map.get(&name_hint.to_lowercase()) {
                return Some(py_str(id));
            }
        }

        if let Some(user_input) = user_input.filter(|u| !u.is_empty()) {
            if !entity_map.is_empty() {
                let input_lower = user_input.to_lowercase();
                for (name, item_id) in entity_map {
                    if input_lower.contains(&name.to_lowercase()) {
                        let item_id = py_str(item_id);
                        tracing::info!(
                            target: LOG_TARGET,
                            "Found '{}' in user input, using ID: {}",
                            name,
                            item_id
                        );
                        return Some(item_id);
                    }
                }
            }
        }

        if let Some(first_id) = self.first_ids.get(&entity_type_lower) {
            tracing::info!(
                target: LOG_TARGET,
                "Using first ID for '{}': {}",
                entity_type,
                first_id
            );
            return Some(first_id.clone());
        }

        None
    }

    /// Get an ID for a parameter based on its name.
    pub fn get_id_for_param(&self, param_name: &str, user_input: Option<&str>) -> Option<String> {
        let entity_type = self.extract_param_entity(param_name)?;
        self.get_id(&entity_type, None, user_input)
    }

    /// Get resolved arguments for ID parameters from the map.
    pub fn get_resolved_arguments(
        &self,
        required_params: &[String],
        properties: &Map<String, Value>,
        user_input: Option<&str>,
    ) -> Map<String, Value> {
        let mut resolved = Map::new();
        let empty = Value::Object(Map::new());

        for param_name in required_params {
            let info = properties.get(param_name).unwrap_or(&empty);
            if self.is_id_param(param_name, info) {
                if let Some(resolved_id) = self.get_id_for_param(param_name, user_input) {
                    if !resolved_id.is_empty() {
                        tracing::info!(
                            target: LOG_TARGET,
                            "Resolved {} -> {}",
                            param_name,
                            resolved_id
                        );
                        resolved.insert(param_name.clone(), Value::String(resolved_id));
                    }
                }
            }
        }

        resolved
    }

    /// Check if a parameter is an ID-type parameter.
    pub fn is_id_param(&self, param_name: &str, param_info: &Value) -> bool {
        if ID_PARAM_RES.iter().any(|re| re.is_match(param_name)) {
            return true;
        }

        let description = str_field(param_info, "description").to_lowercase();
        let id_keywords = [
            "identifier", "uuid", "unique id", "object id", "id of", " id", "team id", "project id",
        ];
        id_keywords.iter().any(|kw| description.contains(kw))
    }

    /// Check if we have data for an entity type.
    pub fn has_entity(&self, entity_type: &str) -> bool {
        self.first_ids.contains_key(&entity_type.to_lowercase())
    }

    /// Clear all stored maps.
    pub fn clear(&mut self) {
        self.entity_maps.clear();
        self.first_ids.clear();
        self.entity_items.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn entity_names_from_tools_and_params() {
        let m = MCPResultMapper::new();
        assert_eq!(m.extract_entity_from_tool_name("list-teams").as_deref(), Some("team"));
        assert_eq!(m.extract_entity_from_tool_name("getProjects").as_deref(), Some("project"));
        assert_eq!(m.extract_entity_from_tool_name("teams"), None);
        assert_eq!(m.extract_param_entity("teamId").as_deref(), Some("team"));
        assert_eq!(m.extract_param_entity("issue_id").as_deref(), Some("issue"));
        assert_eq!(m.extract_param_entity("id").as_deref(), Some("id"));
    }

    #[test]
    fn process_and_resolve_ids() {
        let mut m = MCPResultMapper::new();
        let text = json!({"teams": [{"id": "t1", "name": "Eng"}, {"id": "t2", "name": "Design"}]});
        let result = json!({"content": [{"type": "text", "text": text.to_string()}]});
        let map = m.process_list_result("list_teams", &result, None);
        assert_eq!(map.get("eng"), Some(&json!("t1")));
        assert_eq!(map.get("Design"), Some(&json!("t2")));
        assert!(m.has_entity("team"));
        assert_eq!(m.get_id("team", None, Some("assign to design please")).as_deref(), Some("t2"));
        assert_eq!(m.get_id("team", None, None).as_deref(), Some("t1"));

        let mut props = Map::new();
        props.insert("teamId".into(), json!({"type": "string"}));
        props.insert("title".into(), json!({"type": "string"}));
        let resolved =
            m.get_resolved_arguments(&["teamId".into(), "title".into()], &props, Some("for Eng"));
        assert_eq!(Value::Object(resolved), json!({"teamId": "t1"}));

        m.clear();
        assert!(!m.has_entity("team"));
    }

    #[test]
    fn hex_ids_are_normalized() {
        let m = MCPResultMapper::new();
        let mut item = Map::new();
        item.insert("issueId".into(), json!("0123456789abcdef0123456789abcdef"));
        assert_eq!(
            m.extract_id(&item).as_deref(),
            Some("01234567-89ab-cdef-0123-456789abcdef")
        );
    }
}
